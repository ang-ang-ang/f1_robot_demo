"""Shared streaming interfaces and a LeRobot source for quality operators."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pyarrow.parquet as pq


@dataclasses.dataclass(frozen=True)
class EpisodeMetadata:
    dataset_root: str
    episode_index: int
    frame_count: int
    fps: float
    image_keys: tuple[str, ...]
    numeric_keys: tuple[str, ...]
    feature_names: Mapping[str, tuple[str, ...]]


@dataclasses.dataclass(frozen=True)
class FrameContext:
    episode_index: int
    frame_index: int
    timestamp_s: float
    images_bgr: Mapping[str, np.ndarray]
    numeric: Mapping[str, np.ndarray]


class StreamingQualityOperator(Protocol):
    name: str
    required_image_keys: frozenset[str]
    required_numeric_keys: frozenset[str]

    def start(self, metadata: EpisodeMetadata) -> None:
        """Initialize one episode."""

    def process(self, frame: FrameContext) -> None:
        """Consume one synchronized frame."""

    def finish(self) -> dict[str, Any]:
        """Finalize and return the episode-level report."""


class LeRobotEpisodeSource:
    """Adapt one LeRobot v2 episode to synchronized streaming frames."""

    def __init__(
        self,
        dataset_root: Path,
        episode_index: int,
        *,
        image_keys: Sequence[str],
        numeric_keys: Sequence[str],
        max_frames: int | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.episode_index = episode_index
        self.image_keys = tuple(sorted(set(image_keys)))
        self.numeric_keys = tuple(sorted(set(numeric_keys)))
        self.max_frames = max_frames
        self._captures: dict[str, cv2.VideoCapture] = {}

        info_path = dataset_root / "meta/info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
        self._info = json.loads(info_path.read_text(encoding="utf-8"))
        self._validate_keys()
        self._table = self._load_table()
        self._timestamps = np.asarray(self._table["timestamp"].to_numpy(), dtype=np.float64)
        self._numeric = {key: np.asarray(self._table[key].to_pylist(), dtype=np.float64) for key in self.numeric_keys}
        self._frame_count = len(self._timestamps)
        if self.max_frames is not None:
            if self.max_frames < 1:
                raise ValueError("max_frames must be positive")
            self._frame_count = min(self._frame_count, self.max_frames)

        feature_names = {key: self._feature_names(key) for key in self.numeric_keys}
        self.metadata = EpisodeMetadata(
            dataset_root=str(dataset_root),
            episode_index=episode_index,
            frame_count=self._frame_count,
            fps=float(self._info["fps"]),
            image_keys=self.image_keys,
            numeric_keys=self.numeric_keys,
            feature_names=feature_names,
        )

    def _validate_keys(self) -> None:
        features = self._info.get("features", {})
        missing = [key for key in (*self.image_keys, *self.numeric_keys) if key not in features]
        if missing:
            raise KeyError(f"LeRobot features are missing required keys: {missing}")
        invalid_images = [key for key in self.image_keys if features[key].get("dtype") != "video"]
        if invalid_images:
            raise ValueError(f"Image features are not video-backed: {invalid_images}")

    def _episode_path(self, template_key: str) -> Path:
        template = self._info[template_key]
        relative = template.format(
            episode_chunk=self.episode_index // int(self._info["chunks_size"]),
            episode_index=self.episode_index,
            video_key="{video_key}",
        )
        return self.dataset_root / relative

    def _load_table(self):
        path = self._episode_path("data_path")
        if not path.exists():
            raise FileNotFoundError(f"LeRobot episode parquet not found: {path}")
        columns = ["timestamp", "frame_index", *self.numeric_keys]
        table = pq.read_table(path, columns=columns)
        frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        expected = np.arange(len(frame_indices), dtype=np.int64)
        if not np.array_equal(frame_indices, expected):
            raise ValueError(f"Non-contiguous frame_index in {path}")
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Non-increasing timestamps in {path}")
        return table

    def _feature_names(self, key: str) -> tuple[str, ...]:
        names = self._info["features"][key].get("names")
        if names and isinstance(names[0], list):
            names = names[0]
        if not names:
            width = len(self._table[key][0].as_py())
            return tuple(f"{key}[{index}]" for index in range(width))
        return tuple(str(name) for name in names)

    def _open_videos(self) -> None:
        video_template = self._episode_path("video_path")
        for key in self.image_keys:
            path = Path(str(video_template).format(video_key=key))
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                self.close()
                raise FileNotFoundError(f"Cannot open LeRobot video: {path}")
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            parquet_frames = len(self._timestamps)
            if video_frames and video_frames != parquet_frames:
                self.close()
                raise ValueError(f"Frame count mismatch for {path}: video={video_frames}, parquet={parquet_frames}")
            if video_fps and not np.isclose(video_fps, self.metadata.fps, rtol=0.0, atol=1e-3):
                self.close()
                raise ValueError(f"FPS mismatch for {path}: video={video_fps}, metadata={self.metadata.fps}")
            self._captures[key] = capture

    def frames(self) -> Iterator[FrameContext]:
        self._open_videos()
        for index in range(self._frame_count):
            images = {}
            for key, capture in self._captures.items():
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Video ended before frame {index}: {key}, episode={self.episode_index}")
                images[key] = image
            numeric = {key: values[index] for key, values in self._numeric.items()}
            yield FrameContext(
                episode_index=self.episode_index,
                frame_index=index,
                timestamp_s=float(self._timestamps[index]),
                images_bgr=images,
                numeric=numeric,
            )

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


def run_episode_operators(
    dataset_root: Path,
    episode_index: int,
    operators: Sequence[StreamingQualityOperator],
    *,
    max_frames: int | None = None,
) -> dict[str, Any]:
    image_keys = set().union(*(operator.required_image_keys for operator in operators))
    numeric_keys = set().union(*(operator.required_numeric_keys for operator in operators))
    source = LeRobotEpisodeSource(
        dataset_root,
        episode_index,
        image_keys=sorted(image_keys),
        numeric_keys=sorted(numeric_keys),
        max_frames=max_frames,
    )
    for operator in operators:
        operator.start(source.metadata)
    try:
        for frame in source.frames():
            for operator in operators:
                operator.process(frame)
    finally:
        source.close()
    return {
        "episode_index": episode_index,
        "frame_count": source.metadata.frame_count,
        "fps": source.metadata.fps,
        "operators": {operator.name: operator.finish() for operator in operators},
    }
