"""Independent raw F1 MCAP camera streams for image-quality operators."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np

from data_process import f1_schema, mcap
from data_process.cleaning import TimestampRepairResult, repair_camera_timestamps
from data_process.config import PipelineConfig
from data_process.quality_operator import EpisodeMetadata, FrameContext

HEAD_KEY = "observation.images.head"
LEFT_WRIST_KEY = "observation.images.left_wrist"
RIGHT_WRIST_KEY = "observation.images.right_wrist"
CAMERA_KEYS = (HEAD_KEY, LEFT_WRIST_KEY, RIGHT_WRIST_KEY)
CAMERA_TOPIC_BY_KEY = {
    HEAD_KEY: f1_schema.HEAD_IMAGE_TOPIC,
    LEFT_WRIST_KEY: f1_schema.LEFT_WRIST_IMAGE_TOPIC,
    RIGHT_WRIST_KEY: f1_schema.RIGHT_WRIST_IMAGE_TOPIC,
}


def _require_image_samples(episode: mcap.EpisodeData, topic: str) -> list[mcap.MessageSample]:
    samples = episode.topic(topic)
    if not samples:
        raise KeyError(f"Required camera topic is missing from {episode.root}: {topic}")
    for sample in samples:
        if sample.header_time_ns is None or sample.image is None:
            raise ValueError(
                f"Camera message cannot be decoded as sensor_msgs/CompressedImage: "
                f"topic={topic}, topic_index={sample.topic_index}"
            )
    return samples


class McapIndependentImageSource:
    """Yield every selected camera message independently in MCAP record order.

    No cross-camera matching is performed. Head-camera Header timestamps are
    repaired in memory only to provide a monotonic per-camera report timeline;
    wrist Header timestamps and all source MCAP bytes remain unchanged.
    """

    def __init__(
        self,
        episode_dir: Path,
        episode_index: int,
        *,
        pipeline_config: PipelineConfig,
        max_frames_per_camera: int | None = None,
    ) -> None:
        if max_frames_per_camera is not None and max_frames_per_camera < 1:
            raise ValueError("max_frames_per_camera must be positive")
        if pipeline_config.head_timestamp.topic != f1_schema.HEAD_IMAGE_TOPIC:
            raise ValueError(
                "cleaning.head_timestamp.topic must identify the F1 head camera: "
                f"expected={f1_schema.HEAD_IMAGE_TOPIC}, got={pipeline_config.head_timestamp.topic}"
            )

        self.episode = mcap.load_episode(episode_dir)
        self.episode_index = episode_index
        self._handles: dict[Path, BinaryIO] = {}
        self._samples = {key: _require_image_samples(self.episode, topic) for key, topic in CAMERA_TOPIC_BY_KEY.items()}
        self._head_repair = self._repair_head_timestamps(pipeline_config)
        self._analysis_timestamps = self._build_analysis_timestamps()
        self._selected_counts = {
            key: min(len(samples), max_frames_per_camera) if max_frames_per_camera is not None else len(samples)
            for key, samples in self._samples.items()
        }
        self._events = sorted(
            (
                (sample.record_index, key, source_index)
                for key, samples in self._samples.items()
                for source_index, sample in enumerate(samples[: self._selected_counts[key]])
            ),
            key=lambda item: item[0],
        )
        if not self._events:
            raise ValueError(f"No camera messages selected from {episode_dir}")

        source_details: dict[str, Any] = {
            "analysis_mode": "independent_per_camera",
            "cross_camera_matching": False,
            "time_source": "per-camera Header; head repaired in memory, wrists unchanged",
            "camera_topics": CAMERA_TOPIC_BY_KEY,
            "available_image_count": {key: len(samples) for key, samples in self._samples.items()},
            "analysis_image_count": self._selected_counts,
            "analysis_truncated_by_max_frames": {
                key: self._selected_counts[key] < len(self._samples[key]) for key in CAMERA_KEYS
            },
            "head_timestamp_repair": self._head_repair.diagnostics,
        }
        self.metadata = EpisodeMetadata(
            dataset_root=str(episode_dir.parent),
            episode_index=episode_index,
            frame_count=len(self._events),
            fps=float(1e9 / self._head_repair.nominal_period_ns),
            image_keys=CAMERA_KEYS,
            numeric_keys=(),
            feature_names={},
            episode_name=self.episode.name,
            source_type="raw_mcap_independent_images",
            source_files=tuple(str(path) for path in self.episode.source_files),
            source_details=source_details,
        )

    def _repair_head_timestamps(self, pipeline_config: PipelineConfig) -> TimestampRepairResult:
        timestamps = mcap.header_timestamps(self._samples[HEAD_KEY])
        if timestamps is None:
            raise ValueError(f"Head camera has missing Header timestamps: {self.episode.root}")
        return repair_camera_timestamps(timestamps, pipeline_config.head_timestamp)

    def _build_analysis_timestamps(self) -> dict[str, np.ndarray]:
        result = {HEAD_KEY: self._head_repair.corrected_ns}
        for key in (LEFT_WRIST_KEY, RIGHT_WRIST_KEY):
            timestamps = mcap.header_timestamps(self._samples[key])
            if timestamps is None:
                raise ValueError(f"Camera has missing Header timestamps: {CAMERA_TOPIC_BY_KEY[key]}")
            result[key] = timestamps
        return result

    def _open_files(self) -> None:
        if not self._handles:
            self._handles = {path: path.open("rb") for path in self.episode.source_files}

    def _decode(self, sample: mcap.MessageSample) -> np.ndarray:
        if sample.image is None:
            raise AssertionError("Validated image sample unexpectedly has no image reference")
        payload = mcap.read_image_bytes(sample.image, self._handles)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                f"OpenCV failed to decode image: topic={sample.topic}, topic_index={sample.topic_index}, "
                f"source={sample.source_path}"
            )
        return image

    def _provenance(self, key: str, source_index: int) -> dict[str, Any]:
        sample = self._samples[key][source_index]
        result: dict[str, Any] = {
            "topic": sample.topic,
            "source_file": str(sample.source_path),
            "topic_index": sample.topic_index,
            "record_index": sample.record_index,
            "sequence": sample.sequence,
            "header_time_ns": sample.header_time_ns,
            "analysis_time_ns": int(self._analysis_timestamps[key][source_index]),
            "publish_time_ns": sample.publish_time_ns,
            "log_time_ns": sample.log_time_ns,
        }
        if key == HEAD_KEY:
            result.update(
                {
                    "timestamp_repair_method": self._head_repair.methods[source_index],
                    "timestamp_risk": self._head_repair.risks[source_index],
                    "timestamp_correction_ms": (
                        int(self._head_repair.corrected_ns[source_index])
                        - int(self._head_repair.original_ns[source_index])
                    )
                    / 1e6,
                }
            )
        return result

    def frames(self) -> Iterator[FrameContext]:
        self._open_files()
        first_time_ns = {key: int(timestamps[0]) for key, timestamps in self._analysis_timestamps.items()}
        for _, key, source_index in self._events:
            analysis_time_ns = int(self._analysis_timestamps[key][source_index])
            yield FrameContext(
                episode_index=self.episode_index,
                frame_index=source_index,
                timestamp_s=(analysis_time_ns - first_time_ns[key]) / 1e9,
                images_bgr={key: self._decode(self._samples[key][source_index])},
                numeric={},
                provenance={key: self._provenance(key, source_index)},
            )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


# Compatibility alias: the source now yields independent camera messages.
McapExposureSource = McapIndependentImageSource
