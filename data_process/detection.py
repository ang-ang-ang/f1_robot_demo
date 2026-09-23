"""Reusable object-detection pipeline components for F1 camera data."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class VideoMetadata:
    source: str
    fps: float
    width: int
    height: int
    frame_count: int | None


@dataclasses.dataclass(frozen=True)
class FramePacket:
    index: int
    timestamp_s: float
    image_bgr: np.ndarray


@dataclasses.dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.xyxy
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": [x1, y1, x2, y2],
        }


@dataclasses.dataclass(frozen=True)
class FrameDetections:
    frame_index: int
    timestamp_s: float
    detections: tuple[Detection, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "detections": [item.as_dict() for item in self.detections],
        }


class FrameSource(Protocol):
    metadata: VideoMetadata

    def frames(self) -> Iterator[FramePacket]:
        """Yield frame packets in timestamp order."""

    def close(self) -> None:
        """Release any source resources."""


class DetectionSink(Protocol):
    def open(self, metadata: VideoMetadata) -> None:
        """Prepare sink outputs."""

    def write(self, frame: FramePacket, detections: FrameDetections, annotated_bgr: np.ndarray) -> None:
        """Consume one processed frame."""

    def close(self) -> None:
        """Finalize outputs."""


class VideoFrameSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise FileNotFoundError(f"无法打开输入视频：{path}")

        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        raw_frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.metadata = VideoMetadata(
            source=str(path),
            fps=fps if fps > 0 else 20.0,
            width=width,
            height=height,
            frame_count=raw_frame_count if raw_frame_count > 0 else None,
        )

    def frames(self) -> Iterator[FramePacket]:
        index = 0
        fps = self.metadata.fps if self.metadata.fps > 0 else 20.0
        while True:
            ok, image_bgr = self._capture.read()
            if not ok:
                break
            timestamp_s = index / fps
            yield FramePacket(index=index, timestamp_s=timestamp_s, image_bgr=image_bgr)
            index += 1

    def close(self) -> None:
        self._capture.release()


class UltralyticsYOLOEDetector:
    def __init__(
        self,
        *,
        model_name: str,
        class_prompts: list[str],
        confidence: float,
        iou: float,
        device: str | None,
    ) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as error:
            raise ImportError(
                "未找到 ultralytics。请先进入 conda 的 yoloe 环境，再运行该检测脚本。"
            ) from error

        self.class_prompts = class_prompts
        self._model = YOLOE(model_name)
        self._model.set_classes(class_prompts)
        self._confidence = confidence
        self._iou = iou
        self._device = device

    def detect(self, frame: FramePacket) -> tuple[FrameDetections, np.ndarray]:
        predict_kwargs = {
            "source": frame.image_bgr,
            "conf": self._confidence,
            "iou": self._iou,
            "verbose": False,
        }
        if self._device is not None:
            predict_kwargs["device"] = self._device
        results = self._model.predict(
            **predict_kwargs,
        )
        result = results[0]
        names = result.names or {}
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy_values = boxes.xyxy.detach().cpu().tolist()
            confidence_values = boxes.conf.detach().cpu().tolist()
            class_indices = boxes.cls.detach().cpu().tolist()
            for xyxy, confidence, class_index in zip(xyxy_values, confidence_values, class_indices, strict=True):
                label = _resolve_label(names, int(class_index))
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(confidence),
                        xyxy=tuple(float(value) for value in xyxy),
                    )
                )

        return (
            FrameDetections(
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                detections=tuple(detections),
            ),
            result.plot(),
        )


class AnnotatedVideoSink:
    def __init__(self, output_path: Path, fourcc: str = "mp4v") -> None:
        self.output_path = output_path
        self.fourcc = fourcc
        self._fps = 20.0
        self._frame_size: tuple[int, int] | None = None
        self._writer: cv2.VideoWriter | None = None

    def open(self, metadata: VideoMetadata) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = metadata.fps if metadata.fps > 0 else 20.0

    def write(self, frame: FramePacket, detections: FrameDetections, annotated_bgr: np.ndarray) -> None:
        del frame, detections
        height, width = annotated_bgr.shape[:2]
        frame_size = (width, height)
        if self._writer is None:
            writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*self.fourcc),
                self._fps,
                frame_size,
            )
            if not writer.isOpened():
                raise RuntimeError(f"无法创建输出视频：{self.output_path}")
            self._writer = writer
            self._frame_size = frame_size
        elif frame_size != self._frame_size:
            raise ValueError(f"标注后视频帧尺寸发生变化：{frame_size} != {self._frame_size}")
        self._writer.write(annotated_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class JsonReportSink:
    def __init__(self, output_path: Path, *, source_labels: list[str]) -> None:
        self.output_path = output_path
        self.source_labels = source_labels
        self._metadata: VideoMetadata | None = None
        self._frames: list[dict[str, Any]] = []
        self._class_counts: Counter[str] = Counter()

    def open(self, metadata: VideoMetadata) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = metadata

    def write(self, frame: FramePacket, detections: FrameDetections, annotated_bgr: np.ndarray) -> None:
        del frame, annotated_bgr
        self._frames.append(detections.as_dict())
        self._class_counts.update(item.label for item in detections.detections)

    def close(self) -> None:
        if self._metadata is None:
            return
        report = {
            "format": "f1_detection_report_v1",
            "source": dataclasses.asdict(self._metadata),
            "class_prompts": list(self.source_labels),
            "summary": {
                "frame_count": len(self._frames),
                "detected_class_counts": dict(self._class_counts),
                "frames_with_detections": sum(bool(frame["detections"]) for frame in self._frames),
            },
            "frames": self._frames,
        }
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


def run_detection_pipeline(
    source: FrameSource,
    detector: UltralyticsYOLOEDetector,
    sinks: list[DetectionSink],
    *,
    max_frames: int | None = None,
) -> dict[str, Any]:
    opened_sinks: list[DetectionSink] = []
    processed_frames = 0
    try:
        for sink in sinks:
            sink.open(source.metadata)
            opened_sinks.append(sink)
        for frame in source.frames():
            if max_frames is not None and processed_frames >= max_frames:
                break
            detections, annotated_bgr = detector.detect(frame)
            for sink in sinks:
                sink.write(frame, detections, annotated_bgr)
            processed_frames += 1
    finally:
        source.close()
        for sink in reversed(opened_sinks):
            sink.close()

    return {
        "source": dataclasses.asdict(source.metadata),
        "processed_frames": processed_frames,
    }


def default_output_paths(input_video: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = input_video.stem or "detected"
    return (
        output_dir / f"{stem}_detected.mp4",
        output_dir / f"{stem}_detection_report.json",
    )


def _resolve_label(names: Any, class_index: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_index, class_index))
    if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)
