"""Independent per-camera exposure operator using linear-light luminance."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data_process.quality_operator import EpisodeMetadata, FrameContext
from data_process.quality_operator_config import ExposureOperatorConfig


@dataclasses.dataclass(frozen=True)
class LuminanceMetrics:
    l50: float
    l95: float
    highlight_clipping_ratio: float


class IndependentCameraExposureOperator:
    """Detect overexposure independently in each camera stream."""

    name = "independent_camera_exposure"
    required_numeric_keys: frozenset[str] = frozenset()

    def __init__(
        self,
        camera_keys: Sequence[str],
        config: ExposureOperatorConfig,
        *,
        mask_output_dir: Path | None = None,
    ) -> None:
        if not camera_keys:
            raise ValueError("Exposure analysis requires at least one camera")
        self.camera_keys = tuple(camera_keys)
        self.required_image_keys = frozenset(camera_keys)
        self.config = config
        self.mask_output_dir = mask_output_dir
        srgb = np.arange(256, dtype=np.float32) / 255.0
        self._linear_lut = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        self._metadata: EpisodeMetadata | None = None
        self._rows: list[dict[str, Any]] = []
        self._mask_writers: dict[str, cv2.VideoWriter] = {}
        self._mask_paths: dict[str, Path] = {}

    def start(self, metadata: EpisodeMetadata) -> None:
        missing = self.required_image_keys.difference(metadata.image_keys)
        if missing:
            raise KeyError(f"Exposure operator is missing cameras: {sorted(missing)}")
        self._close_mask_writers()
        self._metadata = metadata
        self._rows = []
        self._mask_paths = {}
        if self.mask_output_dir is not None:
            self.mask_output_dir.mkdir(parents=True, exist_ok=True)

    def _luminance(self, image_bgr: np.ndarray, *, stride: int) -> np.ndarray:
        sampled = image_bgr[::stride, ::stride]
        blue, green, red = cv2.split(sampled)
        return 0.2126 * self._linear_lut[red] + 0.7152 * self._linear_lut[green] + 0.0722 * self._linear_lut[blue]

    def _metrics(self, image_bgr: np.ndarray) -> LuminanceMetrics:
        luminance = self._luminance(image_bgr, stride=self.config.pixel_stride)
        l50, l95 = np.percentile(luminance, [50, 95])
        return LuminanceMetrics(
            l50=float(l50),
            l95=float(l95),
            highlight_clipping_ratio=float(np.mean(luminance > self.config.clip_luminance)),
        )

    @staticmethod
    def _safe_name(camera: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", camera).strip("_")

    def _mask_writer(self, camera: str, image: np.ndarray) -> cv2.VideoWriter:
        writer = self._mask_writers.get(camera)
        if writer is not None:
            return writer
        if self.mask_output_dir is None or self._metadata is None:
            raise RuntimeError("Mask writer requested before exposure operator initialization")
        height, width = image.shape[:2]
        path = self.mask_output_dir / f"{self._safe_name(camera)}_overexposure_mask.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._metadata.fps,
            (width * 2, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot create exposure mask video: {path}")
        self._mask_writers[camera] = writer
        self._mask_paths[camera] = path
        return writer

    def _write_mask_frame(
        self,
        camera: str,
        image_bgr: np.ndarray,
        metrics: LuminanceMetrics,
        candidate: bool,
    ) -> None:
        if self.mask_output_dir is None:
            return
        luminance = self._luminance(image_bgr, stride=1)
        mask = luminance > self.config.clip_luminance
        overlay = image_bgr.copy()
        red = np.zeros_like(image_bgr)
        red[:, :, 2] = 255
        if np.any(mask):
            overlay[mask] = cv2.addWeighted(image_bgr[mask], 0.35, red[mask], 0.65, 0.0)
        mask_image = np.zeros_like(image_bgr)
        mask_image[mask] = (255, 255, 255)
        color = (0, 0, 255) if candidate else (0, 200, 0)
        label = (
            f"candidate={int(candidate)} L50={metrics.l50:.3f} L95={metrics.l95:.3f} "
            f"Rclip={metrics.highlight_clipping_ratio:.2%}"
        )
        cv2.putText(overlay, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(
            mask_image,
            f"Y > {self.config.clip_luminance:.2f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._mask_writer(camera, image_bgr).write(np.hstack([overlay, mask_image]))

    def process(self, frame: FrameContext) -> None:
        unknown = set(frame.images_bgr).difference(self.required_image_keys)
        if unknown:
            raise KeyError(f"Exposure operator received unknown cameras: {sorted(unknown)}")
        for key, image in frame.images_bgr.items():
            value = self._metrics(image)
            provenance = frame.provenance.get(key, {})
            broad_overexposure = (
                value.l50 >= self.config.overexposed_l50_threshold
                and value.l95 >= self.config.overexposed_l95_threshold
            )
            highlight_clipping = value.highlight_clipping_ratio >= self.config.clip_ratio_threshold
            candidate = broad_overexposure or highlight_clipping
            self._write_mask_frame(key, image, value, candidate)
            self._rows.append(
                {
                    "episode_index": frame.episode_index,
                    "frame_index": frame.frame_index,
                    "timestamp_s": frame.timestamp_s,
                    "camera": key,
                    "l50": value.l50,
                    "l95": value.l95,
                    "highlight_clipping_ratio": value.highlight_clipping_ratio,
                    "broad_overexposure_candidate": broad_overexposure,
                    "highlight_clipping_candidate": highlight_clipping,
                    "candidate": candidate,
                    "confirmed": False,
                    "source_topic": provenance.get("topic"),
                    "source_file": provenance.get("source_file"),
                    "source_topic_index": provenance.get("topic_index"),
                    "source_record_index": provenance.get("record_index"),
                    "source_sequence": provenance.get("sequence"),
                    "source_header_time_ns": provenance.get("header_time_ns"),
                    "source_analysis_time_ns": provenance.get("analysis_time_ns"),
                    "source_publish_time_ns": provenance.get("publish_time_ns"),
                    "source_log_time_ns": provenance.get("log_time_ns"),
                    "source_timestamp_repair_method": provenance.get("timestamp_repair_method"),
                    "source_timestamp_risk": provenance.get("timestamp_risk"),
                    "source_timestamp_correction_ms": provenance.get("timestamp_correction_ms"),
                }
            )

    def _confirm_segments(self) -> list[dict[str, Any]]:
        segments = []
        for camera in self.camera_keys:
            camera_rows = [row for row in self._rows if row["camera"] == camera]
            start = None
            previous = None
            for row in [*camera_rows, None]:
                is_contiguous = (
                    row is not None
                    and row["candidate"]
                    and (previous is None or row["frame_index"] == previous["frame_index"] + 1)
                )
                if is_contiguous:
                    if start is None:
                        start = row
                    previous = row
                    continue
                if start is not None and previous is not None:
                    confirmed_rows = [
                        item
                        for item in camera_rows
                        if start["frame_index"] <= item["frame_index"] <= previous["frame_index"]
                    ]
                    if len(confirmed_rows) >= self.config.minimum_consecutive_frames:
                        for item in confirmed_rows:
                            item["confirmed"] = True
                        segments.append(
                            {
                                "camera": camera,
                                "start_frame": start["frame_index"],
                                "end_frame": previous["frame_index"],
                                "start_time_s": start["timestamp_s"],
                                "end_time_s": previous["timestamp_s"],
                                "frame_count": len(confirmed_rows),
                                "broad_overexposure_frame_count": sum(
                                    item["broad_overexposure_candidate"] for item in confirmed_rows
                                ),
                                "highlight_clipping_frame_count": sum(
                                    item["highlight_clipping_candidate"] for item in confirmed_rows
                                ),
                                "max_l50": max(item["l50"] for item in confirmed_rows),
                                "max_l95": max(item["l95"] for item in confirmed_rows),
                                "max_highlight_clipping_ratio": max(
                                    item["highlight_clipping_ratio"] for item in confirmed_rows
                                ),
                                "start_source_topic_index": start["source_topic_index"],
                                "end_source_topic_index": previous["source_topic_index"],
                                "start_source_header_time_ns": start["source_header_time_ns"],
                                "end_source_header_time_ns": previous["source_header_time_ns"],
                                "start_source_log_time_ns": start["source_log_time_ns"],
                                "end_source_log_time_ns": previous["source_log_time_ns"],
                            }
                        )
                start = row if row is not None and row["candidate"] else None
                previous = row if row is not None and row["candidate"] else None
        return segments

    @staticmethod
    def _distribution(values: np.ndarray) -> dict[str, float]:
        percentiles = np.percentile(values, [0, 5, 50, 95, 99, 100])
        return dict(zip(("min", "p05", "p50", "p95", "p99", "max"), (float(x) for x in percentiles)))

    def _close_mask_writers(self) -> None:
        for writer in self._mask_writers.values():
            writer.release()
        self._mask_writers.clear()

    def finish(self) -> dict[str, Any]:
        if self._metadata is None:
            raise RuntimeError("Exposure operator was not started")
        self._close_mask_writers()
        segments = self._confirm_segments()
        camera_summary = {}
        for camera in self.camera_keys:
            rows = [row for row in self._rows if row["camera"] == camera]
            camera_summary[camera] = {
                key: self._distribution(np.asarray([row[key] for row in rows], dtype=np.float64))
                for key in ("l50", "l95", "highlight_clipping_ratio")
            }
            camera_summary[camera]["analyzed_frames"] = len(rows)
            camera_summary[camera]["candidate_frames"] = sum(row["candidate"] for row in rows)
            camera_summary[camera]["confirmed_frames"] = sum(row["confirmed"] for row in rows)
        return {
            "status": "warning" if segments else "pass",
            "method": {
                "comparison_mode": "independent absolute thresholds per camera; no cross-camera baseline",
                "input_transfer_function": "sRGB IEC 61966-2-1 inverse EOTF",
                "luminance": "Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear",
                "broad_overexposure_definition": (
                    f"L50 >= {self.config.overexposed_l50_threshold} and L95 >= {self.config.overexposed_l95_threshold}"
                ),
                "highlight_clipping_definition": (
                    f"ratio(Y > {self.config.clip_luminance}) >= {self.config.clip_ratio_threshold}"
                ),
                "pixel_stride": self.config.pixel_stride,
            },
            "thresholds": dataclasses.asdict(self.config),
            "camera_summary": camera_summary,
            "confirmed_segment_count": len(segments),
            "confirmed_segments": segments,
            "mask_visualizations": {
                "layout": "left=red clipping overlay, right=binary clipping mask",
                "mask_definition": f"Y > {self.config.clip_luminance}",
                "videos": {camera: str(path) for camera, path in self._mask_paths.items()},
            },
            "frames": self._rows,
        }


# Temporary import compatibility for downstream code; behavior is now independent per camera.
CrossCameraExposureOperator = IndependentCameraExposureOperator
