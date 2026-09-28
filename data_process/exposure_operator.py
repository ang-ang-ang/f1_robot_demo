"""Cross-camera exposure consistency operator using linear-light luminance."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
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


class CrossCameraExposureOperator:
    name = "cross_camera_exposure"
    required_numeric_keys: frozenset[str] = frozenset()

    def __init__(self, camera_keys: Sequence[str], config: ExposureOperatorConfig) -> None:
        if len(camera_keys) < 3:
            raise ValueError("Cross-camera exposure comparison requires at least three cameras")
        self.camera_keys = tuple(camera_keys)
        self.required_image_keys = frozenset(camera_keys)
        self.config = config
        srgb = np.arange(256, dtype=np.float32) / 255.0
        self._linear_lut = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        self._metadata: EpisodeMetadata | None = None
        self._rows: list[dict[str, Any]] = []

    def start(self, metadata: EpisodeMetadata) -> None:
        missing = self.required_image_keys.difference(metadata.image_keys)
        if missing:
            raise KeyError(f"Exposure operator is missing cameras: {sorted(missing)}")
        self._metadata = metadata
        self._rows = []

    def _metrics(self, image_bgr: np.ndarray) -> LuminanceMetrics:
        stride = self.config.pixel_stride
        sampled = image_bgr[::stride, ::stride]
        blue, green, red = cv2.split(sampled)
        luminance = 0.2126 * self._linear_lut[red] + 0.7152 * self._linear_lut[green] + 0.0722 * self._linear_lut[blue]
        l50, l95 = np.percentile(luminance, [50, 95])
        return LuminanceMetrics(
            l50=float(l50),
            l95=float(l95),
            highlight_clipping_ratio=float(np.mean(luminance > self.config.clip_luminance)),
        )

    def process(self, frame: FrameContext) -> None:
        metrics = {key: self._metrics(frame.images_bgr[key]) for key in self.camera_keys}
        for key, value in metrics.items():
            peers = [dataclasses.astuple(metrics[peer]) for peer in self.camera_keys if peer != key]
            peer_baseline = np.median(np.asarray(peers, dtype=np.float64), axis=0)
            baseline_l50, baseline_l95, baseline_clip = (float(item) for item in peer_baseline)
            l50_ratio = (value.l50 + 0.01) / (baseline_l50 + 0.01)
            l50_delta = value.l50 - baseline_l50
            l95_delta = value.l95 - baseline_l95
            clip_excess = value.highlight_clipping_ratio - baseline_clip
            broad_brightness = (
                l50_ratio >= self.config.l50_ratio_threshold
                and l50_delta >= self.config.l50_delta_threshold
                and l95_delta >= self.config.l95_delta_threshold
            )
            highlight_clipping = (
                value.highlight_clipping_ratio >= self.config.clip_ratio_threshold
                and clip_excess >= self.config.clip_excess_threshold
                and value.l95 >= self.config.clip_luminance
            )
            self._rows.append(
                {
                    "episode_index": frame.episode_index,
                    "frame_index": frame.frame_index,
                    "timestamp_s": frame.timestamp_s,
                    "camera": key,
                    "l50": value.l50,
                    "l95": value.l95,
                    "highlight_clipping_ratio": value.highlight_clipping_ratio,
                    "peer_l50": baseline_l50,
                    "peer_l95": baseline_l95,
                    "peer_highlight_clipping_ratio": baseline_clip,
                    "l50_ratio": l50_ratio,
                    "l50_delta": l50_delta,
                    "l95_delta": l95_delta,
                    "clip_excess": clip_excess,
                    "broad_brightness_candidate": broad_brightness,
                    "highlight_clipping_candidate": highlight_clipping,
                    "candidate": broad_brightness or highlight_clipping,
                    "confirmed": False,
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
                    length = previous["frame_index"] - start["frame_index"] + 1
                    if length >= self.config.minimum_consecutive_frames:
                        confirmed_rows = [
                            item
                            for item in camera_rows
                            if start["frame_index"] <= item["frame_index"] <= previous["frame_index"]
                        ]
                        for item in confirmed_rows:
                            item["confirmed"] = True
                        segments.append(
                            {
                                "camera": camera,
                                "start_frame": start["frame_index"],
                                "end_frame": previous["frame_index"],
                                "start_time_s": start["timestamp_s"],
                                "end_time_s": previous["timestamp_s"],
                                "frame_count": length,
                                "max_l50_ratio": max(item["l50_ratio"] for item in confirmed_rows),
                                "max_l50_delta": max(item["l50_delta"] for item in confirmed_rows),
                                "max_l95_delta": max(item["l95_delta"] for item in confirmed_rows),
                                "max_highlight_clipping_ratio": max(
                                    item["highlight_clipping_ratio"] for item in confirmed_rows
                                ),
                            }
                        )
                start = row if row is not None and row["candidate"] else None
                previous = row if row is not None and row["candidate"] else None
        return segments

    @staticmethod
    def _distribution(values: np.ndarray) -> dict[str, float]:
        percentiles = np.percentile(values, [0, 5, 50, 95, 99, 100])
        return dict(zip(("min", "p05", "p50", "p95", "p99", "max"), (float(x) for x in percentiles)))

    def finish(self) -> dict[str, Any]:
        if self._metadata is None:
            raise RuntimeError("Exposure operator was not started")
        segments = self._confirm_segments()
        camera_summary = {}
        for camera in self.camera_keys:
            rows = [row for row in self._rows if row["camera"] == camera]
            camera_summary[camera] = {
                key: self._distribution(np.asarray([row[key] for row in rows], dtype=np.float64))
                for key in ("l50", "l95", "highlight_clipping_ratio")
            }
            camera_summary[camera]["candidate_frames"] = sum(row["candidate"] for row in rows)
            camera_summary[camera]["confirmed_frames"] = sum(row["confirmed"] for row in rows)
        return {
            "status": "warning" if segments else "pass",
            "method": {
                "input_transfer_function": "sRGB IEC 61966-2-1 inverse EOTF",
                "luminance": "Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear",
                "clip_definition": f"Y > {self.config.clip_luminance}",
                "pixel_stride": self.config.pixel_stride,
            },
            "thresholds": dataclasses.asdict(self.config),
            "camera_summary": camera_summary,
            "confirmed_segment_count": len(segments),
            "confirmed_segments": segments,
            "frames": self._rows,
        }
