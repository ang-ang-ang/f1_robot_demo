"""Joint position, velocity, acceleration, discontinuity, and spike operator."""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from data_process.quality_operator import EpisodeMetadata, FrameContext
from data_process.quality_operator_config import JointMotionOperatorConfig, JointSignalConfig


@dataclasses.dataclass(frozen=True)
class JointLimits:
    lower_rad: np.ndarray
    upper_rad: np.ndarray
    max_velocity_rad_s: np.ndarray
    position_source: str
    velocity_source: str


def _canonical_joint_name(name: str) -> str:
    lowered = name.lower()
    for side in ("l", "r"):
        for index in range(1, 8):
            if lowered in {f"arm_{side}_j{index}", f"arm_{side}_joint{index}"}:
                return f"arm_{side}_joint{index}"
    return lowered


def load_joint_limits_from_urdf(
    path: Path | None,
    joint_names: tuple[str, ...],
    *,
    fallback_max_velocity_deg_s: float,
) -> JointLimits:
    count = len(joint_names)
    lower = np.full(count, -np.inf, dtype=np.float64)
    upper = np.full(count, np.inf, dtype=np.float64)
    velocity = np.full(count, np.deg2rad(fallback_max_velocity_deg_s), dtype=np.float64)
    position_source = "disabled: no URDF supplied"
    velocity_source = f"provisional fallback {fallback_max_velocity_deg_s:.3f} deg/s"
    if path is None:
        return JointLimits(lower, upper, velocity, position_source, velocity_source)
    if not path.exists():
        raise FileNotFoundError(f"Joint-limit URDF not found: {path}")

    limits_by_name = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        limits_by_name[_canonical_joint_name(joint.attrib["name"])] = limit.attrib
    missing = []
    valid_velocity_count = 0
    for index, name in enumerate(joint_names):
        attributes = limits_by_name.get(_canonical_joint_name(name))
        if attributes is None:
            missing.append(name)
            continue
        if "lower" in attributes and "upper" in attributes:
            lower[index] = float(attributes["lower"])
            upper[index] = float(attributes["upper"])
        urdf_velocity = float(attributes.get("velocity", 0.0))
        if urdf_velocity > 0:
            velocity[index] = urdf_velocity
            valid_velocity_count += 1
    if missing:
        raise ValueError(f"URDF does not define limits for joints: {missing}")
    position_source = str(path)
    if valid_velocity_count == count:
        velocity_source = str(path)
    elif valid_velocity_count:
        velocity_source = f"URDF where positive; fallback {fallback_max_velocity_deg_s:.3f} deg/s elsewhere"
    return JointLimits(lower, upper, velocity, position_source, velocity_source)


class JointMotionOperator:
    required_image_keys: frozenset[str] = frozenset()

    def __init__(
        self,
        feature_key: str,
        signal_config: JointSignalConfig,
        config: JointMotionOperatorConfig,
        *,
        joint_limit_urdf: Path | None,
        arm_dimensions: int = 14,
    ) -> None:
        self.feature_key = feature_key
        self.name = "joint_motion_" + feature_key.replace("observation.", "").replace(".", "_")
        self.required_numeric_keys = frozenset({feature_key})
        self.signal_config = signal_config
        self.config = config
        self.joint_limit_urdf = joint_limit_urdf
        self.arm_dimensions = arm_dimensions
        self._metadata: EpisodeMetadata | None = None
        self._names: tuple[str, ...] = ()
        self._limits: JointLimits | None = None
        self._timestamps: list[float] = []
        self._positions: list[np.ndarray] = []

    def start(self, metadata: EpisodeMetadata) -> None:
        names = metadata.feature_names[self.feature_key][: self.arm_dimensions]
        if len(names) != self.arm_dimensions:
            raise ValueError(f"{self.feature_key} has {len(names)} arm dimensions, expected {self.arm_dimensions}")
        self._metadata = metadata
        self._names = names
        self._limits = load_joint_limits_from_urdf(
            self.joint_limit_urdf,
            names,
            fallback_max_velocity_deg_s=self.signal_config.fallback_max_velocity_deg_s,
        )
        self._timestamps = []
        self._positions = []

    def process(self, frame: FrameContext) -> None:
        values = np.asarray(frame.numeric[self.feature_key], dtype=np.float64)
        arm = values[: self.arm_dimensions]
        if not np.all(np.isfinite(arm)):
            raise ValueError(f"Non-finite values in {self.feature_key} at frame {frame.frame_index}")
        self._timestamps.append(frame.timestamp_s)
        self._positions.append(arm.copy())

    @staticmethod
    def _quantiles(values: np.ndarray) -> dict[str, float]:
        quantiles = np.percentile(values, [50, 95, 99, 99.9, 100])
        return dict(zip(("p50", "p95", "p99", "p99_9", "max"), (float(value) for value in quantiles)))

    def _event(
        self,
        kind: str,
        severity: str,
        frame_index: int,
        joint_index: int,
        value: float,
        threshold: float,
        unit: str,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "severity": severity,
            "frame_index": frame_index,
            "timestamp_s": self._timestamps[frame_index],
            "joint_index": joint_index,
            "joint_name": self._names[joint_index],
            "value": value,
            "threshold": threshold,
            "unit": unit,
        }

    def finish(self) -> dict[str, Any]:
        if self._metadata is None or self._limits is None:
            raise RuntimeError("Joint motion operator was not started")
        positions = np.asarray(self._positions, dtype=np.float64)
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        if len(positions) < 3:
            raise ValueError(f"{self.feature_key} requires at least three frames")
        dt = np.diff(timestamps)
        if np.any(dt <= 0):
            raise ValueError(f"Non-positive timestamp step in {self.feature_key}")
        steps = np.diff(positions, axis=0)
        velocity = steps / dt[:, None]
        acceleration_dt = (dt[:-1] + dt[1:]) / 2.0
        acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
        absolute_steps_deg = np.rad2deg(np.abs(steps))
        absolute_velocity_deg_s = np.rad2deg(np.abs(velocity))
        absolute_acceleration_deg_s2 = np.rad2deg(np.abs(acceleration))
        max_velocity_deg_s = np.rad2deg(self._limits.max_velocity_rad_s)

        events = []
        tolerance_rad = np.deg2rad(self.config.position_limit_tolerance_deg)
        for frame_index, joint_index in np.argwhere(
            (positions < self._limits.lower_rad[None, :] - tolerance_rad)
            | (positions > self._limits.upper_rad[None, :] + tolerance_rad)
        ):
            lower = np.rad2deg(self._limits.lower_rad[joint_index])
            upper = np.rad2deg(self._limits.upper_rad[joint_index])
            events.append(
                self._event(
                    "position_limit",
                    "fail",
                    int(frame_index),
                    int(joint_index),
                    float(np.rad2deg(positions[frame_index, joint_index])),
                    float(
                        lower if positions[frame_index, joint_index] < self._limits.lower_rad[joint_index] else upper
                    ),
                    "deg",
                )
            )

        for step_index, joint_index in np.argwhere(absolute_steps_deg > self.signal_config.max_step_deg):
            events.append(
                self._event(
                    "step_discontinuity",
                    "warning",
                    int(step_index + 1),
                    int(joint_index),
                    float(absolute_steps_deg[step_index, joint_index]),
                    self.signal_config.max_step_deg,
                    "deg",
                )
            )

        velocity_warning = max_velocity_deg_s * self.config.velocity_warning_ratio
        for step_index, joint_index in np.argwhere(absolute_velocity_deg_s > velocity_warning[None, :]):
            value = float(absolute_velocity_deg_s[step_index, joint_index])
            maximum = float(max_velocity_deg_s[joint_index])
            events.append(
                self._event(
                    "velocity_limit",
                    "fail" if value > maximum else "warning",
                    int(step_index + 1),
                    int(joint_index),
                    value,
                    maximum if value > maximum else float(velocity_warning[joint_index]),
                    "deg/s",
                )
            )

        acceleration_warning = self.signal_config.max_acceleration_deg_s2 * self.config.acceleration_warning_ratio
        for acceleration_index, joint_index in np.argwhere(absolute_acceleration_deg_s2 > acceleration_warning):
            value = float(absolute_acceleration_deg_s2[acceleration_index, joint_index])
            maximum = self.signal_config.max_acceleration_deg_s2
            events.append(
                self._event(
                    "acceleration_limit",
                    "fail" if value > maximum else "warning",
                    int(acceleration_index + 2),
                    int(joint_index),
                    value,
                    maximum if value > maximum else acceleration_warning,
                    "deg/s^2",
                )
            )

        entry_deg = np.rad2deg(np.abs(positions[1:-1] - positions[:-2]))
        exit_deg = np.rad2deg(np.abs(positions[2:] - positions[1:-1]))
        recovery_deg = np.rad2deg(np.abs(positions[2:] - positions[:-2]))
        spike_mask = (
            (entry_deg >= self.config.spike_min_step_deg)
            & (exit_deg >= self.config.spike_min_step_deg)
            & (recovery_deg <= self.config.spike_recovery_deg)
        )
        for center_index, joint_index in np.argwhere(spike_mask):
            frame_index = int(center_index + 1)
            events.append(
                {
                    **self._event(
                        "single_frame_spike",
                        "fail",
                        frame_index,
                        int(joint_index),
                        float(max(entry_deg[center_index, joint_index], exit_deg[center_index, joint_index])),
                        self.config.spike_min_step_deg,
                        "deg",
                    ),
                    "recovery_error_deg": float(recovery_deg[center_index, joint_index]),
                }
            )

        events.sort(key=lambda item: (item["frame_index"], item["joint_index"], item["kind"]))
        per_joint = []
        for index, name in enumerate(self._names):
            per_joint.append(
                {
                    "joint_index": index,
                    "joint_name": name,
                    "position_deg": {
                        "min": float(np.rad2deg(positions[:, index].min())),
                        "max": float(np.rad2deg(positions[:, index].max())),
                        "limit_lower": (
                            float(np.rad2deg(self._limits.lower_rad[index]))
                            if np.isfinite(self._limits.lower_rad[index])
                            else None
                        ),
                        "limit_upper": (
                            float(np.rad2deg(self._limits.upper_rad[index]))
                            if np.isfinite(self._limits.upper_rad[index])
                            else None
                        ),
                    },
                    "absolute_step_deg": self._quantiles(absolute_steps_deg[:, index]),
                    "absolute_velocity_deg_s": self._quantiles(absolute_velocity_deg_s[:, index]),
                    "absolute_acceleration_deg_s2": self._quantiles(absolute_acceleration_deg_s2[:, index]),
                    "configured_max_velocity_deg_s": float(max_velocity_deg_s[index]),
                    "configured_max_acceleration_deg_s2": self.signal_config.max_acceleration_deg_s2,
                }
            )
        fail_count = sum(event["severity"] == "fail" for event in events)
        warning_count = sum(event["severity"] == "warning" for event in events)
        return {
            "status": "fail" if fail_count else "warning" if warning_count else "pass",
            "feature_key": self.feature_key,
            "units": "input radians; report degrees",
            "limit_sources": {
                "position": self._limits.position_source,
                "velocity": self._limits.velocity_source,
                "acceleration": "configured threshold; not available in the supplied URDF",
            },
            "thresholds": {
                **dataclasses.asdict(self.signal_config),
                "velocity_warning_ratio": self.config.velocity_warning_ratio,
                "acceleration_warning_ratio": self.config.acceleration_warning_ratio,
                "position_limit_tolerance_deg": self.config.position_limit_tolerance_deg,
                "spike_min_step_deg": self.config.spike_min_step_deg,
                "spike_recovery_deg": self.config.spike_recovery_deg,
            },
            "summary": {
                "frame_count": len(positions),
                "duration_s": float(timestamps[-1] - timestamps[0]),
                "fail_event_count": fail_count,
                "warning_event_count": warning_count,
                "event_counts": {
                    kind: sum(event["kind"] == kind for event in events)
                    for kind in (
                        "position_limit",
                        "step_discontinuity",
                        "velocity_limit",
                        "acceleration_limit",
                        "single_frame_spike",
                    )
                },
            },
            "per_joint": per_joint,
            "events": events,
        }
