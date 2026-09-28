"""Configuration loading for composable LeRobot quality operators."""

from __future__ import annotations

import copy
import dataclasses
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class ExposureOperatorConfig:
    clip_luminance: float
    l50_ratio_threshold: float
    l50_delta_threshold: float
    l95_delta_threshold: float
    clip_ratio_threshold: float
    clip_excess_threshold: float
    minimum_consecutive_frames: int
    pixel_stride: int


@dataclasses.dataclass(frozen=True)
class JointSignalConfig:
    max_step_deg: float
    fallback_max_velocity_deg_s: float
    max_acceleration_deg_s2: float


@dataclasses.dataclass(frozen=True)
class JointMotionOperatorConfig:
    state: JointSignalConfig
    action: JointSignalConfig
    velocity_warning_ratio: float
    acceleration_warning_ratio: float
    position_limit_tolerance_deg: float
    spike_min_step_deg: float
    spike_recovery_deg: float


@dataclasses.dataclass(frozen=True)
class QualityOperatorConfig:
    exposure: ExposureOperatorConfig
    joint_motion: JointMotionOperatorConfig
    source: Path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate(config: QualityOperatorConfig) -> None:
    exposure = config.exposure
    unit_interval = {
        "clip_luminance": exposure.clip_luminance,
        "l50_delta_threshold": exposure.l50_delta_threshold,
        "l95_delta_threshold": exposure.l95_delta_threshold,
        "clip_ratio_threshold": exposure.clip_ratio_threshold,
        "clip_excess_threshold": exposure.clip_excess_threshold,
    }
    for name, value in unit_interval.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"exposure.{name} must be in [0, 1], got {value}")
    if exposure.l50_ratio_threshold <= 1.0:
        raise ValueError("exposure.l50_ratio_threshold must be greater than 1")
    if exposure.minimum_consecutive_frames < 1 or exposure.pixel_stride < 1:
        raise ValueError("exposure frame count and pixel stride must be positive")

    motion = config.joint_motion
    for name, ratio in {
        "velocity_warning_ratio": motion.velocity_warning_ratio,
        "acceleration_warning_ratio": motion.acceleration_warning_ratio,
    }.items():
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"joint_motion.{name} must be in (0, 1], got {ratio}")
    for signal_name, signal in {"state": motion.state, "action": motion.action}.items():
        if min(signal.max_step_deg, signal.fallback_max_velocity_deg_s, signal.max_acceleration_deg_s2) <= 0:
            raise ValueError(f"joint_motion.{signal_name} thresholds must be positive")
    if min(motion.spike_min_step_deg, motion.spike_recovery_deg) < 0:
        raise ValueError("joint spike thresholds must be non-negative")


def load_operator_config(path: Path | None = None) -> QualityOperatorConfig:
    default_path = Path(__file__).with_name("operator_config.toml")
    with default_path.open("rb") as handle:
        data = tomllib.load(handle)
    source = default_path
    if path is not None:
        with path.open("rb") as handle:
            data = _deep_merge(data, tomllib.load(handle))
        source = path

    motion = data["joint_motion"]
    config = QualityOperatorConfig(
        exposure=ExposureOperatorConfig(**data["exposure"]),
        joint_motion=JointMotionOperatorConfig(
            state=JointSignalConfig(**motion["state"]),
            action=JointSignalConfig(**motion["action"]),
            velocity_warning_ratio=motion["velocity_warning_ratio"],
            acceleration_warning_ratio=motion["acceleration_warning_ratio"],
            position_limit_tolerance_deg=motion["position_limit_tolerance_deg"],
            spike_min_step_deg=motion["spike_min_step_deg"],
            spike_recovery_deg=motion["spike_recovery_deg"],
        ),
        source=source,
    )
    _validate(config)
    return config


def operator_config_to_dict(config: QualityOperatorConfig) -> dict[str, Any]:
    result = dataclasses.asdict(config)
    result["source"] = str(config.source)
    return result
