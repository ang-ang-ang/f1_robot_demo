from __future__ import annotations

import copy
import dataclasses
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class TimestampQualityConfig:
    duplicate_epsilon_ms: float
    normal_period_tolerance_ratio: float
    gap_warning_factor: float
    gap_fail_factor: float
    image_gap_floor_ms: float
    state_gap_floor_ms: float
    action_gap_floor_ms: float
    log_publish_latency_warning_ms: float


@dataclasses.dataclass(frozen=True)
class CrossStreamQualityConfig:
    target_fps: int
    max_image_residual_ms: float
    max_state_residual_ms: float
    max_action_residual_ms: float
    max_start_end_skew_ms: float


@dataclasses.dataclass(frozen=True)
class JointMotionQualityConfig:
    max_state_step_deg: float
    max_action_step_deg: float
    max_state_velocity_deg_s: float
    max_action_velocity_deg_s: float
    max_acceleration_deg_s2: float
    max_tracking_p95_deg: float


@dataclasses.dataclass(frozen=True)
class ImageQualityConfig:
    decode_samples_per_topic: int
    hash_samples_per_topic: int
    max_sampled_identical_ratio: float


@dataclasses.dataclass(frozen=True)
class HeadTimestampCleaningConfig:
    topic: str
    nominal_period_ms: float
    period_search_min_ms: float
    period_search_max_ms: float
    normal_step_tolerance_ratio: float
    anchor_residual_tolerance_periods: float
    local_anchor_tolerance_ratio: float
    short_run_max_frames: int
    warning_correction_periods: float
    high_risk_correction_periods: float
    minimum_anchor_count: int


@dataclasses.dataclass(frozen=True)
class ConversionConfig:
    fps: int
    max_image_delta_ms: float
    max_state_delta_ms: float
    max_action_delta_ms: float
    max_action_step_deg: float
    gripper_scale: float


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    timestamp: TimestampQualityConfig
    cross_stream: CrossStreamQualityConfig
    joint_motion: JointMotionQualityConfig
    image: ImageQualityConfig
    head_timestamp: HeadTimestampCleaningConfig
    conversion: ConversionConfig
    source: Path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate(config: PipelineConfig) -> None:
    if config.cross_stream.target_fps <= 0 or config.conversion.fps <= 0:
        raise ValueError("target FPS must be positive")
    if not 0 < config.timestamp.normal_period_tolerance_ratio < 1:
        raise ValueError("quality.timestamp.normal_period_tolerance_ratio must be in (0, 1)")
    if config.timestamp.gap_warning_factor <= 1:
        raise ValueError("quality.timestamp.gap_warning_factor must be greater than 1")
    if config.timestamp.gap_fail_factor <= config.timestamp.gap_warning_factor:
        raise ValueError("quality.timestamp.gap_fail_factor must exceed gap_warning_factor")
    cleaning = config.head_timestamp
    if cleaning.period_search_min_ms <= 0 or cleaning.period_search_max_ms <= cleaning.period_search_min_ms:
        raise ValueError("invalid cleaning head-camera period search range")
    if cleaning.short_run_max_frames < 1 or cleaning.minimum_anchor_count < 2:
        raise ValueError("cleaning run/anchor counts are invalid")
    if cleaning.high_risk_correction_periods < cleaning.warning_correction_periods:
        raise ValueError("high-risk correction threshold must not be below warning threshold")


def load_config(path: Path | None = None) -> PipelineConfig:
    default_path = Path(__file__).with_name("default_config.toml")
    with default_path.open("rb") as handle:
        data = tomllib.load(handle)
    source = default_path
    if path is not None:
        with path.open("rb") as handle:
            data = _deep_merge(data, tomllib.load(handle))
        source = path

    quality = data["quality"]
    cleaning = data["cleaning"]
    config = PipelineConfig(
        timestamp=TimestampQualityConfig(**quality["timestamp"]),
        cross_stream=CrossStreamQualityConfig(**quality["cross_stream"]),
        joint_motion=JointMotionQualityConfig(**quality["joint_motion"]),
        image=ImageQualityConfig(**quality["image"]),
        head_timestamp=HeadTimestampCleaningConfig(**cleaning["head_timestamp"]),
        conversion=ConversionConfig(**data["conversion"]),
        source=source,
    )
    _validate(config)
    return config


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    result = dataclasses.asdict(config)
    result["source"] = str(config.source)
    return result
