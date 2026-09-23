# ruff: noqa: RUF001
"""Convert F1 ROS 2 MCAP episodes to a LeRobot dataset.

The converter is intentionally independent of a ROS installation. It reads the
uncompressed MCAP chunks produced by the F1 recorder and decodes the three ROS 2
message types used by the dataset directly from CDR.

Example:

    uv run examples/f1/convert_f1_mcap_to_lerobot.py \
        --raw-dir test_data/F1_data_test \
        --repo-id your_name/f1_open_box \
        --task "Open the cardboard box with both grippers." \
        --start-time-s 3.0 \
        --end-time-s 83.5

Run with ``--dry-run`` first to inspect synchronization residuals and warnings
without writing a LeRobot dataset.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import shutil
import struct
import subprocess
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal

import cv2
import numpy as np
import tqdm
import tyro

HEAD_IMAGE_TOPIC = "/camera/head/color/image_raw/compressed"
LEFT_WRIST_IMAGE_TOPIC = "/camera/left_wrist/color/image_raw/compressed"
RIGHT_WRIST_IMAGE_TOPIC = "/camera/right_wrist/color/image_raw/compressed"
HAL_JOINT_TOPIC = "/hal/joint_states"
LEAD_JOINT_TOPIC = "/lead/joint_states"
LEFT_GRIPPER_COMMAND_TOPIC = "/motion_ctl/gripper/left"
RIGHT_GRIPPER_COMMAND_TOPIC = "/motion_ctl/gripper/right"
LEFT_GRIPPER_STATE_TOPIC = "/motion_ctl/gripper/left/state"
RIGHT_GRIPPER_STATE_TOPIC = "/motion_ctl/gripper/right/state"
LEFT_TCP_STATE_TOPIC = "/state/left_arm/tcp_pos"
RIGHT_TCP_STATE_TOPIC = "/state/right_arm/tcp_pos"
LEFT_LEAD_TCP_TOPIC = "/lead/left_tcp"
RIGHT_LEAD_TCP_TOPIC = "/lead/right_tcp"

IMAGE_TOPICS = {
    "observation.images.head": HEAD_IMAGE_TOPIC,
    "observation.images.left_wrist": LEFT_WRIST_IMAGE_TOPIC,
    "observation.images.right_wrist": RIGHT_WRIST_IMAGE_TOPIC,
}

STATE_JOINT_NAMES = tuple([f"arm_l_j{index}" for index in range(1, 8)] + [f"arm_r_j{index}" for index in range(1, 8)])
ACTION_JOINT_NAMES = tuple(
    [f"arm_L_joint{index}" for index in range(1, 8)] + [f"arm_R_joint{index}" for index in range(1, 8)]
)
STATE_NAMES = (*STATE_JOINT_NAMES, "left_gripper", "right_gripper")
ACTION_NAMES = (*ACTION_JOINT_NAMES, "left_gripper", "right_gripper")

MCAP_MAGIC = b"\x89MCAP0\r\n"
MCAP_SCHEMA = 0x03
MCAP_CHANNEL = 0x04
MCAP_MESSAGE = 0x05
MCAP_CHUNK = 0x06


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """An episode directory or a directory containing ``episode_*`` directories."""

    repo_id: str
    """LeRobot repository id, for example ``your_name/f1_open_box``."""

    task: str = ""
    """Fallback language instruction used for every episode."""

    episode_config: Path | None = None
    """Optional JSON file containing per-episode task, trim, and include overrides."""

    output_root: Path | None = None
    """Exact local output path. By default LeRobot uses ``HF_LEROBOT_HOME/repo_id``."""

    report_dir: Path = Path("artifacts/f1_conversion_reports")
    """Every run writes a unique JSON and Chinese Markdown report below this directory."""

    fps: int = 20
    """Fixed output rate. Twenty hertz is recommended for this recording."""

    start_time_s: float = 0.0
    """Default trim start relative to the first required source message."""

    end_time_s: float | None = None
    """Default trim end relative to the first required source message."""

    gripper_scale: float = 100.0
    """Raw gripper value mapped to 1.0; the observed recording uses 0..100."""

    max_image_delta_ms: float = 40.0
    """Maximum nearest-image timestamp residual allowed at an output frame."""

    max_state_delta_ms: float = 20.0
    """Maximum nearest-state timestamp residual allowed at an output frame."""

    max_action_delta_ms: float = 60.0
    """Maximum nearest-command timestamp residual allowed at an output frame."""

    max_action_step_deg: float = 25.0
    """Warn when consecutive resampled arm targets exceed this change."""

    image_writer_threads: int = 8
    use_videos: bool = True
    video_codec: Literal["h264", "hevc", "libsvtav1"] = "h264"
    """Codec used when ``use_videos`` is enabled; h264 is widely available."""

    overwrite: bool = False
    push_to_hub: bool = False
    dry_run: bool = False
    """Analyze and validate the conversion without writing output data."""


@dataclasses.dataclass(frozen=True)
class ImageRef:
    path: Path
    offset: int
    size: int
    format: str
    frame_id: str


@dataclasses.dataclass
class NumericSeries:
    timestamps_ns: list[int] = dataclasses.field(default_factory=list)
    values: list[np.ndarray] = dataclasses.field(default_factory=list)
    names: tuple[str, ...] | None = None

    def append(self, timestamp_ns: int, values: np.ndarray, names: tuple[str, ...] | None = None) -> None:
        if names is not None:
            if self.names is None:
                self.names = names
            elif names != self.names:
                raise ValueError(f"Joint names changed within a topic: {self.names} != {names}")
        self.timestamps_ns.append(timestamp_ns)
        self.values.append(np.asarray(values, dtype=np.float64))

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.timestamps_ns:
            raise ValueError("Required numeric topic contains no messages")
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        values = np.stack(self.values)
        order = np.argsort(timestamps, kind="stable")
        return timestamps[order], values[order]


@dataclasses.dataclass
class ImageSeries:
    timestamps_ns: list[int] = dataclasses.field(default_factory=list)
    refs: list[ImageRef] = dataclasses.field(default_factory=list)

    def append(self, timestamp_ns: int, ref: ImageRef) -> None:
        self.timestamps_ns.append(timestamp_ns)
        self.refs.append(ref)

    def finalize(self) -> tuple[np.ndarray, list[ImageRef]]:
        if not self.timestamps_ns:
            raise ValueError("Required image topic contains no messages")
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        order = np.argsort(timestamps, kind="stable")
        return timestamps[order], [self.refs[index] for index in order]


@dataclasses.dataclass
class EpisodeData:
    name: str
    source_files: list[Path]
    numeric: dict[str, NumericSeries] = dataclasses.field(default_factory=dict)
    images: dict[str, ImageSeries] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    data: EpisodeData
    task: str
    target_timestamps_ns: np.ndarray
    image_indices: dict[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray
    report: dict


def _align_cdr(position: int, alignment: int, base: int = 4) -> int:
    return position + (-(position - base) % alignment)


def _read_u16(buffer: bytes | memoryview, position: int) -> tuple[int, int]:
    return struct.unpack_from("<H", buffer, position)[0], position + 2


def _read_u32(buffer: bytes | memoryview, position: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buffer, position)[0], position + 4


def _read_i32(buffer: bytes | memoryview, position: int) -> tuple[int, int]:
    return struct.unpack_from("<i", buffer, position)[0], position + 4


def _read_u64(buffer: bytes | memoryview, position: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", buffer, position)[0], position + 8


def _read_string(buffer: bytes | memoryview, position: int) -> tuple[str, int]:
    length, position = _read_u32(buffer, position)
    raw = bytes(buffer[position : position + length])
    return raw.rstrip(b"\0").decode("utf-8", "replace"), position + length


def _read_blob32(buffer: bytes | memoryview, position: int) -> tuple[bytes, int]:
    length, position = _read_u32(buffer, position)
    return bytes(buffer[position : position + length]), position + length


def _read_blob64(buffer: bytes | memoryview, position: int) -> tuple[memoryview, int, int]:
    length, position = _read_u64(buffer, position)
    data_position = position
    return memoryview(buffer)[position : position + length], position + length, data_position


def _iter_records(buffer: bytes | memoryview):
    position = 0
    while position + 9 <= len(buffer):
        opcode = buffer[position]
        length = struct.unpack_from("<Q", buffer, position + 1)[0]
        data_position = position + 9
        end = data_position + length
        if end > len(buffer):
            raise ValueError(f"Invalid MCAP record at byte {position}: record extends beyond its container")
        yield opcode, memoryview(buffer)[data_position:end], data_position
        position = end
    if position != len(buffer):
        raise ValueError(f"MCAP container has {len(buffer) - position} trailing bytes")


def _decode_cdr_header(buffer: bytes | memoryview) -> tuple[int, str, int]:
    if bytes(buffer[:4]) != b"\x00\x01\x00\x00":
        raise ValueError(f"Only little-endian CDR is supported, got encapsulation {bytes(buffer[:4]).hex()}")
    position = 4
    seconds, position = _read_i32(buffer, position)
    nanoseconds, position = _read_u32(buffer, position)
    frame_id, position = _read_string(buffer, position)
    return seconds * 1_000_000_000 + nanoseconds, frame_id, position


def _decode_joint_state(buffer: bytes | memoryview) -> tuple[int, tuple[str, ...], np.ndarray]:
    timestamp_ns, _, position = _decode_cdr_header(buffer)
    position = _align_cdr(position, 4)
    name_count, position = _read_u32(buffer, position)
    names = []
    for _ in range(name_count):
        position = _align_cdr(position, 4)
        name, position = _read_string(buffer, position)
        names.append(name)

    position = _align_cdr(position, 4)
    value_count, position = _read_u32(buffer, position)
    position = _align_cdr(position, 8)
    values = np.frombuffer(buffer, dtype="<f8", count=value_count, offset=position).copy()
    return timestamp_ns, tuple(names), values


def _decode_gripper_command(buffer: bytes | memoryview) -> np.ndarray:
    if bytes(buffer[:4]) != b"\x00\x01\x00\x00":
        raise ValueError(f"Only little-endian CDR is supported, got encapsulation {bytes(buffer[:4]).hex()}")
    return np.asarray([struct.unpack_from("<d", buffer, 4)[0]], dtype=np.float64)


def _decode_image_ref(
    buffer: bytes | memoryview,
    *,
    cdr_file_offset: int,
    mcap_path: Path,
) -> tuple[int, ImageRef]:
    timestamp_ns, frame_id, position = _decode_cdr_header(buffer)
    position = _align_cdr(position, 4)
    image_format, position = _read_string(buffer, position)
    position = _align_cdr(position, 4)
    size, position = _read_u32(buffer, position)
    if position + size > len(buffer):
        raise ValueError(f"Compressed image payload exceeds its MCAP message in {mcap_path}")
    return timestamp_ns, ImageRef(
        path=mcap_path,
        offset=cdr_file_offset + position,
        size=size,
        format=image_format,
        frame_id=frame_id,
    )


def _read_mcap(path: Path, episode: EpisodeData) -> None:
    schemas: dict[int, str] = {}
    channels: dict[int, tuple[int, str]] = {}

    with path.open("rb") as handle:
        if handle.read(8) != MCAP_MAGIC:
            raise ValueError(f"Not an MCAP file: {path}")

        while True:
            record_offset = handle.tell()
            header = handle.read(9)
            if not header:
                break
            if len(header) == len(MCAP_MAGIC) and header == MCAP_MAGIC:
                break
            if len(header) != 9:
                raise ValueError(f"Truncated MCAP record header in {path}")
            opcode = header[0]
            length = struct.unpack("<Q", header[1:])[0]
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"Truncated MCAP record payload in {path}")
            if opcode != MCAP_CHUNK:
                continue

            position = 0
            _, position = _read_u64(payload, position)
            _, position = _read_u64(payload, position)
            uncompressed_size, position = _read_u64(payload, position)
            _, position = _read_u32(payload, position)
            compression, position = _read_string(payload, position)
            records, position, records_position = _read_blob64(payload, position)
            if compression:
                raise ValueError(
                    f"{path} uses '{compression}' chunk compression. This converter currently supports the "
                    "uncompressed F1 MCAP layout only. Re-record without MCAP chunk compression or decompress it first."
                )
            if len(records) != uncompressed_size:
                raise ValueError(f"Uncompressed chunk size mismatch in {path}")
            if position != len(payload):
                raise ValueError(f"Unexpected data after an MCAP chunk in {path}")

            records_file_offset = record_offset + 9 + records_position
            for inner_opcode, data, data_position in _iter_records(records):
                if inner_opcode == MCAP_SCHEMA:
                    schema_position = 0
                    schema_id, schema_position = _read_u16(data, schema_position)
                    schema_name, schema_position = _read_string(data, schema_position)
                    _, schema_position = _read_string(data, schema_position)
                    _, schema_position = _read_blob32(data, schema_position)
                    schemas[schema_id] = schema_name
                elif inner_opcode == MCAP_CHANNEL:
                    channel_position = 0
                    channel_id, channel_position = _read_u16(data, channel_position)
                    schema_id, channel_position = _read_u16(data, channel_position)
                    topic, channel_position = _read_string(data, channel_position)
                    _, channel_position = _read_string(data, channel_position)
                    channels[channel_id] = (schema_id, topic)
                elif inner_opcode == MCAP_MESSAGE:
                    channel_id = struct.unpack_from("<H", data, 0)[0]
                    log_time_ns = struct.unpack_from("<Q", data, 6)[0]
                    cdr = data[22:]
                    if channel_id not in channels:
                        raise ValueError(f"Message references unknown MCAP channel {channel_id} in {path}")
                    schema_id, topic = channels[channel_id]
                    schema_name = schemas.get(schema_id)
                    cdr_file_offset = records_file_offset + data_position + 22

                    if topic in IMAGE_TOPICS.values():
                        if schema_name != "sensor_msgs/msg/CompressedImage":
                            raise ValueError(f"Unexpected schema {schema_name!r} for image topic {topic}")
                        timestamp_ns, image_ref = _decode_image_ref(
                            cdr,
                            cdr_file_offset=cdr_file_offset,
                            mcap_path=path,
                        )
                        episode.images.setdefault(topic, ImageSeries()).append(timestamp_ns, image_ref)
                    elif topic in {
                        HAL_JOINT_TOPIC,
                        LEAD_JOINT_TOPIC,
                        LEFT_GRIPPER_STATE_TOPIC,
                        RIGHT_GRIPPER_STATE_TOPIC,
                        LEFT_TCP_STATE_TOPIC,
                        RIGHT_TCP_STATE_TOPIC,
                        LEFT_LEAD_TCP_TOPIC,
                        RIGHT_LEAD_TCP_TOPIC,
                    }:
                        if schema_name != "sensor_msgs/msg/JointState":
                            raise ValueError(f"Unexpected schema {schema_name!r} for joint topic {topic}")
                        timestamp_ns, names, values = _decode_joint_state(cdr)
                        episode.numeric.setdefault(topic, NumericSeries()).append(timestamp_ns, values, names)
                    elif topic in {LEFT_GRIPPER_COMMAND_TOPIC, RIGHT_GRIPPER_COMMAND_TOPIC}:
                        if schema_name != "control_msgs/msg/GripperCommand":
                            raise ValueError(f"Unexpected schema {schema_name!r} for gripper topic {topic}")
                        values = _decode_gripper_command(cdr)
                        episode.numeric.setdefault(topic, NumericSeries()).append(log_time_ns, values)


def _find_episode_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    if list(raw_dir.glob("*.mcap")):
        return [raw_dir]
    episode_dirs = sorted(path for path in raw_dir.glob("episode_*") if path.is_dir() and list(path.glob("*.mcap")))
    if not episode_dirs:
        raise FileNotFoundError(f"No MCAP files found under {raw_dir}")
    return episode_dirs


def _load_episode(path: Path) -> EpisodeData:
    source_files = sorted(path.glob("*.mcap"))
    if not source_files:
        raise FileNotFoundError(f"No MCAP files found in {path}")
    episode = EpisodeData(name=path.name, source_files=source_files)
    for source_file in source_files:
        _read_mcap(source_file, episode)
    return episode


def _select_named_columns(series: NumericSeries, required_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    timestamps, values = series.finalize()
    if series.names is None:
        raise ValueError("A named JointState topic did not contain names")
    missing = set(required_names) - set(series.names)
    if missing:
        raise ValueError(f"JointState is missing required names: {sorted(missing)}")
    indices = [series.names.index(name) for name in required_names]
    selected = values[:, indices]
    if not np.isfinite(selected).all():
        raise ValueError("JointState contains NaN or infinite values")
    return timestamps, selected


def _single_column(series: NumericSeries) -> tuple[np.ndarray, np.ndarray]:
    timestamps, values = series.finalize()
    if values.shape[1] != 1:
        raise ValueError(f"Expected one gripper value, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Gripper topic contains NaN or infinite values")
    return timestamps, values[:, 0]


def _repair_image_timestamps(timestamps_ns: np.ndarray) -> tuple[np.ndarray, dict]:
    differences = np.diff(timestamps_ns)
    nonpositive = int(np.sum(differences <= 0))
    if nonpositive == 0:
        return timestamps_ns, {"nonpositive_source_steps": 0, "repair": "none", "max_correction_ms": 0.0}
    if timestamps_ns[-1] <= timestamps_ns[0]:
        raise ValueError("Image timestamps are not recoverable")

    repaired = np.rint(np.linspace(timestamps_ns[0], timestamps_ns[-1], len(timestamps_ns))).astype(np.int64)
    max_correction_ms = float(np.max(np.abs(repaired - timestamps_ns)) / 1e6)
    return repaired, {
        "nonpositive_source_steps": nonpositive,
        "repair": "affine_from_frame_index",
        "max_correction_ms": max_correction_ms,
    }


def _nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(source_ns) < 2:
        raise ValueError("At least two source samples are required")
    right = np.searchsorted(source_ns, target_ns)
    right = np.clip(right, 1, len(source_ns) - 1)
    left = right - 1
    choose_left = np.abs(target_ns - source_ns[left]) <= np.abs(source_ns[right] - target_ns)
    indices = np.where(choose_left, left, right)
    residual_ms = (source_ns[indices] - target_ns) / 1e6
    return indices, residual_ms


def _series_stats(timestamps_ns: np.ndarray) -> dict:
    differences_ms = np.diff(timestamps_ns).astype(np.float64) / 1e6
    duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    return {
        "count": len(timestamps_ns),
        "effective_hz": float((len(timestamps_ns) - 1) / duration_s),
        "dt_ms": {
            "median": float(np.median(differences_ms)),
            "p95": float(np.percentile(differences_ms, 95)),
            "max": float(np.max(differences_ms)),
            "nonpositive": int(np.sum(differences_ms <= 0)),
        },
    }


def _residual_stats(residual_ms: np.ndarray) -> dict:
    absolute = np.abs(residual_ms)
    return {
        "median_abs_ms": float(np.median(absolute)),
        "p95_abs_ms": float(np.percentile(absolute, 95)),
        "max_abs_ms": float(np.max(absolute)),
    }


def _validate_gripper(values: np.ndarray, scale: float, topic: str) -> np.ndarray:
    if scale <= 0:
        raise ValueError("gripper_scale must be positive")
    tolerance = scale * 1e-3
    if values.min() < -tolerance or values.max() > scale + tolerance:
        raise ValueError(
            f"{topic} values [{values.min()}, {values.max()}] are outside the configured [0, {scale}] range"
        )
    return np.clip(values / scale, 0.0, 1.0)


def _make_episode_plan(
    data: EpisodeData,
    *,
    task: str,
    fps: int,
    start_time_s: float,
    end_time_s: float | None,
    gripper_scale: float,
    max_image_delta_ms: float,
    max_state_delta_ms: float,
    max_action_delta_ms: float,
    max_action_step_deg: float,
) -> EpisodePlan:
    if fps <= 0:
        raise ValueError("fps must be positive")
    required_numeric = {
        HAL_JOINT_TOPIC,
        LEAD_JOINT_TOPIC,
        LEFT_GRIPPER_COMMAND_TOPIC,
        RIGHT_GRIPPER_COMMAND_TOPIC,
        LEFT_GRIPPER_STATE_TOPIC,
        RIGHT_GRIPPER_STATE_TOPIC,
    }
    missing_numeric = required_numeric - set(data.numeric)
    missing_images = set(IMAGE_TOPICS.values()) - set(data.images)
    if missing_numeric or missing_images:
        raise ValueError(f"{data.name} is missing required topics: {sorted(missing_numeric | missing_images)}")

    state_times, state_joints_deg = _select_named_columns(data.numeric[HAL_JOINT_TOPIC], STATE_JOINT_NAMES)
    action_times, action_joints_deg = _select_named_columns(data.numeric[LEAD_JOINT_TOPIC], ACTION_JOINT_NAMES)
    left_state_times, left_state = _single_column(data.numeric[LEFT_GRIPPER_STATE_TOPIC])
    right_state_times, right_state = _single_column(data.numeric[RIGHT_GRIPPER_STATE_TOPIC])
    left_action_times, left_action = _single_column(data.numeric[LEFT_GRIPPER_COMMAND_TOPIC])
    right_action_times, right_action = _single_column(data.numeric[RIGHT_GRIPPER_COMMAND_TOPIC])

    image_timestamps: dict[str, np.ndarray] = {}
    source_report: dict[str, dict] = {}
    for feature, topic in IMAGE_TOPICS.items():
        timestamps, _ = data.images[topic].finalize()
        source_report[topic] = _series_stats(timestamps)
        timestamps, repair_report = _repair_image_timestamps(timestamps)
        source_report[topic]["timestamp_repair"] = repair_report
        image_timestamps[feature] = timestamps

    for topic, timestamps in {
        HAL_JOINT_TOPIC: state_times,
        LEAD_JOINT_TOPIC: action_times,
        LEFT_GRIPPER_STATE_TOPIC: left_state_times,
        RIGHT_GRIPPER_STATE_TOPIC: right_state_times,
        LEFT_GRIPPER_COMMAND_TOPIC: left_action_times,
        RIGHT_GRIPPER_COMMAND_TOPIC: right_action_times,
    }.items():
        source_report[topic] = _series_stats(timestamps)

    all_timestamps = [
        state_times,
        action_times,
        left_state_times,
        right_state_times,
        left_action_times,
        right_action_times,
        *image_timestamps.values(),
    ]
    source_origin_ns = min(int(timestamps[0]) for timestamps in all_timestamps)
    common_start_ns = max(int(timestamps[0]) for timestamps in all_timestamps)
    common_end_ns = min(int(timestamps[-1]) for timestamps in all_timestamps)
    requested_start_ns = source_origin_ns + round(start_time_s * 1e9)
    requested_end_ns = common_end_ns if end_time_s is None else source_origin_ns + round(end_time_s * 1e9)
    start_ns = max(common_start_ns, requested_start_ns)
    end_ns = min(common_end_ns, requested_end_ns)
    if end_ns <= start_ns:
        raise ValueError(
            f"Invalid trim for {data.name}: common range is "
            f"[{(common_start_ns - source_origin_ns) / 1e9:.3f}, {(common_end_ns - source_origin_ns) / 1e9:.3f}] s"
        )

    period_ns = 1e9 / fps
    frame_count = math.floor((end_ns - start_ns) / period_ns) + 1
    target_timestamps_ns = np.rint(start_ns + np.arange(frame_count) * period_ns).astype(np.int64)

    state_indices, state_residual = _nearest_indices(state_times, target_timestamps_ns)
    left_state_indices, left_state_residual = _nearest_indices(left_state_times, target_timestamps_ns)
    right_state_indices, right_state_residual = _nearest_indices(right_state_times, target_timestamps_ns)
    action_indices, action_residual = _nearest_indices(action_times, target_timestamps_ns)
    left_action_indices, left_action_residual = _nearest_indices(left_action_times, target_timestamps_ns)
    right_action_indices, right_action_residual = _nearest_indices(right_action_times, target_timestamps_ns)

    image_indices = {}
    residual_report = {}
    for feature, timestamps in image_timestamps.items():
        indices, residual = _nearest_indices(timestamps, target_timestamps_ns)
        if np.max(np.abs(residual)) > max_image_delta_ms:
            raise ValueError(
                f"{data.name} {feature} maximum timestamp residual is {np.max(np.abs(residual)):.2f} ms, "
                f"above max_image_delta_ms={max_image_delta_ms}"
            )
        image_indices[feature] = indices
        residual_report[feature] = _residual_stats(residual)

    combined_state_residual = np.maximum.reduce(
        [np.abs(state_residual), np.abs(left_state_residual), np.abs(right_state_residual)]
    )
    combined_action_residual = np.maximum.reduce(
        [np.abs(action_residual), np.abs(left_action_residual), np.abs(right_action_residual)]
    )
    if combined_state_residual.max() > max_state_delta_ms:
        raise ValueError(
            f"{data.name} maximum state timestamp residual is {combined_state_residual.max():.2f} ms, "
            f"above max_state_delta_ms={max_state_delta_ms}"
        )
    if combined_action_residual.max() > max_action_delta_ms:
        raise ValueError(
            f"{data.name} maximum action timestamp residual is {combined_action_residual.max():.2f} ms, "
            f"above max_action_delta_ms={max_action_delta_ms}"
        )
    residual_report["state"] = _residual_stats(combined_state_residual)
    residual_report["action"] = _residual_stats(combined_action_residual)

    state = np.concatenate(
        [
            np.deg2rad(state_joints_deg[state_indices]),
            _validate_gripper(left_state[left_state_indices], gripper_scale, LEFT_GRIPPER_STATE_TOPIC)[:, None],
            _validate_gripper(right_state[right_state_indices], gripper_scale, RIGHT_GRIPPER_STATE_TOPIC)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    action = np.concatenate(
        [
            np.deg2rad(action_joints_deg[action_indices]),
            _validate_gripper(left_action[left_action_indices], gripper_scale, LEFT_GRIPPER_COMMAND_TOPIC)[:, None],
            _validate_gripper(right_action[right_action_indices], gripper_scale, RIGHT_GRIPPER_COMMAND_TOPIC)[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    action_steps_deg = np.rad2deg(np.abs(np.diff(action[:, :14], axis=0)))
    max_action_step = float(action_steps_deg.max(initial=0.0))
    per_frame_action_step = action_steps_deg.max(axis=1, initial=0.0)
    action_jump_indices = np.flatnonzero(per_frame_action_step > max_action_step_deg)
    tracking_error_deg = np.rad2deg(np.abs(action[:, :14] - state[:, :14]))
    warnings = []
    if max_action_step > max_action_step_deg:
        index = np.unravel_index(np.argmax(action_steps_deg), action_steps_deg.shape)[0] + 1
        warnings.append(
            f"输出帧 {index} 的重采样动作跳变达到 {max_action_step:.2f}°；请检查示教初始化、离合重定位或 Episode 切分。"
        )
    head_repair = source_report[HEAD_IMAGE_TOPIC]["timestamp_repair"]
    if head_repair["nonpositive_source_steps"]:
        warnings.append(
            f"检测并按帧序号修复了 {head_repair['nonpositive_source_steps']} 个非递增头部相机时间戳，"
            f"最大修正量为 {head_repair['max_correction_ms']:.2f} ms。"
        )

    report = {
        "episode": data.name,
        "task": task,
        "source_files": [str(path) for path in data.source_files],
        "source_range_s": {
            "common_start": (common_start_ns - source_origin_ns) / 1e9,
            "common_end": (common_end_ns - source_origin_ns) / 1e9,
        },
        "selected_range_s": {
            "start": (start_ns - source_origin_ns) / 1e9,
            "end": (target_timestamps_ns[-1] - source_origin_ns) / 1e9,
        },
        "output": {"fps": fps, "frames": frame_count, "duration_s": (frame_count - 1) / fps},
        "source_topics": source_report,
        "sampling_residuals": residual_report,
        "action_quality": {
            "jump_threshold_deg": max_action_step_deg,
            "jump_count": len(action_jump_indices),
            "max_arm_action_step_deg": max_action_step,
            "largest_jumps": [
                {
                    "output_frame": int(index + 1),
                    "relative_time_s": float((index + 1) / fps),
                    "max_joint_step_deg": float(per_frame_action_step[index]),
                }
                for index in np.argsort(per_frame_action_step)[-10:][::-1]
            ],
        },
        "leader_follower_tracking": {
            "median_abs_deg": np.median(tracking_error_deg, axis=0).tolist(),
            "p95_abs_deg": np.percentile(tracking_error_deg, 95, axis=0).tolist(),
            "max_abs_deg": tracking_error_deg.max(axis=0).tolist(),
            "global_p95_abs_deg": float(np.percentile(tracking_error_deg, 95)),
            "worst_joint_p95_deg": float(np.percentile(tracking_error_deg, 95, axis=0).max()),
        },
        "ranges": {
            "state_min": state.min(axis=0).tolist(),
            "state_max": state.max(axis=0).tolist(),
            "action_min": action.min(axis=0).tolist(),
            "action_max": action.max(axis=0).tolist(),
            "max_arm_action_step_deg": max_action_step,
        },
        "warnings": warnings,
    }
    return EpisodePlan(
        data=data,
        task=task,
        target_timestamps_ns=target_timestamps_ns,
        image_indices=image_indices,
        state=state,
        action=action,
        report=report,
    )


def _read_image(ref: ImageRef, handles: dict[Path, BinaryIO]) -> np.ndarray:
    handle = handles[ref.path]
    handle.seek(ref.offset)
    compressed = handle.read(ref.size)
    if len(compressed) != ref.size:
        raise ValueError(f"Could not read complete image payload from {ref.path} at byte {ref.offset}")
    image_bgr = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode image from {ref.path} at byte {ref.offset}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _encode_video_frames_ffmpeg(
    images_dir: Path | str,
    video_path: Path | str,
    fps: int,
    *,
    codec: Literal["h264", "hevc", "libsvtav1"],
    overwrite: bool = False,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("Video output requires ffmpeg. Use --no-use-videos or install ffmpeg.")
    encoder = {"h264": "libx264", "hevc": "libx265", "libsvtav1": "libsvtav1"}[codec]
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(Path(images_dir) / "frame_%06d.png"),
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-g",
        "2",
        "-crf",
        "23",
        str(video_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffmpeg could not encode {video_path} with {encoder}. "
            "Use --no-use-videos or select an installed --video-codec."
        ) from error


def _episode_settings(
    episode_name: str,
    episode_config: dict,
    args: Args,
) -> tuple[bool, str, float, float | None]:
    override = episode_config.get(episode_name, {})
    unknown = set(override) - {
        "include",
        "task",
        "start_time_s",
        "end_time_s",
        "success",
        "failure_reason",
        "notes",
    }
    if unknown:
        raise ValueError(f"Unknown keys for {episode_name} in episode_config: {sorted(unknown)}")
    include = bool(override.get("include", True))
    task = str(override.get("task", args.task)).strip()
    start_time_s = float(override.get("start_time_s", args.start_time_s))
    end_value = override.get("end_time_s", args.end_time_s)
    end_time_s = None if end_value is None else float(end_value)
    if include and not task:
        raise ValueError(
            f"No task instruction configured for {episode_name}. Pass --task or add task to --episode-config."
        )
    return include, task, start_time_s, end_time_s


def _load_episode_config(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(isinstance(value, dict) for value in data.values()):
        raise ValueError("episode_config must be a JSON object mapping episode directory names to objects")
    return data


def _probe_image_shapes(plan: EpisodePlan) -> dict[str, tuple[int, int, int]]:
    shapes = {}
    with ExitStack() as stack:
        handles = {path: stack.enter_context(path.open("rb")) for path in plan.data.source_files}
        for feature, topic in IMAGE_TOPICS.items():
            _, refs = plan.data.images[topic].finalize()
            ref = refs[int(plan.image_indices[feature][0])]
            shapes[feature] = _read_image(ref, handles).shape
    return shapes


def _create_dataset(args: Args, image_shapes: dict[str, tuple[int, int, int]]):
    from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_module
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    if args.use_videos:
        lerobot_dataset_module.encode_video_frames = functools.partial(
            _encode_video_frames_ffmpeg,
            codec=args.video_codec,
        )

    output_path = args.output_root if args.output_root is not None else HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": [list(STATE_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": [list(ACTION_NAMES)],
        },
    }
    for feature, shape in image_shapes.items():
        height, width, channels = shape
        if channels != 3:
            raise ValueError(f"Expected three-channel image for {feature}, got shape {shape}")
        features[feature] = {
            "dtype": "video" if args.use_videos else "image",
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }

    dataset = lerobot_dataset_module.LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root,
        robot_type="f1_dual_arm",
        fps=args.fps,
        features=features,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
    )
    return dataset, Path(output_path)


def _write_episode(dataset, plan: EpisodePlan, expected_shapes: dict[str, tuple[int, int, int]]) -> None:
    finalized_images = {feature: plan.data.images[topic].finalize()[1] for feature, topic in IMAGE_TOPICS.items()}
    with ExitStack() as stack:
        handles = {path: stack.enter_context(path.open("rb")) for path in plan.data.source_files}
        for frame_index in tqdm.trange(len(plan.target_timestamps_ns), desc=plan.data.name):
            frame = {
                "observation.state": plan.state[frame_index],
                "action": plan.action[frame_index],
                "task": plan.task,
            }
            for feature, refs in finalized_images.items():
                ref = refs[int(plan.image_indices[feature][frame_index])]
                image = _read_image(ref, handles)
                if image.shape != expected_shapes[feature]:
                    raise ValueError(
                        f"Image shape changed for {feature}: expected {expected_shapes[feature]}, got {image.shape}"
                    )
                frame[feature] = image
            dataset.add_frame(frame)
    dataset.save_episode()


def _new_report_run_dir(args: Args) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    repo_slug = args.repo_id.replace("/", "__").replace(" ", "_")
    run_dir = args.report_dir / f"{timestamp}_{repo_slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _render_conversion_markdown(report: dict) -> str:
    run = report["run"]
    lines = [
        "# F1 MCAP → LeRobot 转换与质量报告",
        "",
        f"- 运行状态：`{run['status']}`",
        f"- UTC 时间：`{run['started_at_utc']}`",
        f"- 输入：`{run['raw_dir']}`",
        f"- 输出仓库：`{report['repo_id']}`",
        f"- 模式：`{run.get('mode', 'unknown')}`",
        f"- 目标频率：`{run['config']['fps']} Hz`",
        "",
        "## Episode 汇总",
        "",
        "| Episode | 任务 | 选择区间 / s | 输出帧数 | 图像最大误差 / ms | 状态最大误差 / ms | 动作最大误差 / ms | 告警数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in report.get("episodes", []):
        selected = episode["selected_range_s"]
        residuals = episode["sampling_residuals"]
        image_error = max(residuals[feature]["max_abs_ms"] for feature in IMAGE_TOPICS)
        task = episode["task"].replace("|", "\\|")
        lines.append(
            f"| `{episode['episode']}` | {task} | {selected['start']:.3f}–{selected['end']:.3f} | "
            f"{episode['output']['frames']} | {image_error:.3f} | {residuals['state']['max_abs_ms']:.3f} | "
            f"{residuals['action']['max_abs_ms']:.3f} | {len(episode['warnings'])} |"
        )

    for episode in report.get("episodes", []):
        lines.extend(["", f"## {episode['episode']}", ""])
        lines.append(f"- 任务：{episode['task']}")
        lines.append(f"- 输出：`{episode['output']['frames']}` 帧，`{episode['output']['duration_s']:.3f} s`")
        lines.append(f"- 最大重采样动作步长：`{episode['ranges']['max_arm_action_step_deg']:.3f}°`")
        lines.extend(
            [
                "",
                "### 源话题时序",
                "",
                "| Topic | 数量 | 有效频率 / Hz | 中位周期 / ms | P95 / ms | 最大间隔 / ms | 非递增次数 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for topic, stats in episode["source_topics"].items():
            timing = stats["dt_ms"]
            lines.append(
                f"| `{topic}` | {stats['count']} | {stats['effective_hz']:.3f} | {timing['median']:.3f} | "
                f"{timing['p95']:.3f} | {timing['max']:.3f} | {timing['nonpositive']} |"
            )
        lines.extend(["", "### 告警", ""])
        if episode["warnings"]:
            lines.extend(f"- {warning}" for warning in episode["warnings"])
        else:
            lines.append("- 无。")

    if run.get("error"):
        lines.extend(["", "## 失败原因", "", f"```text\n{run['error']}\n```"])
    lines.extend(
        [
            "",
            "## 字段约定",
            "",
            f"- 状态顺序：`{', '.join(report['state_names'])}`",
            f"- 动作顺序：`{', '.join(report['action_names'])}`",
            "- 关节单位：弧度。",
            "- 夹爪单位：归一化 `[0, 1]`。",
            "- 采样方法：固定频率时间轴上的最近时间戳匹配。",
            "",
        ]
    )
    return "\n".join(lines)


def _save_conversion_report(report: dict, directory: Path, stem: str = "conversion_report") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (directory / f"{stem}.md").write_text(_render_conversion_markdown(report))


def main(args: Args) -> None:
    report_run_dir = _new_report_run_dir(args)
    conversion_report = {
        "format": "f1_mcap_to_lerobot_v1",
        "repo_id": args.repo_id,
        "run": {
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "raw_dir": str(args.raw_dir),
            "report_dir": str(report_run_dir),
            "config": {
                "fps": args.fps,
                "start_time_s": args.start_time_s,
                "end_time_s": args.end_time_s,
                "gripper_scale": args.gripper_scale,
                "max_image_delta_ms": args.max_image_delta_ms,
                "max_state_delta_ms": args.max_state_delta_ms,
                "max_action_delta_ms": args.max_action_delta_ms,
                "max_action_step_deg": args.max_action_step_deg,
                "use_videos": args.use_videos,
                "video_codec": args.video_codec,
                "dry_run": args.dry_run,
            },
        },
        "state_names": list(STATE_NAMES),
        "action_names": list(ACTION_NAMES),
        "units": {"arm_joints": "radian", "grippers": "normalized_0_to_1"},
        "sampling": "nearest timestamp after fixed-rate resampling",
        "episodes": [],
    }
    plans = []
    try:
        episode_config = _load_episode_config(args.episode_config)
        episode_dirs = _find_episode_dirs(args.raw_dir)
        conversion_report["run"]["mode"] = "single" if len(episode_dirs) == 1 else "batch"
        for episode_dir in episode_dirs:
            include, task, start_time_s, end_time_s = _episode_settings(episode_dir.name, episode_config, args)
            if not include:
                print(f"Skipping {episode_dir.name} (include=false)")
                continue
            print(f"Reading {episode_dir.name} ...")
            data = _load_episode(episode_dir)
            plan = _make_episode_plan(
                data,
                task=task,
                fps=args.fps,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                gripper_scale=args.gripper_scale,
                max_image_delta_ms=args.max_image_delta_ms,
                max_state_delta_ms=args.max_state_delta_ms,
                max_action_delta_ms=args.max_action_delta_ms,
                max_action_step_deg=args.max_action_step_deg,
            )
            annotations = {
                key: episode_config.get(episode_dir.name, {}).get(key)
                for key in ("success", "failure_reason", "notes")
                if key in episode_config.get(episode_dir.name, {})
            }
            plan.report["annotations"] = annotations
            if annotations.get("success") is False:
                plan.report["warnings"].append(
                    "该 Episode 被标记为失败但仍被选中转换；普通行为克隆会把失败动作作为正样本。"
                )
            plans.append(plan)
            conversion_report["episodes"].append(plan.report)
            output = plan.report["output"]
            print(
                f"  {output['frames']} frames at {output['fps']} Hz, duration {output['duration_s']:.2f} s; "
                f"warnings={len(plan.report['warnings'])}"
            )
            for warning in plan.report["warnings"]:
                print(f"  WARNING: {warning}")

        if not plans:
            raise ValueError("No episodes selected for conversion")

        warning_count = sum(len(plan.report["warnings"]) for plan in plans)
        conversion_report["quality_summary"] = {
            "status": "warning" if warning_count else "pass",
            "episode_count": len(plans),
            "warning_count": warning_count,
        }
        image_shapes = _probe_image_shapes(plans[0])
        for plan in plans[1:]:
            current_shapes = _probe_image_shapes(plan)
            if current_shapes != image_shapes:
                raise ValueError(
                    f"Image shapes changed between episodes: {image_shapes} != {current_shapes} in {plan.data.name}"
                )
        conversion_report["image_shapes_hwc"] = {feature: list(shape) for feature, shape in image_shapes.items()}
        if args.dry_run:
            conversion_report["run"]["status"] = "dry_run_validated"
            _save_conversion_report(conversion_report, report_run_dir)
            print(json.dumps(conversion_report, indent=2, ensure_ascii=False))
            print(f"Saved dry-run reports to {report_run_dir}")
            return

        conversion_report["run"]["status"] = "validated"
        _save_conversion_report(conversion_report, report_run_dir)
        dataset, output_path = _create_dataset(args, image_shapes)
        for plan in plans:
            _write_episode(dataset, plan, image_shapes)
        if hasattr(dataset, "consolidate"):
            dataset.consolidate()
        conversion_report["run"]["status"] = "completed"
        conversion_report["run"]["output_path"] = str(output_path)
        conversion_report["run"]["completed_at_utc"] = datetime.now(UTC).isoformat()
        _save_conversion_report(conversion_report, report_run_dir)
        _save_conversion_report(conversion_report, output_path / "meta", stem="f1_conversion")
        print(json.dumps(conversion_report, indent=2, ensure_ascii=False))
        print(f"Saved LeRobot dataset to {output_path}")
        print(f"Saved conversion reports to {report_run_dir} and {output_path / 'meta'}")

        if args.push_to_hub:
            dataset.push_to_hub()
    except Exception as error:
        conversion_report["run"]["status"] = "failed"
        conversion_report["run"]["error"] = f"{type(error).__name__}: {error}"
        conversion_report["run"]["failed_at_utc"] = datetime.now(UTC).isoformat()
        _save_conversion_report(conversion_report, report_run_dir)
        print(f"Saved failed-run reports to {report_run_dir}")
        raise


if __name__ == "__main__":
    main(tyro.cli(Args))
