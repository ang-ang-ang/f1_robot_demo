from __future__ import annotations

import dataclasses
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

from data_process import f1_schema

MCAP_MAGIC = b"\x89MCAP0\r\n"
MCAP_SCHEMA = 0x03
MCAP_CHANNEL = 0x04
MCAP_MESSAGE = 0x05
MCAP_CHUNK = 0x06
MCAP_DATA_END = 0x0F


@dataclasses.dataclass(frozen=True)
class ImageReference:
    path: Path
    offset: int
    size: int
    format: str
    frame_id: str


@dataclasses.dataclass
class MessageSample:
    source_path: Path
    record_index: int
    topic_index: int
    topic: str
    schema_name: str | None
    sequence: int
    log_time_ns: int
    publish_time_ns: int
    cdr_file_offset: int
    cdr_size: int
    header_time_ns: int | None = None
    names: tuple[str, ...] | None = None
    values: np.ndarray | None = None
    image: ImageReference | None = None

    @property
    def primary_time_ns(self) -> int:
        if self.header_time_ns is not None:
            return self.header_time_ns
        if self.publish_time_ns > 0:
            return self.publish_time_ns
        return self.log_time_ns


@dataclasses.dataclass(frozen=True)
class McapChunkInfo:
    source_path: Path
    records_offset: int
    records_size: int
    uncompressed_crc: int
    compression: str


@dataclasses.dataclass
class EpisodeData:
    name: str
    root: Path
    source_files: list[Path]
    topics: dict[str, list[MessageSample]] = dataclasses.field(default_factory=dict)
    chunks: list[McapChunkInfo] = dataclasses.field(default_factory=list)
    data_section_crcs: dict[Path, int] = dataclasses.field(default_factory=dict)
    message_count: int = 0

    def topic(self, name: str) -> list[MessageSample]:
        return self.topics.get(name, [])


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


def _iter_records(buffer: bytes | memoryview) -> Iterator[tuple[int, memoryview, int]]:
    position = 0
    while position + 9 <= len(buffer):
        opcode = buffer[position]
        length = struct.unpack_from("<Q", buffer, position + 1)[0]
        data_position = position + 9
        end = data_position + length
        if end > len(buffer):
            raise ValueError(f"Invalid MCAP record at byte {position}: record exceeds its container")
        yield opcode, memoryview(buffer)[data_position:end], data_position
        position = end
    if position != len(buffer):
        raise ValueError(f"MCAP container has {len(buffer) - position} trailing bytes")


def _decode_cdr_header(buffer: bytes | memoryview) -> tuple[int, str, int]:
    if bytes(buffer[:4]) != b"\x00\x01\x00\x00":
        raise ValueError(f"Only little-endian CDR is supported, got {bytes(buffer[:4]).hex()}")
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
    end = position + value_count * 8
    if end > len(buffer):
        raise ValueError("JointState position array exceeds the CDR message")
    values = np.frombuffer(buffer, dtype="<f8", count=value_count, offset=position).copy()
    return timestamp_ns, tuple(names), values


def _decode_gripper_command(buffer: bytes | memoryview) -> np.ndarray:
    if bytes(buffer[:4]) != b"\x00\x01\x00\x00":
        raise ValueError(f"Only little-endian CDR is supported, got {bytes(buffer[:4]).hex()}")
    return np.asarray([struct.unpack_from("<d", buffer, 4)[0]], dtype=np.float64)


def _decode_image(
    buffer: bytes | memoryview,
    *,
    cdr_file_offset: int,
    source_path: Path,
) -> tuple[int, ImageReference]:
    timestamp_ns, frame_id, position = _decode_cdr_header(buffer)
    position = _align_cdr(position, 4)
    image_format, position = _read_string(buffer, position)
    position = _align_cdr(position, 4)
    size, position = _read_u32(buffer, position)
    if position + size > len(buffer):
        raise ValueError(f"Compressed image exceeds its MCAP message in {source_path}")
    return timestamp_ns, ImageReference(
        path=source_path,
        offset=cdr_file_offset + position,
        size=size,
        format=image_format,
        frame_id=frame_id,
    )


def _decode_known_message(sample: MessageSample, cdr: memoryview) -> None:
    if sample.schema_name == "sensor_msgs/msg/CompressedImage":
        sample.header_time_ns, sample.image = _decode_image(
            cdr,
            cdr_file_offset=sample.cdr_file_offset,
            source_path=sample.source_path,
        )
    elif sample.schema_name == "sensor_msgs/msg/JointState":
        sample.header_time_ns, sample.names, sample.values = _decode_joint_state(cdr)
    elif sample.schema_name == "control_msgs/msg/GripperCommand":
        sample.values = _decode_gripper_command(cdr)


def _read_mcap(path: Path, episode: EpisodeData, record_index: int) -> int:
    schemas: dict[int, str] = {}
    channels: dict[int, tuple[int, str]] = {}
    topic_counts = {topic: len(samples) for topic, samples in episode.topics.items()}
    with path.open("rb") as handle:
        if handle.read(8) != MCAP_MAGIC:
            raise ValueError(f"Not an MCAP file: {path}")
        while True:
            outer_offset = handle.tell()
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
            if opcode == MCAP_DATA_END:
                episode.data_section_crcs[path] = struct.unpack_from("<I", payload, 0)[0]
                continue
            if opcode != MCAP_CHUNK:
                continue

            position = 0
            _, position = _read_u64(payload, position)
            _, position = _read_u64(payload, position)
            uncompressed_size, position = _read_u64(payload, position)
            uncompressed_crc, position = _read_u32(payload, position)
            compression, position = _read_string(payload, position)
            records, position, records_position = _read_blob64(payload, position)
            if compression:
                raise ValueError(
                    f"{path} uses {compression!r} chunk compression; this F1 pipeline supports uncompressed chunks only"
                )
            if len(records) != uncompressed_size:
                raise ValueError(f"Uncompressed chunk size mismatch in {path}")
            if position != len(payload):
                raise ValueError(f"Unexpected bytes after MCAP chunk in {path}")
            records_file_offset = outer_offset + 9 + records_position
            episode.chunks.append(
                McapChunkInfo(
                    source_path=path,
                    records_offset=records_file_offset,
                    records_size=len(records),
                    uncompressed_crc=uncompressed_crc,
                    compression=compression,
                )
            )

            for inner_opcode, data, data_position in _iter_records(records):
                if inner_opcode == MCAP_SCHEMA:
                    schema_position = 0
                    schema_id, schema_position = _read_u16(data, schema_position)
                    schema_name, schema_position = _read_string(data, schema_position)
                    _, schema_position = _read_string(data, schema_position)
                    _, schema_position = _read_blob32(data, schema_position)
                    schemas[schema_id] = schema_name
                    continue
                if inner_opcode == MCAP_CHANNEL:
                    channel_position = 0
                    channel_id, channel_position = _read_u16(data, channel_position)
                    schema_id, channel_position = _read_u16(data, channel_position)
                    topic, channel_position = _read_string(data, channel_position)
                    _, channel_position = _read_string(data, channel_position)
                    channels[channel_id] = (schema_id, topic)
                    continue
                if inner_opcode != MCAP_MESSAGE:
                    continue

                channel_id = struct.unpack_from("<H", data, 0)[0]
                if channel_id not in channels:
                    raise ValueError(f"Message references unknown channel {channel_id} in {path}")
                schema_id, topic = channels[channel_id]
                topic_index = topic_counts.get(topic, 0)
                topic_counts[topic] = topic_index + 1
                cdr_file_offset = records_file_offset + data_position + 22
                sample = MessageSample(
                    source_path=path,
                    record_index=record_index,
                    topic_index=topic_index,
                    topic=topic,
                    schema_name=schemas.get(schema_id),
                    sequence=struct.unpack_from("<I", data, 2)[0],
                    log_time_ns=struct.unpack_from("<Q", data, 6)[0],
                    publish_time_ns=struct.unpack_from("<Q", data, 14)[0],
                    cdr_file_offset=cdr_file_offset,
                    cdr_size=len(data) - 22,
                )
                if topic in f1_schema.IMAGE_TOPICS or sample.schema_name in {
                    "sensor_msgs/msg/JointState",
                    "control_msgs/msg/GripperCommand",
                }:
                    _decode_known_message(sample, data[22:])
                episode.topics.setdefault(topic, []).append(sample)
                record_index += 1
    return record_index


def find_episode_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    if list(raw_dir.glob("*.mcap")):
        return [raw_dir]
    episode_dirs = sorted(path for path in raw_dir.glob("episode_*") if path.is_dir() and list(path.glob("*.mcap")))
    if not episode_dirs:
        raise FileNotFoundError(f"No MCAP episodes found under {raw_dir}")
    return episode_dirs


def load_episode(episode_dir: Path) -> EpisodeData:
    source_files = sorted(episode_dir.glob("*.mcap"))
    if not source_files:
        raise FileNotFoundError(f"No MCAP files found in {episode_dir}")
    episode = EpisodeData(name=episode_dir.name, root=episode_dir, source_files=source_files)
    record_index = 0
    for source_file in source_files:
        record_index = _read_mcap(source_file, episode, record_index)
    episode.message_count = record_index
    return episode


def primary_timestamps(samples: list[MessageSample]) -> np.ndarray:
    return np.asarray([sample.primary_time_ns for sample in samples], dtype=np.int64)


def header_timestamps(samples: list[MessageSample]) -> np.ndarray | None:
    if not samples or any(sample.header_time_ns is None for sample in samples):
        return None
    return np.asarray([sample.header_time_ns for sample in samples], dtype=np.int64)


def read_image_bytes(reference: ImageReference, handles: dict[Path, BinaryIO]) -> bytes:
    handle = handles[reference.path]
    handle.seek(reference.offset)
    payload = handle.read(reference.size)
    if len(payload) != reference.size:
        raise ValueError(f"Truncated image payload in {reference.path}")
    return payload
