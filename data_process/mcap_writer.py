from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from data_process.mcap import MCAP_MAGIC

MCAP_HEADER = 0x01
MCAP_FOOTER = 0x02
MCAP_SCHEMA = 0x03
MCAP_CHANNEL = 0x04
MCAP_MESSAGE = 0x05
MCAP_CHUNK = 0x06
MCAP_METADATA = 0x0C
MCAP_DATA_END = 0x0F


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _blob32(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def _blob64(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _map(values: dict[str, str]) -> bytes:
    payload = b"".join(_string(key) + _string(value) for key, value in values.items())
    return struct.pack("<I", len(payload)) + payload


def _record(opcode: int, payload: bytes) -> bytes:
    return bytes([opcode]) + struct.pack("<Q", len(payload)) + payload


class McapWriter:
    """Small uncompressed chunked MCAP writer for lossless CDR message copying."""

    def __init__(
        self,
        path: Path,
        *,
        profile: str = "ros2",
        library: str = "openpi-f1-resampler",
        chunk_size: int = 64 * 1024 * 1024,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("wb")
        self._chunk_size = chunk_size
        self._records = bytearray()
        self._chunk_start_ns: int | None = None
        self._chunk_end_ns: int | None = None
        self._next_schema_id = 1
        self._next_channel_id = 1
        self._finished = False
        self._handle.write(MCAP_MAGIC)
        self._write_outer(MCAP_HEADER, _string(profile) + _string(library))

    def _write_outer(self, opcode: int, payload: bytes) -> None:
        self._handle.write(_record(opcode, payload))

    def _append_record(self, opcode: int, payload: bytes, *, log_time_ns: int | None = None) -> None:
        record = _record(opcode, payload)
        if self._records and len(self._records) + len(record) > self._chunk_size:
            self._flush_chunk()
        self._records.extend(record)
        if log_time_ns is not None:
            self._chunk_start_ns = (
                log_time_ns if self._chunk_start_ns is None else min(self._chunk_start_ns, log_time_ns)
            )
            self._chunk_end_ns = log_time_ns if self._chunk_end_ns is None else max(self._chunk_end_ns, log_time_ns)

    def register_schema(self, *, name: str, encoding: str, data: bytes) -> int:
        schema_id = self._next_schema_id
        self._next_schema_id += 1
        payload = struct.pack("<H", schema_id) + _string(name) + _string(encoding) + _blob32(data)
        self._append_record(MCAP_SCHEMA, payload)
        return schema_id

    def register_json_schema(self, *, name: str, schema: dict[str, Any]) -> int:
        return self.register_schema(
            name=name,
            encoding="jsonschema",
            data=json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        )

    def register_channel(
        self,
        *,
        topic: str,
        message_encoding: str,
        schema_id: int,
        metadata: dict[str, str] | None = None,
    ) -> int:
        channel_id = self._next_channel_id
        self._next_channel_id += 1
        payload = (
            struct.pack("<HH", channel_id, schema_id)
            + _string(topic)
            + _string(message_encoding)
            + _map(metadata or {})
        )
        self._append_record(MCAP_CHANNEL, payload)
        return channel_id

    def add_message(
        self,
        *,
        channel_id: int,
        sequence: int,
        log_time_ns: int,
        publish_time_ns: int,
        data: bytes,
    ) -> None:
        payload = struct.pack("<HIQQ", channel_id, sequence, log_time_ns, publish_time_ns) + data
        self._append_record(MCAP_MESSAGE, payload, log_time_ns=log_time_ns)

    def add_json_message(
        self,
        *,
        channel_id: int,
        sequence: int,
        log_time_ns: int,
        message: dict[str, Any],
    ) -> None:
        self.add_message(
            channel_id=channel_id,
            sequence=sequence,
            log_time_ns=log_time_ns,
            publish_time_ns=log_time_ns,
            data=json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        )

    def add_metadata(self, *, name: str, metadata: dict[str, str]) -> None:
        self._flush_chunk()
        self._write_outer(MCAP_METADATA, _string(name) + _map(metadata))

    def _flush_chunk(self) -> None:
        if not self._records:
            return
        records = bytes(self._records)
        start_ns = 0 if self._chunk_start_ns is None else self._chunk_start_ns
        end_ns = 0 if self._chunk_end_ns is None else self._chunk_end_ns
        payload = struct.pack("<QQQI", start_ns, end_ns, len(records), 0) + _string("") + _blob64(records)
        self._write_outer(MCAP_CHUNK, payload)
        self._records.clear()
        self._chunk_start_ns = None
        self._chunk_end_ns = None

    def finish(self) -> None:
        if self._finished:
            return
        self._flush_chunk()
        self._write_outer(MCAP_DATA_END, struct.pack("<I", 0))
        self._write_outer(MCAP_FOOTER, struct.pack("<QQI", 0, 0, 0))
        self._handle.write(MCAP_MAGIC)
        self._handle.close()
        self._finished = True

    def __enter__(self) -> McapWriter:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is None:
            self.finish()
        else:
            self._handle.close()
