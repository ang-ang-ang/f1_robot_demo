from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from data_process.mcap import MCAP_MAGIC, load_episode
from data_process.mcap_writer import McapWriter
from data_process.resampling import nearest_indices
from data_process.resampling_analysis import _timeline_stats


def test_nearest_indices_preserve_signed_residual_and_choose_left_on_tie() -> None:
    source = np.asarray([0, 40, 80, 120], dtype=np.int64) * 1_000_000
    target = np.asarray([20, 55, 115], dtype=np.int64) * 1_000_000

    indices, residual_ns = nearest_indices(source, target)

    np.testing.assert_array_equal(indices, [0, 1, 3])
    np.testing.assert_array_equal(residual_ns / 1e6, [-20, -15, 5])


def test_timeline_stats_count_duplicates_negative_steps_and_period_gaps() -> None:
    timestamps = np.asarray([0, 33, 33, 20, 53, 153], dtype=np.int64) * 1_000_000

    stats, events = _timeline_stats(
        timestamps,
        duplicate_epsilon_ms=0.05,
        gap_warning_factor=2.2,
        gap_fail_factor=5.0,
    )

    assert stats["duplicate_count"] == 1
    assert stats["negative_count"] == 1
    assert stats["nonpositive_count"] == 2
    assert stats["warning_gap_count"] == 1
    assert len(events) == 1
    assert events[0]["period_multiple"] > 3


def test_minimal_chunked_writer_round_trip(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode_000001"
    path = episode_dir / "aligned_20hz.mcap"
    with McapWriter(path, chunk_size=128) as writer:
        schema_id = writer.register_json_schema(
            name="test.Sample",
            schema={"type": "object", "properties": {"value": {"type": "number"}}},
        )
        channel_id = writer.register_channel(topic="/test", message_encoding="json", schema_id=schema_id)
        for index in range(3):
            writer.add_json_message(
                channel_id=channel_id,
                sequence=index,
                log_time_ns=1_000_000_000 + index * 50_000_000,
                message={"value": index},
            )

    episode = load_episode(episode_dir)

    assert episode.message_count == 3
    assert len(episode.topic("/test")) == 3
    assert episode.topic("/test")[0].schema_name == "test.Sample"
    assert [sample.sequence for sample in episode.topic("/test")] == [0, 1, 2]


def test_metadata_is_written_outside_chunks(tmp_path: Path) -> None:
    path = tmp_path / "metadata.mcap"
    with McapWriter(path) as writer:
        schema_id = writer.register_json_schema(name="test.Sample", schema={"type": "object"})
        channel_id = writer.register_channel(topic="/test", message_encoding="json", schema_id=schema_id)
        writer.add_metadata(name="test.metadata", metadata={"source": "unit-test"})
        writer.add_json_message(
            channel_id=channel_id,
            sequence=0,
            log_time_ns=1_000_000_000,
            message={},
        )

    payload = path.read_bytes()
    position = len(MCAP_MAGIC)
    outer_opcodes: list[int] = []
    chunk_opcodes: list[int] = []
    while position < len(payload) - len(MCAP_MAGIC):
        opcode = payload[position]
        length = struct.unpack_from("<Q", payload, position + 1)[0]
        record = memoryview(payload)[position + 9 : position + 9 + length]
        outer_opcodes.append(opcode)
        if opcode == 0x06:
            chunk_position = 8 + 8 + 8 + 4
            compression_length = struct.unpack_from("<I", record, chunk_position)[0]
            chunk_position += 4 + compression_length
            records_length = struct.unpack_from("<Q", record, chunk_position)[0]
            chunk_position += 8
            records_end = chunk_position + records_length
            while chunk_position < records_end:
                chunk_opcode = record[chunk_position]
                chunk_length = struct.unpack_from("<Q", record, chunk_position + 1)[0]
                chunk_opcodes.append(chunk_opcode)
                chunk_position += 9 + chunk_length
        position += 9 + length

    assert 0x0C in outer_opcodes
    assert 0x0C not in chunk_opcodes
