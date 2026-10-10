# ruff: noqa: RUF001
from __future__ import annotations

import csv
import json
import re
import shutil
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from data_process import f1_schema
from data_process.mcap import EpisodeData, MessageSample, find_episode_dirs, load_episode, read_message_bytes
from data_process.mcap_writer import McapWriter

McapMode = Literal["aligned", "unique"]


def nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source_ns.ndim != 1 or target_ns.ndim != 1 or len(source_ns) < 2:
        raise ValueError("nearest sampling requires one-dimensional timelines and at least two source samples")
    if np.any(np.diff(source_ns) <= 0):
        raise ValueError("source timestamps must be strictly increasing before resampling")
    right = np.searchsorted(source_ns, target_ns, side="left")
    right = np.clip(right, 1, len(source_ns) - 1)
    left = right - 1
    choose_left = np.abs(target_ns - source_ns[left]) <= np.abs(source_ns[right] - target_ns)
    indices = np.where(choose_left, left, right)
    return indices, source_ns[indices] - target_ns


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "stream"


def _read_cleaning_rows(episode_dir: Path) -> tuple[dict[int, dict[str, str]], Path | None]:
    manifest_path = episode_dir / "f1_cleaning_manifest.json"
    candidates = [episode_dir / "f1_timestamp_corrections.csv"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        corrections = manifest.get("timestamp_corrections")
        if corrections:
            candidate = Path(corrections)
            candidates.insert(0, candidate if candidate.is_absolute() else episode_dir / candidate)
        report_dir = manifest.get("report_dir")
        if report_dir:
            report_path = Path(report_dir)
            candidates.append(report_path / "timestamp_corrections.csv")
            if not report_path.is_absolute():
                candidates.extend(
                    parent / report_path / "timestamp_corrections.csv" for parent in episode_dir.resolve().parents
                )
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {int(row["mcap_record_index"]): row for row in rows}, candidate
    return {}, None


def _optional_int(value: str | int | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(value)


def _optional_float(value: str | float | None) -> float | None:
    if value in {None, "", "None"}:
        return None
    return float(value)


def _sample_row(
    episode: EpisodeData,
    sample: MessageSample,
    source_index: int,
    cleaning_row: dict[str, str] | None,
) -> dict[str, Any]:
    cleaned_header_ns = sample.header_time_ns
    original_header_ns = cleaned_header_ns
    repair_method = "original"
    risk = "none"
    correction_ms = 0.0
    if cleaning_row is not None:
        original_header_ns = _optional_int(cleaning_row.get("header_time_ns"))
        repair_method = cleaning_row.get("repair_method") or "original"
        risk = cleaning_row.get("risk") or "none"
        correction_ms = _optional_float(cleaning_row.get("correction_ms")) or 0.0
    original_primary_ns = original_header_ns if original_header_ns is not None else sample.primary_time_ns
    cleaned_primary_ns = sample.primary_time_ns
    return {
        "source_index": source_index,
        "mcap_record_index": sample.record_index,
        "source_file": str(sample.source_path.relative_to(episode.root)),
        "sequence": sample.sequence,
        "original_header_time_ns": original_header_ns,
        "cleaned_header_time_ns": cleaned_header_ns,
        "publish_time_ns": sample.publish_time_ns,
        "log_time_ns": sample.log_time_ns,
        "original_primary_time_ns": original_primary_ns,
        "cleaned_primary_time_ns": cleaned_primary_ns,
        "cleaning_correction_ms": correction_ms,
        "repair_method": repair_method,
        "risk": risk,
    }


def build_episode_mapping(
    episode: EpisodeData,
    *,
    fps: float,
    topics: list[str],
    require_cleaning_manifest: bool,
) -> tuple[dict[str, Any], dict[str, list[MessageSample]]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    manifest_path = episode.root / "f1_cleaning_manifest.json"
    if require_cleaning_manifest and not manifest_path.exists():
        raise ValueError(f"{episode.name} is not a cleaned episode: missing {manifest_path.name}")
    cleaning_rows, cleaning_rows_path = _read_cleaning_rows(episode.root)
    if manifest_path.exists() and not cleaning_rows:
        manifest = json.loads(manifest_path.read_text())
        changed_count = int(manifest.get("diagnostics", {}).get("changed_frame_count", 0))
        if changed_count:
            raise FileNotFoundError(
                f"{episode.name} reports {changed_count} cleaned timestamps but its timestamp_corrections.csv "
                "cannot be found; keep the cleaning audit file with the cleaned episode or repair the manifest path"
            )
    streams: dict[str, Any] = {}
    stream_samples: dict[str, list[MessageSample]] = {}
    timelines: dict[str, np.ndarray] = {}
    for topic in topics:
        samples = episode.topic(topic)
        if len(samples) < 2:
            raise ValueError(f"{episode.name} {topic} has fewer than two samples")
        timeline = np.asarray([sample.primary_time_ns for sample in samples], dtype=np.int64)
        if np.any(np.diff(timeline) <= 0):
            count = int(np.sum(np.diff(timeline) <= 0))
            raise ValueError(f"{episode.name} {topic} still has {count} non-increasing primary timestamps")
        rows = [
            _sample_row(episode, sample, index, cleaning_rows.get(sample.record_index))
            for index, sample in enumerate(samples)
        ]
        streams[topic] = {
            "category": f1_schema.topic_category(topic),
            "time_source": "header"
            if all(sample.header_time_ns is not None for sample in samples)
            else "publish_or_log",
            "source_count": len(samples),
            "source_samples": rows,
            "target_mappings": [],
        }
        stream_samples[topic] = samples
        timelines[topic] = timeline

    common_start_ns = max(int(timeline[0]) for timeline in timelines.values())
    common_end_ns = min(int(timeline[-1]) for timeline in timelines.values())
    if common_end_ns <= common_start_ns:
        raise ValueError(f"{episode.name} has no common time range across {len(topics)} streams")
    period_ns = round(1e9 / fps)
    target_count = int((common_end_ns - common_start_ns) // period_ns) + 1
    target_ns = common_start_ns + np.arange(target_count, dtype=np.int64) * period_ns

    for topic, timeline in timelines.items():
        indices, residual_ns = nearest_indices(timeline, target_ns)
        previous_index: int | None = None
        mappings = []
        for target_index, (source_index, residual) in enumerate(zip(indices, residual_ns, strict=True)):
            source_row = streams[topic]["source_samples"][int(source_index)]
            mappings.append(
                {
                    "target_index": target_index,
                    "target_time_ns": int(target_ns[target_index]),
                    "source_index": int(source_index),
                    "cleaned_time_ns": int(timeline[source_index]),
                    "original_time_ns": int(source_row["original_primary_time_ns"]),
                    "residual_ns": int(residual),
                    "residual_ms": float(residual / 1e6),
                    "raw_to_target_residual_ms": float(
                        (int(source_row["original_primary_time_ns"]) - int(target_ns[target_index])) / 1e6
                    ),
                    "reused_previous_source": previous_index == int(source_index),
                }
            )
            previous_index = int(source_index)
        unique_indices = sorted({int(index) for index in indices})
        streams[topic]["target_mappings"] = mappings
        streams[topic]["selected_unique_source_indices"] = unique_indices
        streams[topic]["selected_unique_count"] = len(unique_indices)
        streams[topic]["omitted_source_count"] = len(timeline) - len(unique_indices)
        streams[topic]["reused_target_count"] = sum(row["reused_previous_source"] for row in mappings)

    return (
        {
            "format": "f1_mcap_resampling_mapping_v1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "episode": episode.name,
            "source_episode": str(episode.root),
            "cleaning_manifest": str(manifest_path) if manifest_path.exists() else None,
            "cleaning_corrections": str(cleaning_rows_path) if cleaning_rows_path is not None else None,
            "fps": fps,
            "period_ns": period_ns,
            "strategy": "nearest",
            "time_basis": "header_else_publish_else_log",
            "common_start_ns": common_start_ns,
            "common_end_ns": common_end_ns,
            "target_count": target_count,
            "topics": topics,
            "streams": streams,
        },
        stream_samples,
    )


def _stream_metric_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "target_index": {"type": "integer"},
            "target_time_ns": {"type": "integer"},
            "target_elapsed_s": {"type": "number"},
            "source_index": {"type": "integer"},
            "source_time_ns": {"type": "integer"},
            "original_time_ns": {"type": "integer"},
            "residual_ms": {"type": "number"},
            "abs_residual_ms": {"type": "number"},
            "raw_to_target_residual_ms": {"type": "number"},
            "cleaning_correction_ms": {"type": "number"},
            "reused_previous_source": {"type": "boolean"},
            "repair_method": {"type": "string"},
            "risk": {"type": "string"},
        },
        "required": ["target_index", "target_time_ns", "source_index", "residual_ms"],
    }


def _cross_stream_schema(properties: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "target_index": {"type": "integer"},
            "target_time_ns": {"type": "integer"},
            "target_elapsed_s": {"type": "number"},
            **{property_name: {"type": "number"} for property_name in properties},
        },
        "required": ["target_index", "target_time_ns"],
    }


def write_resampled_mcap(
    path: Path,
    episode: EpisodeData,
    mapping: dict[str, Any],
    stream_samples: dict[str, list[MessageSample]],
    *,
    mode: McapMode,
    include_diagnostics: bool,
) -> None:
    if mode not in {"aligned", "unique"}:
        raise ValueError(f"unsupported MCAP mode: {mode}")
    handles: dict[Path, Any] = {}
    with ExitStack() as stack:
        for source_file in episode.source_files:
            handles[source_file] = stack.enter_context(source_file.open("rb"))
        with McapWriter(path) as writer:
            schema_ids: dict[tuple[str, str, bytes], int] = {}
            output_channels: dict[str, int] = {}
            for topic in mapping["topics"]:
                samples = stream_samples[topic]
                source_channels = [episode.channels[(sample.source_path, sample.channel_id)] for sample in samples]
                first_channel = source_channels[0]
                first_schema = episode.schemas.get((first_channel.source_path, first_channel.schema_id))
                if first_schema is None:
                    raise ValueError(f"{episode.name} {topic} has no schema")
                schema_key = (first_schema.name, first_schema.encoding, first_schema.data)
                schema_id = schema_ids.get(schema_key)
                if schema_id is None:
                    schema_id = writer.register_schema(
                        name=first_schema.name,
                        encoding=first_schema.encoding,
                        data=first_schema.data,
                    )
                    schema_ids[schema_key] = schema_id
                for channel in source_channels[1:]:
                    schema = episode.schemas.get((channel.source_path, channel.schema_id))
                    if (
                        schema is None
                        or (schema.name, schema.encoding, schema.data) != schema_key
                        or channel.message_encoding != first_channel.message_encoding
                    ):
                        raise ValueError(f"{episode.name} {topic} changes schema or encoding across source files")
                output_channels[topic] = writer.register_channel(
                    topic=topic,
                    message_encoding=first_channel.message_encoding,
                    schema_id=schema_id,
                    metadata=first_channel.metadata,
                )

            diagnostic_channels: dict[str, int] = {}
            cross_channel: int | None = None
            cross_properties: dict[tuple[str, str], str] = {}
            if include_diagnostics:
                metric_schema_id = writer.register_json_schema(
                    name="openpi.f1.ResamplingMetric",
                    schema=_stream_metric_schema(),
                )
                for topic in mapping["topics"]:
                    diagnostic_channels[topic] = writer.register_channel(
                        topic=f"/f1/resampling/{_slug(topic)}",
                        message_encoding="json",
                        schema_id=metric_schema_id,
                        metadata={"source_topic": topic},
                    )
                for left_index, left_topic in enumerate(mapping["topics"]):
                    for right_topic in mapping["topics"][left_index + 1 :]:
                        cross_properties[(left_topic, right_topic)] = f"{_slug(left_topic)}__{_slug(right_topic)}_ms"
                cross_schema_id = writer.register_json_schema(
                    name="openpi.f1.CrossStreamSkew",
                    schema=_cross_stream_schema(list(cross_properties.values())),
                )
                cross_channel = writer.register_channel(
                    topic="/f1/resampling/cross_stream",
                    message_encoding="json",
                    schema_id=cross_schema_id,
                )
                writer.add_metadata(
                    name="openpi.f1.resampling",
                    metadata={
                        "mapping_format": mapping["format"],
                        "mcap_mode": mode,
                        "fps": str(mapping["fps"]),
                        "payload_header_policy": "preserve_cleaned_source_header",
                    },
                )

            if mode == "aligned":
                for target_index in range(mapping["target_count"]):
                    target_time_ns = mapping["common_start_ns"] + target_index * mapping["period_ns"]
                    selected_times: dict[str, int] = {}
                    for topic in mapping["topics"]:
                        stream = mapping["streams"][topic]
                        selection = stream["target_mappings"][target_index]
                        source_index = selection["source_index"]
                        sample = stream_samples[topic][source_index]
                        source_row = stream["source_samples"][source_index]
                        writer.add_message(
                            channel_id=output_channels[topic],
                            sequence=target_index,
                            log_time_ns=target_time_ns,
                            publish_time_ns=target_time_ns,
                            data=read_message_bytes(sample, handles),
                        )
                        selected_times[topic] = selection["cleaned_time_ns"]
                        if include_diagnostics:
                            writer.add_json_message(
                                channel_id=diagnostic_channels[topic],
                                sequence=target_index,
                                log_time_ns=target_time_ns,
                                message={
                                    "target_index": target_index,
                                    "target_time_ns": target_time_ns,
                                    "target_elapsed_s": target_index / mapping["fps"],
                                    "source_index": source_index,
                                    "source_time_ns": selection["cleaned_time_ns"],
                                    "original_time_ns": selection["original_time_ns"],
                                    "residual_ms": selection["residual_ms"],
                                    "abs_residual_ms": abs(selection["residual_ms"]),
                                    "raw_to_target_residual_ms": selection["raw_to_target_residual_ms"],
                                    "cleaning_correction_ms": source_row["cleaning_correction_ms"],
                                    "reused_previous_source": selection["reused_previous_source"],
                                    "repair_method": source_row["repair_method"],
                                    "risk": source_row["risk"],
                                },
                            )
                    if include_diagnostics and cross_channel is not None:
                        cross_message: dict[str, Any] = {
                            "target_index": target_index,
                            "target_time_ns": target_time_ns,
                            "target_elapsed_s": target_index / mapping["fps"],
                        }
                        for pair, property_name in cross_properties.items():
                            cross_message[property_name] = (selected_times[pair[0]] - selected_times[pair[1]]) / 1e6
                        writer.add_json_message(
                            channel_id=cross_channel,
                            sequence=target_index,
                            log_time_ns=target_time_ns,
                            message=cross_message,
                        )
            else:
                selected: list[tuple[int, str, MessageSample]] = []
                for topic in mapping["topics"]:
                    for source_index in mapping["streams"][topic]["selected_unique_source_indices"]:
                        sample = stream_samples[topic][source_index]
                        selected.append((sample.log_time_ns, topic, sample))
                for _, topic, sample in sorted(selected, key=lambda item: (item[0], item[2].record_index)):
                    writer.add_message(
                        channel_id=output_channels[topic],
                        sequence=sample.sequence,
                        log_time_ns=sample.log_time_ns,
                        publish_time_ns=sample.publish_time_ns,
                        data=read_message_bytes(sample, handles),
                    )


def resample_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    fps: float = 20.0,
    topics: list[str] | None = None,
    mcap_mode: McapMode = "aligned",
    include_diagnostics: bool = True,
    require_cleaning_manifest: bool = True,
    overwrite: bool = False,
    mapping_only: bool = False,
) -> dict[str, Any]:
    selected_topics = list(topics or f1_schema.REQUIRED_TRAINING_TOPICS)
    if len(set(selected_topics)) != len(selected_topics):
        raise ValueError("topics must be unique")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    episode_reports = []
    for episode_dir in find_episode_dirs(raw_dir):
        print(f"Resampling {episode_dir.name} ...")
        episode = load_episode(episode_dir)
        mapping, stream_samples = build_episode_mapping(
            episode,
            fps=fps,
            topics=selected_topics,
            require_cleaning_manifest=require_cleaning_manifest,
        )
        episode_output = output_dir / episode.name
        episode_output.mkdir()
        mapping_path = episode_output / "resampling_mapping.json"
        mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, separators=(",", ":")))
        fps_label = f"{fps:g}".replace(".", "p")
        mcap_path = episode_output / (
            f"aligned_{fps_label}hz.mcap" if mcap_mode == "aligned" else "selected_unique.mcap"
        )
        if not mapping_only:
            write_resampled_mcap(
                mcap_path,
                episode,
                mapping,
                stream_samples,
                mode=mcap_mode,
                include_diagnostics=include_diagnostics and mcap_mode == "aligned",
            )
        episode_reports.append(
            {
                "episode": episode.name,
                "source": str(episode_dir),
                "mapping": str(mapping_path),
                "mcap": None if mapping_only else str(mcap_path),
                "target_count": mapping["target_count"],
                "common_duration_s": (mapping["common_end_ns"] - mapping["common_start_ns"]) / 1e9,
            }
        )
    manifest = {
        "format": "f1_mcap_resampling_run_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(raw_dir),
        "output": str(output_dir),
        "fps": fps,
        "mcap_mode": mcap_mode,
        "mapping_only": mapping_only,
        "topics": selected_topics,
        "episodes": episode_reports,
        "semantics": {
            "aligned": "one selected payload per target and stream; MCAP envelope log/publish time is the target time",
            "payload_header": "the cleaned source Header inside each CDR payload is preserved",
            "mapping": "resampling_mapping.json is the authoritative source lineage and timing-error record",
        },
    }
    (output_dir / "resampling_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
