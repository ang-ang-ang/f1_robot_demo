"""Detect and repair short, isolated F1 JointState spikes.

The repair is deliberately conservative.  A value is changed only when a
short run of abnormal steps is bounded by two trustworthy samples and linear
interpolation gives a clearly different value.  A persistent step, a boundary
event, or a fast but plausible movement is reported for manual review and is
left untouched.

The source MCAP is never modified in place.  Write mode copies an episode and
patches only the serialized ``position`` doubles in an uncompressed MCAP whose
chunk and data-section CRCs are zero.  Use ``--dry-run`` before writing.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
import struct
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import tyro
except ImportError:  # pragma: no cover - exercised only without project dependencies.
    tyro = None

from data_process import f1_schema
from data_process.mcap import EpisodeData, MessageSample, find_episode_dirs, load_episode


RepairStatus = Literal["clean", "repaired", "review", "failed"]
@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """单个 MCAP/episode 目录，或包含 ``episode_*`` 目录的数据集根目录。"""

    output_dir: Path | None = None
    """清洗后数据集根目录；省略时只能生成报告，等价于 dry-run。"""

    report_dir: Path = Path("artifacts/f1_joint_state_repairs")
    """JSON、Markdown 和逐帧修正 CSV 的输出目录。"""

    topic: str = f1_schema.HAL_JOINT_TOPIC
    """默认修复从机状态；可传 ``/lead/joint_states`` 检查示教动作。"""

    joint_prefix: str = "arm_"
    """默认只处理 F1 双臂 14 个关节；传空字符串可处理全部 JointState position。"""

    max_step_deg: float = 15.0
    """超过该单步角度的边被标为异常候选。"""

    max_velocity_deg_s: float = 300.0
    """超过该角速度的边被标为异常候选。"""

    max_repair_run_frames: int = 6
    """一次最多自动修复的连续异常样本数。"""

    max_anchor_delta_deg: float = 15.0
    """异常段两侧锚点允许的最大净位移；防止把持续性动作阶跃当成坏点。"""

    min_correction_deg: float = 5.0
    """插值残差低于该值时不修改，只保留质量告警。"""

    dry_run: bool = False
    """只分析并写报告，不复制或修改 MCAP；未提供 output_dir 时也会自动启用。"""

    overwrite: bool = False
    """允许覆盖已有的清洗后 episode 目录。"""


@dataclasses.dataclass(frozen=True)
class RepairEvent:
    joint: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    max_step_deg: float
    max_velocity_deg_s: float
    max_interpolation_residual_deg: float
    method: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _align_cdr(position: int, alignment: int, base: int = 4) -> int:
    return position + (-(position - base) % alignment)


def _validate_args(args: Args) -> None:
    if args.max_step_deg <= 0 or args.max_velocity_deg_s <= 0:
        raise ValueError("max_step_deg and max_velocity_deg_s must be positive")
    if args.max_repair_run_frames < 1:
        raise ValueError("max_repair_run_frames must be at least 1")
    if args.max_anchor_delta_deg < 0 or args.min_correction_deg <= 0:
        raise ValueError("max_anchor_delta_deg must be non-negative and min_correction_deg positive")


def _new_run_dir(report_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = report_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "episodes").mkdir()
    return run_dir


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _selected_indices(names: tuple[str, ...], prefix: str) -> list[int]:
    indices = [index for index, name in enumerate(names) if not prefix or name.startswith(prefix)]
    if not indices:
        raise ValueError(f"No JointState names match joint_prefix={prefix!r}; names={list(names)!r}")
    return indices


def _timestamp_metrics(timestamps_ns: np.ndarray) -> dict[str, Any]:
    differences_ms = np.diff(timestamps_ns).astype(np.float64) / 1e6
    duration_s = float((timestamps_ns[-1] - timestamps_ns[0]) / 1e9)
    return {
        "count": int(len(timestamps_ns)),
        "duration_s": duration_s,
        "effective_hz": float((len(timestamps_ns) - 1) / duration_s) if duration_s else 0.0,
        "dt_ms": {
            "median": float(np.median(differences_ms)),
            "p95": float(np.percentile(differences_ms, 95)),
            "max": float(np.max(differences_ms)),
            "nonpositive": int(np.sum(differences_ms <= 0)),
        },
    }


def _position_value_offset(sample: MessageSample, names: tuple[str, ...], value_count: int) -> int:
    """Return the absolute file offset of the first JointState position double."""

    # sensor_msgs/msg/JointState CDR: encapsulation, Header, name sequence,
    # then the position sequence.  The parser uses the same alignment rules.
    with sample.source_path.open("rb") as handle:
        handle.seek(sample.cdr_file_offset)
        cdr = handle.read(sample.cdr_size)
    if len(cdr) != sample.cdr_size or bytes(cdr[:4]) != b"\x00\x01\x00\x00":
        raise ValueError(f"Unsupported or truncated JointState CDR in {sample.source_path}")

    position = 4
    if position + 8 > len(cdr):
        raise ValueError("JointState Header is truncated")
    position += 8  # builtin_interfaces/Time: sec + nanosec
    if position + 4 > len(cdr):
        raise ValueError("JointState frame_id length is missing")
    frame_length = struct.unpack_from("<I", cdr, position)[0]
    position += 4 + frame_length
    position = _align_cdr(position, 4)
    if position + 4 > len(cdr):
        raise ValueError("JointState name sequence length is missing")
    encoded_name_count = struct.unpack_from("<I", cdr, position)[0]
    position += 4
    if encoded_name_count != len(names):
        raise ValueError(
            f"JointState name count changed at record {sample.record_index}: "
            f"encoded={encoded_name_count}, decoded={len(names)}"
        )
    for _ in range(encoded_name_count):
        position = _align_cdr(position, 4)
        if position + 4 > len(cdr):
            raise ValueError("JointState name length is missing")
        length = struct.unpack_from("<I", cdr, position)[0]
        position += 4 + length
    position = _align_cdr(position, 4)
    if position + 4 > len(cdr):
        raise ValueError("JointState position sequence length is missing")
    encoded_value_count = struct.unpack_from("<I", cdr, position)[0]
    position += 4
    if encoded_value_count != value_count:
        raise ValueError(
            f"JointState position count changed at record {sample.record_index}: "
            f"encoded={encoded_value_count}, decoded={value_count}"
        )
    position = _align_cdr(position, 8)
    end = position + value_count * 8
    if end > len(cdr):
        raise ValueError("JointState position sequence exceeds its CDR message")
    return sample.cdr_file_offset + position


def _safe_to_patch(episode: EpisodeData) -> None:
    compressed = [chunk for chunk in episode.chunks if chunk.compression]
    nonzero_chunk_crc = [chunk for chunk in episode.chunks if chunk.uncompressed_crc]
    nonzero_data_crc = [path for path, value in episode.data_section_crcs.items() if value]
    if compressed or nonzero_chunk_crc or nonzero_data_crc:
        raise ValueError(
            "JointState patching requires uncompressed chunks with zero chunk/data CRCs; "
            "use a full MCAP reader/writer rewrite for this recording"
        )


def _copy_episode(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output episode already exists: {destination}; pass --overwrite to replace it")
        shutil.rmtree(destination)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {"artifacts", "quality_reports", "conversion_reports", "cleaning_reports"}}
        ignored.update(name for name in names if name == "__pycache__")
        return ignored

    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=ignore)


def _load_joint_series(
    episode: EpisodeData,
    topic: str,
    joint_prefix: str,
) -> tuple[list[MessageSample], tuple[str, ...], np.ndarray, np.ndarray, list[int]]:
    samples = episode.topic(topic)
    if not samples:
        raise ValueError(f"Required topic is missing: {topic}")
    if any(sample.values is None or sample.names is None for sample in samples):
        raise ValueError(f"Topic {topic} contains an undecodable JointState sample")
    names = tuple(samples[0].names or ())
    if not names:
        raise ValueError(f"Topic {topic} has no JointState names")
    if any(tuple(sample.names or ()) != names for sample in samples):
        raise ValueError(f"Topic {topic} changes JointState names/order between messages")
    values = np.stack([sample.values for sample in samples]).astype(np.float64, copy=False)
    timestamps = np.asarray([sample.header_time_ns for sample in samples], dtype=np.int64)
    if not np.isfinite(values).all():
        raise ValueError(f"Topic {topic} contains NaN or Inf")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Topic {topic} has non-increasing Header timestamps")
    return samples, names, values, timestamps, _selected_indices(names, joint_prefix)


def _analyse_joint(
    *,
    joint: str,
    joint_index: int,
    values: np.ndarray,
    timestamps_ns: np.ndarray,
    args: Args,
) -> tuple[list[RepairEvent], list[RepairEvent], np.ndarray]:
    times_s = timestamps_ns.astype(np.float64) / 1e9
    dt_s = np.diff(times_s)
    differences = np.diff(values[:, joint_index])
    velocities = differences / dt_s
    bad_edges = (np.abs(differences) > args.max_step_deg) | (np.abs(velocities) > args.max_velocity_deg_s)
    repair_events: list[RepairEvent] = []
    review_events: list[RepairEvent] = []
    corrected = values[:, joint_index].copy()

    for edge_start, edge_end in _contiguous_runs(np.flatnonzero(bad_edges)):
        left = edge_start
        right = edge_end + 1
        run_length = right - left - 1
        max_step = float(np.max(np.abs(differences[edge_start : edge_end + 1])))
        max_velocity = float(np.max(np.abs(velocities[edge_start : edge_end + 1])))
        base_kwargs = {
            "joint": joint,
            # The event covers the abnormal edge(s), with adjacent samples as
            # the interpolation anchors when a repair is possible.
            "start_frame": left,
            "end_frame": right,
            "start_time_s": float(times_s[left] - times_s[0]),
            "end_time_s": float(times_s[right] - times_s[0]),
            "max_step_deg": max_step,
            "max_velocity_deg_s": max_velocity,
        }
        # ``left`` and ``right`` are the samples joined by the abnormal edge
        # run.  A single abnormal edge has no return sample, so it is kept for
        # review instead of being silently rewritten.
        if left == 0 or right >= len(corrected) - 1:
            review_events.append(RepairEvent(**base_kwargs, max_interpolation_residual_deg=0.0, method="manual_review", reason="boundary_event"))
            continue
        if run_length == 0:
            review_events.append(
                RepairEvent(
                    **base_kwargs,
                    max_interpolation_residual_deg=0.0,
                    method="manual_review",
                    reason="persistent_step",
                )
            )
            continue
        if run_length > args.max_repair_run_frames:
            review_events.append(RepairEvent(**base_kwargs, max_interpolation_residual_deg=0.0, method="manual_review", reason="run_too_long"))
            continue

        anchor_delta = abs(float(corrected[right] - corrected[left]))
        interpolation = np.linspace(corrected[left], corrected[right], right - left + 1)[1:-1]
        original_interior = corrected[left + 1 : right].copy()
        residual = np.abs(original_interior - interpolation)
        max_residual = float(np.max(residual, initial=0.0))
        base_kwargs["max_interpolation_residual_deg"] = max_residual
        # A single-sample spike should return to the pre-event level.  For a
        # multi-sample run, require both small net displacement and an
        # interpolation residual large enough to justify changing values.
        if run_length == 1 and anchor_delta <= args.max_anchor_delta_deg and max_residual >= args.min_correction_deg:
            corrected[left + 1 : right] = interpolation
            repair_events.append(RepairEvent(**base_kwargs, method="linear_interpolation"))
            continue
        if anchor_delta > args.max_anchor_delta_deg:
            review_events.append(RepairEvent(**base_kwargs, method="manual_review", reason="persistent_step"))
            continue
        if max_residual < args.min_correction_deg:
            review_events.append(RepairEvent(**base_kwargs, method="manual_review", reason="small_interpolation_residual"))
            continue

        corrected[left + 1 : right] = interpolation
        repair_events.append(RepairEvent(**base_kwargs, method="linear_interpolation"))

    return repair_events, review_events, corrected


def _analyse_episode(episode: EpisodeData, args: Args) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    samples, names, values, timestamps_ns, selected_indices = _load_joint_series(episode, args.topic, args.joint_prefix)
    repairs: list[RepairEvent] = []
    reviews: list[RepairEvent] = []
    corrected_by_joint: dict[int, np.ndarray] = {}
    for joint_index in selected_indices:
        joint_repairs, joint_reviews, corrected = _analyse_joint(
            joint=names[joint_index],
            joint_index=joint_index,
            values=values,
            timestamps_ns=timestamps_ns,
            args=args,
        )
        repairs.extend(joint_repairs)
        reviews.extend(joint_reviews)
        if not np.array_equal(corrected, values[:, joint_index]):
            corrected_by_joint[joint_index] = corrected

    selected = values[:, selected_indices]
    corrected_selected = selected.copy()
    for local_index, joint_index in enumerate(selected_indices):
        if joint_index in corrected_by_joint:
            corrected_selected[:, local_index] = corrected_by_joint[joint_index]
    dt_s = np.diff(timestamps_ns.astype(np.float64) / 1e9)
    original_steps = np.diff(selected, axis=0)
    corrected_steps = np.diff(corrected_selected, axis=0)
    original_velocities = original_steps / dt_s[:, None]
    corrected_velocities = corrected_steps / dt_s[:, None]
    changed_values = int(np.sum(np.abs(corrected_selected - selected) > 0))
    # A report containing any manual-review event must remain review even when
    # another joint had a safe isolated repair.  This prevents a partially
    # repaired Episode from being mistaken for a fully accepted one.
    status: RepairStatus = "review" if reviews else "repaired" if changed_values else "clean"
    report = {
        "episode": episode.name,
        "topic": args.topic,
        "source_files": [str(path) for path in episode.source_files],
        "sample_count": len(samples),
        "joint_names": [names[index] for index in selected_indices],
        "timestamp": _timestamp_metrics(timestamps_ns),
        "thresholds": {
            "max_step_deg": args.max_step_deg,
            "max_velocity_deg_s": args.max_velocity_deg_s,
            "max_repair_run_frames": args.max_repair_run_frames,
            "max_anchor_delta_deg": args.max_anchor_delta_deg,
            "min_correction_deg": args.min_correction_deg,
        },
        "source_quality": {
            "max_step_deg": float(np.max(np.abs(original_steps), initial=0.0)),
            "max_velocity_deg_s": float(np.max(np.abs(original_velocities), initial=0.0)),
        },
        "corrected_quality": {
            "max_step_deg": float(np.max(np.abs(corrected_steps), initial=0.0)),
            "max_velocity_deg_s": float(np.max(np.abs(corrected_velocities), initial=0.0)),
        },
        "status": status,
        "changed_value_count": changed_values,
        "repair_event_count": len(repairs),
        "review_event_count": len(reviews),
        "repairs": [event.as_dict() for event in repairs],
        "manual_review": [event.as_dict() for event in reviews],
    }
    return report, corrected_by_joint


def _write_patches(
    episode: EpisodeData,
    output_episode: Path,
    samples: list[MessageSample],
    names: tuple[str, ...],
    corrected_by_joint: dict[int, np.ndarray],
) -> int:
    if not corrected_by_joint:
        return 0
    offsets = {
        sample.record_index: _position_value_offset(
            sample,
            names,
            len(sample.values) if sample.values is not None else 0,
        )
        for sample in samples
    }
    handles: dict[Path, Any] = {}
    changed = 0
    try:
        for source_file in episode.source_files:
            relative = source_file.relative_to(episode.root)
            handles[source_file] = (output_episode / relative).open("r+b")
        for sample_index, sample in enumerate(samples):
            if sample.record_index not in offsets:
                continue
            handle = handles[sample.source_path]
            for joint_index, corrected_values in corrected_by_joint.items():
                original_value = float(sample.values[joint_index])
                corrected_value = float(corrected_values[sample_index])
                if original_value == corrected_value:
                    continue
                offset = offsets[sample.record_index] + joint_index * 8
                handle.seek(offset)
                encoded = handle.read(8)
                if len(encoded) != 8:
                    raise ValueError(f"Truncated position value at {sample.source_path}:{offset}")
                source_value = struct.unpack("<d", encoded)[0]
                if source_value != original_value:
                    raise ValueError(
                        f"Source value changed before patching at {sample.source_path}:{offset}; "
                        f"expected {original_value}, found {source_value}"
                    )
                handle.seek(offset)
                handle.write(struct.pack("<d", corrected_value))
                changed += 1
    finally:
        for handle in handles.values():
            handle.close()
    return changed


def _write_event_csv(path: Path, events: Iterable[dict[str, Any]]) -> None:
    rows = list(events)
    if not rows:
        path.write_text("episode,joint,start_frame,end_frame,start_time_s,end_time_s,max_step_deg,max_velocity_deg_s,max_interpolation_residual_deg,method,reason\n")
        return
    fieldnames = ["episode", *rows[0].keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# F1 JointState 跳变修复报告",
        "",
        f"- 生成时间：`{report['generated_at_utc']}`",
        f"- 输入：`{report['input']}`",
        f"- Topic：`{report['topic']}`",
        f"- 模式：`{'dry-run' if report['dry_run'] else 'write'}`",
        "",
        "## 汇总",
        "",
        "| Episode | 状态 | 样本数 | 修复值数 | 自动修复事件 | 人工复核事件 | 原始最大步长 / ° | 修复后最大步长 / ° |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in report["episodes"]:
        lines.append(
            f"| `{episode['episode']}` | `{episode['status']}` | {episode.get('sample_count', '-')} | "
            f"{episode.get('changed_value_count', '-')} | {episode.get('repair_event_count', '-')} | "
            f"{episode.get('review_event_count', '-')} | "
            f"{episode.get('source_quality', {}).get('max_step_deg', float('nan')):.3f} | "
            f"{episode.get('corrected_quality', {}).get('max_step_deg', float('nan')):.3f} |"
        )
    for episode in report["episodes"]:
        lines.extend(["", f"## {episode['episode']}", ""])
        if episode.get("error"):
            lines.append(f"- 失败：`{episode['error']}`")
            continue
        lines.append(f"- 目标关节：`{', '.join(episode['joint_names'])}`")
        if episode["repairs"]:
            lines.extend(["", "### 自动修复", ""])
            for event in episode["repairs"]:
                lines.append(
                    f"- `{event['joint']}` 帧 `{event['start_frame']}..{event['end_frame']}`，"
                    f"相对时间 `{event['start_time_s']:.3f}..{event['end_time_s']:.3f} s`，"
                    f"最大插值残差 `{event['max_interpolation_residual_deg']:.3f}°`。"
                )
        if episode["manual_review"]:
            lines.extend(["", "### 保留并需人工复核", ""])
            for event in episode["manual_review"]:
                lines.append(
                    f"- `{event['joint']}` 帧 `{event['start_frame']}..{event['end_frame']}`：`{event['reason']}`，"
                    f"最大步长 `{event['max_step_deg']:.3f}°`，最大速度 `{event['max_velocity_deg_s']:.1f}°/s`。"
                )
        if not episode["repairs"] and not episode["manual_review"]:
            lines.append("- 未发现超过当前阈值且满足孤立异常修复条件的关节状态事件。")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 自动修复只对短暂、可由两侧样本约束的孤立异常做线性插值。",
            "- 持续性阶跃、序列边界事件和过长异常段不会被静默平滑，而是保留原始值并列入人工复核。",
            "- 写模式只复制 Episode 并修改 JointState position 的 double；原始 MCAP 不会原地修改。",
            "",
        ]
    )
    return "\n".join(lines)


def main(args: Args) -> None:
    _validate_args(args)
    run_dir = _new_run_dir(args.report_dir)
    episode_dirs = find_episode_dirs(args.raw_dir)
    output_root = args.output_dir
    dry_run = args.dry_run or output_root is None
    report: dict[str, Any] = {
        "format": "f1_joint_state_repair_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.raw_dir),
        "output": None if dry_run else str(output_root),
        "report_dir": str(run_dir),
        "topic": args.topic,
        "dry_run": dry_run,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dataclasses.asdict(args).items()
        },
        "episodes": [],
    }
    all_events: list[dict[str, Any]] = []

    for episode_dir in episode_dirs:
        try:
            episode = load_episode(episode_dir)
            episode_report, corrected_by_joint = _analyse_episode(episode, args)
            if not dry_run and corrected_by_joint:
                _safe_to_patch(episode)
                assert output_root is not None
                destination = output_root / episode_dir.name if len(episode_dirs) > 1 or list(args.raw_dir.glob("episode_*")) else output_root
                _copy_episode(episode_dir, destination, overwrite=args.overwrite)
                samples, names, _values, _timestamps, _indices = _load_joint_series(episode, args.topic, args.joint_prefix)
                patched = _write_patches(episode, destination, samples, names, corrected_by_joint)
                episode_report["output_episode"] = str(destination)
                episode_report["patched_value_count"] = patched
            elif not dry_run:
                assert output_root is not None
                destination = output_root / episode_dir.name if len(episode_dirs) > 1 or list(args.raw_dir.glob("episode_*")) else output_root
                _copy_episode(episode_dir, destination, overwrite=args.overwrite)
                episode_report["output_episode"] = str(destination)
                episode_report["patched_value_count"] = 0
            report["episodes"].append(episode_report)
            all_events.extend({"episode": episode_report["episode"], **event} for event in episode_report["repairs"])
            all_events.extend({"episode": episode_report["episode"], **event} for event in episode_report["manual_review"])
        except Exception as error:
            failure = {
                "episode": episode_dir.name,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "repairs": [],
                "manual_review": [],
            }
            report["episodes"].append(failure)

    if not report["episodes"]:
        raise ValueError(f"No episodes found under {args.raw_dir}")
    report["summary"] = {
        "episode_count": len(report["episodes"]),
        "failed_episodes": sum(episode.get("status") == "failed" for episode in report["episodes"]),
        "repaired_episodes": sum(episode.get("status") == "repaired" for episode in report["episodes"]),
        "review_episodes": sum(episode.get("status") == "review" for episode in report["episodes"]),
        "clean_episodes": sum(episode.get("status") == "clean" for episode in report["episodes"]),
        "repair_event_count": sum(len(episode.get("repairs", [])) for episode in report["episodes"]),
        "review_event_count": sum(len(episode.get("manual_review", [])) for episode in report["episodes"]),
    }
    (run_dir / "joint_state_repair_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "joint_state_repair_report.md").write_text(_render_markdown(report), encoding="utf-8")
    _write_event_csv(run_dir / "joint_state_events.csv", all_events)
    for episode in report["episodes"]:
        (run_dir / "episodes" / f"{episode['episode']}.json").write_text(
            json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"报告目录：{run_dir}")
    if not dry_run:
        print(f"清洗数据：{output_root}")


def _fallback_cli(argv: list[str] | None = None) -> Args:
    """Parse the public CLI without requiring the optional ``tyro`` package."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Args.report_dir)
    parser.add_argument("--topic", default=Args.topic)
    parser.add_argument("--joint-prefix", default=Args.joint_prefix)
    parser.add_argument("--max-step-deg", type=float, default=Args.max_step_deg)
    parser.add_argument("--max-velocity-deg-s", type=float, default=Args.max_velocity_deg_s)
    parser.add_argument("--max-repair-run-frames", type=int, default=Args.max_repair_run_frames)
    parser.add_argument("--max-anchor-delta-deg", type=float, default=Args.max_anchor_delta_deg)
    parser.add_argument("--min-correction-deg", type=float, default=Args.min_correction_deg)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    namespace = parser.parse_args(argv)
    return Args(**vars(namespace))


if __name__ == "__main__":
    main(tyro.cli(Args) if tyro is not None else _fallback_cli(sys.argv[1:]))
