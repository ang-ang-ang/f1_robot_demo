# ruff: noqa: BLE001, RUF001
"""Audit raw F1 JointState signals without modifying source MCAP files.

Run ``python -m data_process.audit_f1_joint_states --help`` for its public
batch interface. The tool checks both the leader target and follower state
streams when present, reports structural/timestamp issues, and preserves the
raw joint values in its evidence reports.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from data_process import f1_schema
from data_process.config import load_config
from data_process.mcap import find_episode_dirs, load_episode
from data_process.plot_f1_joint_state_jumps import _detect_events
from data_process.quality_operator_config import load_operator_config
from data_process.repair_f1_joint_state_jumps import Args, _analyse_episode, _load_joint_series


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], empty_fields: list[str]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or empty_fields
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_signal(
    episode: Any,
    topic: str,
    signal: Any,
    motion_config: Any,
    timing_config: Any,
) -> dict[str, Any]:
    samples = episode.topic(topic)
    expected = f1_schema.ACTION_JOINT_NAMES if topic == f1_schema.LEAD_JOINT_TOPIC else f1_schema.STATE_JOINT_NAMES
    names = tuple(samples[0].names or ()) if samples else ()
    structural = {
        "missing_topic": not bool(samples),
        "undecodable_frames": [i for i, s in enumerate(samples) if s.names is None or s.values is None],
        "name_order_change_frames": [i for i, s in enumerate(samples) if tuple(s.names or ()) != names],
        "shape_mismatch_frames": [
            i for i, s in enumerate(samples) if s.values is not None and len(s.values) != len(s.names or ())
        ],
        "duplicate_names": sorted({name for name in names if names.count(name) > 1}),
        "missing_arm_names": sorted(set(expected) - set(names)),
        "unexpected_arm_names": sorted(name for name in names if name.startswith("arm_") and name not in expected),
        "missing_header_frames": [i for i, s in enumerate(samples) if s.header_time_ns is None],
        "nonfinite_value_count": sum(
            int(np.count_nonzero(~np.isfinite(s.values))) for s in samples if s.values is not None
        ),
        "nonfinite_frames": [
            i for i, s in enumerate(samples) if s.values is not None and not np.isfinite(s.values).all()
        ],
    }
    report = {
        "episode": episode.name,
        "topic": topic,
        "sample_count": len(samples),
        "raw_names": names,
        "raw_dimension": len(names),
        "structural": structural,
        "status": "invalid",
        "thresholds": dataclasses.asdict(signal),
        "position_limits": "unverified: no robot URDF/controller joint limits supplied",
    }
    if any(bool(value) for value in structural.values()) or len(samples) < 3:
        report["reason"] = "Invalid/missing JointState data or fewer than three samples"
        return report
    stamps = np.asarray([s.header_time_ns for s in samples], dtype=np.int64)
    dt_ns = np.diff(stamps)
    if np.any(dt_ns <= 0):
        report["nonpositive_timestamp_edges"] = np.flatnonzero(dt_ns <= 0).tolist()
        report["reason"] = "Non-increasing Header timestamps; velocity/acceleration not evaluated"
        return report

    _, names, full_values, stamps, selected = _load_joint_series(episode, topic, "arm_")
    names = tuple(names[i] for i in selected)
    values = full_values[:, selected]
    times, bad, merged = _detect_events(
        values, stamps, names, max_step_deg=signal.max_step_deg, max_velocity_deg_s=signal.fallback_max_velocity_deg_s
    )
    # Relative integer-nanosecond differences avoid precision loss at Unix epoch scale.
    dt = np.diff(stamps).astype(np.float64) / 1e9
    steps = np.diff(values, axis=0)
    velocity = steps / dt[:, None]
    # Same midpoint-velocity acceleration definition as joint_motion_operator.py.
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]
    abs_steps, abs_velocity, abs_acceleration = np.abs(steps), np.abs(velocity), np.abs(acceleration)
    step_exceeded = abs_steps > signal.max_step_deg
    velocity_exceeded = abs_velocity > signal.fallback_max_velocity_deg_s
    velocity_near = (
        abs_velocity > signal.fallback_max_velocity_deg_s * motion_config.velocity_warning_ratio
    ) & ~velocity_exceeded
    acceleration_exceeded = abs_acceleration > signal.max_acceleration_deg_s2
    acceleration_near = (
        abs_acceleration > signal.max_acceleration_deg_s2 * motion_config.acceleration_warning_ratio
    ) & ~acceleration_exceeded
    spike = (
        (abs_steps[:-1] >= motion_config.spike_min_step_deg)
        & (abs_steps[1:] >= motion_config.spike_min_step_deg)
        & (np.abs(values[2:] - values[:-2]) <= motion_config.spike_recovery_deg)
    )

    median_ms = float(np.median(dt) * 1000)
    gap_floor = (
        timing_config.action_gap_floor_ms if topic == f1_schema.LEAD_JOINT_TOPIC else timing_config.state_gap_floor_ms
    )
    gap_warning = max(gap_floor, median_ms * timing_config.gap_warning_factor)
    gap_fail = max(gap_floor, median_ms * timing_config.gap_fail_factor)
    dt_ms = dt * 1000
    gap_edges = np.flatnonzero(dt_ms > gap_warning)
    near_duplicate = np.flatnonzero(dt_ms <= timing_config.duplicate_epsilon_ms)
    timestamp = {
        "origin_ns": int(stamps[0]),
        "duration_s": float(times[-1]),
        "effective_hz": float((len(samples) - 1) / times[-1]),
        "median_dt_ms": median_ms,
        "min_dt_ms": float(dt_ms.min()),
        "max_dt_ms": float(dt_ms.max()),
        "p95_dt_ms": float(np.percentile(dt_ms, 95)),
        "nonpositive_count": 0,
        "near_duplicate_count": len(near_duplicate),
        "near_duplicate_from_frames": near_duplicate.tolist(),
        "gap_warning_threshold_ms": gap_warning,
        "gap_fail_threshold_ms": gap_fail,
        "gap_count": len(gap_edges),
        "severe_gap_count": int(np.count_nonzero(dt_ms > gap_fail)),
        "gaps": [
            {
                "from_frame": int(i),
                "to_frame": int(i + 1),
                "start_time_s": float(times[i]),
                "end_time_s": float(times[i + 1]),
                "interval_ms": float(dt_ms[i]),
                "severity": "fail" if dt_ms[i] > gap_fail else "warning",
            }
            for i in gap_edges
        ],
    }

    events: list[dict[str, Any]] = []

    def add_event(kind: str, severity: str, start: int, end: int, joint: int, value: float, threshold: float) -> None:
        events.append(
            {
                "episode": episode.name,
                "topic": topic,
                "kind": kind,
                "severity": severity,
                "joint": names[joint],
                "from_frame": start,
                "to_frame": end,
                "start_time_s": float(times[start]),
                "end_time_s": float(times[end]),
                "from_deg": float(values[start, joint]),
                "to_deg": float(values[end, joint]),
                "value": float(value),
                "threshold": float(threshold),
            }
        )

    for i, j in np.argwhere(bad):
        add_event("joint_jump", "review", int(i), int(i + 1), j, abs_steps[i, j], signal.max_step_deg)
        events[-1].update(
            {
                "signed_step_deg": float(steps[i, j]),
                "interval_ms": float(dt_ms[i]),
                "abs_velocity_deg_s": float(abs_velocity[i, j]),
                "step_exceeded": bool(step_exceeded[i, j]),
                "velocity_exceeded": bool(velocity_exceeded[i, j]),
            }
        )
    for i, j in np.argwhere(velocity_near):
        add_event(
            "velocity_near_limit",
            "warning",
            int(i),
            int(i + 1),
            j,
            abs_velocity[i, j],
            signal.fallback_max_velocity_deg_s * motion_config.velocity_warning_ratio,
        )
    for i, j in np.argwhere(acceleration_exceeded | acceleration_near):
        add_event(
            "acceleration",
            "review" if acceleration_exceeded[i, j] else "warning",
            int(i),
            int(i + 2),
            j,
            abs_acceleration[i, j],
            signal.max_acceleration_deg_s2
            if acceleration_exceeded[i, j]
            else signal.max_acceleration_deg_s2 * motion_config.acceleration_warning_ratio,
        )
    for i, j in np.argwhere(spike):
        add_event(
            "single_frame_spike",
            "review",
            int(i),
            int(i + 2),
            j,
            max(abs_steps[i, j], abs_steps[i + 1, j]),
            motion_config.spike_min_step_deg,
        )
        events[-1].update(
            {
                "center_frame": int(i + 1),
                "center_deg": float(values[i + 1, j]),
                "recovery_error_deg": float(abs(values[i + 2, j] - values[i, j])),
            }
        )
    events.sort(key=lambda item: (item["from_frame"], item["joint"], item["kind"]))
    largest_edge, largest_joint = np.unravel_index(np.argmax(abs_steps), abs_steps.shape)
    per_joint = [
        {
            "joint": name,
            "min_deg": float(values[:, j].min()),
            "max_deg": float(values[:, j].max()),
            "std_deg": float(values[:, j].std()),
            "constant": bool(np.ptp(values[:, j]) < 1e-8),
            "max_step_deg": float(abs_steps[:, j].max()),
            "max_velocity_deg_s": float(abs_velocity[:, j].max()),
            "max_acceleration_deg_s2": float(abs_acceleration[:, j].max()),
            "jump_edge_count": int(bad[:, j].sum()),
            "acceleration_over_count": int(acceleration_exceeded[:, j].sum()),
            "single_frame_spike_count": int(spike[:, j].sum()),
        }
        for j, name in enumerate(names)
    ]
    repair, _ = _analyse_episode(
        episode,
        Args(
            raw_dir=episode.root,
            topic=topic,
            max_step_deg=signal.max_step_deg,
            max_velocity_deg_s=signal.fallback_max_velocity_deg_s,
            dry_run=True,
        ),
    )
    report.update(
        {
            "status": "review"
            if bad.any()
            or acceleration_exceeded.any()
            or spike.any()
            or len(near_duplicate)
            or timestamp["severe_gap_count"]
            else "warning"
            if velocity_near.any() or acceleration_near.any() or len(gap_edges)
            else "clean_under_checked_thresholds",
            "joint_count": len(names),
            "timestamp": timestamp,
            "per_joint": per_joint,
            "max_step_deg": float(abs_steps.max()),
            "max_velocity_deg_s": float(abs_velocity.max()),
            "max_acceleration_deg_s2": float(abs_acceleration.max()),
            "jump_interval_count": len(merged),
            "jump_joint_edge_count": int(bad.sum()),
            "step_over_count": int(step_exceeded.sum()),
            "velocity_over_count": int(velocity_exceeded.sum()),
            "velocity_near_count": int(velocity_near.sum()),
            "acceleration_over_count": int(acceleration_exceeded.sum()),
            "acceleration_near_count": int(acceleration_near.sum()),
            "single_frame_spike_count": int(spike.sum()),
            "merged_jump_intervals": [event.as_dict() for event in merged],
            "events": events,
            "largest_step": {
                "joint": names[largest_joint],
                "from_frame": int(largest_edge),
                "to_frame": int(largest_edge + 1),
                "from_deg": float(values[largest_edge, largest_joint]),
                "to_deg": float(values[largest_edge + 1, largest_joint]),
                "time_s": float(times[largest_edge]),
                "signed_step_deg": float(steps[largest_edge, largest_joint]),
                "interval_ms": float(dt_ms[largest_edge]),
            },
            "repair_dry_run": {
                key: repair[key]
                for key in (
                    "status",
                    "repair_event_count",
                    "review_event_count",
                    "changed_value_count",
                    "repairs",
                    "manual_review",
                )
            },
        }
    )
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="One Episode directory or a directory containing episode_* subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/f1_joint_audits"),
        help="Directory for JSON, CSV and Markdown reports; source MCAP files are never modified.",
    )
    return parser.parse_args(argv)


def _source_files(episodes: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for episode_dir in episodes
        for path in sorted(episode_dir.glob("*.mcap"))
    ]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    motion = load_operator_config().joint_motion
    timing = load_config().timestamp
    episodes = find_episode_dirs(args.raw_dir)
    all_reports, rows, events, joint_ranges = [], [], [], []
    sources = _source_files(episodes)
    for index, directory in enumerate(episodes, 1):
        print(f"[{index}/{len(episodes)}] {directory.name}", flush=True)
        ep_report = {"episode": directory.name, "source_dir": str(directory.resolve())}
        try:
            episode = load_episode(directory)
            ep_report["message_count"] = episode.message_count
            ep_report["topic_counts"] = {name: len(samples) for name, samples in episode.topics.items()}
            ep_report["signals"] = []
            for topic, signal in (
                (f1_schema.LEAD_JOINT_TOPIC, motion.action),
                (f1_schema.HAL_JOINT_TOPIC, motion.state),
            ):
                result = _audit_signal(episode, topic, signal, motion, timing)
                ep_report["signals"].append(result)
                row = {"episode": episode.name, "topic": topic}
                keys = (
                    "status",
                    "sample_count",
                    "joint_count",
                    "jump_interval_count",
                    "jump_joint_edge_count",
                    "max_step_deg",
                    "max_velocity_deg_s",
                    "max_acceleration_deg_s2",
                    "step_over_count",
                    "velocity_over_count",
                    "velocity_near_count",
                    "acceleration_over_count",
                    "acceleration_near_count",
                    "single_frame_spike_count",
                )
                row.update({key: result.get(key) for key in keys})
                row.update(
                    {
                        "timestamp_gap_count": result.get("timestamp", {}).get("gap_count"),
                        "severe_gap_count": result.get("timestamp", {}).get("severe_gap_count"),
                    }
                )
                rows.append(row)
                events.extend(event for event in result.get("events", []))
                joint_ranges.extend(
                    {"episode": episode.name, "topic": topic, **joint} for joint in result.get("per_joint", [])
                )
                print(
                    f"  {topic}: {result['status']} N={result['sample_count']} jumps={result.get('jump_interval_count')} "
                    f"step={result.get('max_step_deg')} speed={result.get('max_velocity_deg_s')} "
                    f"accel={result.get('max_acceleration_deg_s2')} gaps={row['timestamp_gap_count']}",
                    flush=True,
                )
        except Exception as exc:
            ep_report["error"] = f"{type(exc).__name__}: {exc}"
            rows.append({"episode": directory.name, "topic": "episode", "status": "error", "error": ep_report["error"]})
            print(ep_report["error"], flush=True)
        all_reports.append(ep_report)
        _write_json(output / "episodes" / f"{directory.name}.json", ep_report)

    for source in sources:
        stat = Path(source["path"]).stat()
        source["size_mtime_unchanged_after_audit"] = (
            stat.st_size == source["size_bytes"] and stat.st_mtime_ns == source["mtime_ns"]
        )
    report = {
        "format": "f1_raw_joint_audit_v1",
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "root": str(args.raw_dir.resolve()),
        "discovered_episode_count": len(episodes),
        "sources": sources,
        "method": "Original record order; zero-based per-topic frames; F1 positions in degrees. Relative Header nanoseconds for derivatives. No MCAP writes.",
        "limits": "No URDF found: mechanical position limits unverified. Engineering motion thresholds do not prove physical defects.",
        "motion_config": dataclasses.asdict(motion),
        "timing_config": dataclasses.asdict(timing),
        "rows": rows,
        "episodes": all_reports,
    }
    _write_json(output / "joint_audit.json", report)
    _write_csv(output / "episode_summary.csv", rows, ["episode", "topic", "status"])
    _write_csv(output / "joint_events.csv", events, ["episode", "topic", "kind", "joint", "from_frame", "to_frame"])
    _write_csv(output / "joint_ranges.csv", joint_ranges, ["episode", "topic", "joint"])
    lines = [
        "# 关节点位数据检测报告",
        "",
        f"- 输入：`{args.raw_dir.resolve()}`",
        f"- 检测时间：{report['generated_at']}",
        f"- Episode 数：{len(episodes)}。",
        "- 检查原始 MCAP 中左右臂 14 个关节，保留原始记录顺序；帧号从 0 开始，时间相对各 Topic 第一条 Header。",
        "- 动作门限：单步 25°、速度 500°/s；状态门限：单步 15°、速度 300°/s。加速度门限：5000°/s²，速度/加速度的 80% 起预警。",
        "- 尖峰：相邻进出步长均至少 5°，且恢复误差不超过 1°。时间间隔按 default_config.toml 的倍数和下限检查。",
        "- 名称/维度、缺失值、非有限数和非递增时间戳检查覆盖完整 JointState；运动统计覆盖 14 臂关节。",
        "- 未提供当前机器人的 URDF/控制器限位，角度机械越限未验证；clean 仅指已检查指标未触发。原始 MCAP 未写入。",
        "",
        "| Episode | Topic | 帧数 | 跳变段 | 最大步长 ° | 最大速度 °/s | 最大加速度 °/s² | 加速度超限点 | 尖峰点 | 时间间隔告警 | 结果 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["status"] in ("error", "invalid"):
            lines.append(f"| {row['episode']} | {row['topic']} | - | - | - | - | - | - | - | - | {row['status']} |")
            continue
        lines.append(
            f"| {row['episode']} | {row['topic']} | {row['sample_count']} | {row['jump_interval_count']} | "
            f"{row['max_step_deg']:.3f} | {row['max_velocity_deg_s']:.3f} | {row['max_acceleration_deg_s2']:.1f} | {row['acceleration_over_count']} | "
            f"{row['single_frame_spike_count']} | {row['timestamp_gap_count']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "跳变段按时间合并所有关节；关节事件按关节×相邻帧统计；加速度用三帧计算。一次动作可能触发多个指标，不能将计数相加视为独立故障。",
            "",
            "## 跳变定位",
            "",
            "| Episode | Topic | 关节 | 帧 | 时间 s | 原值 → 新值 ° | 单步 ° | 速度 °/s |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for event in events:
        if event["kind"] == "joint_jump":
            lines.append(
                f"| {event['episode']} | {event['topic']} | {event['joint']} | {event['from_frame']} → {event['to_frame']} | "
                f"{event['start_time_s']:.6f}–{event['end_time_s']:.6f} | {event['from_deg']:.3f} → {event['to_deg']:.3f} | "
                f"{event['signed_step_deg']:+.3f} | {event['abs_velocity_deg_s']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 附件",
            "",
            "- `joint_audit.json`：完整检测、每条数据的元信息核对、逐关节统计和 dry-run 分类。",
            "- `episode_summary.csv`：逐 Episode/Topic 汇总。",
            "- `joint_events.csv`：跳变、速度预警、加速度和尖峰事件。",
            "- `joint_ranges.csv`：每关节 min/max/std 及运动指标。",
            "- `episodes/`：每条 Episode 的 JSON。",
            "",
        ]
    )
    (output / "joint_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
