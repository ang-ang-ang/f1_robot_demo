# ruff: noqa: RUF001
from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def find_mapping_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / "resampling_mapping.json"
    if direct.exists():
        return [direct]
    files = sorted(path.glob("episode_*/resampling_mapping.json"))
    if not files:
        files = sorted(path.rglob("resampling_mapping.json"))
    if not files:
        raise FileNotFoundError(f"No resampling_mapping.json found under {path}")
    return files


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return dict.fromkeys(("min", "mean", "median", "p95", "p99", "max", "std"))
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _timeline_stats(
    timestamps_ns: np.ndarray,
    *,
    duplicate_epsilon_ms: float,
    gap_warning_factor: float,
    gap_fail_factor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    differences_ms = np.diff(timestamps_ns).astype(np.float64) / 1e6
    positive = differences_ms[differences_ms > duplicate_epsilon_ms]
    nominal_ms = float(np.median(positive)) if len(positive) else None
    if nominal_ms is None:
        warning_ms = None
        fail_ms = None
        multiples = np.zeros(len(differences_ms), dtype=np.float64)
    else:
        warning_ms = gap_warning_factor * nominal_ms
        fail_ms = gap_fail_factor * nominal_ms
        multiples = differences_ms / nominal_ms
    rounded_counts: dict[str, int] = defaultdict(int)
    for difference_ms, multiple in zip(differences_ms, multiples, strict=True):
        if difference_ms <= duplicate_epsilon_ms:
            rounded_counts["nonpositive_or_duplicate"] += 1
        else:
            rounded = max(1, round(multiple))
            rounded_counts[f"{rounded}T" if rounded < 5 else ">=5T"] += 1
    events = []
    if nominal_ms is not None:
        events.extend(
            {
                "edge_after_source_index": int(edge_index),
                "gap_ms": float(differences_ms[edge_index]),
                "period_multiple": float(multiples[edge_index]),
                "severity": "fail"
                if fail_ms is not None and differences_ms[edge_index] > fail_ms
                else "warning"
                if warning_ms is not None and differences_ms[edge_index] > warning_ms
                else "info",
            }
            for edge_index in np.flatnonzero(differences_ms > 1.5 * nominal_ms)
        )
    duration_s = float((timestamps_ns[-1] - timestamps_ns[0]) / 1e9) if len(timestamps_ns) > 1 else 0.0
    return (
        {
            "count": len(timestamps_ns),
            "duration_s": duration_s,
            "effective_hz": float((len(timestamps_ns) - 1) / duration_s) if duration_s > 0 else 0.0,
            "nominal_period_ms": nominal_ms,
            "gap_warning_ms": warning_ms,
            "gap_fail_ms": fail_ms,
            "dt_ms": _distribution(differences_ms),
            "negative_count": int(np.sum(differences_ms < -duplicate_epsilon_ms)),
            "nonpositive_count": int(np.sum(differences_ms <= duplicate_epsilon_ms)),
            "duplicate_count": int(np.sum(np.abs(differences_ms) <= duplicate_epsilon_ms)),
            "warning_gap_count": 0 if warning_ms is None else int(np.sum(differences_ms > warning_ms)),
            "fail_gap_count": 0 if fail_ms is None else int(np.sum(differences_ms > fail_ms)),
            "period_multiple_counts": dict(sorted(rounded_counts.items())),
        },
        events,
    )


def _residual_stats(values_ms: np.ndarray, thresholds_ms: Iterable[float]) -> dict[str, Any]:
    absolute = np.abs(values_ms)
    result: dict[str, Any] = {
        "signed_ms": _distribution(values_ms),
        "absolute_ms": _distribution(absolute),
        "past_source_count": int(np.sum(values_ms < 0)),
        "exact_source_count": int(np.sum(values_ms == 0)),
        "future_source_count": int(np.sum(values_ms > 0)),
    }
    result["threshold_exceedance"] = {
        f"{threshold:g}ms": {
            "count": int(np.sum(absolute > threshold)),
            "ratio": float(np.mean(absolute > threshold)),
        }
        for threshold in thresholds_ms
    }
    return result


def analyze_mapping(
    mapping: dict[str, Any],
    *,
    duplicate_epsilon_ms: float,
    gap_warning_factor: float,
    gap_fail_factor: float,
    residual_thresholds_ms: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stream_reports: dict[str, Any] = {}
    gap_rows = []
    for topic, stream in mapping["streams"].items():
        source_samples = stream["source_samples"]
        original_ns = np.asarray([row["original_primary_time_ns"] for row in source_samples], dtype=np.int64)
        cleaned_ns = np.asarray([row["cleaned_primary_time_ns"] for row in source_samples], dtype=np.int64)
        original_stats, original_events = _timeline_stats(
            original_ns,
            duplicate_epsilon_ms=duplicate_epsilon_ms,
            gap_warning_factor=gap_warning_factor,
            gap_fail_factor=gap_fail_factor,
        )
        cleaned_stats, cleaned_events = _timeline_stats(
            cleaned_ns,
            duplicate_epsilon_ms=duplicate_epsilon_ms,
            gap_warning_factor=gap_warning_factor,
            gap_fail_factor=gap_fail_factor,
        )
        mappings = stream["target_mappings"]
        residual_ms = np.asarray([row["residual_ms"] for row in mappings], dtype=np.float64)
        raw_residual_ms = np.asarray([row["raw_to_target_residual_ms"] for row in mappings], dtype=np.float64)
        correction_ms = np.asarray([row["cleaning_correction_ms"] for row in source_samples], dtype=np.float64)
        stream_reports[topic] = {
            "category": stream["category"],
            "time_source": stream["time_source"],
            "original_timeline": original_stats,
            "cleaned_timeline": cleaned_stats,
            "cleaning_correction_ms": {
                **_distribution(np.abs(correction_ms)),
                "changed_count": int(np.sum(correction_ms != 0)),
            },
            "resampling_residual": _residual_stats(residual_ms, residual_thresholds_ms),
            "raw_to_target_residual": _residual_stats(raw_residual_ms, residual_thresholds_ms),
            "source_count": stream["source_count"],
            "target_count": len(mappings),
            "selected_unique_count": stream["selected_unique_count"],
            "omitted_source_count": stream["omitted_source_count"],
            "omitted_source_ratio": stream["omitted_source_count"] / stream["source_count"],
            "reused_target_count": stream["reused_target_count"],
            "reused_target_ratio": stream["reused_target_count"] / len(mappings),
        }
        for timeline_name, events in (("original", original_events), ("cleaned", cleaned_events)):
            gap_rows.extend(
                {
                    "episode": mapping["episode"],
                    "topic": topic,
                    "timeline": timeline_name,
                    **event,
                }
                for event in events
            )

    pair_reports: dict[str, Any] = {}
    topics = mapping["topics"]
    selected_times = {
        topic: np.asarray(
            [row["cleaned_time_ns"] for row in mapping["streams"][topic]["target_mappings"]],
            dtype=np.int64,
        )
        for topic in topics
    }
    for left_index, left_topic in enumerate(topics):
        for right_topic in topics[left_index + 1 :]:
            skew_ms = (selected_times[left_topic] - selected_times[right_topic]).astype(np.float64) / 1e6
            pair_reports[f"{left_topic} | {right_topic}"] = {
                "left_topic": left_topic,
                "right_topic": right_topic,
                "skew": _residual_stats(skew_ms, residual_thresholds_ms),
            }

    target_ns = mapping["common_start_ns"] + np.arange(mapping["target_count"], dtype=np.int64) * mapping["period_ns"]
    target_stats, _ = _timeline_stats(
        target_ns,
        duplicate_epsilon_ms=duplicate_epsilon_ms,
        gap_warning_factor=gap_warning_factor,
        gap_fail_factor=gap_fail_factor,
    )
    return (
        {
            "episode": mapping["episode"],
            "mapping_format": mapping["format"],
            "fps": mapping["fps"],
            "target_timeline": target_stats,
            "streams": stream_reports,
            "cross_stream_pairs": pair_reports,
        },
        gap_rows,
    )


def _flatten_stream_row(episode: str, topic: str, report: dict[str, Any]) -> dict[str, Any]:
    residual = report["resampling_residual"]
    return {
        "episode": episode,
        "topic": topic,
        "category": report["category"],
        "source_count": report["source_count"],
        "target_count": report["target_count"],
        "original_nominal_period_ms": report["original_timeline"]["nominal_period_ms"],
        "original_nonpositive_count": report["original_timeline"]["nonpositive_count"],
        "original_warning_gap_count": report["original_timeline"]["warning_gap_count"],
        "cleaned_nominal_period_ms": report["cleaned_timeline"]["nominal_period_ms"],
        "cleaned_nonpositive_count": report["cleaned_timeline"]["nonpositive_count"],
        "cleaned_warning_gap_count": report["cleaned_timeline"]["warning_gap_count"],
        "residual_signed_median_ms": residual["signed_ms"]["median"],
        "residual_abs_median_ms": residual["absolute_ms"]["median"],
        "residual_abs_p95_ms": residual["absolute_ms"]["p95"],
        "residual_abs_p99_ms": residual["absolute_ms"]["p99"],
        "residual_abs_max_ms": residual["absolute_ms"]["max"],
        "future_source_count": residual["future_source_count"],
        "past_source_count": residual["past_source_count"],
        "reused_target_count": report["reused_target_count"],
        "reused_target_ratio": report["reused_target_ratio"],
        "omitted_source_count": report["omitted_source_count"],
        "omitted_source_ratio": report["omitted_source_ratio"],
    }


def _flatten_pair_row(episode: str, pair: dict[str, Any]) -> dict[str, Any]:
    skew = pair["skew"]
    return {
        "episode": episode,
        "left_topic": pair["left_topic"],
        "right_topic": pair["right_topic"],
        "signed_median_ms": skew["signed_ms"]["median"],
        "abs_median_ms": skew["absolute_ms"]["median"],
        "abs_p95_ms": skew["absolute_ms"]["p95"],
        "abs_p99_ms": skew["absolute_ms"]["p99"],
        "abs_max_ms": skew["absolute_ms"]["max"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# F1 20 Hz 时间戳映射统计报告",
        "",
        f"- 生成时间：`{report['generated_at_utc']}`",
        f"- 输入映射数：`{len(report['episodes'])}`",
        "",
    ]
    for episode in report["episodes"]:
        lines.extend(
            [
                f"## {episode['episode']}",
                "",
                "### 流内周期与重采样误差",
                "",
                "| 流 | 原始非递增 | 清洗后非递增 | 清洗后 Gap | 残差中位数 / ms | 绝对 P95 / ms | 最大绝对误差 / ms | 复用率 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for topic, stream in episode["streams"].items():
            residual = stream["resampling_residual"]
            lines.append(
                f"| `{topic}` | {stream['original_timeline']['nonpositive_count']} | "
                f"{stream['cleaned_timeline']['nonpositive_count']} | "
                f"{stream['cleaned_timeline']['warning_gap_count']} | "
                f"{residual['signed_ms']['median']:.3f} | {residual['absolute_ms']['p95']:.3f} | "
                f"{residual['absolute_ms']['max']:.3f} | {stream['reused_target_ratio']:.3%} |"
            )
        lines.extend(
            [
                "",
                "### 流间时间差",
                "",
                "| 左流 | 右流 | 有符号中位数 / ms | 绝对 P95 / ms | 绝对 P99 / ms | 最大绝对值 / ms |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        pairs = sorted(
            episode["cross_stream_pairs"].values(),
            key=lambda item: item["skew"]["absolute_ms"]["p95"],
            reverse=True,
        )
        for pair in pairs:
            skew = pair["skew"]
            lines.append(
                f"| `{pair['left_topic']}` | `{pair['right_topic']}` | "
                f"{skew['signed_ms']['median']:.3f} | {skew['absolute_ms']['p95']:.3f} | "
                f"{skew['absolute_ms']['p99']:.3f} | {skew['absolute_ms']['max']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 说明",
            "",
            "- `residual = cleaned_source_time - target_time`；负值表示选中目标时刻之前的样本，正值表示选中未来样本。",
            "- 流间时间差定义为 `left_selected_time - right_selected_time`，不能等同于硬件曝光同步误差。",
            "- Gap 默认在超过 `2.2T` 时告警、超过 `5T` 时失败；详细事件见 `gap_events.csv`。",
            "- 插入 aligned MCAP 的诊断话题可直接在 Foxglove Plot 和 State Transitions 面板中查看。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_episode_plots(mapping: dict[str, Any], episode_report: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    topics = mapping["topics"]
    target_seconds = np.arange(mapping["target_count"], dtype=np.float64) / mapping["fps"]

    figure, axis = plt.subplots(figsize=(15, 7))
    for topic in topics:
        values = [row["residual_ms"] for row in mapping["streams"][topic]["target_mappings"]]
        axis.plot(target_seconds, values, linewidth=0.8, label=topic)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_title(f"{mapping['episode']} — 20 Hz signed resampling residual")
    axis.set_xlabel("Target timeline / s")
    axis.set_ylabel("Source - target / ms")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "residual_timeline.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(15, 7))
    for topic in topics:
        cleaned = np.asarray(
            [row["cleaned_primary_time_ns"] for row in mapping["streams"][topic]["source_samples"]],
            dtype=np.int64,
        )
        nominal = episode_report["streams"][topic]["cleaned_timeline"]["nominal_period_ms"]
        if nominal is None:
            continue
        axis.plot(np.arange(len(cleaned) - 1), np.diff(cleaned) / 1e6 / nominal, linewidth=0.7, label=topic)
    axis.axhline(1, color="black", linewidth=0.6)
    axis.axhline(2.2, color="orange", linewidth=0.8, linestyle="--", label="2.2T warning")
    axis.axhline(5, color="red", linewidth=0.8, linestyle="--", label="5T fail")
    axis.set_ylim(bottom=0)
    axis.set_title(f"{mapping['episode']} — cleaned period multiples")
    axis.set_xlabel("Source edge index")
    axis.set_ylabel("dt / nominal period")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "gap_timeline.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 7))
    for topic in topics:
        values = np.abs([row["residual_ms"] for row in mapping["streams"][topic]["target_mappings"]])
        axis.hist(values, bins=60, histtype="step", linewidth=1.2, label=topic)
    axis.set_title(f"{mapping['episode']} — absolute residual distribution")
    axis.set_xlabel("Absolute residual / ms")
    axis.set_ylabel("Frames")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "residual_histogram.png", dpi=160)
    plt.close(figure)

    image_topics = [topic for topic in topics if mapping["streams"][topic]["category"] == "image"]
    figure, axis = plt.subplots(figsize=(15, 7))
    plotted = False
    for left_index, left_topic in enumerate(image_topics):
        left = np.asarray(
            [row["cleaned_time_ns"] for row in mapping["streams"][left_topic]["target_mappings"]], dtype=np.int64
        )
        for right_topic in image_topics[left_index + 1 :]:
            right = np.asarray(
                [row["cleaned_time_ns"] for row in mapping["streams"][right_topic]["target_mappings"]],
                dtype=np.int64,
            )
            axis.plot(target_seconds, (left - right) / 1e6, linewidth=0.8, label=f"{left_topic} - {right_topic}")
            plotted = True
    if plotted:
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(f"{mapping['episode']} — camera pairwise skew")
        axis.set_xlabel("Target timeline / s")
        axis.set_ylabel("Pairwise skew / ms")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / "camera_pairwise_skew.png", dpi=160)
    plt.close(figure)


def analyze_mapping_files(
    input_path: Path,
    output_dir: Path,
    *,
    duplicate_epsilon_ms: float = 0.05,
    gap_warning_factor: float = 2.2,
    gap_fail_factor: float = 5.0,
    residual_thresholds_ms: list[float] | None = None,
    write_plots: bool = True,
) -> dict[str, Any]:
    thresholds = residual_thresholds_ms or [10.0, 20.0, 33.333, 40.0, 60.0]
    mappings = [(path, json.loads(path.read_text())) for path in find_mapping_files(input_path)]
    episodes = []
    stream_rows = []
    pair_rows = []
    gap_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, mapping in mappings:
        episode_report, episode_gap_rows = analyze_mapping(
            mapping,
            duplicate_epsilon_ms=duplicate_epsilon_ms,
            gap_warning_factor=gap_warning_factor,
            gap_fail_factor=gap_fail_factor,
            residual_thresholds_ms=thresholds,
        )
        episode_report["mapping"] = str(path)
        episodes.append(episode_report)
        stream_rows.extend(
            _flatten_stream_row(mapping["episode"], topic, stream)
            for topic, stream in episode_report["streams"].items()
        )
        pair_rows.extend(
            _flatten_pair_row(mapping["episode"], pair) for pair in episode_report["cross_stream_pairs"].values()
        )
        gap_rows.extend(episode_gap_rows)
        if write_plots:
            _write_episode_plots(mapping, episode_report, output_dir / "episodes" / mapping["episode"])

    report = {
        "format": "f1_mcap_resampling_analysis_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "config": {
            "duplicate_epsilon_ms": duplicate_epsilon_ms,
            "gap_warning_factor": gap_warning_factor,
            "gap_fail_factor": gap_fail_factor,
            "residual_thresholds_ms": thresholds,
        },
        "episodes": episodes,
    }
    (output_dir / "resampling_statistics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (output_dir / "resampling_statistics.md").write_text(_render_markdown(report))
    _write_csv(output_dir / "stream_statistics.csv", stream_rows)
    _write_csv(output_dir / "pairwise_statistics.csv", pair_rows)
    _write_csv(output_dir / "gap_events.csv", gap_rows)
    return report
