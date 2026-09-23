# ruff: noqa: RUF001
from __future__ import annotations

import csv
import dataclasses
import json
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from data_process.config import HeadTimestampCleaningConfig, PipelineConfig, config_to_dict
from data_process.mcap import EpisodeData, MessageSample, find_episode_dirs, header_timestamps, load_episode

RepairMethod = Literal["original", "local_interpolation", "robust_regression", "monotonic_guard"]
RiskLevel = Literal["none", "warning", "high"]


@dataclasses.dataclass(frozen=True)
class TimestampRepairResult:
    original_ns: np.ndarray
    corrected_ns: np.ndarray
    methods: tuple[RepairMethod, ...]
    risks: tuple[RiskLevel, ...]
    nominal_period_ns: int
    predicted_ns: np.ndarray
    diagnostics: dict[str, Any]


def _estimate_nominal_period_ns(timestamps_ns: np.ndarray, config: HeadTimestampCleaningConfig) -> int:
    if config.nominal_period_ms > 0:
        return round(config.nominal_period_ms * 1e6)
    differences_ms = np.diff(timestamps_ns).astype(np.float64) / 1e6
    candidates = differences_ms[
        (differences_ms >= config.period_search_min_ms) & (differences_ms <= config.period_search_max_ms)
    ]
    if len(candidates) < max(3, config.minimum_anchor_count // 4):
        positive = differences_ms[differences_ms > 0]
        if len(positive) < 3:
            raise ValueError("Not enough positive camera timestamp steps to estimate a nominal period")
        preliminary = float(np.percentile(positive, 35))
        candidates = positive[(positive >= preliminary * 0.7) & (positive <= preliminary * 1.3)]
    if not len(candidates):
        raise ValueError("No camera timestamp steps fall inside the configured nominal-period search range")
    return round(float(np.median(candidates)) * 1e6)


def _robust_frame_index_fit(
    timestamps_ns: np.ndarray,
    nominal_period_ns: int,
    config: HeadTimestampCleaningConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    indices = np.arange(len(timestamps_ns), dtype=np.float64)
    origin_ns = int(timestamps_ns[0])
    relative_ns = timestamps_ns.astype(np.float64) - origin_ns
    period = float(nominal_period_ns)
    intercept = float(np.median(relative_ns - indices * period))
    prediction = intercept + indices * period
    tolerance = config.anchor_residual_tolerance_periods * nominal_period_ns
    anchors = np.abs(relative_ns - prediction) <= tolerance

    for _ in range(4):
        if np.sum(anchors) < max(2, config.minimum_anchor_count):
            break
        centered_indices = indices - float(np.mean(indices[anchors]))
        design = np.column_stack([centered_indices[anchors], np.ones(np.sum(anchors))])
        slope, centered_intercept = np.linalg.lstsq(design, relative_ns[anchors], rcond=None)[0]
        prediction = centered_indices * slope + centered_intercept
        residual = relative_ns - prediction
        median_residual = float(np.median(residual[anchors]))
        mad = float(np.median(np.abs(residual[anchors] - median_residual)))
        robust_limit = max(tolerance, 4.5 * 1.4826 * mad)
        next_anchors = np.abs(residual - median_residual) <= robust_limit
        if np.array_equal(next_anchors, anchors):
            break
        anchors = next_anchors

    if np.sum(anchors) < 2:
        raise ValueError("Unable to find enough valid Header anchors for robust timestamp repair")
    centered_indices = indices - float(np.mean(indices[anchors]))
    design = np.column_stack([centered_indices[anchors], np.ones(np.sum(anchors))])
    slope, centered_intercept = np.linalg.lstsq(design, relative_ns[anchors], rcond=None)[0]
    prediction = centered_indices * slope + centered_intercept
    return np.rint(prediction + origin_ns).astype(np.int64), anchors, float(slope)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split_points)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _step_anomalies(
    timestamps_ns: np.ndarray,
    nominal_period_ns: int,
    tolerance_ratio: float,
) -> dict[str, Any]:
    differences = np.diff(timestamps_ns).astype(np.int64)
    tolerance = tolerance_ratio * nominal_period_ns
    normal = np.abs(differences - nominal_period_ns) <= tolerance
    rounded_multiple = np.rint(differences / nominal_period_ns).astype(np.int64)
    multiple_counts: dict[str, int] = {}
    for multiple in sorted(set(rounded_multiple.tolist())):
        mask = (rounded_multiple == multiple) & ~normal
        if np.any(mask):
            multiple_counts[str(int(multiple))] = int(np.sum(mask))
    anomaly_indices = np.flatnonzero(~normal)
    return {
        "normal_count": int(np.sum(normal)),
        "anomaly_count": int(np.sum(~normal)),
        "duplicate_count": int(np.sum(differences == 0)),
        "negative_count": int(np.sum(differences < 0)),
        "multiple_counts": multiple_counts,
        "largest_anomalies": [
            {
                "edge_after_frame": int(index),
                "step_ms": float(differences[index] / 1e6),
                "period_multiple": float(differences[index] / nominal_period_ns),
            }
            for index in anomaly_indices[
                np.argsort(np.abs(differences[anomaly_indices] - nominal_period_ns))[-20:][::-1]
            ]
        ],
    }


def repair_camera_timestamps(
    timestamps_ns: np.ndarray,
    config: HeadTimestampCleaningConfig,
) -> TimestampRepairResult:
    if timestamps_ns.ndim != 1 or len(timestamps_ns) < 3:
        raise ValueError("At least three one-dimensional camera timestamps are required")
    original = timestamps_ns.astype(np.int64, copy=True)
    nominal_period_ns = _estimate_nominal_period_ns(original, config)
    predicted, anchors, fitted_period_ns = _robust_frame_index_fit(original, nominal_period_ns, config)
    residual = original.astype(np.float64) - predicted.astype(np.float64)
    tolerance = config.anchor_residual_tolerance_periods * nominal_period_ns
    invalid = np.abs(residual) > tolerance

    differences = np.diff(original).astype(np.int64)
    normal_tolerance = config.normal_step_tolerance_ratio * nominal_period_ns
    abnormal_edges = np.abs(differences - nominal_period_ns) > normal_tolerance
    for edge in np.flatnonzero(abnormal_edges):
        left_residual = abs(residual[edge])
        right_residual = abs(residual[edge + 1])
        if differences[edge] <= 0 or max(left_residual, right_residual) > tolerance:
            invalid[edge + (right_residual >= left_residual)] = True

    corrected = original.copy()
    methods: list[RepairMethod] = ["original"] * len(original)
    run_reports = []
    for start, end in _contiguous_runs(invalid):
        left = start - 1
        right = end + 1
        run_length = end - start + 1
        bounded = left >= 0 and right < len(original)
        local_fit = False
        if bounded and run_length <= config.short_run_max_frames:
            observed_span = original[right] - original[left]
            expected_span = (right - left) * nominal_period_ns
            local_fit = abs(observed_span - expected_span) <= config.local_anchor_tolerance_ratio * expected_span
        if local_fit:
            local_values = np.rint(np.linspace(original[left], original[right], right - left + 1)).astype(np.int64)
            corrected[start : end + 1] = local_values[start - left : end - left + 1]
            methods[start : end + 1] = ["local_interpolation"] * run_length
            method: RepairMethod = "local_interpolation"
        else:
            corrected[start : end + 1] = predicted[start : end + 1]
            methods[start : end + 1] = ["robust_regression"] * run_length
            method = "robust_regression"
        run_reports.append(
            {
                "start_frame": start,
                "end_frame": end,
                "frames": run_length,
                "bounded": bounded,
                "method": method,
            }
        )

    monotonic_guard_count = 0
    minimum_step_ns = max(1, round(nominal_period_ns * (1 - config.normal_step_tolerance_ratio)))
    for index in range(1, len(corrected)):
        if corrected[index] <= corrected[index - 1]:
            corrected[index] = max(int(predicted[index]), int(corrected[index - 1] + minimum_step_ns))
            methods[index] = "monotonic_guard"
            monotonic_guard_count += 1

    correction_ns = corrected - original
    correction_periods = np.abs(correction_ns) / nominal_period_ns
    risks: list[RiskLevel] = []
    for value in correction_periods:
        if value >= config.high_risk_correction_periods:
            risks.append("high")
        elif value >= config.warning_correction_periods:
            risks.append("warning")
        else:
            risks.append("none")
    corrected_differences = np.diff(corrected)
    if np.any(corrected_differences <= 0):
        raise AssertionError("Timestamp repair failed to produce a strictly increasing sequence")

    diagnostics = {
        "nominal_period_ms": nominal_period_ns / 1e6,
        "fitted_period_ms": fitted_period_ns / 1e6,
        "estimated_hz": 1e9 / fitted_period_ns,
        "anchor_count": int(np.sum(anchors)),
        "anchor_ratio": float(np.mean(anchors)),
        "invalid_frame_count": int(np.sum(invalid)),
        "changed_frame_count": int(np.sum(correction_ns != 0)),
        "method_counts": dict(sorted({method: methods.count(method) for method in set(methods)}.items())),
        "max_abs_correction_ms": float(np.max(np.abs(correction_ns)) / 1e6),
        "p95_abs_correction_ms": float(np.percentile(np.abs(correction_ns), 95) / 1e6),
        "warning_frame_count": risks.count("warning"),
        "high_risk_frame_count": risks.count("high"),
        "monotonic_guard_count": monotonic_guard_count,
        "repair_runs": run_reports,
        "source_steps": _step_anomalies(original, nominal_period_ns, config.normal_step_tolerance_ratio),
        "corrected_steps": _step_anomalies(corrected, nominal_period_ns, config.normal_step_tolerance_ratio),
    }
    return TimestampRepairResult(
        original_ns=original,
        corrected_ns=corrected,
        methods=tuple(methods),
        risks=tuple(risks),
        nominal_period_ns=nominal_period_ns,
        predicted_ns=predicted,
        diagnostics=diagnostics,
    )


def _validate_patch_compatibility(episode: EpisodeData) -> None:
    compressed = [chunk for chunk in episode.chunks if chunk.compression]
    nonzero_chunk_crc = [chunk for chunk in episode.chunks if chunk.uncompressed_crc]
    nonzero_data_crc = [path for path, value in episode.data_section_crcs.items() if value]
    if compressed or nonzero_chunk_crc or nonzero_data_crc:
        raise ValueError(
            "Equal-size MCAP patching requires uncompressed chunks with zero chunk/data CRCs; "
            "use an MCAP reader/writer rewrite path for this recording"
        )


def _copy_episode(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Cleaned episode already exists: {destination}")
        shutil.rmtree(destination)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {"conversion_reports", "quality_reports", "cleaning_reports"}}
        ignored.update(name for name in names if name == "__pycache__")
        return ignored

    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=ignore)


def _write_header_patch(handle: Any, sample: MessageSample, corrected_ns: int) -> None:
    if sample.header_time_ns is None:
        raise ValueError(f"Cannot patch a message without Header: {sample.topic}")
    handle.seek(sample.cdr_file_offset + 4)
    existing = handle.read(8)
    if len(existing) != 8:
        raise ValueError(f"Truncated CDR Header in {sample.source_path}")
    seconds, nanoseconds = struct.unpack("<iI", existing)
    existing_ns = seconds * 1_000_000_000 + nanoseconds
    if existing_ns != sample.header_time_ns:
        raise ValueError(
            f"Source timestamp changed before patching: expected {sample.header_time_ns}, found {existing_ns}"
        )
    corrected_seconds, corrected_nanoseconds = divmod(int(corrected_ns), 1_000_000_000)
    handle.seek(sample.cdr_file_offset + 4)
    handle.write(struct.pack("<iI", corrected_seconds, corrected_nanoseconds))


def _frame_rows(
    episode: EpisodeData,
    samples: list[MessageSample],
    repair: TimestampRepairResult,
) -> list[dict[str, Any]]:
    rows = []
    for index, (sample, original_ns, corrected_ns, method, risk) in enumerate(
        zip(
            samples,
            repair.original_ns,
            repair.corrected_ns,
            repair.methods,
            repair.risks,
            strict=True,
        )
    ):
        rows.append(
            {
                "episode": episode.name,
                "topic": sample.topic,
                "source_file": sample.source_path.name,
                "topic_frame_index": index,
                "mcap_record_index": sample.record_index,
                "sequence": sample.sequence,
                "header_time_ns": int(original_ns),
                "corrected_header_time_ns": int(corrected_ns),
                "correction_ms": float((corrected_ns - original_ns) / 1e6),
                "correction_periods": float(abs(corrected_ns - original_ns) / repair.nominal_period_ns),
                "repair_method": method,
                "risk": risk,
                "publish_time_ns": sample.publish_time_ns,
                "log_time_ns": sample.log_time_ns,
                "log_minus_publish_ms": (sample.log_time_ns - sample.publish_time_ns) / 1e6,
                "header_minus_publish_ms": (int(original_ns) - sample.publish_time_ns) / 1e6,
                "corrected_header_minus_publish_ms": (int(corrected_ns) - sample.publish_time_ns) / 1e6,
            }
        )
    return rows


def _write_episode_report(
    report_dir: Path,
    episode_report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cleaning_report.json").write_text(json.dumps(episode_report, indent=2, ensure_ascii=False))
    if rows:
        with (report_dir / "timestamp_corrections.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# F1 MCAP 时间戳清洗报告",
        "",
        f"- 生成时间：`{report['generated_at_utc']}`",
        f"- 输入：`{report['input']}`",
        f"- 输出：`{report['output']}`",
        f"- 模式：`{'dry-run' if report['dry_run'] else 'write'}`",
        f"- Episode 数：`{len(report['episodes'])}`",
        "",
        "## 汇总",
        "",
        "| Episode | 帧数 | 名义周期 / ms | 修改帧 | 最大修正 / ms | 告警帧 | 高风险帧 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in report["episodes"]:
        diagnostics = episode["diagnostics"]
        lines.append(
            f"| `{episode['episode']}` | {episode['frames']} | {diagnostics['nominal_period_ms']:.6f} | "
            f"{diagnostics['changed_frame_count']} | {diagnostics['max_abs_correction_ms']:.3f} | "
            f"{diagnostics['warning_frame_count']} | {diagnostics['high_risk_frame_count']} |"
        )
    lines.extend(
        [
            "",
            "## 算法约束",
            "",
            "- 始终保持 MCAP 消息记录顺序，不按错误 Header 排序。",
            "- 正常 Header 用于估计周期和鲁棒帧序号模型；短异常段优先局部插值，长段使用鲁棒回归。",
            "- publish/log 只进入审计 CSV，不用于替代采集时刻 Header。",
            "- 清洗不做 20 Hz 重采样；固定频率最近邻采样只在 LeRobot 转换阶段执行。",
            "- 原 MCAP 不会原地修改；输出文件仅等长改写 CDR Header 的 sec/nanosec 字段。",
            "",
        ]
    )
    return "\n".join(lines)


def clean_dataset(
    raw_dir: Path,
    output_dir: Path,
    report_dir: Path,
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    raw_resolved = raw_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == raw_resolved or raw_resolved in output_resolved.parents:
        raise ValueError("output_dir must not be the input directory or a child of it")
    episode_reports = []
    for episode_dir in find_episode_dirs(raw_dir):
        episode = load_episode(episode_dir)
        _validate_patch_compatibility(episode)
        samples = episode.topic(config.head_timestamp.topic)
        timestamps = header_timestamps(samples)
        if timestamps is None:
            raise ValueError(
                f"{episode.name} has no complete Header timestamp series for {config.head_timestamp.topic}"
            )
        repair = repair_camera_timestamps(timestamps, config.head_timestamp)
        rows = _frame_rows(episode, samples, repair)
        destination = output_dir / episode.name
        episode_report = {
            "episode": episode.name,
            "topic": config.head_timestamp.topic,
            "source_dir": str(episode_dir),
            "output_dir": None if dry_run else str(destination),
            "frames": len(samples),
            "diagnostics": repair.diagnostics,
        }
        episode_report_dir = report_dir / "episodes" / episode.name
        _write_episode_report(episode_report_dir, episode_report, rows)
        if not dry_run:
            _copy_episode(episode_dir, destination, overwrite=overwrite)
            handles: dict[Path, Any] = {}
            try:
                for source_file in episode.source_files:
                    relative = source_file.relative_to(episode.root)
                    handles[source_file] = (destination / relative).open("r+b")
                for sample, corrected_ns in zip(samples, repair.corrected_ns, strict=True):
                    if corrected_ns != sample.header_time_ns:
                        _write_header_patch(handles[sample.source_path], sample, int(corrected_ns))
            finally:
                for handle in handles.values():
                    handle.close()
            manifest = {
                "format": "f1_mcap_cleaning_manifest_v1",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_episode": str(episode_dir),
                "topic": config.head_timestamp.topic,
                "diagnostics": repair.diagnostics,
                "report_dir": str(episode_report_dir),
            }
            (destination / "f1_cleaning_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        episode_reports.append(episode_report)

    report = {
        "format": "f1_mcap_timestamp_cleaning_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(raw_dir),
        "output": str(output_dir),
        "report_dir": str(report_dir),
        "dry_run": dry_run,
        "config": config_to_dict(config),
        "episodes": episode_reports,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (report_dir / "cleaning_report.md").write_text(_render_markdown(report))
    return report
