# ruff: noqa: BLE001, RUF001, SLF001
"""Generate detailed quality reports for one or more F1 MCAP episodes."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal

import cv2
import numpy as np
import tyro

from data_process import convert_f1_mcap_to_lerobot as converter


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """单个 Episode 目录，或包含多个 ``episode_*`` 目录的数据集根目录。"""

    output_dir: Path = Path("artifacts/f1_quality_reports")
    """每次运行会在该目录下创建唯一的报告子目录。"""

    task: str = ""
    """所有 Episode 的默认任务文本；也可通过 episode_config 分别设置。"""

    episode_config: Path | None = None
    """可选 JSON，支持 include、task、裁剪范围、success、failure_reason 和 notes。"""

    fps: int = 20
    start_time_s: float = 0.0
    end_time_s: float | None = None
    gripper_scale: float = 100.0

    max_image_delta_ms: float = 40.0
    max_state_delta_ms: float = 20.0
    max_action_delta_ms: float = 60.0
    max_action_step_deg: float = 25.0
    max_image_gap_ms: float = 70.0
    max_state_gap_ms: float = 30.0
    max_action_gap_ms: float = 120.0
    max_tracking_p95_deg: float = 10.0

    image_samples: int = 24
    """每个相机均匀抽取并解码的图像数量。"""

    save_contact_sheets: bool = True
    fail_on_high_risk: bool = False
    """存在高风险失败项时以非零状态退出，适合放入数据 CI。"""


RiskStatus = Literal["pass", "warning", "fail", "info"]
RiskSeverity = Literal["low", "medium", "high"]


def _risk(
    risk_id: str,
    *,
    status: RiskStatus,
    severity: RiskSeverity,
    summary: str,
    evidence: str,
    recommendation: str,
) -> dict:
    return {
        "id": risk_id,
        "status": status,
        "severity": severity,
        "summary": summary,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _new_run_dir(args: Args) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    input_slug = args.raw_dir.name.replace(" ", "_") or "f1_data"
    run_dir = args.output_dir / f"{timestamp}_{input_slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "episodes").mkdir()
    return run_dir


def _timestamp_metrics(timestamps_ns: np.ndarray) -> dict:
    differences_ms = np.diff(timestamps_ns).astype(np.float64) / 1e6
    duration_s = float((timestamps_ns[-1] - timestamps_ns[0]) / 1e9)
    return {
        "count": len(timestamps_ns),
        "duration_s": duration_s,
        "effective_hz": float((len(timestamps_ns) - 1) / duration_s) if duration_s else 0.0,
        "dt_ms": {
            "min": float(differences_ms.min()),
            "median": float(np.median(differences_ms)),
            "p95": float(np.percentile(differences_ms, 95)),
            "p99": float(np.percentile(differences_ms, 99)),
            "max": float(differences_ms.max()),
            "mean": float(differences_ms.mean()),
            "std": float(differences_ms.std()),
            "nonpositive": int(np.sum(differences_ms <= 0)),
        },
    }


def _array_metrics(values: np.ndarray, names: tuple[str, ...] | None) -> dict:
    finite = np.isfinite(values)
    standard_deviation = np.nanstd(values, axis=0)
    return {
        "shape": list(values.shape),
        "names": list(names) if names is not None else None,
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "finite_ratio": float(finite.mean()),
        "constant_dimensions": [int(index) for index in np.flatnonzero(standard_deviation < 1e-8)],
        "min": np.nanmin(values, axis=0).tolist(),
        "max": np.nanmax(values, axis=0).tolist(),
        "mean": np.nanmean(values, axis=0).tolist(),
        "std": standard_deviation.tolist(),
    }


def _read_compressed(ref: converter.ImageRef, handles: dict[Path, BinaryIO]) -> bytes:
    handle = handles[ref.path]
    handle.seek(ref.offset)
    data = handle.read(ref.size)
    if len(data) != ref.size:
        raise ValueError(f"图像数据读取不完整：{ref.path}:{ref.offset}")
    return data


def _save_contact_sheet(images: list[np.ndarray], labels: list[str], path: Path) -> None:
    thumbnails = []
    for image_rgb, label in zip(images, labels, strict=True):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        width = 320
        height = 180
        thumbnail = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(thumbnail, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
        thumbnails.append(thumbnail)
    if not thumbnails:
        return
    columns = 4
    rows = []
    for start in range(0, len(thumbnails), columns):
        row = thumbnails[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(thumbnails[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows))


def _image_metrics(
    topic: str,
    series: converter.ImageSeries,
    *,
    sample_count: int,
    save_contact_sheet: bool,
    contact_sheet_path: Path,
) -> tuple[dict, list[dict]]:
    timestamps_ns, refs = series.finalize()
    repaired_timestamps, repair = converter._repair_image_timestamps(timestamps_ns)
    count = min(max(sample_count, 1), len(refs))
    sample_indices = np.unique(np.linspace(0, len(refs) - 1, count, dtype=np.int64))
    samples = []
    images = []
    labels = []
    hashes = Counter()
    decode_failures = []
    with ExitStack() as stack:
        handles = {path: stack.enter_context(path.open("rb")) for path in {ref.path for ref in refs}}
        for index in sample_indices:
            ref = refs[int(index)]
            compressed = _read_compressed(ref, handles)
            digest = hashlib.blake2b(compressed, digest_size=8).hexdigest()
            hashes[digest] += 1
            try:
                image = converter._read_image(ref, handles)
            except Exception as error:
                decode_failures.append({"index": int(index), "error": str(error)})
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            relative_s = float((repaired_timestamps[int(index)] - repaired_timestamps[0]) / 1e9)
            sample = {
                "index": int(index),
                "relative_time_s": relative_s,
                "shape": list(image.shape),
                "brightness_mean": float(gray.mean()),
                "dark_pixel_ratio": float(np.mean(gray < 10)),
                "bright_pixel_ratio": float(np.mean(gray > 245)),
                "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "encoded_size_bytes": ref.size,
                "format": ref.format,
                "frame_id": ref.frame_id,
            }
            samples.append(sample)
            images.append(image)
            labels.append(f"#{index} t={relative_s:.1f}s")

    if save_contact_sheet:
        _save_contact_sheet(images, labels, contact_sheet_path)
    metrics = {
        "topic": topic,
        "timestamp": _timestamp_metrics(timestamps_ns),
        "timestamp_repair": repair,
        "formats": dict(Counter(ref.format for ref in refs)),
        "frame_ids": dict(Counter(ref.frame_id for ref in refs)),
        "encoded_size_bytes": {
            "min": min(ref.size for ref in refs),
            "median": float(np.median([ref.size for ref in refs])),
            "max": max(ref.size for ref in refs),
            "total": sum(ref.size for ref in refs),
        },
        "sample_count_requested": count,
        "sample_count_decoded": len(samples),
        "sample_unique_hashes": len(hashes),
        "sample_duplicate_hashes": sum(value - 1 for value in hashes.values()),
        "decode_failures": decode_failures,
        "sample_shapes": dict(Counter(str(sample["shape"]) for sample in samples)),
        "sample_summary": {},
        "contact_sheet": str(contact_sheet_path) if save_contact_sheet else None,
        "samples": samples,
    }
    for key in ["brightness_mean", "dark_pixel_ratio", "bright_pixel_ratio", "laplacian_variance"]:
        values = np.asarray([sample[key] for sample in samples], dtype=np.float64)
        if len(values):
            metrics["sample_summary"][key] = {
                "min": float(values.min()),
                "mean": float(values.mean()),
                "max": float(values.max()),
            }
    return metrics, decode_failures


def _topic_gap_limit(topic: str, args: Args) -> float:
    if topic in converter.IMAGE_TOPICS.values():
        return args.max_image_gap_ms
    if topic in {
        converter.LEAD_JOINT_TOPIC,
        converter.LEFT_GRIPPER_COMMAND_TOPIC,
        converter.RIGHT_GRIPPER_COMMAND_TOPIC,
        converter.LEFT_LEAD_TCP_TOPIC,
        converter.RIGHT_LEAD_TCP_TOPIC,
    }:
        return args.max_action_gap_ms
    return args.max_state_gap_ms


def _analyze_episode(
    data: converter.EpisodeData,
    plan: converter.EpisodePlan,
    args: Args,
    run_dir: Path,
    annotations: dict,
) -> dict:
    risks = []
    topic_metrics = {}
    for topic, series in sorted(data.numeric.items()):
        timestamps_ns, values = series.finalize()
        timing = _timestamp_metrics(timestamps_ns)
        arrays = _array_metrics(values, series.names)
        topic_metrics[topic] = {"timestamp": timing, "values": arrays}
        if arrays["nan_count"] or arrays["inf_count"]:
            risks.append(
                _risk(
                    "numeric_non_finite",
                    status="fail",
                    severity="high",
                    summary=f"{topic} 存在 NaN 或 Inf",
                    evidence=f"NaN={arrays['nan_count']}，Inf={arrays['inf_count']}",
                    recommendation="丢弃或修复对应 Episode；不要把非有限值写入 LeRobot。",
                )
            )
        gap_limit = _topic_gap_limit(topic, args)
        max_gap = timing["dt_ms"]["max"]
        if timing["dt_ms"]["nonpositive"]:
            risks.append(
                _risk(
                    "numeric_timestamp_nonpositive",
                    status="fail",
                    severity="high",
                    summary=f"{topic} 时间戳非严格递增",
                    evidence=f"非递增次数={timing['dt_ms']['nonpositive']}",
                    recommendation="修复消息时间戳来源；数值状态和动作不应依赖帧序号重建。",
                )
            )
        if max_gap > gap_limit:
            risks.append(
                _risk(
                    "numeric_timestamp_gap",
                    status="warning",
                    severity="medium",
                    summary=f"{topic} 存在较大消息间隔",
                    evidence=f"最大间隔={max_gap:.3f} ms，门限={gap_limit:.3f} ms",
                    recommendation="检查发布线程阻塞、QoS、CPU/磁盘负载和控制器调度。",
                )
            )

    image_metrics = {}
    for feature, topic in converter.IMAGE_TOPICS.items():
        contact_path = run_dir / f"{data.name}_{feature.replace('.', '_')}.jpg"
        metrics, decode_failures = _image_metrics(
            topic,
            data.images[topic],
            sample_count=args.image_samples,
            save_contact_sheet=args.save_contact_sheets,
            contact_sheet_path=contact_path,
        )
        image_metrics[feature] = metrics
        timestamp = metrics["timestamp"]["dt_ms"]
        if timestamp["nonpositive"]:
            risks.append(
                _risk(
                    "camera_timestamp_nonpositive",
                    status="warning",
                    severity="high" if feature == "observation.images.head" else "medium",
                    summary=f"{feature} 原始时间戳非严格递增",
                    evidence=(
                        f"非递增次数={timestamp['nonpositive']}，最大修正="
                        f"{metrics['timestamp_repair']['max_correction_ms']:.3f} ms"
                    ),
                    recommendation="本次可按帧序号修复；后续应从相机驱动修正真实采集时间戳。",
                )
            )
        if timestamp["max"] > args.max_image_gap_ms:
            risks.append(
                _risk(
                    "camera_timestamp_gap",
                    status="warning",
                    severity="medium",
                    summary=f"{feature} 原始时间戳存在大间隔",
                    evidence=f"最大间隔={timestamp['max']:.3f} ms，门限={args.max_image_gap_ms:.3f} ms",
                    recommendation="检查是否丢帧、时间戳冻结或相机发布线程拥塞。",
                )
            )
        if decode_failures:
            risks.append(
                _risk(
                    "image_decode_failure",
                    status="fail",
                    severity="high",
                    summary=f"{feature} 抽样图像解码失败",
                    evidence=f"失败数={len(decode_failures)}/{metrics['sample_count_requested']}",
                    recommendation="定位损坏消息；完整转换会在解码失败处停止。",
                )
            )
        if len(metrics["sample_shapes"]) > 1:
            risks.append(
                _risk(
                    "image_shape_changed",
                    status="fail",
                    severity="high",
                    summary=f"{feature} 图像尺寸发生变化",
                    evidence=str(metrics["sample_shapes"]),
                    recommendation="一个 LeRobot 图像字段必须保持固定分辨率。",
                )
            )

    residuals = plan.report["sampling_residuals"]
    for feature in converter.IMAGE_TOPICS:
        value = residuals[feature]["max_abs_ms"]
        risks.append(
            _risk(
                "image_alignment_residual",
                status="pass" if value <= args.max_image_delta_ms else "fail",
                severity="high",
                summary=f"{feature} 重采样对齐误差",
                evidence=f"最大绝对误差={value:.3f} ms，门限={args.max_image_delta_ms:.3f} ms",
                recommendation="超限时不要放宽门限掩盖问题，应检查时间源或降低输出 FPS。",
            )
        )
    for key, limit, risk_id in [
        ("state", args.max_state_delta_ms, "state_alignment_residual"),
        ("action", args.max_action_delta_ms, "action_alignment_residual"),
    ]:
        value = residuals[key]["max_abs_ms"]
        risks.append(
            _risk(
                risk_id,
                status="pass" if value <= limit else "fail",
                severity="high",
                summary=f"{key} 重采样对齐误差",
                evidence=f"最大绝对误差={value:.3f} ms，门限={limit:.3f} ms",
                recommendation="超限时检查数据流频率、时间戳和录制阻塞。",
            )
        )

    action_steps_deg = np.rad2deg(np.abs(np.diff(plan.action[:, :14], axis=0)))
    per_frame_action_step = action_steps_deg.max(axis=1, initial=0.0)
    jump_indices = np.flatnonzero(per_frame_action_step > args.max_action_step_deg)
    action_quality = {
        "max_arm_action_step_deg": float(per_frame_action_step.max(initial=0.0)),
        "jump_threshold_deg": args.max_action_step_deg,
        "jump_count": len(jump_indices),
        "largest_jumps": [
            {
                "output_frame": int(index + 1),
                "relative_time_s": float((index + 1) / args.fps),
                "max_joint_step_deg": float(per_frame_action_step[index]),
            }
            for index in np.argsort(per_frame_action_step)[-10:][::-1]
        ],
    }
    risks.append(
        _risk(
            "action_discontinuity",
            status="warning" if len(jump_indices) else "pass",
            severity="high",
            summary="相邻动作目标跳变检查",
            evidence=(
                f"最大步长={action_quality['max_arm_action_step_deg']:.3f}°，"
                f"超过 {args.max_action_step_deg:.3f}° 的帧数={len(jump_indices)}"
            ),
            recommendation="排除示教接管、离合重定位、急停恢复和异常目标阶跃。",
        )
    )

    tracking_error_deg = np.rad2deg(np.abs(plan.action[:, :14] - plan.state[:, :14]))
    tracking = {
        "joint_names": list(converter.STATE_JOINT_NAMES),
        "median_abs_deg": np.median(tracking_error_deg, axis=0).tolist(),
        "p95_abs_deg": np.percentile(tracking_error_deg, 95, axis=0).tolist(),
        "max_abs_deg": tracking_error_deg.max(axis=0).tolist(),
        "global_p95_abs_deg": float(np.percentile(tracking_error_deg, 95)),
        "worst_joint_p95_deg": float(np.percentile(tracking_error_deg, 95, axis=0).max()),
    }
    risks.append(
        _risk(
            "leader_follower_tracking",
            status="pass" if tracking["worst_joint_p95_deg"] <= args.max_tracking_p95_deg else "warning",
            severity="medium",
            summary="示教目标与从机状态跟随误差",
            evidence=(
                f"全局 P95={tracking['global_p95_abs_deg']:.3f}°，最差关节 P95={tracking['worst_joint_p95_deg']:.3f}°"
            ),
            recommendation="误差过大时检查控制延迟、限速、饱和、碰撞和示教动作过快。",
        )
    )

    if not plan.task or plan.task == "__TASK_NOT_SET__":
        risks.append(
            _risk(
                "task_label_missing",
                status="fail",
                severity="high",
                summary="缺少任务语言标签",
                evidence="metadata.yaml 和 MCAP 中没有可用于 Pi0.5 的任务文本。",
                recommendation="通过 --task 或 --episode-config 为每个 Episode 提供明确任务描述。",
            )
        )
    success = annotations.get("success")
    if success is None:
        risks.append(
            _risk(
                "success_label_unverified",
                status="warning",
                severity="medium",
                summary="无法自动确认任务成功",
                evidence="episode_config 中没有 success 标记，MCAP 本身也没有结果字段。",
                recommendation="人工复核抽帧/视频，并记录 success、failure_reason 和裁剪边界。",
            )
        )
    elif success is False:
        risks.append(
            _risk(
                "demonstration_failed",
                status="fail",
                severity="high",
                summary="该示教被标记为失败",
                evidence=f"failure_reason={annotations.get('failure_reason')!r}",
                recommendation="普通行为克隆训练应设置 include=false；仅在专门利用失败轨迹的方法中保留。",
            )
        )
    else:
        risks.append(
            _risk(
                "success_label_verified",
                status="pass",
                severity="medium",
                summary="该示教已标记成功",
                evidence="episode_config.success=true",
                recommendation="仍建议抽样复核成功定义是否一致。",
            )
        )

    if converter.LEFT_TCP_STATE_TOPIC in data.numeric and converter.LEFT_LEAD_TCP_TOPIC in data.numeric:
        _, measured_tcp = data.numeric[converter.LEFT_TCP_STATE_TOPIC].finalize()
        _, lead_tcp = data.numeric[converter.LEFT_LEAD_TCP_TOPIC].finalize()
        measured_scale = float(np.median(np.abs(measured_tcp[:, :3])))
        lead_scale = float(np.median(np.abs(lead_tcp[:, :3])))
        scale_ratio = measured_scale / max(lead_scale, 1e-12)
        tcp_metrics = {
            "measured_translation_median_abs": measured_scale,
            "lead_translation_median_abs": lead_scale,
            "translation_scale_ratio": scale_ratio,
            "lead_orientation_max_step": float(np.abs(np.diff(lead_tcp[:, 3:6], axis=0)).max(initial=0.0)),
        }
        if scale_ratio > 100 or tcp_metrics["lead_orientation_max_step"] > 180:
            risks.append(
                _risk(
                    "tcp_unit_and_wrap",
                    status="warning",
                    severity="high",
                    summary="从机与示教 TCP 存在单位或欧拉角环绕风险",
                    evidence=(
                        f"平移量级比={scale_ratio:.1f}，示教姿态最大单步变化="
                        f"{tcp_metrics['lead_orientation_max_step']:.3f}"
                    ),
                    recommendation="统一米/毫米和角度/弧度，并用连续旋转表示后再训练 TCP 动作。",
                )
            )
    else:
        tcp_metrics = None

    high_failures = sum(risk["status"] == "fail" and risk["severity"] == "high" for risk in risks)
    warnings = sum(risk["status"] == "warning" for risk in risks)
    overall_status = "fail" if high_failures else "warning" if warnings else "pass"
    return {
        "episode": data.name,
        "overall_status": overall_status,
        "task": None if plan.task == "__TASK_NOT_SET__" else plan.task,
        "annotations": annotations,
        "source_files": [str(path) for path in data.source_files],
        "selected_range_s": plan.report["selected_range_s"],
        "output_preview": plan.report["output"],
        "risk_summary": {"high_failures": high_failures, "warnings": warnings, "items": len(risks)},
        "risks": risks,
        "sampling_residuals": residuals,
        "action_quality": action_quality,
        "leader_follower_tracking": tracking,
        "tcp_quality": tcp_metrics,
        "numeric_topics": topic_metrics,
        "images": image_metrics,
    }


def _render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# F1 数据质量评估报告",
        "",
        f"- 生成时间：`{report['generated_at_utc']}`",
        f"- 输入目录：`{report['input']}`",
        f"- 模式：`{report['mode']}`",
        f"- Episode 数：`{summary['episode_count']}`",
        f"- 总体结果：`{summary['overall_status']}`",
        "",
        "## 总览",
        "",
        "| Episode | 结果 | 任务 | 区间 / s | 预览帧数 | 高风险失败 | 告警 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for episode in report["episodes"]:
        selected = episode.get("selected_range_s", {})
        interval = f"{selected['start']:.3f}-{selected['end']:.3f}" if selected else "-"
        task = (episode.get("task") or "未设置").replace("|", "\\|")
        risk_summary = episode["risk_summary"]
        lines.append(
            f"| `{episode['episode']}` | `{episode['overall_status']}` | {task} | {interval} | "
            f"{episode.get('output_preview', {}).get('frames', '-')} | {risk_summary['high_failures']} | "
            f"{risk_summary['warnings']} |"
        )

    for episode in report["episodes"]:
        lines.extend(["", f"## {episode['episode']}", ""])
        lines.extend(
            [
                "### 主要风险",
                "",
                "| 状态 | 严重度 | 风险项 | 证据 | 建议 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for risk in episode["risks"]:
            values = [risk["status"], risk["severity"], risk["summary"], risk["evidence"], risk["recommendation"]]
            values = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append(f"| `{values[0]}` | `{values[1]}` | {values[2]} | {values[3]} | {values[4]} |")

        lines.extend(
            [
                "",
                "### 重采样误差",
                "",
                "| 数据流 | 中位绝对误差 / ms | P95 / ms | 最大绝对误差 / ms |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key, metrics in episode.get("sampling_residuals", {}).items():
            lines.append(
                f"| `{key}` | {metrics['median_abs_ms']:.3f} | {metrics['p95_abs_ms']:.3f} | "
                f"{metrics['max_abs_ms']:.3f} |"
            )

        lines.extend(
            [
                "",
                "### 数值话题时序",
                "",
                "| Topic | 数量 | 频率 / Hz | 中位周期 / ms | P95 / ms | P99 / ms | 最大间隔 / ms | 非递增 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for topic, metrics in episode.get("numeric_topics", {}).items():
            timestamp = metrics["timestamp"]
            dt = timestamp["dt_ms"]
            lines.append(
                f"| `{topic}` | {timestamp['count']} | {timestamp['effective_hz']:.3f} | {dt['median']:.3f} | "
                f"{dt['p95']:.3f} | {dt['p99']:.3f} | {dt['max']:.3f} | {dt['nonpositive']} |"
            )

        lines.extend(
            [
                "",
                "### 图像抽样",
                "",
                "| 图像字段 | 解码数 | 尺寸 | 亮度均值 | 拉普拉斯方差均值 | 原始时间戳非递增 | 最大时间戳修正 / ms |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for feature, metrics in episode.get("images", {}).items():
            summary_values = metrics["sample_summary"]
            brightness = summary_values.get("brightness_mean", {}).get("mean", float("nan"))
            blur = summary_values.get("laplacian_variance", {}).get("mean", float("nan"))
            repair = metrics["timestamp_repair"]
            lines.append(
                f"| `{feature}` | {metrics['sample_count_decoded']} | {metrics['sample_shapes']} | "
                f"{brightness:.3f} | {blur:.3f} | {repair['nonpositive_source_steps']} | "
                f"{repair['max_correction_ms']:.3f} |"
            )

        tracking = episode.get("leader_follower_tracking")
        action = episode.get("action_quality")
        if tracking and action:
            lines.extend(
                [
                    "",
                    "### 动作与跟随",
                    "",
                    f"- 最大相邻关节目标步长：`{action['max_arm_action_step_deg']:.3f}°`。",
                    f"- 超过动作跳变门限的帧数：`{action['jump_count']}`。",
                    f"- 主从跟随全局 P95：`{tracking['global_p95_abs_deg']:.3f}°`。",
                    f"- 最差关节 P95：`{tracking['worst_joint_p95_deg']:.3f}°`。",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def _write_reports(report: dict, run_dir: Path) -> None:
    (run_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (run_dir / "quality_report.md").write_text(_render_markdown(report))
    for episode in report["episodes"]:
        (run_dir / "episodes" / f"{episode['episode']}.json").write_text(
            json.dumps(episode, indent=2, ensure_ascii=False)
        )
    with (run_dir / "risks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode", "status", "severity", "id", "summary", "evidence", "recommendation"],
        )
        writer.writeheader()
        for episode in report["episodes"]:
            for risk in episode["risks"]:
                writer.writerow({"episode": episode["episode"], **risk})


def main(args: Args) -> None:
    run_dir = _new_run_dir(args)
    episode_config = converter._load_episode_config(args.episode_config)
    episode_dirs = converter._find_episode_dirs(args.raw_dir)
    conversion_args = converter.Args(
        raw_dir=args.raw_dir,
        repo_id="quality/analysis",
        task=args.task or "__TASK_NOT_SET__",
        episode_config=args.episode_config,
        fps=args.fps,
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
        gripper_scale=args.gripper_scale,
        max_image_delta_ms=float("inf"),
        max_state_delta_ms=float("inf"),
        max_action_delta_ms=float("inf"),
        max_action_step_deg=args.max_action_step_deg,
    )
    episodes = []
    for episode_dir in episode_dirs:
        include, task, start_time_s, end_time_s = converter._episode_settings(
            episode_dir.name,
            episode_config,
            conversion_args,
        )
        if not include:
            print(f"跳过 {episode_dir.name}：include=false")
            continue
        print(f"评估 {episode_dir.name} ...")
        try:
            data = converter._load_episode(episode_dir)
            plan = converter._make_episode_plan(
                data,
                task=task,
                fps=args.fps,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                gripper_scale=args.gripper_scale,
                max_image_delta_ms=float("inf"),
                max_state_delta_ms=float("inf"),
                max_action_delta_ms=float("inf"),
                max_action_step_deg=args.max_action_step_deg,
            )
            annotations = {
                key: episode_config.get(episode_dir.name, {}).get(key)
                for key in ("success", "failure_reason", "notes")
                if key in episode_config.get(episode_dir.name, {})
            }
            episodes.append(_analyze_episode(data, plan, args, run_dir, annotations))
        except Exception as error:
            episodes.append(
                {
                    "episode": episode_dir.name,
                    "overall_status": "fail",
                    "task": None if task == "__TASK_NOT_SET__" else task,
                    "risk_summary": {"high_failures": 1, "warnings": 0, "items": 1},
                    "risks": [
                        _risk(
                            "analysis_failed",
                            status="fail",
                            severity="high",
                            summary="Episode 解析或评估失败",
                            evidence=f"{type(error).__name__}: {error}",
                            recommendation="检查 MCAP 完整性、必需话题、schema、时间范围和夹爪标定。",
                        )
                    ],
                }
            )

    if not episodes:
        raise ValueError("没有选中任何 Episode")
    high_failures = sum(episode["risk_summary"]["high_failures"] for episode in episodes)
    warnings = sum(episode["risk_summary"]["warnings"] for episode in episodes)
    overall_status = "fail" if high_failures else "warning" if warnings else "pass"
    report = {
        "format": "f1_data_quality_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(args.raw_dir),
        "mode": "single" if len(episode_dirs) == 1 else "batch",
        "config": dataclasses.asdict(args) | {"raw_dir": str(args.raw_dir), "output_dir": str(args.output_dir)},
        "summary": {
            "overall_status": overall_status,
            "episode_count": len(episodes),
            "high_failures": high_failures,
            "warnings": warnings,
        },
        "episodes": episodes,
    }
    if args.episode_config is not None:
        report["config"]["episode_config"] = str(args.episode_config)
    _write_reports(report, run_dir)
    print(f"总体结果：{overall_status}")
    print(f"报告目录：{run_dir}")
    print(f"JSON：{run_dir / 'quality_report.json'}")
    print(f"Markdown：{run_dir / 'quality_report.md'}")
    print(f"风险 CSV：{run_dir / 'risks.csv'}")
    if args.fail_on_high_risk and high_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main(tyro.cli(Args))
