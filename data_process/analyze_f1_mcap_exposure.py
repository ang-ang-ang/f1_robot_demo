# ruff: noqa: RUF001
"""Run independent per-camera exposure checks on raw F1 MCAP episodes."""

from __future__ import annotations

import csv
import dataclasses
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tyro

from data_process.config import config_to_dict, load_config
from data_process.exposure_operator import IndependentCameraExposureOperator
from data_process.mcap import find_episode_dirs
from data_process.mcap_quality_source import CAMERA_KEYS, McapIndependentImageSource
from data_process.quality_operator import run_streaming_operators
from data_process.quality_operator_config import load_operator_config, operator_config_to_dict


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path = Path("test_data/episode_all")
    """包含 episode_* 子目录的原始 MCAP 根目录，也可以直接指向单个 Episode。"""

    output_dir: Path = Path("artifacts/episode_all/raw_exposure")
    """每次运行会在该目录下创建带 UTC 时间戳的报告目录。"""

    config: Path | None = None
    """曝光算子 TOML 覆盖配置；默认使用 data_process/operator_config.toml。"""

    pipeline_config: Path | None = None
    """头相机时间戳修复 TOML 覆盖配置；默认使用 data_process/default_config.toml。"""

    episodes: list[str] = dataclasses.field(default_factory=list)
    """可选 Episode 目录名，例如 episode_000001；留空时分析全部。"""

    max_frames: int | None = None
    """每个 Episode 的每路相机最多解码前 N 帧；仅建议用于快速验证。"""

    write_exposure_mask_videos: bool = False
    """为每路相机输出红色过曝叠加图与二值 Mask 并排的校验视频。"""


def _new_run_dir(args: Args) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / f"{timestamp}_{args.raw_dir.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "episodes").mkdir()
    return run_dir


def _select_episode_dirs(raw_dir: Path, requested: list[str]) -> list[Path]:
    available = find_episode_dirs(raw_dir)
    if not requested:
        return available
    by_name = {path.name: path for path in available}
    missing = sorted(set(requested).difference(by_name))
    if missing:
        raise ValueError(f"Requested episodes are missing: {missing}; available={sorted(by_name)}")
    return [by_name[name] for name in requested]


def _episode_index(path: Path, fallback: int) -> int:
    match = re.fullmatch(r"episode_(\d+)", path.name)
    return int(match.group(1)) if match else fallback


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_episode_outputs(episode_dir: Path, episode: dict[str, Any]) -> None:
    exposure = episode["operators"][IndependentCameraExposureOperator.name]
    _write_csv(episode_dir / "exposure_frames.csv", exposure["frames"])
    _write_csv(episode_dir / "exposure_segments.csv", exposure["confirmed_segments"])
    (episode_dir / "operator_report.json").write_text(
        json.dumps(episode, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _overall_status(episodes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    if failures:
        return "fail"
    if any(episode["operators"][IndependentCameraExposureOperator.name]["status"] == "warning" for episode in episodes):
        return "warning"
    return "pass"


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# F1 原始 MCAP 三路独立曝光质量报告",
        "",
        f"- 原始数据：`{report['raw_dir']}`",
        f"- 总体状态：`{summary['status']}`",
        f"- 成功/失败 Episode：`{summary['successful_episode_count']}` / `{summary['failed_episode_count']}`",
        f"- 实际分析图像数：`{summary['analyzed_image_count']}`",
        f"- 曝光配置：`{report['operator_config']['source']}`",
        f"- 时间戳修复配置：`{report['pipeline_config']['source']}`",
        "",
        "## Episode 汇总",
        "",
        "| Episode | 头/左腕/右腕原始帧 | 头/左腕/右腕分析帧 | 曝光状态 | 异常片段 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for episode in report["episodes"]:
        details = episode["source_details"]
        available = details["available_image_count"]
        analyzed = details["analysis_image_count"]
        exposure = episode["operators"][IndependentCameraExposureOperator.name]
        lines.append(
            f"| `{episode['episode_name']}` | "
            f"{available[CAMERA_KEYS[0]]}/{available[CAMERA_KEYS[1]]}/{available[CAMERA_KEYS[2]]} | "
            f"{analyzed[CAMERA_KEYS[0]]}/{analyzed[CAMERA_KEYS[1]]}/{analyzed[CAMERA_KEYS[2]]} | "
            f"`{exposure['status']}` | {exposure['confirmed_segment_count']} |"
        )
    for failure in report["failures"]:
        lines.append(f"| `{failure['episode_name']}` | - | - | `fail` | - |")

    for episode in report["episodes"]:
        exposure = episode["operators"][IndependentCameraExposureOperator.name]
        repair = episode["source_details"]["head_timestamp_repair"]
        lines.extend(
            [
                "",
                f"## {episode['episode_name']}",
                "",
                f"- 头相机 Header 修复帧：`{repair['changed_frame_count']}`；高风险修正帧："
                f"`{repair['high_risk_frame_count']}`；最大修正：`{repair['max_abs_correction_ms']:.3f} ms`。",
                "- 三路相机分别使用自己的帧序号与时间轴检测；不做跨相机基线比较，也不会因同步残差跳过图像。",
                "",
                "### 曝光异常片段",
                "",
                "| 相机 | 原始 topic 帧范围 | 相对时间 (s) | 原始 Header 时间范围 (ns) | 帧数 | 最大 L50 | 最大 L95 | 最大截断比例 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        if not exposure["confirmed_segments"]:
            lines.append("| 无 | - | - | - | - | - | - | - |")
        for segment in exposure["confirmed_segments"]:
            lines.append(
                f"| `{segment['camera']}` | {segment['start_source_topic_index']}–"
                f"{segment['end_source_topic_index']} | {segment['start_time_s']:.3f}–"
                f"{segment['end_time_s']:.3f} | {segment['start_source_header_time_ns']}–"
                f"{segment['end_source_header_time_ns']} | {segment['frame_count']} | "
                f"{segment['max_l50']:.3f} | {segment['max_l95']:.3f} | "
                f"{segment['max_highlight_clipping_ratio']:.3%} |"
            )

    if report["failures"]:
        lines.extend(["", "## 失败 Episode", ""])
        for failure in report["failures"]:
            lines.append(f"- `{failure['episode_name']}`：`{failure['error_type']}` — {failure['error']}")

    lines.extend(
        [
            "",
            "## 方法与解释",
            "",
            "- 算子直接读取 MCAP 中的 JPEG/CompressedImage，不经过 LeRobot 转码，原始 MCAP 不会被修改。",
            "- 头相机 Header 的重复、负步长和周期倍数异常仅在内存中按清洗算法修复；报告保留原始 Header、publish_time、log_time、修复方法和修正量。",
            "- 三路相机独立使用绝对阈值判定，不比较其他相机，因此视场和白色物体占比差异不会直接构成异常。",
            "- 大面积过曝要求本路同时满足 L50 和 L95 门限；高光截断要求本路 R_clip 超过门限。",
            "- `--max-frames` 会分别截断每路相机；正式质检应不设置该参数。",
            "- 使用 `--write-exposure-mask-videos` 可输出左侧红色截断区域叠加、右侧二值 Mask 的逐路视频。",
            "",
            "## 输出文件",
            "",
            "- `quality_report.json`：批量汇总、每路图像计数、时间戳修复诊断和算子结果。",
            "- `episodes/*/operator_report.json`：单 Episode 完整机器可读结果。",
            "- `episodes/*/exposure_frames.csv`：逐相机原始帧指标及 MCAP 定位字段。",
            "- `episodes/*/exposure_segments.csv`：确认异常片段及其原始 topic 帧和时间戳范围。",
            "- `episodes/*/exposure_masks/*.mp4`：可选逐路过曝区域校验视频。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(args: Args) -> None:
    operator_config = load_operator_config(args.config)
    pipeline_config = load_config(args.pipeline_config)
    episode_dirs = _select_episode_dirs(args.raw_dir, args.episodes)
    run_dir = _new_run_dir(args)
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for fallback_index, episode_dir in enumerate(episode_dirs):
        print(f"分析原始 MCAP {episode_dir.name} ...")
        try:
            episode_output_dir = run_dir / "episodes" / episode_dir.name
            episode_output_dir.mkdir(parents=True)
            source = McapIndependentImageSource(
                episode_dir,
                _episode_index(episode_dir, fallback_index),
                pipeline_config=pipeline_config,
                max_frames_per_camera=args.max_frames,
            )
            episode = run_streaming_operators(
                source,
                [
                    IndependentCameraExposureOperator(
                        CAMERA_KEYS,
                        operator_config.exposure,
                        mask_output_dir=(
                            episode_output_dir / "exposure_masks" if args.write_exposure_mask_videos else None
                        ),
                    )
                ],
            )
            _write_episode_outputs(episode_output_dir, episode)
            episodes.append(episode)
            exposure = episode["operators"][IndependentCameraExposureOperator.name]
            print(
                f"  {episode['frame_count']} 帧，状态={exposure['status']}，"
                f"异常片段={exposure['confirmed_segment_count']}"
            )
        except Exception as error:  # noqa: BLE001 - preserve other episodes and record the per-episode failure
            failures.append(
                {
                    "episode_name": episode_dir.name,
                    "episode_dir": str(episode_dir),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"  失败：{type(error).__name__}: {error}")

    report = {
        "format": "f1_raw_mcap_independent_exposure_quality_v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "raw_dir": str(args.raw_dir),
        "run_dir": str(run_dir),
        "operator_config": operator_config_to_dict(operator_config),
        "pipeline_config": config_to_dict(pipeline_config),
        "arguments": {
            "episodes": args.episodes,
            "max_frames": args.max_frames,
            "write_exposure_mask_videos": args.write_exposure_mask_videos,
        },
        "summary": {
            "status": _overall_status(episodes, failures),
            "requested_episode_count": len(episode_dirs),
            "successful_episode_count": len(episodes),
            "failed_episode_count": len(failures),
            "analyzed_image_count": sum(episode["frame_count"] for episode in episodes),
            "confirmed_segment_count": sum(
                episode["operators"][IndependentCameraExposureOperator.name]["confirmed_segment_count"]
                for episode in episodes
            ),
        },
        "episodes": episodes,
        "failures": failures,
    }
    (run_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "quality_report.md").write_text(_markdown(report), encoding="utf-8")

    print(f"总体状态：{report['summary']['status']}")
    print(f"报告目录：{run_dir}")
    if failures:
        raise RuntimeError(f"{len(failures)} raw MCAP episode(s) failed; see {run_dir / 'quality_report.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
