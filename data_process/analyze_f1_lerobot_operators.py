# ruff: noqa: RUF001
"""Run composable image and joint quality operators on an F1 LeRobot dataset."""

from __future__ import annotations

import csv
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tyro

from data_process.exposure_operator import CrossCameraExposureOperator
from data_process.joint_motion_operator import JointMotionOperator
from data_process.quality_operator import run_episode_operators
from data_process.quality_operator_config import load_operator_config, operator_config_to_dict

CAMERA_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_root: Path = Path("artifacts/F1_data_SOP_openbox/lerobot")
    """LeRobot 数据集根目录。"""

    output_dir: Path = Path("artifacts/F1_data_SOP_openbox/operator_quality")
    """每次运行会在该目录创建带 UTC 时间戳的报告子目录。"""

    config: Path | None = None
    """可选 TOML，只需写需要覆盖的算子参数。"""

    episodes: list[int] = dataclasses.field(default_factory=list)
    """要分析的 LeRobot Episode 索引；留空时分析全部 Episode。"""

    run_exposure: bool = True
    run_joint_state: bool = True
    run_joint_action: bool = True

    joint_limit_urdf: Path | None = Path("f1p01_00000000_20260920/urdf/f1p01_00000000_20260918.urdf")
    """位置限位来源；URDF 中 velocity<=0 时自动使用 TOML 的临时速度上限。"""

    max_frames: int | None = None
    """仅处理每条 Episode 的前 N 帧，适合快速验证；正式报告应留空。"""


def _new_run_dir(args: Args) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / f"{timestamp}_{args.dataset_root.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "episodes").mkdir()
    return run_dir


def _episode_indices(dataset_root: Path, requested: list[int]) -> list[int]:
    episodes_path = dataset_root / "meta/episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"LeRobot episodes metadata not found: {episodes_path}")
    available = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            available.append(int(json.loads(line)["episode_index"]))
    selected = requested or available
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"Requested episodes are not present in the dataset: {missing}; available={available}")
    return sorted(set(selected))


def _operators(args: Args, config):
    operators = []
    if args.run_exposure:
        operators.append(CrossCameraExposureOperator(CAMERA_KEYS, config.exposure))
    if args.run_joint_state:
        operators.append(
            JointMotionOperator(
                "observation.state",
                config.joint_motion.state,
                config.joint_motion,
                joint_limit_urdf=args.joint_limit_urdf,
            )
        )
    if args.run_joint_action:
        operators.append(
            JointMotionOperator(
                "action",
                config.joint_motion.action,
                config.joint_motion,
                joint_limit_urdf=args.joint_limit_urdf,
            )
        )
    if not operators:
        raise ValueError("At least one operator must be enabled")
    return operators


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
    operators = episode["operators"]
    exposure = operators.get("cross_camera_exposure")
    if exposure is not None:
        _write_csv(episode_dir / "exposure_frames.csv", exposure["frames"])
    for name, result in operators.items():
        if name.startswith("joint_motion_"):
            _write_csv(episode_dir / f"{name}_events.csv", result["events"])
            distribution_rows = []
            for joint in result["per_joint"]:
                row = {
                    "joint_index": joint["joint_index"],
                    "joint_name": joint["joint_name"],
                    "position_min_deg": joint["position_deg"]["min"],
                    "position_max_deg": joint["position_deg"]["max"],
                    "position_limit_lower_deg": joint["position_deg"]["limit_lower"],
                    "position_limit_upper_deg": joint["position_deg"]["limit_upper"],
                    "configured_max_velocity_deg_s": joint["configured_max_velocity_deg_s"],
                    "configured_max_acceleration_deg_s2": joint["configured_max_acceleration_deg_s2"],
                }
                for metric in ("absolute_step_deg", "absolute_velocity_deg_s", "absolute_acceleration_deg_s2"):
                    for quantile, value in joint[metric].items():
                        row[f"{metric}_{quantile}"] = value
                distribution_rows.append(row)
            _write_csv(episode_dir / f"{name}_distributions.csv", distribution_rows)
    (episode_dir / "operator_report.json").write_text(
        json.dumps(episode, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _status(episodes: list[dict[str, Any]]) -> str:
    statuses = [result["status"] for episode in episodes for result in episode["operators"].values()]
    return "fail" if "fail" in statuses else "warning" if "warning" in statuses else "pass"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# F1 LeRobot 可组合质量算子报告",
        "",
        f"- 数据集：`{report['dataset_root']}`",
        f"- 总体状态：`{report['summary']['status']}`",
        f"- Episode 数：`{report['summary']['episode_count']}`",
        f"- 配置：`{report['config']['source']}`",
        "",
        "## Episode 汇总",
        "",
        "| Episode | 帧数 | 曝光状态/片段 | State 失败/告警 | Action 失败/告警 |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    for episode in report["episodes"]:
        operators = episode["operators"]
        exposure = operators.get("cross_camera_exposure")
        state = operators.get("joint_motion_state")
        action = operators.get("joint_motion_action")
        exposure_text = "未运行" if exposure is None else f"{exposure['status']}/{exposure['confirmed_segment_count']}"
        state_text = (
            "未运行"
            if state is None
            else f"{state['summary']['fail_event_count']}/{state['summary']['warning_event_count']}"
        )
        action_text = (
            "未运行"
            if action is None
            else f"{action['summary']['fail_event_count']}/{action['summary']['warning_event_count']}"
        )
        lines.append(
            f"| {episode['episode_index']} | {episode['frame_count']} | {exposure_text} | {state_text} | {action_text} |"
        )

    for episode in report["episodes"]:
        lines.extend(["", f"## Episode {episode['episode_index']} 明细", ""])
        operators = episode["operators"]
        exposure = operators.get("cross_camera_exposure")
        if exposure is not None:
            lines.extend(
                [
                    "### 曝光异常片段",
                    "",
                    "| 相机 | 起止帧 | 起止时间 (s) | 帧数 | 最大 L50 比值 | 最大截断比例 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            if exposure["confirmed_segments"]:
                for segment in exposure["confirmed_segments"]:
                    lines.append(
                        f"| `{segment['camera']}` | {segment['start_frame']}–{segment['end_frame']} | "
                        f"{segment['start_time_s']:.3f}–{segment['end_time_s']:.3f} | {segment['frame_count']} | "
                        f"{segment['max_l50_ratio']:.3f} | {segment['max_highlight_clipping_ratio']:.3%} |"
                    )
            else:
                lines.append("| 无 | - | - | - | - | - |")

        for name, title in (("joint_motion_state", "State"), ("joint_motion_action", "Action")):
            result = operators.get(name)
            if result is None:
                continue
            counts = result["summary"]["event_counts"]
            lines.extend(
                [
                    "",
                    f"### {title} 关节事件",
                    "",
                    f"- 状态：`{result['status']}`；失败 `{result['summary']['fail_event_count']}`，"
                    f"告警 `{result['summary']['warning_event_count']}`。",
                    f"- 事件计数：位置限位 `{counts['position_limit']}`，步长跳变 `{counts['step_discontinuity']}`，"
                    f"速度 `{counts['velocity_limit']}`，加速度 `{counts['acceleration_limit']}`，"
                    f"单帧跳点 `{counts['single_frame_spike']}`。",
                    f"- 位置限位来源：`{result['limit_sources']['position']}`。",
                    f"- 速度限位来源：`{result['limit_sources']['velocity']}`。",
                ]
            )

    lines.extend(
        [
            "",
            "## 解释与限制",
            "",
            "- 曝光指标先按标准 sRGB 逆传递函数线性化，再计算 Rec.709 亮度；不是直接对 JPEG 灰度求均值。",
            "- 跨相机异常必须同时满足亮度差/比例或高光截断条件，并持续达到最小帧数；单帧候选仍保留在 CSV。",
            "- 三个相机视场不同，白色物体占比也会不同，因此该算子是数据风险筛查，不替代人工查看异常片段。",
            "- 位置限位来自 URDF；当前 F1 URDF 的 velocity 为 0，报告中的速度上限是 TOML 临时值，不是已确认硬件规格。",
            "- 加速度上限同样是可调工程门限。取得控制器真实 max_velocity/max_acceleration 后必须更新配置。",
            "- `single_frame_spike` 要求进入和退出步长都大、前后帧又恢复接近，可区分孤立跳点和持续真实运动。",
            "",
            "## 输出文件",
            "",
            "- `quality_report.json`：完整机器可读汇总。",
            "- `episodes/*/exposure_frames.csv`：逐帧、逐相机亮度与候选/确认标记。",
            "- `episodes/*/joint_motion_*_distributions.csv`：逐关节步长、速度、加速度分位数。",
            "- `episodes/*/joint_motion_*_events.csv`：逐个限位、跳变、超速、超加速度和单帧尖峰事件。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(args: Args) -> None:
    config = load_operator_config(args.config)
    run_dir = _new_run_dir(args)
    indices = _episode_indices(args.dataset_root, args.episodes)
    episodes = []
    try:
        for episode_index in indices:
            print(f"分析 LeRobot episode_{episode_index:06d} ...")
            episode = run_episode_operators(
                args.dataset_root,
                episode_index,
                _operators(args, config),
                max_frames=args.max_frames,
            )
            episode_dir = run_dir / "episodes" / f"episode_{episode_index:06d}"
            episode_dir.mkdir(parents=True)
            _write_episode_outputs(episode_dir, episode)
            episodes.append(episode)
        report = {
            "format": "f1_lerobot_quality_operators_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_root": str(args.dataset_root),
            "run_dir": str(run_dir),
            "config": operator_config_to_dict(config),
            "arguments": {
                "episodes": indices,
                "run_exposure": args.run_exposure,
                "run_joint_state": args.run_joint_state,
                "run_joint_action": args.run_joint_action,
                "joint_limit_urdf": str(args.joint_limit_urdf) if args.joint_limit_urdf else None,
                "max_frames": args.max_frames,
            },
            "summary": {
                "status": _status(episodes),
                "episode_count": len(episodes),
                "frame_count": sum(episode["frame_count"] for episode in episodes),
            },
            "episodes": episodes,
        }
        (run_dir / "quality_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (run_dir / "quality_report.md").write_text(_markdown(report), encoding="utf-8")
    except Exception as error:
        (run_dir / "failed_run.json").write_text(
            json.dumps(
                {"error_type": type(error).__name__, "error": str(error)},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    print(f"总体状态：{report['summary']['status']}")
    print(f"分析帧数：{report['summary']['frame_count']}")
    print(f"报告目录：{run_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
