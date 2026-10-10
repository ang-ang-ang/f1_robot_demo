# ruff: noqa: RUF001
"""Create an audited fixed-rate MCAP view from cleaned F1 episodes."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import tyro

from data_process.resampling import resample_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """清洗后的单个 Episode，或包含多个 Episode 的目录。"""

    output_dir: Path
    """20 Hz MCAP、映射 JSON 和运行 manifest 的输出目录。"""

    fps: float = 20.0
    topics: list[str] | None = None
    """默认处理 LeRobot 训练必需话题；可显式传入话题列表。"""

    mcap_mode: Literal["aligned", "unique"] = "aligned"
    """aligned 输出严格目标频率回放视图；unique 仅保留唯一选中的原消息。"""

    include_diagnostics: bool = True
    """在 aligned MCAP 中加入可供 Foxglove 绘图的 JSON 诊断话题。"""

    require_cleaning_manifest: bool = True
    """默认只接受带 f1_cleaning_manifest.json 的清洗数据。"""

    overwrite: bool = False
    mapping_only: bool = False
    """只生成映射 JSON，不写 MCAP。"""


def main(args: Args) -> None:
    manifest = resample_dataset(
        args.raw_dir,
        args.output_dir,
        fps=args.fps,
        topics=args.topics,
        mcap_mode=args.mcap_mode,
        include_diagnostics=args.include_diagnostics,
        require_cleaning_manifest=args.require_cleaning_manifest,
        overwrite=args.overwrite,
        mapping_only=args.mapping_only,
    )
    print(f"处理 Episode：{len(manifest['episodes'])}")
    print(f"输出目录：{args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
