# ruff: noqa: RUF001
"""Repair F1 head-camera Header timestamps while preserving MCAP record order."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

from data_process.cleaning import clean_dataset
from data_process.config import load_config


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """单个 episode 目录，或包含多个 episode_* 的目录。"""

    output_dir: Path
    """清洗后 MCAP 的数据集根目录；原始数据不会被修改。"""

    report_dir: Path = Path("artifacts/f1_cleaning")
    """清洗报告和逐帧 timestamp_corrections.csv 输出目录。"""

    config: Path | None = None
    """可选 TOML 覆盖文件。"""

    overwrite: bool = False
    dry_run: bool = False
    """只计算修复方案和报告，不复制或修改 MCAP。"""


def main(args: Args) -> None:
    report = clean_dataset(
        args.raw_dir,
        args.output_dir,
        args.report_dir,
        load_config(args.config),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"处理 Episode：{len(report['episodes'])}")
    print(f"清洗报告：{args.report_dir / 'cleaning_report.md'}")
    if not args.dry_run:
        print(f"清洗数据：{args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
