# ruff: noqa: RUF001
"""Analyze F1 fixed-rate mapping lineage and synchronization errors."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

from data_process.resampling_analysis import analyze_mapping_files


@dataclasses.dataclass(frozen=True)
class Args:
    input_path: Path
    """单个 resampling_mapping.json、Episode 输出目录或批量输出根目录。"""

    output_dir: Path
    duplicate_epsilon_ms: float = 0.05
    gap_warning_factor: float = 2.2
    gap_fail_factor: float = 5.0
    residual_thresholds_ms: list[float] | None = None
    write_plots: bool = True


def main(args: Args) -> None:
    report = analyze_mapping_files(
        args.input_path,
        args.output_dir,
        duplicate_epsilon_ms=args.duplicate_epsilon_ms,
        gap_warning_factor=args.gap_warning_factor,
        gap_fail_factor=args.gap_fail_factor,
        residual_thresholds_ms=args.residual_thresholds_ms,
        write_plots=args.write_plots,
    )
    print(f"分析 Episode：{len(report['episodes'])}")
    print(f"统计报告：{args.output_dir / 'resampling_statistics.md'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
