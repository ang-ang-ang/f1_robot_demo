# ruff: noqa: RUF001
"""Run an open-vocabulary detector on the F1 head-camera video."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

from data_process.detection import (
    AnnotatedVideoSink,
    JsonReportSink,
    UltralyticsYOLOEDetector,
    VideoFrameSource,
    default_output_paths,
    run_detection_pipeline,
)


@dataclasses.dataclass(frozen=True)
class Args:
    input_video: Path = Path("1000000.mp4")
    """默认直接读取当前目录下的 1000000.mp4。"""

    output_dir: Path = Path("artifacts/f1_detection")
    """输出检测框视频和 JSON 检测报告。"""

    output_video: Path | None = None
    """可选，显式指定标注后视频路径。"""

    report_path: Path | None = None
    """可选，显式指定机器可读检测报告路径。"""

    model_name: str = "yoloe-11s.pt"
    """Ultralytics YOLOE 权重名；默认自动下载小模型。"""

    class_prompts: list[str] = dataclasses.field(default_factory=lambda: ["cardboard box", "robot arm"])
    """开放词汇类别提示词，默认检测纸箱和机器人手臂。"""

    confidence: float = 0.2
    iou: float = 0.5
    device: str | None = None
    """可选，如 cuda:0、cpu。默认由 Ultralytics 自动选择。"""

    max_frames: int | None = None
    """可选，限制处理帧数，便于快速试跑。"""


def main(args: Args) -> None:
    if not args.input_video.exists():
        raise FileNotFoundError(f"未找到输入视频：{args.input_video}")

    output_video, report_path = default_output_paths(args.input_video, args.output_dir)
    if args.output_video is not None:
        output_video = args.output_video
    if args.report_path is not None:
        report_path = args.report_path

    source = VideoFrameSource(args.input_video)
    detector = UltralyticsYOLOEDetector(
        model_name=args.model_name,
        class_prompts=args.class_prompts,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
    )
    sinks = [
        AnnotatedVideoSink(output_video),
        JsonReportSink(report_path, source_labels=args.class_prompts),
    ]
    summary = run_detection_pipeline(source, detector, sinks, max_frames=args.max_frames)

    print(f"输入视频：{args.input_video}")
    print(f"检测类别：{', '.join(args.class_prompts)}")
    print(f"处理帧数：{summary['processed_frames']}")
    print(f"标注视频：{output_video}")
    print(f"检测报告：{report_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
