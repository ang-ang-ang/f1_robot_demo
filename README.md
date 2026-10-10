# F1 机器人数据质量检测、清洗与 LeRobot 转换

本仓库提供一套面向 F1 双臂机器人采集数据的数据处理工具，覆盖：

- 原始 ROS 2 MCAP 数据质量检测；
- 三路相机独立曝光检测与过曝 Mask 校验；
- 时间戳、消息周期、跨流同步和关节运动风险分析；
- 头部相机异常 Header 时间戳清洗；
- 固定频率映射、派生 MCAP、误差统计和 Foxglove 可视化；
- 原始/清洗后 MCAP 转 LeRobot 数据集；
- LeRobot 转换后的曝光、关节限位、速度、加速度和单帧跳点复检。

所有命令都应从项目根目录执行：

```bash
cd ~/f1_robot_demo
```

> 原始 MCAP 不会被原地修改。质检、清洗、映射、转换和转换后复检分别保存报告，便于追踪每个 Episode 的数据来源和处理过程。

## 1. 推荐处理流程

```text
原始 MCAP
  ├─ 原始综合质量检测：时间戳、Gap、数值、动作跳变、主从跟随、图像抽检
  ├─ 原始曝光检测：三路相机分别检测 L50/L95/高光截断比例
  ↓
头相机 Header 时间戳清洗
  ├─ 清洗后综合质量复检
  ├─ 清洗后曝光复检
  ↓
固定 20 Hz 映射与同步误差分析
  ├─ resampling_mapping.json
  ├─ aligned_20hz.mcap
  ├─ 流内 residual / Gap / 复用率 / 丢弃率
  └─ 流间 pairwise skew 与 Foxglove 诊断话题
  ↓
MCAP → LeRobot 转换
  ├─ 先 dry-run
  ├─ 再正式生成 Parquet/MP4/meta
  ↓
LeRobot 转换后复检
  ├─ 独立曝光算子
  ├─ State/Action 关节运动算子
  └─ 人工抽检、任务成功标注和训练数据门禁
```

固定频率派生 MCAP 主要用于审计、同步分析和 Foxglove 回放。正式 LeRobot 转换应读取清洗后的 MCAP，并使用相同的 `--fps`；不要把 `aligned_20hz.mcap` 再交给转换器做第二次重采样。

## 2. 环境准备

项目使用 Python 3.11 和 `uv` 管理核心依赖：

```bash
uv sync
uv run python -c "import cv2, lerobot, numpy, torch, torchcodec; print('environment ok')"
```

LeRobot 视频写出依赖系统 FFmpeg：

```bash
ffmpeg -version
ffprobe -version
```

查看任意入口的完整参数：

```bash
uv run python -m data_process.analyze_f1_data_quality --help
uv run python -m data_process.clean_f1_mcap --help
uv run python -m data_process.convert_f1_mcap_to_lerobot --help
```

开放词汇目标检测入口 `detect_f1_objects.py` 使用 Ultralytics YOLOE。若 `ultralytics` 只安装在 Conda 的 `yoloe` 环境中，应从仓库根目录使用该环境的 Python 运行，不要误以为 `uv run` 会自动读取 Conda 包：

```bash
conda activate yoloe
python -m data_process.detect_f1_objects --help
```

## 3. 输入数据结构

单个 Episode：

```text
episode_000001/
├── metadata.yaml
└── episode_000001_0.mcap
```

批量数据：

```text
data_1009/
├── episode_000000/
│   ├── metadata.yaml
│   └── episode_000000_0.mcap
├── episode_000001/
│   ├── metadata.yaml
│   └── episode_000001_0.mcap
└── ...
```

大多数入口的 `--raw-dir` 同时支持单个 Episode 目录和包含多个 `episode_*` 子目录的数据集根目录。一个 Episode 可以包含多个 MCAP 分片。

当前自定义解析与等长清洗实现面向本项目的未压缩 MCAP Chunk。头相机清洗写出还要求 Chunk/Data CRC 为 0；不满足条件时程序会拒绝写入，而不是静默生成损坏文件。

## 4. 主要入口一览

| 入口 | 输入 | 主要功能 | 主要输出 |
| --- | --- | --- | --- |
| `data_process.analyze_f1_data_quality` | 原始或清洗后 MCAP | 综合时序、数值、图像、动作与跟随质量检测 | `quality_report.json/.md`、`risks.csv`、抽帧图 |
| `data_process.analyze_f1_mcap_exposure` | 原始或清洗后 MCAP | 三路相机独立曝光检测，保留 MCAP 原始定位信息 | 曝光逐帧 CSV、异常段 CSV、可选 Mask 视频 |
| `data_process.clean_f1_mcap` | 原始 MCAP | 修复头相机 Header 时间戳，不重采样 | 清洗 MCAP、manifest、逐帧修正 CSV、清洗报告 |
| `data_process.resample_f1_mcap` | 清洗后 MCAP | 固定频率映射、派生 MCAP、Foxglove 诊断话题 | `resampling_mapping.json`、`aligned_20hz.mcap` |
| `data_process.analyze_f1_resampling` | 映射 JSON 或重采样目录 | 周期 Gap、流内 residual、流间 skew 和静态图 | JSON/Markdown/CSV/PNG |
| `data_process.convert_f1_mcap_to_lerobot` | 原始或清洗后 MCAP，推荐清洗后 | 固定频率采样并生成 LeRobot 数据集 | Parquet、MP4、meta、转换报告 |
| `data_process.analyze_f1_lerobot_operators` | LeRobot 数据集 | 转换后曝光和 State/Action 关节运动复检 | 算子报告、事件/分布 CSV、可选 Mask 视频 |
| `data_process.detect_f1_objects` | MP4 | YOLOE 开放词汇纸箱/机械臂检测 | 检测框视频、JSON 检测结果 |

## 5. 原始 MCAP 综合质量检测

入口：

```text
data_process.analyze_f1_data_quality
```

批量检测：

```bash
uv run python -m data_process.analyze_f1_data_quality \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/raw_quality
```

单条检测并指定裁剪区间：

```bash
uv run python -m data_process.analyze_f1_data_quality \
  --raw-dir test_data/data_1009/episode_000000 \
  --output-dir artifacts/data_1009/raw_quality \
  --start-time-s 2.0 \
  --end-time-s 45.0
```

主要检查项：

- 各 Topic 消息数、持续时间、有效频率和周期分布；
- Header 时间戳重复、倒退、非递增和大间隔；
- 图像、状态、动作在固定频率预览时间轴上的最近邻对齐误差；
- 数值维度变化、关节名称变化、NaN 和 Inf；
- 三路压缩图像抽样解码、分辨率变化和接触表；
- 相邻动作目标跳变；
- 示教目标与从机状态的 P95 跟随误差；
- TCP 平移单位量级和欧拉角 `-180°/+180°` 环绕风险；
- 任务文本、成功标签和失败轨迹标记。

每次运行创建独立报告目录：

```text
artifacts/data_1009/raw_quality/<UTC时间>_data_1009/
├── quality_report.json
├── quality_report.md
├── risks.csv
├── episodes/
│   └── episode_*.json
└── episode_*_observation_images_*.jpg
```

如果当前阶段尚未完成任务文本和成功标注，报告中的 `task_label_missing`、`success_label_unverified` 可以作为待办项保留；最终进入训练集前必须处理。需要将高风险失败用于数据 CI 时增加：

```bash
--fail-on-high-risk
```

该入口的门限通过 CLI 参数覆盖，例如：

```bash
--max-image-gap-ms 70 \
--max-state-gap-ms 30 \
--max-action-gap-ms 120 \
--max-action-step-deg 25
```

详细定义见 [data_process/DATA_QUALITY_GUIDE.md](data_process/DATA_QUALITY_GUIDE.md)。

## 6. 原始 MCAP 三路独立曝光检测

入口：

```text
data_process.analyze_f1_mcap_exposure
```

先对每路相机前 100 帧快速验证：

```bash
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/raw_exposure \
  --max-frames 100
```

正式分析不要设置 `--max-frames`：

```bash
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/raw_exposure
```

只分析指定 Episode，并生成过曝区域 Mask 校验视频：

```bash
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/data_1009 \
  --episodes episode_000000 episode_000001 \
  --output-dir artifacts/data_1009/raw_exposure \
  --write-exposure-mask-videos
```

曝光算子不会比较三路相机谁更亮，而是对每路图像独立执行：

1. JPEG/sRGB 归一化并按标准 sRGB 逆传递函数线性化；
2. 计算 `Y = 0.2126R + 0.7152G + 0.0722B`；
3. 计算亮度中位数 `L50`、亮度 95 分位数 `L95`；
4. 计算 `R_clip = N(Y > 0.98) / N_pixel`；
5. 候选异常连续达到默认 3 帧后，确认为异常片段。

默认候选条件：

```text
大面积过曝：L50 >= 0.40 且 L95 >= 0.90
高光截断：R_clip >= 5%
```

输出目录：

```text
artifacts/data_1009/raw_exposure/<UTC时间>_data_1009/
├── quality_report.json
├── quality_report.md
└── episodes/episode_000000/
    ├── operator_report.json
    ├── exposure_frames.csv
    ├── exposure_segments.csv
    └── exposure_masks/                 # 可选
        ├── observation_images_head_overexposure_mask.mp4
        ├── observation_images_left_wrist_overexposure_mask.mp4
        └── observation_images_right_wrist_overexposure_mask.mp4
```

`exposure_frames.csv` 保留 Topic 帧号、MCAP record index、Header、publish time、log time、头相机修复方法和修正量，可用于在 Foxglove 中回到原始位置。

默认参数位于 [data_process/operator_config.toml](data_process/operator_config.toml)，算法与 Mask 说明见 [data_process/QUALITY_OPERATORS.md](data_process/QUALITY_OPERATORS.md)。

## 7. 头部相机时间戳清洗

入口：

```text
data_process.clean_f1_mcap
```

清洗只修复头部相机 `CompressedImage.header.stamp`，不进行 20 Hz 重采样，不改变消息数量和 MCAP 消息记录顺序，也不会修改左右腕相机、关节、夹爪或图像 JPEG 内容。

算法流程：

1. 保持 MCAP 消息记录顺序；
2. 同时保存 Header、publish time 和 log time；
3. 从正常 Header 段估计名义相机周期；
4. 检测重复、负步长、周期倍数和偏离鲁棒帧序号模型的异常；
5. 短异常段使用相邻可信 Header 做局部线性插值；
6. 长异常段和边界异常段使用鲁棒帧序号回归；
7. 最后只对仍非递增的位置应用单调保护；
8. 为每一帧记录原时间、修复时间、修正量、修复方法和风险等级。

先运行 dry-run：

```bash
uv run python -m data_process.clean_f1_mcap \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/cleaned_mcap \
  --report-dir artifacts/data_1009/cleaning \
  --dry-run
```

重点检查：

- `nominal_period_ms` 是否接近相机真实周期；
- `changed_frame_count`；
- `max_abs_correction_ms`；
- `warning_frame_count` 和 `high_risk_frame_count`；
- `corrected_steps.anomaly_count` 是否降为 0；
- 大修正是否集中在真实时间戳异常段，而不是正常图像段。

确认后正式写出：

```bash
uv run python -m data_process.clean_f1_mcap \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/cleaned_mcap \
  --report-dir artifacts/data_1009/cleaning
```

输出包括：

```text
artifacts/data_1009/cleaning/
├── cleaning_report.json
├── cleaning_report.md
└── episodes/episode_000000/
    ├── cleaning_report.json
    └── timestamp_corrections.csv

artifacts/data_1009/cleaned_mcap/episode_000000/
├── metadata.yaml
├── episode_000000_0.mcap
├── f1_cleaning_manifest.json
└── f1_timestamp_corrections.csv
```

不要只复制清洗后的 `.mcap` 而丢失 `f1_cleaning_manifest.json` 和 `f1_timestamp_corrections.csv`，后续映射需要它们恢复原始/清洗后时间 lineage。

默认清洗参数位于 [data_process/default_config.toml](data_process/default_config.toml) 的 `[cleaning.head_timestamp]`。

### 7.1 清洗后复检

```bash
uv run python -m data_process.analyze_f1_data_quality \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --output-dir artifacts/data_1009/cleaned_quality

uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --output-dir artifacts/data_1009/cleaned_exposure
```

清洗后应重点确认：

- 头相机 Header 已严格递增；
- 头相机异常周期数量显著下降；
- 左右腕相机和所有非目标 Topic 没有发生变化；
- 关节跳变、曝光异常、图像损坏等非时间戳问题仍被保留，而不是被“清洗”掩盖。

## 8. 固定频率映射、派生 MCAP 与同步分析

### 8.1 生成映射和派生 MCAP

入口：

```text
data_process.resample_f1_mcap
```

默认使用 20 Hz 最近邻映射：

```bash
uv run python -m data_process.resample_f1_mcap \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --output-dir artifacts/data_1009/resampled_20hz \
  --fps 20 \
  --mcap-mode aligned
```

如只想先检查映射 JSON、不写大体积 MCAP，可使用单独的预览目录：

```bash
uv run python -m data_process.resample_f1_mcap \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --output-dir artifacts/data_1009/resampling_preview \
  --fps 20 \
  --mapping-only
```

每个 Episode 输出：

```text
artifacts/data_1009/resampled_20hz/episode_000000/
├── aligned_20hz.mcap
└── resampling_mapping.json
```

`resampling_mapping.json` 是时间映射的审计真值，包含：

- 每条源消息的 record index、sequence、原始/清洗 Header、publish/log time；
- 清洗修正量、修复方法和风险等级；
- 每个目标时刻选中的 `source_index`；
- `cleaned_source_time - target_time` 有符号 residual；
- 原始 Header 相对目标时间的 residual；
- 源消息复用次数和未被选择的源消息数量。

`--mcap-mode aligned` 的语义：

- MCAP envelope 的 `log_time/publish_time` 使用固定目标时间，便于同步播放；
- CDR payload 内 ROS `header.stamp` 保留清洗后的真实源时间；
- 每个目标时刻、每个选定流写一条消息；
- 稀疏流可能重复使用同一个 payload，复用情况记录在映射和诊断话题中。

`--mcap-mode unique` 只保留真正被选择的唯一源消息，并保留原 envelope 时间，因此不保证每个 Topic 都严格达到目标频率。

当前数据建议先保持 20 Hz。不要因为约 30 Hz 相机映射到 20 Hz 后出现规则锯齿 residual 就直接提高 FPS；规则 `±半个源周期` 折返是异步最近邻采样的正常相位现象。是否改成 25 Hz 应根据真机快速接触任务的成功率、动作执行周期和 action horizon 做对照实验。

### 8.2 分析映射误差

入口：

```text
data_process.analyze_f1_resampling
```

```bash
uv run python -m data_process.analyze_f1_resampling \
  --input-path artifacts/data_1009/resampled_20hz \
  --output-dir artifacts/data_1009/resampling_analysis
```

输出：

```text
artifacts/data_1009/resampling_analysis/
├── resampling_statistics.json
├── resampling_statistics.md
├── stream_statistics.csv
├── pairwise_statistics.csv
├── gap_events.csv
└── episodes/episode_*/
    ├── residual_timeline.png
    ├── residual_histogram.png
    ├── gap_timeline.png
    └── camera_pairwise_skew.png
```

核心定义：

```text
residual(A) = selected_cleaned_time(A) - target_time
pairwise_skew(A, B) = selected_cleaned_time(A) - selected_cleaned_time(B)
```

正 residual 表示离线最近邻选择了目标时刻之后的源样本，负 residual 表示选择了历史样本。它不是相机曝光延迟或网络延迟。

### 8.3 Foxglove 可视化

打开 `aligned_20hz.mcap` 后可添加三个 Image 面板，并在 Plot 中查看：

```text
/f1/resampling/camera_head_color_image_raw_compressed.residual_ms
/f1/resampling/camera_head_color_image_raw_compressed.abs_residual_ms
/f1/resampling/camera_head_color_image_raw_compressed.cleaning_correction_ms
/f1/resampling/cross_stream.camera_head_color_image_raw_compressed__camera_left_wrist_color_image_raw_compressed_ms
```

State Transitions 面板可查看：

```text
/f1/resampling/camera_head_color_image_raw_compressed.reused_previous_source
/f1/resampling/camera_head_color_image_raw_compressed.repair_method
/f1/resampling/camera_head_color_image_raw_compressed.risk
```

Plot 时间基准选择 `Log time`，因为派生 MCAP 的 envelope log time 就是固定目标时间。需要核对真实源时刻时读取诊断字段或图像 payload 的 Header。

## 9. MCAP 转 LeRobot

入口：

```text
data_process.convert_f1_mcap_to_lerobot
```

推荐输入清洗后的 MCAP：

```text
artifacts/data_1009/cleaned_mcap
```

转换器会在共同有效区间上生成固定频率时间轴，分别对三路图像、状态和动作进行最近邻匹配，并输出：

| LeRobot 字段 | MCAP 来源 |
| --- | --- |
| `observation.images.head` | `/camera/head/color/image_raw/compressed` |
| `observation.images.left_wrist` | `/camera/left_wrist/color/image_raw/compressed` |
| `observation.images.right_wrist` | `/camera/right_wrist/color/image_raw/compressed` |
| `observation.state[:14]` | `/hal/joint_states`，度转弧度 |
| `observation.state[14:]` | 左右夹爪状态，默认 `0..100` 转 `0..1` |
| `action[:14]` | `/lead/joint_states`，绝对关节目标，度转弧度 |
| `action[14:]` | 左右夹爪命令，默认 `0..100` 转 `0..1` |
| `task` | `--task` 或 Episode 配置文件 |

### 9.1 Episode 配置文件

批量转换建议创建自己的配置文件，例如 `configs/data_1009_episodes.json`：

```json
{
  "episode_000000": {
    "include": true,
    "task": "打开纸箱的箱盖，使箱口完全打开。",
    "start_time_s": 2.0,
    "end_time_s": 45.0,
    "success": true,
    "notes": "已人工复核。"
  },
  "episode_000001": {
    "include": false,
    "task": "打开纸箱的箱盖，使箱口完全打开。",
    "success": false,
    "failure_reason": "右夹爪未抓住箱盖。"
  }
}
```

`success=false` 不会自动改写 `include`，不用于失败轨迹训练时必须显式设置 `include=false`。

### 9.2 先 dry-run

```bash
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --repo-id local/data_1009 \
  --task "打开纸箱的箱盖，使箱口完全打开。" \
  --output-root artifacts/data_1009/lerobot \
  --report-dir artifacts/data_1009/conversion_reports \
  --fps 20 \
  --dry-run
```

使用逐 Episode 配置时，将统一的 `--task` 替换或补充为：

```bash
--episode-config configs/data_1009_episodes.json
```

Dry-run 会完成 MCAP/CDR 解析、必需 Topic、维度、关节名、图像解码、公共区间、固定频率采样、对齐误差、动作跳变和主从跟随检查，但不会写 LeRobot Parquet/视频。

### 9.3 正式转换

确认 dry-run 报告后移除 `--dry-run`：

```bash
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --repo-id local/data_1009 \
  --task "打开纸箱的箱盖，使箱口完全打开。" \
  --output-root artifacts/data_1009/lerobot \
  --report-dir artifacts/data_1009/conversion_reports \
  --fps 20
```

转换数据集典型结构：

```text
artifacts/data_1009/lerobot/
├── data/
├── videos/
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── f1_conversion.json
    └── f1_conversion.md
```

每次运行无论成功、dry-run 还是失败，都会在 `--report-dir` 下创建独立的 `conversion_report.json/.md`。不要只看命令退出状态，应检查每个 Episode 的采样 residual、动作跳变、图像尺寸和警告。

默认使用 H.264 视频：

```text
--use-videos --video-codec h264
```

调试可用 `--no-use-videos`，但完整数据集会显著占用更多空间。

## 10. LeRobot 转换后质量算子

入口：

```text
data_process.analyze_f1_lerobot_operators
```

先快速验证：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/data_1009/lerobot \
  --output-dir artifacts/data_1009/operator_quality \
  --max-frames 100
```

正式运行全部 Episode：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/data_1009/lerobot \
  --output-dir artifacts/data_1009/operator_quality
```

生成曝光 Mask 视频：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/data_1009/lerobot \
  --output-dir artifacts/data_1009/operator_quality \
  --episodes 0 1 \
  --write-exposure-mask-videos
```

只运行曝光算子：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/data_1009/lerobot \
  --output-dir artifacts/data_1009/operator_quality \
  --no-run-joint-state \
  --no-run-joint-action
```

关节运动算子分别检查 `observation.state` 和 `action`：

- URDF 位置上下限；
- 单帧关节步长；
- 按真实时间间隔计算的速度；
- 按相邻速度区间中心计算的加速度；
- “进入大跳、下一帧恢复”的单帧跳点模式；
- 每个关节的 P50/P95/P99/P99.9 和最大值分布。

默认 URDF：

```text
f1p01_00000000_20260920/urdf/f1p01_00000000_20260918.urdf
```

当前 URDF 中部分 velocity 为 0，程序会使用 `operator_config.toml` 的临时速度上限。拿到机器人真实逐关节限位、最大速度和最大加速度后，应通过正确 URDF 或 TOML 覆盖替换这些工程门限。

输出包括：

```text
artifacts/data_1009/operator_quality/<UTC时间>_lerobot/
├── quality_report.json
├── quality_report.md
└── episodes/episode_000000/
    ├── operator_report.json
    ├── exposure_frames.csv
    ├── exposure_segments.csv
    ├── joint_motion_state_events.csv
    ├── joint_motion_state_distributions.csv
    ├── joint_motion_action_events.csv
    └── joint_motion_action_distributions.csv
```

## 11. 可选：纸箱和机器人手臂目标检测

该入口不是转换门禁的一部分，主要用于生成可视化检测结果和后续视觉算子原型：

```bash
conda activate yoloe
python -m data_process.detect_f1_objects \
  --input-video artifacts/data_1009/lerobot/videos/chunk-000/observation.images.head/episode_000000.mp4 \
  --output-dir artifacts/data_1009/object_detection \
  --class-prompts "cardboard box" "robot arm" \
  --device cuda:0
```

输出为检测框视频和逐帧 JSON。检测器、输入源和输出 Sink 已拆分在 `data_process/detection.py`，后续可替换为原始 MCAP 图像 Source 或把检测结果接入其他算子。

## 12. 配置和门限调整

### 12.1 时间戳、清洗和转换默认值

[data_process/default_config.toml](data_process/default_config.toml) 包含：

- `[quality.timestamp]`：重复容差、周期容差、Gap 告警/失败倍数；
- `[quality.cross_stream]`：目标 FPS 和图像/状态/动作 residual 门限；
- `[quality.joint_motion]`：综合质检的步长、速度、加速度和跟随误差门限；
- `[quality.image]`：图像解码和哈希抽样数量；
- `[cleaning.head_timestamp]`：周期搜索、锚点、短段插值和风险阈值；
- `[conversion]`：转换 FPS、重采样误差和夹爪归一化参数。

清洗入口支持只写覆盖项的 TOML：

```toml
[cleaning.head_timestamp]
short_run_max_frames = 4
warning_correction_periods = 0.75
high_risk_correction_periods = 1.5
```

```bash
uv run python -m data_process.clean_f1_mcap \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/cleaned_mcap \
  --report-dir artifacts/data_1009/cleaning \
  --config configs/f1_cleaning.local.toml \
  --dry-run
```

### 12.2 曝光和 LeRobot 关节算子默认值

[data_process/operator_config.toml](data_process/operator_config.toml) 包含曝光和关节算子门限。建议创建局部覆盖文件，不直接修改默认文件：

```toml
[exposure]
overexposed_l50_threshold = 0.45
overexposed_l95_threshold = 0.92
clip_ratio_threshold = 0.08
minimum_consecutive_frames = 5

[joint_motion.state]
fallback_max_velocity_deg_s = 240.0

[joint_motion.action]
fallback_max_velocity_deg_s = 360.0
```

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/data_1009/lerobot \
  --output-dir artifacts/data_1009/operator_quality \
  --config configs/f1_operator.local.toml
```

门限应通过机器人硬件规格、健康数据分布和人工确认异常共同标定，不能只为了让报告变绿而放宽。

## 13. 一套可直接执行的 data_1009 流程

```bash
# 1. 原始综合质量检测
uv run python -m data_process.analyze_f1_data_quality \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/raw_quality

# 2. 原始三路曝光检测
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/raw_exposure

# 3. 清洗 dry-run
uv run python -m data_process.clean_f1_mcap \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/cleaned_mcap \
  --report-dir artifacts/data_1009/cleaning \
  --dry-run

# 4. 正式写出清洗 MCAP
uv run python -m data_process.clean_f1_mcap \
  --raw-dir test_data/data_1009 \
  --output-dir artifacts/data_1009/cleaned_mcap \
  --report-dir artifacts/data_1009/cleaning

# 5. 清洗后综合复检
uv run python -m data_process.analyze_f1_data_quality \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --output-dir artifacts/data_1009/cleaned_quality

# 6. 生成 20 Hz 映射与 Foxglove 派生 MCAP
uv run python -m data_process.resample_f1_mcap \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --output-dir artifacts/data_1009/resampled_20hz \
  --fps 20

# 7. 分析重采样和多流同步误差
uv run python -m data_process.analyze_f1_resampling \
  --input-path artifacts/data_1009/resampled_20hz \
  --output-dir artifacts/data_1009/resampling_analysis

# 8. LeRobot 转换 dry-run
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --repo-id local/data_1009 \
  --task "打开纸箱的箱盖，使箱口完全打开。" \
  --output-root artifacts/data_1009/lerobot \
  --report-dir artifacts/data_1009/conversion_reports \
  --fps 20 \
  --dry-run

# 9. 正式转换：确认 dry-run 后执行
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir artifacts/data_1009/cleaned_mcap \
  --repo-id local/data_1009 \
  --task "打开纸箱的箱盖，使箱口完全打开。" \
  --output-root artifacts/data_1009/lerobot \
  --report-dir artifacts/data_1009/conversion_reports \
  --fps 20

# 10. LeRobot 转换后复检
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/data_1009/lerobot \
  --output-dir artifacts/data_1009/operator_quality
```

若输出目录已经存在，清洗、重采样或转换入口会默认拒绝覆盖。确认旧结果可以删除或替换后，再显式增加 `--overwrite`。报告目录通常按 UTC 时间创建子目录，可以保留多次运行结果。

## 14. 数据进入训练前的建议门禁

### 必须失败并隔离

- 必需 Topic 缺失；
- MCAP/CDR 无法解析；
- 图像解码失败或同一字段分辨率变化；
- NaN、Inf 或数值维度变化；
- 清洗后仍存在头相机非递增时间戳；
- 图像、状态或动作对齐误差超过确认后的硬门限；
- 任务失败且未显式排除；
- LeRobot 输出结构或视频帧数不一致。

### 高风险人工复核

- 时间戳修正超过 1～2 个相机周期；
- 关节位置、速度、加速度或单帧跳点异常；
- 动作和状态主从跟随误差过大；
- 多流时间差在关键接触动作附近达到一帧以上；
- 连续曝光异常、遮挡、模糊或相机画面冻结；
- 示教接管、离合重定位、急停和失败恢复；
- 重采样连续复用同一个源消息。

自动报告不能判断任务是否真正完成、是否发生轻微碰撞、物体是否抓偏或放错位置。最终仍需结合视频、Foxglove 和任务成功定义进行人工抽检。

## 15. 详细文档

- [data_process/README.md](data_process/README.md)：MCAP 字段映射、转换算法、报告和 OpenPI 接入细节；
- [data_process/DATA_QUALITY_GUIDE.md](data_process/DATA_QUALITY_GUIDE.md)：综合质量指标、门限和风险解释；
- [data_process/QUALITY_OPERATORS.md](data_process/QUALITY_OPERATORS.md)：曝光、Mask、关节运动算子和扩展接口；
- [data_process/data_wash.md](data_process/data_wash.md)：数据清洗与质量体系的设计记录；
- [data_process/default_config.toml](data_process/default_config.toml)：时间戳、清洗和转换默认门限；
- [data_process/operator_config.toml](data_process/operator_config.toml)：曝光和关节算子默认门限。

## 16. 开发验证

```bash
uv run ruff check data_process
uv run ruff format --check data_process
uv run python -m pytest -q data_process/tests
```

新增质量算子时，应保持“Source 负责解析和来源追踪、Operator 负责判定、Runner 负责组合和输出”的结构，避免把业务阈值写入 MCAP/LeRobot 数据读取层。
