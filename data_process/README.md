# F1 机器人 MCAP → LeRobot 数据工具

本目录提供一套面向 F1 双臂机器人的数据处理工具，用于把 ROS 2 MCAP 采集数据转换成 OpenPI/Pi0.5
可读取的 LeRobot v2 数据集，并在转换前后保存可追溯的数据质量报告。

## 1. 文件说明

| 文件 | 用途 |
| --- | --- |
| `convert_f1_mcap_to_lerobot.py` | 单条/批量 MCAP 转 LeRobot，自动保存每次转换报告 |
| `analyze_f1_data_quality.py` | 独立数据质量评估，不生成 LeRobot 数据 |
| `clean_f1_mcap.py` | 单条/批量 MCAP 头相机时间戳清洗入口 |
| `cleaning.py` | 局部插值、鲁棒帧序号回归、逐帧风险记录和 MCAP 等长补丁 |
| `mcap.py` | 按 MCAP 记录顺序读取 Header、publish time、log time 和 CDR 字节偏移 |
| `default_config.toml` | 质检、时间戳清洗和转换默认门限 |
| `DATA_QUALITY_GUIDE.md` | 质量指标、风险门限和报告解读说明 |

推荐流程不是“直接转换”，而是：

1. 查看原始 Episode 和任务结果；
2. 编写每条 Episode 的任务、裁剪和成功标注；
3. 运行独立质量评估；
4. 对时间戳清洗先运行 `--dry-run`，检查逐帧修正量和高风险帧；
5. 写出清洗后的新 MCAP，原始 MCAP 不做原地修改；
6. 对清洗数据重新运行质量评估；
7. 运行转换器 `--dry-run`；
8. 正式生成 LeRobot 数据；
9. 读取 LeRobot 数据集做最终抽样检查；
10. 计算 OpenPI normalization stats 后再训练。

### 1.1 环境准备

本仓库使用 Python 3.11 和 `uv` 管理依赖。首次运行或 `pyproject.toml` / `uv.lock` 更新后，在仓库根目录执行：

```bash
uv sync
uv run python -c "import lerobot, torch, torchcodec; print(lerobot.__version__, torch.__version__)"
```

正式转换依赖 LeRobot、PyTorch、TorchCodec 和系统 FFmpeg。相关版本已经在 `pyproject.toml` 与 `uv.lock`
中锁定；不要只在系统 Python 或 Conda `base` 环境中执行 `pip install lerobot`，否则 `uv run` 使用的项目
虚拟环境仍可能找不到该模块。所有质检、清洗和转换命令都应从仓库根目录通过 `uv run` 执行。

可用以下命令确认视频编码工具存在：

```bash
ffmpeg -version
ffprobe -version
```

## 2. 原始目录结构

转换器同时支持单条和批量输入。

### 2.1 单条 Episode

```text
episode_000001/
├── metadata.yaml
└── episode_000001_0.mcap
```

此时 `--raw-dir` 指向 `episode_000001`。

### 2.2 批量 Episode

```text
F1_data/
├── episode_000001/
│   ├── metadata.yaml
│   └── episode_000001_0.mcap
├── episode_000002/
│   ├── metadata.yaml
│   └── episode_000002_0.mcap
└── episode_000003/
    ├── metadata.yaml
    └── episode_000003_0.mcap
```

此时 `--raw-dir` 指向 `F1_data`，工具会按目录名排序处理所有 `episode_*`。

一个 Episode 内允许有多个 MCAP 分片，工具会读取并按时间戳重新排序；当前解析器要求 MCAP chunk 本身未压缩，
但 `CompressedImage` 内的 JPEG 图像不受影响。

## 3. 字段映射

| LeRobot 字段 | 原始 Topic | 处理方式 |
| --- | --- | --- |
| `observation.images.head` | `/camera/head/color/image_raw/compressed` | JPEG 解码，BGR → RGB |
| `observation.images.left_wrist` | `/camera/left_wrist/color/image_raw/compressed` | JPEG 解码，BGR → RGB |
| `observation.images.right_wrist` | `/camera/right_wrist/color/image_raw/compressed` | JPEG 解码，BGR → RGB |
| `observation.state[:14]` | `/hal/joint_states` | 按名称选择左右臂 14 关节，度 → 弧度 |
| `observation.state[14:]` | 左右夹爪 `state` | 默认 `0..100` → `0..1` |
| `action[:14]` | `/lead/joint_states` | 绝对关节目标，按名称排序，度 → 弧度 |
| `action[14:]` | 左右夹爪命令 | 默认 `0..100` → `0..1` |
| `task` | CLI 或 Episode 配置文件 | 每帧保存任务语言文本 |

状态和动作顺序固定为：

```text
left_joint_1 ... left_joint_7,
right_joint_1 ... right_joint_7,
left_gripper,
right_gripper
```

共 16 维。TCP Topic 只用于质量诊断，不默认写入模型输入。当前数据中的从机 TCP 和示教 TCP 很可能分别使用
毫米和米，欧拉角还存在 `-180°/+180°` 环绕；在统一单位、坐标系和旋转表示前，不能直接把它们混入动作。

## 4. 准备 Episode 配置文件

批量转换强烈建议创建 JSON 文件，例如 `examples/f1/episodes.local.json`：

```json
{
  "episode_000001": {
    "include": true,
    "task": "Open the cardboard box with both grippers.",
    "start_time_s": 3.0,
    "end_time_s": 83.5,
    "success": true,
    "notes": "人工复核：四个箱盖均已打开，双臂正常退出。"
  },
  "episode_000002": {
    "include": true,
    "task": "Load the part into the fixture.",
    "start_time_s": 2.2,
    "end_time_s": 51.8,
    "success": true
  },
  "episode_000003": {
    "include": false,
    "task": "Pick up the part from the tray.",
    "success": false,
    "failure_reason": "The right gripper dropped the part."
  }
}
```

支持字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `include` | bool | 是否转换/评估该 Episode |
| `task` | string | Pi0.5 使用的任务文本 |
| `start_time_s` | float | 相对原始共同起点的裁剪开始时间 |
| `end_time_s` | float/null | 裁剪结束时间；不填时使用公共有效结束时间 |
| `success` | bool | 人工或自动成功判定 |
| `failure_reason` | string | 失败原因 |
| `notes` | string | 复核说明、场景信息等 |

`success=false` 不会自动覆盖 `include`。如果失败轨迹不用于专门的负样本方法，应明确设置 `include=false`。

### 4.1 头相机时间戳清洗

清洗器不会直接转换为 LeRobot，也不会修改原始 MCAP。它按 MCAP 消息记录顺序保留 Header、publish time、
log time，短异常段使用局部插值，长异常段使用鲁棒帧序号回归，并为每一帧记录修正量、修复方法和风险等级。

先执行 dry-run：

```bash
uv run python -m data_process.clean_f1_mcap \
  --raw-dir test_data/F1_data_SOP_openbox/ \
  --output-dir artifacts/F1_data_SOP_openbox/cleaned_mcap \
  --report-dir artifacts/F1_data_SOP_openbox/cleaning \
  --dry-run
```

检查以下输出：

```text
artifacts/F1_data_SOP_openbox/cleaning/
├── cleaning_report.json
├── cleaning_report.md
└── episodes/
    ├── episode_000004/
    │   ├── cleaning_report.json
    │   └── timestamp_corrections.csv
    └── episode_000006/
        ├── cleaning_report.json
        └── timestamp_corrections.csv
```

重点检查 `max_abs_correction_ms`、`warning_frame_count`、`high_risk_frame_count` 和
`corrected_steps.anomaly_count`。高风险帧不会被静默删除，必须结合视频和动作数据人工复核。

确认方案后去掉 `--dry-run`，清洗结果会写入 `--output-dir`。当前等长补丁模式只支持未压缩 Chunk 且
Chunk/Data CRC 为 0 的 F1 MCAP；不满足条件时程序会拒绝写入，避免生成损坏文件。

## 5. 转换前先做独立质量评估

单条 Episode：

```bash
uv run python -m data_process.analyze_f1_data_quality \
  --raw-dir test_data/F1_data_test/episode_000001 \
  --task "Open the cardboard box with both grippers." \
  --start-time-s 3.0 \
  --end-time-s 83.5
```

批量评估：

```bash
uv run python -m data_process.analyze_f1_data_quality \
  --raw-dir test_data/F1_data_test \
  --episode-config examples/f1/episodes.local.json
```

默认写入：

```text
artifacts/f1_quality_reports/<UTC时间>_<输入目录名>/
├── quality_report.json
├── quality_report.md
├── risks.csv
├── episodes/
│   ├── episode_000001.json
│   └── episode_000002.json
├── episode_000001_observation_images_head.jpg
├── episode_000001_observation_images_left_wrist.jpg
└── episode_000001_observation_images_right_wrist.jpg
```

完整指标和默认门限见 `DATA_QUALITY_GUIDE.md`。

建议对同一批数据运行两次：

1. 不裁剪，发现示教接管、启动动作跳变和原始空闲段；
2. 使用最终裁剪配置，确认真正进入训练的数据通过质量门限。

## 6. 单条转换

### 6.1 Dry-run

```bash
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir test_data/F1_data_test/episode_000001 \
  --repo-id your_name/f1_open_box \
  --task "Open the cardboard box with both grippers." \
  --start-time-s 3.0 \
  --end-time-s 83.5 \
  --dry-run
```

Dry-run 会完成：

- MCAP/schema/CDR 解析；
- 必需 Topic 检查；
- 关节名称与维度检查；
- NaN/Inf 和夹爪范围检查；
- 公共时间区间计算；
- 相机异常时间戳修复；
- 固定 20 Hz 时间轴预采样；
- 图像/状态/动作对齐误差检查；
- 动作跳变与主从跟随误差统计；
- 首帧图像解码和分辨率检查；
- JSON 与中文 Markdown 报告保存。

### 6.2 正式转换

确认 dry-run 和独立质检结果后，移除 `--dry-run`：

```bash
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir test_data/F1_data_test/episode_000001 \
  --repo-id your_name/f1_open_box \
  --task "Open the cardboard box with both grippers." \
  --start-time-s 3.0 \
  --end-time-s 83.5
```

默认 LeRobot 输出路径：

```text
$HF_LEROBOT_HOME/your_name/f1_open_box
```

若使用 `--output-root`，它表示数据集的精确目录。之后训练时必须让 OpenPI 能从同一个路径找到该 `repo_id`；
通常最简单的方式是不传 `--output-root`，而是正确设置 `HF_LEROBOT_HOME`。

## 7. 批量转换

先 dry-run：

```bash
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir test_data/F1_data_test \
  --repo-id your_name/f1_multitask \
  --episode-config examples/f1/episodes.local.json \
  --dry-run
```

正式转换：

```bash
uv run python -m data_process.convert_f1_mcap_to_lerobot \
  --raw-dir test_data/F1_data_test \
  --repo-id your_name/f1_multitask \
  --episode-config examples/f1/episodes.local.json
```

批量数据中每条 Episode 都会独立执行裁剪和质量计算，再写入同一个 LeRobot 数据集。不同任务可以使用不同 `task`，
但应避免某个超长任务以大量帧压倒其他任务；必要时按任务拆分数据集或在训练采样器中做均衡。

## 8. 每次转换自动保存的报告

无论是 dry-run、成功转换还是失败转换，工具都会创建唯一报告目录：

```text
artifacts/f1_conversion_reports/<UTC时间>_<repo_id>/
├── conversion_report.json
└── conversion_report.md
```

可通过以下参数修改报告根目录：

```bash
--report-dir /path/to/conversion_reports
```

正式转换成功后，同一份最终报告还会复制到：

```text
<LeRobot数据集>/meta/f1_conversion.json
<LeRobot数据集>/meta/f1_conversion.md
```

转换报告包含：

- `single` 或 `batch` 模式；
- 所有运行参数和质量门限；
- 输入文件和 Episode 数量；
- 每个 Topic 的消息数、频率、中位周期、P95、最大间隔、非递增次数；
- 头相机时间戳修复次数和最大修正量；
- 选择的裁剪范围；
- 输出帧数和时长；
- 三路图像、状态、动作的重采样残差；
- 动作范围、动作跳变；
- 示教目标和从机状态的跟随误差；
- 状态/动作字段顺序和单位；
- `success`、失败原因和人工备注；
- 失败转换的异常类型和错误信息。

## 9. 转换算法

### 9.1 MCAP 解析

工具直接读取 MCAP chunk、内嵌 ROS 2 schema 和 CDR 数据，不依赖 ROS 2、`rosbag2` 或 Python MCAP 包。

### 9.2 时间源选择

- 图像和 `JointState` 使用消息 Header 时间戳；
- `GripperCommand` 没有 Header，使用 MCAP log time；
- 所有数据先按时间戳稳定排序；
- 不允许按消息数组序号直接拼接不同 Topic。

### 9.3 头相机时间戳修复

若图像 Header 存在重复或倒退，工具使用 Episode 首尾时间和帧序号构建单调均匀时间轴，并在报告中记录：

- 原始非递增次数；
- 修复方法；
- 最大时间修正量。

这只是兼容已有数据。后续采集应从相机驱动修复真实曝光时间戳，不能长期依赖离线重建。

### 9.4 固定频率重采样

默认输出 20 Hz。原因是当前数据约为：

- 相机：30 Hz；
- 从机状态：50 Hz；
- 示教端/动作：26.6 Hz，且周期抖动明显。

工具在公共有效区间上生成固定时间轴，对图像、状态、动作分别做最近时间戳匹配，并检查最大残差。

### 9.5 单位和范围

- 14 个关节角默认从度转换为弧度；
- 夹爪默认除以 `--gripper-scale 100`；
- 超出夹爪范围时转换失败，而不是静默截断严重异常；
- 图像转换为 RGB `uint8`；
- 正式写入时每一帧都执行图像解码和固定尺寸检查。

## 10. 常用门限参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--fps` | 20 | LeRobot 输出频率 |
| `--max-image-delta-ms` | 40 | 图像到目标时刻的最大允许误差 |
| `--max-state-delta-ms` | 20 | 状态到目标时刻的最大允许误差；兼容约 30 Hz 状态流的半周期最近邻误差 |
| `--max-action-delta-ms` | 60 | 动作到目标时刻的最大允许误差 |
| `--max-action-step-deg` | 25 | 相邻输出动作的关节跳变告警门限 |
| `--gripper-scale` | 100 | 归一化到 1.0 的原始夹爪值 |

不要仅为了让报告变绿而任意放宽门限。应先判断异常来自：

- 时间戳来源错误；
- 相机丢帧；
- 发布线程阻塞；
- MCAP 写盘拥塞；
- 示教接管或离合切换；
- 控制器限速；
- 真实任务动作确实很快。

## 11. 视频编码

默认使用系统 FFmpeg 和 H.264：

```text
--use-videos --video-codec h264
```

可选：

```text
--video-codec hevc
--video-codec libsvtav1
```

调试时可使用 `--no-use-videos`，但 PNG 图像模式占用空间明显更大，不建议用于完整数据集。

## 12. 接入 OpenPI/Pi0.5

Pi0.5 使用固定图像槽位：

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
```

机器人专用输入变换应映射：

```text
observation.images.head        -> base_0_rgb
observation.images.left_wrist  -> left_wrist_0_rgb
observation.images.right_wrist -> right_wrist_0_rgb
observation.state              -> state
action                         -> actions
task                           -> prompt
```

数据原生动作是 16 维。建议保留 Pi0.5 的 32 维动作头，由 `PadStatesAndActions` 补齐，推理输出时只返回
`actions[:, :16]`。

如果把关节动作转换成相对当前状态的 delta，而保持夹爪为绝对动作，可使用：

```python
delta_action_mask = transforms.make_bool_mask(14, -2)
```

训练前必须为 F1 数据计算新的 normalization stats，不能复用 ALOHA、DROID 或其他机器人平台的统计量。

## 13. 必须人工确认的事项

- 原始关节单位确实是度；
- 左右关节名称和控制器下发顺序一致；
- 夹爪 `0` 和 `100` 分别代表开还是闭；
- 左右夹爪极性是否一致；
- 每条 Episode 是否真正成功；
- 裁剪后是否还含初始化、人工接管、急停恢复和长空闲段；
- 任务文本是否准确描述整条轨迹；
- 训练、验证、测试是否按 Episode/采集 Session 划分，而不是随机拆帧。
