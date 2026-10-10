# F1 MCAP/LeRobot 曝光与关节质量算子

本文说明可组合质量检测算子。算子通过统一的 `FrameContext` 接收图像和数值特征：

- `McapIndependentImageSource` 直接读取原始 MCAP，按各相机自己的 Topic 顺序输出全部图像并保留消息来源；
- `LeRobotEpisodeSource` 读取转换后的 LeRobot 视频与 Parquet；
- `IndependentCameraExposureOperator` 对每路相机使用独立绝对阈值，不比较其他相机；
- `JointMotionOperator` 当前用于 LeRobot state/action，后续可接入原始 MCAP 数值 Source，不需要复制判定逻辑。

## 1. 直接检测原始 MCAP 曝光

### 1.1 快速验证

对 `test_data/episode_all` 中每条 Episode 的每路相机先解码 100 帧：

```bash
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/episode_all \
  --output-dir artifacts/episode_all/raw_exposure \
  --max-frames 100
```

正式报告不要设置 `--max-frames`：

```bash
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/episode_all \
  --output-dir artifacts/episode_all/raw_exposure
```

只分析指定 Episode：

```bash
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/episode_all \
  --episodes episode_000001 episode_000004 \
  --output-dir artifacts/episode_all/raw_exposure
```

生成过曝区域 Mask 校验视频：

```bash
uv run python -m data_process.analyze_f1_mcap_exposure \
  --raw-dir test_data/episode_all/episode_000001 \
  --output-dir artifacts/episode_all/raw_exposure \
  --write-exposure-mask-videos
```

### 1.2 原始三路图像独立分析原则

原始算子不再把三路图像同步后互相比较，而是遵循以下规则：

1. 保持每个 Topic 在 MCAP 中的原始消息顺序，不修改源文件；
2. 头相机 Header 的重复、倒退和周期倍数异常使用现有清洗算法在内存中修复；
3. 左右腕相机使用原始 Header 时间，不使用 publish/log time 替代物理采集时间；
4. 每路图像都用自己的 `topic_index` 判断连续片段，不要求另外两路存在同一时刻的图像；
5. 三路相机视场和物体占比不同不会直接构成异常，因为判定不再使用 peer baseline；
6. publish time 和 log time 仅用于报告定位、录包延迟分析和可视化工具回看。

报告在 `episodes/<episode>/exposure_frames.csv` 中保留：

- `source_topic_index`、`source_record_index`、`source_sequence`；
- 原始 `source_header_time_ns`、`source_publish_time_ns`、`source_log_time_ns`；
- 当前流用于报告的 `source_analysis_time_ns`；
- 头相机修复方法、风险等级和修正毫秒数。

因此检测到异常后，可以用 Topic 索引或 log time 在 Foxglove/PlotJuggler 中定位，而不是只能定位到转换后视频帧。

### 1.3 原始曝光输出

```text
artifacts/episode_all/raw_exposure/<UTC时间>_episode_all/
├── quality_report.json
├── quality_report.md
└── episodes/
    └── episode_000001/
        ├── operator_report.json
        ├── exposure_frames.csv
        ├── exposure_segments.csv
        └── exposure_masks/                 # 使用可选参数时生成
            ├── observation_images_head_overexposure_mask.mp4
            ├── observation_images_left_wrist_overexposure_mask.mp4
            └── observation_images_right_wrist_overexposure_mask.mp4
```

`quality_report.json` 包含每路原始图像数、实际分析数和头相机时间戳修复诊断。`--max-frames` 分别限制每路相机的
JPEG 解码与曝光计算量；正式结论必须用完整运行结果。

## 2. 检测 LeRobot 数据

先用少量帧验证环境与路径：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/F1_data_SOP_openbox/lerobot \
  --max-frames 100
```

正式分析全部 Episode：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/F1_data_SOP_openbox/lerobot \
  --output-dir artifacts/F1_data_SOP_openbox/operator_quality
```

只运行曝光算子：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/F1_data_SOP_openbox/lerobot \
  --no-run-joint-state \
  --no-run-joint-action
```

只分析指定 Episode：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/F1_data_SOP_openbox/lerobot \
  --episodes 0 1
```

## 3. 独立曝光算子

### 3.1 亮度计算

输入视频是 sRGB/JPEG，不能直接把编码后的 RGB 值代入亮度公式。算子先逐通道执行标准 sRGB 逆传递函数：

```text
C_linear = C_srgb / 12.92                              C_srgb <= 0.04045
C_linear = ((C_srgb + 0.055) / 1.055) ^ 2.4           其他情况
```

再计算线性光亮度：

```text
Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear
```

每张图输出：

- `l50`：亮度中位数，反映画面主体和大面积区域；
- `l95`：亮度 95 分位数，反映亮部但不被极少数高亮像素完全支配；
- `highlight_clipping_ratio`：`Y > 0.98` 的像素比例。

### 3.2 单路绝对阈值判定

每张图只根据本路指标判断。以下任一条件成立即为候选：

1. 大面积过曝：`L50 >= 0.40` 且 `L95 >= 0.90`；
2. 高光截断：`R_clip = N(Y > 0.98) / N_pixel >= 5%`。

候选必须连续至少 3 帧才确认为异常片段。LeRobot 20 Hz 数据中 3 帧约为 150 ms；原始 MCAP 约 30 Hz 时约为
100 ms。该连续性要求用于滤掉解码噪声、反光闪烁和单帧曝光调整。

这些绝对阈值是保守的初始工程值。在线性光空间中 `L50=0.40` 已表示至少一半画面相当明亮，而 `L95=0.90` 要求亮部
接近传感器上限；两者联合使用可避免仅有小块白色纸箱时触发大面积过曝。`R_clip=5%` 用于检测大量像素堆积在右端极值。
不同相机的镜头、曝光策略和动态范围仍可能不同，应根据人工确认结果分别调参，不能把这些值当成通用物理标准。

`pixel_stride=1` 默认使用所有像素。大批量筛查可在覆盖配置中设置为 2，但这时计算的是规则网格抽样估计值。

### 3.3 Mask 校验视频

启用 `--write-exposure-mask-videos` 后，每路相机生成一个并排视频：

- 左侧：原图，满足 `Y > clip_luminance` 的像素以红色覆盖，并显示 L50、L95、R_clip 和候选状态；
- 右侧：相同阈值形成的二值 Mask，白色为高光截断区域，黑色为其他区域。

Mask 表示的是“达到截断亮度阈值的像素”，不是物体分割，也不代表白色区域一定错误。它的用途是判断异常来自灯具反光、
纸箱白面、小范围金属高光，还是整幅图像曝光失控。视频会显著增加计算量和磁盘占用，建议先对指定 Episode 或
`--max-frames` 小样本开启。

## 4. 关节运动算子

算子分别分析：

- `observation.state`：真实从机状态；
- `action`：示教/目标动作。

输入为弧度，报告统一转换成度。逐关节输出位置范围以及绝对步长、速度、加速度的 P50、P95、P99、P99.9 和最大值。

### 4.1 位置限位

默认从 `f1p01_00000000_20260920/urdf/f1p01_00000000_20260918.urdf` 读取 14 个机械臂关节上下限，
允许 `0.1°` 数值容差。部署到其他机器人版本时必须通过 `--joint-limit-urdf` 指定对应本体的 URDF。

### 4.2 速度

速度采用真实时间间隔计算：

```text
v[i] = (q[i] - q[i-1]) / (t[i] - t[i-1])
```

超过 `0.8 × V_max` 记为 warning，超过 `V_max` 记为 fail。当前 URDF 的 arm velocity 均错误地写成 0，不能作为
真实硬件限制，因此默认临时使用：

- state：`300°/s`；
- action：`500°/s`。

这些值沿用当前质量管线的工程门限，不代表 F1 控制器规格。拿到驱动器/控制器逐关节 `V_max` 后，应优先修复 URDF，
或修改覆盖 TOML。

### 4.3 加速度

加速度使用相邻速度及两个区间中心的时间差计算：

```text
a[i] = (v[i] - v[i-1]) / ((dt[i] + dt[i-1]) / 2)
```

默认 `A_max=5000°/s²`，`0.8 × A_max` 开始 warning。该值同样是待硬件确认的工程阈值。

### 4.4 单帧跳点

中心帧同时满足以下条件时标为 `single_frame_spike`：

- 相对前一帧变化至少 `5°`；
- 相对后一帧变化至少 `5°`；
- 前一帧和后一帧之间恢复误差不超过 `1°`。

这个三点模式针对“只有中间一帧错误、下一帧恢复”的采集异常。持续运动、真实阶跃或离合重定位不会仅靠该指标直接判为
单帧跳点，但可能触发步长、速度或加速度事件。

## 5. 参数覆盖

不要直接改算法代码。创建例如 `configs/f1_operator.local.toml`，只覆盖已确认的参数：

```toml
[exposure]
overexposed_l50_threshold = 0.45
overexposed_l95_threshold = 0.92
minimum_consecutive_frames = 5
clip_ratio_threshold = 0.08

[joint_motion.state]
fallback_max_velocity_deg_s = 240.0

[joint_motion.action]
fallback_max_velocity_deg_s = 360.0

[joint_motion]
spike_min_step_deg = 4.0
```

然后运行：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/F1_data_SOP_openbox/lerobot \
  --config configs/f1_operator.local.toml
```

## 6. 代码扩展

- `quality_operator.py`：统一 `FrameContext`、算子协议、LeRobot Source 和执行器；
- `mcap_quality_source.py`：原始 MCAP 逐路图像解码、时间处理与来源追踪，不包含曝光判定；
- `exposure_operator.py`：逐相机独立亮度判定和可选 Mask 视频；
- `joint_motion_operator.py`：只依赖一个关节特征；
- `quality_operator_config.py`：默认配置和局部 TOML 覆盖；
- `analyze_f1_lerobot_operators.py`：组合算子、选择 Episode、写 JSON/Markdown/CSV。
- `analyze_f1_mcap_exposure.py`：原始 MCAP 曝光批量入口和原始消息定位报告。

新增其他 MCAP 检测项时，应继续让 Source 只负责解析、时间处理、单位规范化和来源追踪，不要在 Source 中实现曝光、速度或
异常判断。下游离线报告、实时消费或清洗决策均可复用每个算子的 `start/process/finish` 生命周期。
