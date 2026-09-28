# F1 LeRobot 曝光与关节质量算子

本文说明 `analyze_f1_lerobot_operators.py` 中新增的可组合检测算子。算子通过统一的
`FrameContext` 接收同步图像和数值特征，因此后续只需新增 MCAP Source 适配器，即可复用相同算法、阈值和报告结构，
不需要把曝光或关节逻辑复制进 MCAP 解析代码。

## 1. 快速运行

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

## 2. 曝光一致性算子

### 2.1 亮度计算

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

### 2.2 跨相机判定

对每个相机，把另外两个相机指标的中位数作为同一时刻的 peer baseline。以下任一条件成立即为候选：

1. 广泛偏亮：`L50 ratio >= 1.60`、`L50 delta >= 0.08` 且 `L95 delta >= 0.05`；
2. 高光截断：本相机 `R_clip >= 5%`、比其他相机高至少 `3%`，且 `L95 >= 0.98`。

候选必须连续至少 3 帧才确认为异常片段。20 Hz 数据中 3 帧约为 150 ms，可以滤掉解码噪声、反光闪烁和单帧曝光调整。

这些默认值由当前两条开箱数据抽样校准：正常 L50 大多位于 `0.09～0.23`，头相机 clipping ratio 的 P95
约为 `2.6%`；因此 `5%` 是明显但不过于敏感的高光截断起点。由于三个相机视场不同，纸箱白面或灯具可能只占某一路
的大面积区域，任何确认片段仍需要结合图像人工复核。

`pixel_stride=1` 默认使用所有像素。大批量筛查可在覆盖配置中设置为 2，但这时计算的是规则网格抽样估计值。

## 3. 关节运动算子

算子分别分析：

- `observation.state`：真实从机状态；
- `action`：示教/目标动作。

输入为弧度，报告统一转换成度。逐关节输出位置范围以及绝对步长、速度、加速度的 P50、P95、P99、P99.9 和最大值。

### 3.1 位置限位

默认从 `f1p01_00000000_20260920/urdf/f1p01_00000000_20260918.urdf` 读取 14 个机械臂关节上下限，
允许 `0.1°` 数值容差。部署到其他机器人版本时必须通过 `--joint-limit-urdf` 指定对应本体的 URDF。

### 3.2 速度

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

### 3.3 加速度

加速度使用相邻速度及两个区间中心的时间差计算：

```text
a[i] = (v[i] - v[i-1]) / ((dt[i] + dt[i-1]) / 2)
```

默认 `A_max=5000°/s²`，`0.8 × A_max` 开始 warning。该值同样是待硬件确认的工程阈值。

### 3.4 单帧跳点

中心帧同时满足以下条件时标为 `single_frame_spike`：

- 相对前一帧变化至少 `5°`；
- 相对后一帧变化至少 `5°`；
- 前一帧和后一帧之间恢复误差不超过 `1°`。

这个三点模式针对“只有中间一帧错误、下一帧恢复”的采集异常。持续运动、真实阶跃或离合重定位不会仅靠该指标直接判为
单帧跳点，但可能触发步长、速度或加速度事件。

## 4. 参数覆盖

不要直接改算法代码。创建例如 `configs/f1_operator.local.toml`，只覆盖已确认的参数：

```toml
[exposure]
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

## 5. 代码扩展

- `quality_operator.py`：统一 `FrameContext`、算子协议、LeRobot Source 和执行器；
- `exposure_operator.py`：只依赖三路同步图像；
- `joint_motion_operator.py`：只依赖一个关节特征；
- `quality_operator_config.py`：默认配置和局部 TOML 覆盖；
- `analyze_f1_lerobot_operators.py`：组合算子、选择 Episode、写 JSON/Markdown/CSV。

新增 MCAP 支持时，应实现一个产生相同 `FrameContext` 的 Source：负责 MCAP 解码和时间对齐，但不要在 Source 中实现曝光、
速度或异常判断。下游算子需要实时消费时，也可以直接复用每个算子的 `start/process/finish` 生命周期。

