# F1 关节跳变检测、审计与保守修复

本文说明原始 F1 ROS 2 MCAP 中 `sensor_msgs/msg/JointState.position` 的检测、证据生成和保守修复流程。
工具面向左右臂 14 个关节，原始读数单位为度。每个命令均从仓库根目录运行；除明确提供 `--output-dir`
的修复写模式外，所有命令只读取 MCAP，不会原地改写源文件。

## 1. 工具和职责

| 文件 | 职责 |
| --- | --- |
| `audit_f1_joint_states.py` | 批量审计原始 `/lead/joint_states` 与 `/hal/joint_states`：结构、时间戳、步长、速度、加速度、尖峰和逐关节范围 |
| `repair_f1_joint_state_jumps.py` | 对可证明是短暂、孤立坏点的状态序列进行保守线性插值；其余事件保留为人工复核 |
| `plot_f1_joint_state_jumps.py` | 生成正常运动蓝线、异常边红线，并嵌入最近头部相机截图 |
| `operator_config.toml` | 两类关节流的步长、速度、加速度和尖峰默认工程门限 |
| `requirements-plot.txt` | 单独运行曲线图所需的 NumPy、Matplotlib、OpenCV 依赖 |

`/lead/joint_states` 是示教端/目标动作，`/hal/joint_states` 是从机状态。两者出现不同结果时，不能把动作目标的
跳变直接解释为实际机械臂发生同等幅度的物理跳变。

## 2. 默认规则

默认值来自 `operator_config.toml`，用于工程筛查，并不等同于控制器或硬件规格。

| 信号 | 单步门限 | 速度门限 | 加速度门限 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `/lead/joint_states` | 25° | 500°/s | 5000°/s² | 动作目标；速度/加速度超过 80% 时预警 |
| `/hal/joint_states` | 15° | 300°/s | 5000°/s² | 从机状态；速度/加速度超过 80% 时预警 |

对相邻 Header 时间戳 `t[i]` 和原始角度 `q[i]`，工具计算：

```text
step[i]         = q[i + 1] - q[i]
velocity[i]     = step[i] / (t[i + 1] - t[i])
acceleration[i] = (velocity[i] - velocity[i - 1]) / ((dt[i] + dt[i - 1]) / 2)
```

任一关节单步或速度超门限时，被视为跳变候选。单帧尖峰还须满足：进入与退出均至少 `5°`，且前后两帧的恢复误差不超过
`1°`。它用于识别单帧错误，不能把持续性阶跃、真实快速运动或接管重定位误判为可自动修复的尖峰。

位置机械限位需要当前 F1 本体 URDF 或控制器逐关节规格。未提供时，脚本会明确报告为 `unverified`，不会用臆测范围
判定 MCAP 不合格。

## 3. 快速使用

### 3.1 批量原始关节审计

对一个 Episode 或包含多个 `episode_*` 子目录的根目录运行：

```bash
uv run python -m data_process.audit_f1_joint_states \
  --raw-dir data/F1_data \
  --output-dir artifacts/f1_joint_audit
```

审计器会检查两个 JointState Topic 是否存在、关节名/顺序/维度是否稳定、是否有 NaN/Inf、Header 是否递增，
再对 14 个 `arm_` 关节计算运动指标。报告目录为：

```text
artifacts/f1_joint_audit/
├── joint_audit.json       # 完整机器可读报告、源文件大小/mtime、逐关节统计
├── joint_audit.md         # 每个 Episode 和 Topic 的人工复核表
├── episode_summary.csv    # Episode × Topic 汇总
├── joint_events.csv       # 跳变、速度预警、加速度和尖峰事件
├── joint_ranges.csv       # 每关节 min/max/std 与最大运动指标
└── episodes/
    ├── episode_000001.json
    └── episode_000002.json
```

`joint_audit.json` 会在运行前记录每个源 MCAP 的大小和修改时间，结束时再次核对；`size_mtime_unchanged_after_audit=true`
表示该只读检查未改变对应源文件。

### 3.2 保守修复候选检查

默认检查从机状态。先以 dry-run 运行；没有 `--output-dir` 时也会自动进入 dry-run：

```bash
uv run python -m data_process.repair_f1_joint_state_jumps \
  --raw-dir data/F1_data \
  --report-dir artifacts/f1_joint_state_repair \
  --dry-run
```

检查动作目标时，显式选择 Topic 与动作门限：

```bash
uv run python -m data_process.repair_f1_joint_state_jumps \
  --raw-dir data/F1_data \
  --topic /lead/joint_states \
  --max-step-deg 25 \
  --max-velocity-deg-s 500 \
  --report-dir artifacts/f1_lead_joint_review \
  --dry-run
```

输出含 `joint_state_repair_report.{json,md}`、`joint_state_events.csv` 和各 Episode JSON。关注：

- `repair_event_count`：满足严格条件、可候选插值的事件数；
- `review_event_count`：保留原值、必须人工确认的事件数；
- `status`：只要存在 `manual_review`，Episode 就保持 `review`，即使部分点已被安全修复；
- `max_step_deg`、`max_velocity_deg_s`：当前检测门限下的最大原始变化。

只有当视频、控制日志和从机响应共同证明事件是采集孤立坏点时，才可以运行写模式：

```bash
uv run python -m data_process.repair_f1_joint_state_jumps \
  --raw-dir data/F1_data \
  --output-dir artifacts/F1_data_joint_repaired \
  --report-dir artifacts/f1_joint_state_repair
```

写模式会复制每个 Episode，再改写复制品中的 `JointState.position` 双精度值；它绝不修改 `--raw-dir`。为防止破坏
MCAP 完整性，当前写模式仅接受未压缩 Chunk 且 Chunk/Data CRC 为 0 的记录。已有输出目录时，必须显式传入
`--overwrite`。

## 4. 自动修复条件和人工复核

检测到越限边不代表应立即修复。修复器仅修改同时满足以下条件的候选：

1. 异常段不是 Episode 边界；
2. 两端存在可靠锚点；
3. 异常段不超过 `max_repair_run_frames=6` 帧；
4. 两端锚点的净位移不超过 `max_anchor_delta_deg=15°`；
5. 插值相对原值的最大残差至少 `min_correction_deg=5°`。

否则会保留原始读数并写为 `manual_review`。常见原因包括：

| `reason` | 含义与处置 |
| --- | --- |
| `boundary_event` | 发生在序列首尾，缺少一侧锚点；保留并查看采集起止阶段 |
| `persistent_step` | 连续阶跃或锚点净位移过大；可能是接管、初始化或真实动作 |
| `run_too_long` | 连续异常帧超过自动修复上限；需要结合视频判断 |
| `small_interpolation_residual` | 插值与原值差异不足 5°；记录风险但没有明确的修复收益 |

不应通过放宽门限或平滑曲线来“通过”数据。初始化、离合、急停恢复、控制模式切换、真实快速操作和消息损坏都可能触发相同的数值模式。

## 5. 曲线与头部视频证据

为单条存在跳变的 Episode 生成 14 关节对照图：

```bash
uv run --no-project --python 3.11 \
  --with-requirements data_process/requirements-plot.txt \
  python -m data_process.plot_f1_joint_state_jumps \
  --raw-dir data/F1_data/episode_000001 \
  --output-dir artifacts/joint_jump_visualization/episode_000001 \
  --topic /lead/joint_states \
  --max-step-deg 25 \
  --max-velocity-deg-s 500
```

图左列是自动选择的正常运动参考窗口：曲线为蓝色，且至少一个关节有 `1°` 的有效运动。右列展示包含最大跳变的
上下文：正常边仍为蓝色，触发门限的对应关节边和端点为红色。每列顶部都会嵌入距离关节窗口最近的
`/camera/head/color/image_raw/compressed` 截图，并显示 Topic 帧号、相对时间和 Header 时间差。

输出为：

```text
artifacts/joint_jump_visualization/episode_000001/
├── joint_jump_vs_normal.png
├── joint_jump_vs_normal.json
├── joint_jump_events.csv
├── joint_angle_samples.csv
├── head_camera_normal_reference.jpg
└── head_camera_jump_reference.jpg
```

头部截图用于核对手部、目标物和初始化/接管场景。相机与关节 Topic 的频率不同，且数据可能有重复图像 Header；
最近邻截图不证明硬件时钟已同步。图中红色只表示超过当前工程门限，不证明该读数必然是传感器坏点。

## 6. 批次判读与训练准入

建议按以下顺序处理异常：

1. 查看 `episode_summary.csv`，分开比较 `/lead/joint_states` 和 `/hal/joint_states`；
2. 对 `joint_events.csv` 中的动作跳变，检查对应头部/腕部视频、从机响应和控制日志；
3. 首先识别采集起始的初始化或接管片段，再检查任务操作阶段的跳变和高加速度段；
4. 仅对已证实的单帧采集坏点进行保守修复，其他事件记录裁剪边界或保留为人工复核；
5. 对修复后的复制数据重新审计，并在 Episode 配置中记录 `include`、任务文本、裁剪范围、成功状态和复核依据；
6. 完成时间对齐、图像质量、关节限位和任务成功检查后，再进入 LeRobot 转换与训练集划分。

`clean_under_checked_thresholds` 表示当前已检查的结构、时间与运动门限没有触发，不代表整条 Episode 已完成所有数据质量验收。
同样，动作目标跳变而从机状态稳定时，可能是控制器限速、主从延迟或 leader Topic 并非实际下发命令；应使用真实时间轴的
主从跟随误差与控制日志继续确认。

## 7. 当前限制

- 机械位置范围只有在提供当前机器人版本的 URDF 或控制器规格后才能判定；
- 速度和加速度默认为工程阈值，需由硬件团队确认或用覆盖配置替换；
- 审计按每个 Topic 的 Header 时间计算，不做跨 Topic 延迟补偿；
- 图像截图是辅助人工复核证据，不替代完整视频、控制日志或任务成功判断；
- 报告产物、真实 MCAP、视频和包含隐私场景的截图应存入实验制品存储，通常不应提交到 Git。
