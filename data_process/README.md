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
| `repair_f1_joint_state_jumps.py` | 检测并保守修复短暂、孤立的 JointState 跳变；持续性动作只记录人工复核 |
| `plot_f1_joint_state_jumps.py` | 直接从原始 MCAP 绘制正常运动（蓝）与检测到的关节跳变（红）证据图 |
| `audit_f1_joint_states.py` | 批量审计原始 `/lead/joint_states` 与 `/hal/joint_states` 的结构、时间戳、步长、速度、加速度、尖峰和逐关节范围 |
| `README_JOINT_STATE_JUMPS.md` | 关节跳变检测、可视化、保守修复、报告解读和训练准入说明 |
| `requirements-plot.txt` | 独立绘图环境的 NumPy / Matplotlib / OpenCV 依赖，不需安装训练依赖 |
| `mcap.py` | 按 MCAP 记录顺序读取 Header、publish time、log time 和 CDR 字节偏移 |
| `mcap_writer.py` | 无损复制 ROS 2 CDR payload，并写出未压缩 Chunk MCAP 和 JSON 诊断话题 |
| `resample_f1_mcap.py` | 清洗后 MCAP 的固定频率映射、派生 MCAP 和 Foxglove 诊断入口 |
| `resampling.py` | 公共时间轴、最近邻映射、原始/清洗/目标时间 lineage 和 MCAP 写出 |
| `analyze_f1_resampling.py` | 周期 Gap、流内残差、流间偏差和可视化分析入口 |
| `resampling_analysis.py` | JSON/Markdown/CSV 统计与 PNG 图生成 |
| `default_config.toml` | 质检、时间戳清洗和转换默认门限 |
| `DATA_QUALITY_GUIDE.md` | 质量指标、风险门限和报告解读说明 |
| `analyze_f1_mcap_exposure.py` | 直接对原始 MCAP 三路图像分别运行曝光检测并保存原始帧定位信息 |
| `analyze_f1_lerobot_operators.py` | 对 LeRobot 逐相机曝光和关节运动质量进行可组合分析 |
| `mcap_quality_source.py` | 原始 MCAP 三路图像独立遍历、JPEG 解码和逐帧来源追踪适配器 |
| `exposure_operator.py` | MCAP/LeRobot 共用的独立曝光检测和过曝 Mask 可视化算法 |
| `operator_config.toml` | 曝光、关节限位/速度/加速度/单帧跳点默认门限 |
| `QUALITY_OPERATORS.md` | 可组合质量算子的算法、阈值依据、运行与扩展说明 |

推荐流程不是“直接转换”，而是：

1. 查看原始 Episode 和任务结果；
2. 编写每条 Episode 的任务、裁剪和成功标注；
3. 运行原始时序/关节质量评估；
4. 直接对原始 MCAP 运行三相机曝光算子，保留异常图像的原始 Topic 索引和时间戳；
5. 对时间戳清洗先运行 `--dry-run`，检查逐帧修正量和高风险帧；
6. 写出清洗后的新 MCAP，原始 MCAP 不做原地修改；
7. 对清洗数据重新运行质量评估和原始 MCAP 曝光算子；
8. 生成 20 Hz 映射和派生 MCAP，分析周期 Gap、流内残差和流间偏差；
9. 用 Foxglove 或静态图定位时间误差尖峰；
10. 运行转换器 `--dry-run`；
11. 正式生成 LeRobot 数据；
12. 运行 LeRobot 曝光与关节质量算子，比较转换前后结果；
13. 读取 LeRobot 数据集做最终抽样检查；
14. 计算 OpenPI normalization stats 后再训练。

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

### 4.2 JointState 跳变检测与保守修复

`repair_f1_joint_state_jumps.py` 用于处理 `sensor_msgs/msg/JointState.position` 中可证明为短暂孤立异常的跳变。
它默认检查从机状态 `/hal/joint_states` 的 14 个 `arm_` 关节，默认门限为单步 `15°` 或速度 `300°/s`；只有
异常段被前后可靠锚点包围、长度不超过 6 帧、锚点净位移不超过 `15°`，且插值修正至少为 `5°` 时才会改写数据。

持续性阶跃、示教初始化、离合/接管、Episode 边界和长异常段会被保留为 `manual_review`，不会被静默平滑。先运行
dry-run：

```bash
uv run python -m data_process.repair_f1_joint_state_jumps \
  --raw-dir data/260915_data \
  --report-dir artifacts/260915_joint_state_repair \
  --dry-run
```

输出目录包含：

```text
artifacts/260915_joint_state_repair/<UTC时间>/
├── joint_state_repair_report.json
├── joint_state_repair_report.md
├── joint_state_events.csv
└── episodes/
    ├── episode_000096_openbox.json
    └── episode_000097_rollbox.json
```

应优先查看 `status`、`repair_event_count`、`review_event_count`、`max_step_deg` 和
`max_velocity_deg_s`。`review` 不等于可以写出修复结果；必须结合三路视频、任务阶段和控制日志判断事件是否是
真实操作。要检查示教动作而不写入数据，可显式指定 Topic：

```bash
uv run python -m data_process.repair_f1_joint_state_jumps \
  --raw-dir data/260915_data \
  --topic /lead/joint_states \
  --max-step-deg 25 \
  --max-velocity-deg-s 500 \
  --report-dir artifacts/260915_lead_joint_review \
  --dry-run
```

只有人工确认报告中的异常确实是采集孤立坏点后，才可写入新目录：

```bash
uv run python -m data_process.repair_f1_joint_state_jumps \
  --raw-dir data/260915_data \
  --output-dir artifacts/260915_data_joint_repaired \
  --report-dir artifacts/260915_joint_state_repair
```

写模式绝不修改 `--raw-dir`：它复制每个 Episode，仅改写已确认需要修复的 `JointState.position` 双精度值，并且只接受
未压缩、Chunk/Data CRC 均为 0 的 MCAP。若输出已存在，必须在确认旧结果可以替换后显式添加 `--overwrite`。

#### 4.2.1 已核验的 `data/` 数据范围（2026-09-28）

以下是当时已核验的样本范围。后来加入的 `data/episode_all/` 共 9 条 MCAP，不包含在该历史结论中；
本次可视化使用已核验的 `episode_000096_openbox`，没有据此判定新增整批数据已经合格。

| 数据范围 | 结论 |
| --- | --- |
| `260915_data/episode_000096_openbox` | 可解析；13 个采集 Topic。`/hal/joint_states` 共 3,564 帧，最大单步 `5.561°`、最大速度约 `168.7°/s`，低于当前从机状态修复门限。 |
| `260915_data/episode_000097_rollbox` | 可解析；13 个采集 Topic。`/hal/joint_states` 共 2,941 帧，最大单步 `5.717°`、最大速度约 `173.5°/s`，低于当前从机状态修复门限。 |
| `F1_data_SOP_openbox/episode_000004`、`episode_000006` | 可解析；各有 13 个采集 Topic。从机状态最大单步分别为 `5.480°`、`5.378°`，当前未发现自动修复候选。 |
| `260915_data/reference data/roll2g3` | 可解析但仅含 `/hal/joint_states` 和两路夹爪状态，缺少相机与动作 Topic；可作状态诊断，不可直接转换为训练 Episode。 |
| `260915_data/playback_*` 及其余 `reference data` 小文件 | 多数在 MCAP 尾部有截断 record，现有解析器会拒绝读取；应隔离并从原始采集端重新导出/校验，不能直接清洗或转换。 |

按本工具的状态默认门限（单步 `15°`、速度 `300°/s`）检查 2026-09-15 两条完整真机 Episode 的
`/lead/joint_states`，发现持续性大跳变：openbox 最大单步 `65.384°`，rollbox 最大单步 `55.280°`。按动作的
方案门限（单步 `25°`、速度 `500°/s`）复跑后仍分别有 9、4 个 `manual_review` 事件；当前自动修复数为 0。它们可能
对应初始化、接管或真实快速动作，在人工对照视频前不得平滑或纳入普通训练集。

按动作方案门限检查 SOP `episode_000004`、`episode_000006` 的 `/lead/joint_states`，也发现分别 5、3 个
`manual_review` 事件（最大单步分别为 `25.507°`、`23.032°`，后者由速度超限触发）；当前同样全部保留，没有写出清洗 MCAP。四条完整 Episode 的
`/hal/joint_states` 均为 `clean`，因此本轮没有需要自动修改的从机状态值。

#### 4.2.2 跳变与正常运动可视化

为便于人工复核，`plot_f1_joint_state_jumps.py` 直接读取原始 MCAP，按与动作质检相同的门限检测每个相邻
JointState 边。正常连续边绘制为蓝色；触发单步或速度门限的**对应关节边及端点**绘制为红色。它不会修改 MCAP，
并会自动从最长的无跳变连续区间选择一段有实际运动的读数作对照（至少三帧、至少一个关节变化 `1°`）。
蓝色仅表示未触发当前跳变门限，不能代替位置限位、加速度、任务成功等完整验收。

图的每一列顶部还嵌入一张头部相机截图，来源为 `/camera/head/color/image_raw/compressed`。脚本按 Header 时间戳，
为正常窗口和跳变窗口分别选择最近的头部图像，并在标题中标出图像帧号、相对时间和时间差。头部图像与关节流采样频率
不同，且原始数据存在重复的头部 Header 时间戳，因此截图用于显示手部/箱体场景，不宣称与关节读数逐纳秒同步。

以下命令复现当前真机 `episode_000096_openbox` 的图。该 Episode 的 `/lead/joint_states` 在帧 `7 → 8`
（相对时间约 `0.249 → 0.298 s`）出现最大 `65.384°` 的跳变，涉及 `arm_L_joint1`、`arm_L_joint2`、
`arm_L_joint3`、`arm_L_joint4` 和 `arm_L_joint7`。其中左臂第 4 关节在 `49.446 ms` 内从 `-44.616°`
变到 `-110.000°`；正常对照段为 `42.697–45.636 s`，最大单步 `1.747°`、最大速度 `45.200°/s`。
两列都是 `/lead/joint_states` 的示教目标读数；图中的跳变不代表从机 `/hal/joint_states` 实际运动了相同幅度。

推荐使用轻量独立环境（命令从仓库根目录运行）：

```bash
uv run --no-project --python 3.11 --with-requirements data_process/requirements-plot.txt \
  python -m data_process.plot_f1_joint_state_jumps \
  --raw-dir data/260915_data/episode_000096_openbox \
  --output-dir artifacts/joint_jump_visualization \
  --topic /lead/joint_states \
  --max-step-deg 25 \
  --max-velocity-deg-s 500
```

已有 Python 3.11/3.12 分析环境时，也可先运行 `python -m pip install -r data_process/requirements-plot.txt`，
再用 `python -m data_process.plot_f1_joint_state_jumps` 加上上述参数。Windows PowerShell 可使用下面的单行命令：

```powershell
uv run --no-project --python 3.11 --with-requirements data_process/requirements-plot.txt python -m data_process.plot_f1_joint_state_jumps --raw-dir data/260915_data/episode_000096_openbox --output-dir artifacts/joint_jump_visualization
```

CLI 默认从 `operator_config.toml` 按 Topic 读取门限：`/lead/joint_states` 为 `25°` / `500°/s`，
`/hal/joint_states` 为 `15°` / `300°/s`；可用 `--config` 或显式门限覆盖。选择最大单步事件作图；没有检测到跳变时
明确报错，不生成虚假的异常示例。帧号从 0 开始，时间来自该 Topic 的 Header，角度沿用项目规定的原始度数，未做平滑或重采样。

输出文件：

```text
artifacts/joint_jump_visualization/
├── joint_jump_vs_normal.png       # 左列正常蓝线；右列异常边/端点红色、其余上下文蓝色
├── joint_jump_vs_normal.json      # 源文件、最大跳变原值、窗口、统计和门限
├── joint_jump_events.csv          # 按时间合并的异常片段，当前为 4 段
├── joint_angle_samples.csv        # 两个窗口内的 14 关节原始角度、时间戳与异常标记
├── head_camera_normal_reference.jpg # 正常窗口最近的头部截图
└── head_camera_jump_reference.jpg   # 跳变窗口最近的头部截图
```

绘图报告按时间合并为 4 段，修复报告按关节统计为 9 个 `manual_review` 事件，计数口径不同。
`artifacts/` 是忽略提交的本地产物；同事可运行以上命令生成下图：

![关节跳变与正常运动对比](../artifacts/joint_jump_visualization/joint_jump_vs_normal.png)

图中的红色只表示达到工程检测门限的读数，不证明它一定是传感器坏点；例如初始化、示教接管或真实快速动作也会触发。
头部截图可帮助确认手部是否正在接近箱体、发生接管或处于初始化阶段；图仍应与完整视频和控制日志一起用于决定裁剪、
人工复核或是否允许使用保守插值修复。

绘图所需直接依赖固定在 `requirements-plot.txt`；主项目 `pyproject.toml` / `uv.lock` 保持本次上游版本。

#### 4.2.3 `data/episode_all` 批次核验与代表性跳变

针对后来加入的 `data/episode_all/`，本次按动作质检门限逐条检查了 9 条 Episode 的
`/lead/joint_states`（单步 `25°` 或速度 `500°/s`），并同时按状态门限检查了
`/hal/joint_states`（单步 `15°` 或速度 `300°/s`）。扫描直接读取原始 MCAP；头部图像数量只用于确认
每条 Episode 都具备视频对照，不参与关节跳变判定。批次汇总保存在
`artifacts/episode_all_joint_jump_scan/episode_all_joint_jump_scan.{json,csv,md}`。

| Episode | `/lead/joint_states` 帧数 | 动作异常段 | 最大单步 / ° | 最大速度 / °/s | `/hal/joint_states` 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `episode_000000` | 1,939 | 2 | 45.388 | 1,295.479 | clean |
| `episode_000001` | 1,389 | 2 | 32.996 | 1,123.583 | clean |
| `episode_000002` | 1,550 | 1 | 28.963 | 680.186 | clean |
| `episode_000003` | 1,573 | 2 | 26.999 | 942.814 | clean |
| `episode_000004` | 1,399 | 1 | 24.565 | 750.164 | clean |
| `episode_000005` | 1,824 | 3 | 33.777 | 947.861 | clean |
| `episode_000006` | 1,558 | 2 | 27.731 | 645.213 | clean |
| `episode_000055` | 1,204 | 2 | 25.947 | 628.392 | clean |
| `episode_000057` | 1,684 | 1 | 23.783 | 747.570 | clean |

本批次选取 `episode_000000` 作为代表性跳变样本。它的 `/lead/joint_states` 在相对时间
`0.324388–0.359423 s`（第 `8 → 9` 帧）触发最大事件，涉及 `arm_L_joint2`、`arm_L_joint3` 和
`arm_L_joint7`。最大单步是 `arm_L_joint7` 从 `-45.388°` 变为 `0.000°`，步长 `+45.388°`，
相邻 Header 间隔约 `35.035 ms`，对应约 `1,295.479°/s`。同一 Episode 的第二段事件为
`arm_R_joint5` 第 `14 → 15` 帧，步长 `37.515°`。

按保守修复器以 `--dry-run` 生成逐关节报告时，4 个事件均被判为 `manual_review/persistent_step`，
自动修复值数为 `0`；这表示它们可能是初始化、示教接管或真实快速动作，不能仅凭曲线认定为传感器坏点，
也不能直接平滑。正式报告目录为：

```text
artifacts/episode_all_joint_jump_detection/20261002T154053263438Z/
├── joint_state_repair_report.md
├── joint_state_repair_report.json
├── joint_state_events.csv
└── episodes/episode_000000.json
```

本次代表性曲线和最近邻头部截图如下；左列为未越限的正常运动（蓝色），右列为跳变上下文（触发边及端点为红色）：

![`episode_000000` 关节跳变与正常运动对比](../artifacts/episode_all_joint_jump_visualization/episode_000000/joint_jump_vs_normal.png)

对应产物包括：

- `joint_jump_vs_normal.json`：门限、源 MCAP、选中事件、窗口和头部截图时间差；
- `joint_jump_events.csv`：按时间合并的异常片段；
- `joint_angle_samples.csv`：正常/异常窗口的 14 关节原始度数和异常边标记；
- `head_camera_normal_reference.jpg`、`head_camera_jump_reference.jpg`：用于观察手部/箱体场景的头部截图。

上述所有报告和图像均由原始数据派生，`data/episode_all/episode_000000/episode_000000_0.mcap` 未被修改。

#### 4.2.4 `data/episode_00001--10` 关节点位核验（2026-10-09）

目录实际包含 `episode_000000–episode_000010` 共 11 条 MCAP。本次以 `episode_000001–episode_000010`
为 10 条主检查对象，`episode_000000` 作为附加样本单列。均可解析，消息数量与各自的 `metadata.yaml` 一致。
主检查读取 10,238 帧 `/lead/joint_states` 和 11,785 帧 `/hal/joint_states`。

检查包括完整 JointState 的关节名/维度、NaN/Inf、Header 时间连续性，以及双臂 14 关节的步长、速度、加速度、单帧尖峰和角度 min/max。
沿用动作门限 `25° / 500°/s`、状态门限 `15° / 300°/s`、加速度门限 `5000°/s²`；速度/加速度从门限的 80% 起预警。
原始数值按项目约定使用度，时间取相对 Header 纳秒差；没有重采样、平滑或 MCAP 写入。缺少当前机器人 URDF/控制器限位，
机械位置限位仍未验证。

| Episode | 动作跳变段 | 动作最大步长 / ° | 动作最大速度 / °/s | 跳变发生时间 / s | 从机状态 |
| --- | ---: | ---: | ---: | --- | --- |
| `episode_000001` | 1 | 28.160 | 781.446 | 0.274–0.310 | 已检查项未越限 |
| `episode_000002` | 1 | 30.272 | 1,019.099 | 0.475–0.505 | 已检查项未越限 |
| `episode_000003` | 1 | 28.763 | 855.096 | 0.435–0.468 | 已检查项未越限 |
| `episode_000004` | 1 | 27.350 | 843.789 | 0.150–0.182 | 已检查项未越限 |
| `episode_000005` | 2 | 30.986 | 980.835 | 0.226–0.257；24.492–24.545 | 已检查项未越限 |
| `episode_000006` | 1 | 30.579 | 655.719 | 0.380–0.426 | 已检查项未越限 |
| `episode_000007` | 1 | 25.094 | 795.228 | 0.406–0.438 | 已检查项未越限 |
| `episode_000008` | 1 | 26.880 | 602.074 | 19.436–19.480 | 已检查项未越限 |
| `episode_000009` | 1 | 27.388 | 707.217 | 0.171–0.210 | 已检查项未越限 |
| `episode_000010` | 1 | 27.004 | 730.585 | 0.196–0.233 | 已检查项未越限 |

主检查 10 条均有动作目标异常，共 11 个时间段、41 条逐关节跳变边，dry-run 全部为 `manual_review/persistent_step`，
自动修复候选为 0。9 段发生在起始 0.51 s 内，可能与初始化/接管有关；另两段发生在操作过程中，必须单独复核：

- `episode_000005`：`arm_R_joint4` 第 `659 → 660` 帧，从 `-77.383°` 到 `-105.294°`，步长 `-27.910°`、速度 `531.759°/s`。
- `episode_000008`：`arm_R_joint4` 第 `504 → 505` 帧，从 `-75.011°` 到 `-101.891°`，步长 `-26.880°`、速度 `602.074°/s`。

动作目标还出现 347 个加速度超限点（关节×三帧计数，可能与跳变重叠），其中 156 个窗口起点晚于 1 s，
不能仅裁掉开头就认为数据通过。10 条从机状态的最大步长 `5.651°`、最大速度 `171.191°/s`、最大加速度 `2997.375°/s²` 均未越限。
两类关节流没有数值/结构错误、非递增时间戳、超配置门限时间间隔和符合当前恢复门限的单帧尖峰。

附加 `episode_000000` 在 `29.609–29.651 s`、`775 → 776` 帧出现 `arm_R_joint4` 的 `-40.469°` 动作跳变，
速度 `962.601°/s`；其从机状态同样未触发当前检查门限，该条不计入上表统计。

完整产物位于 `artifacts/episode_00001--10_joint_audit_20261009/`：

- [检测结论与逐关节跳变定位](../artifacts/episode_00001--10_joint_audit_20261009/joint_audit.md)
- [完整 JSON](../artifacts/episode_00001--10_joint_audit_20261009/joint_audit.json)、[逐 Episode 汇总](../artifacts/episode_00001--10_joint_audit_20261009/episode_summary.csv)、[事件明细](../artifacts/episode_00001--10_joint_audit_20261009/joint_events.csv)、[逐关节范围与统计](../artifacts/episode_00001--10_joint_audit_20261009/joint_ranges.csv)
- [`episode_000008` 红蓝曲线与头部截图](../artifacts/episode_00001--10_joint_audit_20261009/episode_000008_visualization/joint_jump_vs_normal.png)

复现本次只读扫描（NumPy/PyYAML 的 Python 3.11 环境，从仓库根目录执行）：

```powershell
python -m artifacts.audit_joint_points --raw-dir data/episode_00001--10 --output-dir artifacts/episode_00001--10_joint_audit_rerun
```

扫描入口 `artifacts/audit_joint_points.py` 和结果均为本地产物。机械限位、完整视频复核、主从延迟及任务验收仍未完成；
动作目标跳变不直接证明从机发生相同幅度的实际运动，也不应未经确认自动插值。

### 4.3 LeRobot 曝光与关节质量算子

GitHub 上游已加入面向**转换后 LeRobot 数据集**的可组合质量算子：

- 三相机曝光算子：先做 sRGB 线性化，再用跨相机中位数基线判断持续过亮或高光截断；
- 关节运动算子：分别检查 `observation.state` 和 `action` 的位置范围、步长、真实时间速度、加速度和单帧跳点；
- 统一 JSON、Markdown 和 CSV 报告，配置写在 `operator_config.toml`，算法与阈值说明见 `QUALITY_OPERATORS.md`。

先完成 MCAP 清洗与 LeRobot 转换，再运行：

```bash
uv run python -m data_process.analyze_f1_lerobot_operators \
  --dataset-root artifacts/F1_data_SOP_openbox/lerobot \
  --output-dir artifacts/F1_data_SOP_openbox/operator_quality
```

本地目前没有上游默认路径的 F1 URDF，运行前需用 `--joint-limit-urdf <实际路径>` 指定当前机器人版本。
速度与加速度默认仍是待硬件确认的工程阈值，不能代替控制器规格。本次确认了这些上游文件完整合入并通过语法编译；
当前验证环境缺少 `pyarrow`、`tyro`，未对 LeRobot 质量算子执行端到端数据验证。


新生成的清洗 Episode 会同时包含 `f1_cleaning_manifest.json` 和 `f1_timestamp_corrections.csv`，后者用于
恢复每帧的原始 Header、清洗后 Header、修正方法和风险等级。不要只复制清洗后的 MCAP 而丢失这两个文件。

### 4.4 固定 20 Hz 映射和派生 MCAP

建议先只生成映射，确认公共时间区间、残差和输出体积：

```bash
uv run python -m data_process.resample_f1_mcap \
  --raw-dir artifacts/F1_data_SOP_openbox/cleaned_mcap \
  --output-dir artifacts/F1_data_SOP_openbox/resampled_20hz \
  --mapping-only
```

检查通过后去掉 `--mapping-only`，每个 Episode 会输出：

```text
episode_000004/
├── aligned_20hz.mcap
└── resampling_mapping.json
```

`resampling_mapping.json` 是时间 lineage 的审计真值：

- `source_samples` 保存源消息的 record index、sequence、原始 Header、清洗后 Header、publish/log、修复方法和风险；
- `target_mappings` 保存每个 20 Hz 目标时刻选择的 `source_index`、`cleaned-target` 和 `original-target` 有符号残差；
- `selected_unique_source_indices` 保存真正参与输出的源消息；
- `reused_target_count` 统计连续目标时刻复用同一源消息的次数；
- `omitted_source_count` 统计降采样后未被选择的源消息。

默认 `--mcap-mode aligned` 会在每个目标时刻为每个流写一条消息：MCAP envelope 的 `log_time/publish_time`
改为严格的目标时间，但 CDR payload 内的 ROS `header.stamp` 保留清洗后的源时间。该模式适合固定频率回放和
Foxglove 排查；稀疏流可能复用 payload，复用信息会进入映射和诊断话题。

如果只希望保留唯一选中的原消息，并保持原始 envelope 时间：

```bash
uv run python -m data_process.resample_f1_mcap \
  --raw-dir artifacts/F1_data_SOP_openbox/cleaned_mcap \
  --output-dir artifacts/F1_data_SOP_openbox/resampled_unique \
  --mcap-mode unique
```

该模式输出 `selected_unique.mcap`，不会伪造低频流，因此不保证每个 Topic 都严格达到 20 Hz。

分析映射 JSON：

```bash
uv run python -m data_process.analyze_f1_resampling \
  --input-path artifacts/F1_data_SOP_openbox/resampled_20hz \
  --output-dir artifacts/F1_data_SOP_openbox/resampling_analysis
```

输出包括：

- `resampling_statistics.json/.md`：逐 Episode、逐流和所有流对的详细统计；
- `stream_statistics.csv`：原始/清洗周期、Gap、重采样残差、复用率和丢弃率；
- `pairwise_statistics.csv`：所有流对的有符号中位数、绝对 P95/P99 和最大偏差；
- `gap_events.csv`：超过 `1.5T` 的周期 Gap 位置、倍数和风险；
- `residual_timeline.png`、`residual_histogram.png`、`gap_timeline.png` 和 `camera_pairwise_skew.png`。

核心定义：

```text
residual = cleaned_source_time - target_time
pairwise_skew(A, B) = selected_time(A) - selected_time(B)
```

残差为负表示选择历史样本，为正表示离线最近邻选择了未来样本。流间偏差描述被选中时间戳之间的差异，
不能证明相机在硬件层面同时曝光。

`aligned_20hz.mcap` 还会加入 `/f1/resampling/<stream_slug>` 和 `/f1/resampling/cross_stream` JSON 诊断话题。
在 Foxglove 中增加三路 Image 面板、一个 Plot 面板和一个 State Transitions 面板；Plot 的 Timestamp 选择
`Log time`，然后绘制例如：

```text
/f1/resampling/camera_head_color_image_raw_compressed.residual_ms
/f1/resampling/camera_left_wrist_color_image_raw_compressed.residual_ms
/f1/resampling/cross_stream.camera_head_color_image_raw_compressed__camera_left_wrist_color_image_raw_compressed_ms
```

State Transitions 可查看 `risk`、`repair_method` 和 `reused_previous_source`。点击 Plot 尖峰可跳到对应播放时间，
同时检查三路图像。Foxglove 参考：

- Plot：<https://docs.foxglove.dev/docs/visualization/panels/plot>
- State Transitions：<https://docs.foxglove.dev/docs/visualization/panels/state-transitions>
- JSON Schema：<https://docs.foxglove.dev/docs/getting-started/custom/custom-schema-encodings>
- 
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

## 14. 数据质检清洗方案对照追踪

本节用于跟踪本目录代码相对于 `data_wash.md`（数据质检清洗方案）和
`DATA_QUALITY_GUIDE.md` 的实现范围。状态只描述当前仓库中的代码/文档能力，不代表某一批数据已经通过验收。
已核验范围包括 `data/` 中的 4 条完整真机 Episode，具体范围见 4.2.1；关节跳变检查与绘图已有回归验证。`artifacts/`
中的报告是本地运行产物，不应提交到仓库。历史处理结果见 `docs/F1_QUALITY_AND_CONVERSION_SUMMARY.md`，不能替代对新数据的复跑。

### 14.1 已完成或已有可复用实现

| 方案项 | 状态 | 当前实现与证据 |
| --- | --- | --- |
| MCAP、ROS 2 schema、CDR 解析 | 已完成 | `mcap.py` 解析未压缩 MCAP chunk、schema/channel/message、Header、publish/log time 和图像字节偏移；`f1_schema.py` 集中维护 Topic 与关节名。 |
| 必需 Topic、字段名和维度检查 | 已完成 | `convert_f1_mcap_to_lerobot.py` 检查三路图像、左右臂关节、夹爪状态/命令、关节名称、单列夹爪和 16 维输出；缺失项会终止当前 Episode。 |
| 图像格式/分辨率检查 | 已完成 | 质检抽样解码并记录尺寸；正式转换逐帧解码，跨 Episode 比较尺寸，异常时失败。 |
| Topic 时间统计 | 已完成 | `analyze_f1_data_quality.py` 和转换报告记录数量、覆盖时长、有效频率、周期中位数/P95/P99/最大值、标准差和非递增次数。 |
| 数值 NaN/Inf 检查 | 已完成 | 质检报告对数值 Topic 统计 NaN/Inf；转换前对选中的关节和夹爪字段执行有限值检查。 |
| 图像解码与基础视觉线索 | 已完成 | 抽样报告包含解码失败、亮度、极暗/极亮像素比例、Laplacian 方差、压缩哈希重复数和联系表；正式转换覆盖全部输出帧。 |
| LeRobot 曝光与关节运动算子 | 上游实现已合入；待本机端到端验证 | GitHub 上游新增 `analyze_f1_lerobot_operators.py`、`exposure_operator.py`、`joint_motion_operator.py` 和 `operator_config.toml`。三相机曝光检查使用线性光亮度和连续段判定；State/Action 分别检查 URDF 位置限位、步长、真实时间速度、加速度和单帧跳点，并输出 JSON/Markdown/CSV。 |
| 固定频率时间轴与多流最近邻对齐 | 已完成 | 转换器默认 20 Hz，计算三路图像、状态、动作的中位/P95/最大重采样残差，并按门限拒绝超限 Episode。 |
| 动作跳变、主从跟随误差和可视化证据 | 已完成 | 报告记录动作最大步长、跳变帧和最大 10 个位置，以及每关节/全局主从误差 P95；动作跳变和跟随偏差会形成风险项。`repair_f1_joint_state_jumps.py` 对 JointState 逐关节计算单步/速度，并对可证明的短暂异常做保守插值；`plot_f1_joint_state_jumps.py` 从原始 MCAP 输出正常蓝线与异常红线的逐关节对比图。 |
| 任务、裁剪、成功/失败标注 | 已完成 | `episode_config` 支持 `include`、`task`、起止时间、`success`、`failure_reason`、`notes`；缺少任务或标记失败会按规则告警/失败。 |
| 头相机 Header 时间戳清洗 | 已完成 | `cleaning.py` 支持 dry-run、局部插值、鲁棒帧序号回归、单调保护、逐帧修正 CSV、风险等级和汇总报告；写模式复制到新目录，不原地改原始 MCAP。 |
| 清洗后 MCAP 的安全写出约束 | 已完成 | 仅对未压缩 chunk 且 chunk/data CRC 为 0 的文件做等长 Header 补丁，否则拒绝写入，避免生成不可验证的 MCAP。 |
| LeRobot 转换和转换报告 | 已完成 | 支持单条/批量、dry-run、图像视频编码、16 维 state/action、RGB 图像、task 字段、失败报告和 `meta/f1_conversion.{json,md}`。 |
| 可追溯报告输出 | 已完成 | 质检输出 JSON/Markdown/`risks.csv`/Episode JSON/联系表；清洗输出汇总报告、Episode 报告和 `timestamp_corrections.csv`。 |
| 辅助对象检测管线 | 已有实现 | `detect_f1_objects.py` + `detection.py` 提供 YOLOE 开放词汇检测、标注视频和 JSON 报告，但尚未接入主质检/标注验收链路。 |

### 14.2 已有统计但尚未形成完整验收规则

这些项目已有部分代码或数据字段，仍应视为“未闭环”，不能直接勾选为完成：

| 方案项 | 当前覆盖 | 缺口 |
| --- | --- | --- |
| 时间 Jitter | 报告有周期标准差，清洗器有周期异常统计 | 没有按相机/状态/动作分别输出 Jitter 阈值判定，也没有统一的 P0/P1 风险策略。 |
| 图像黑屏、过曝、模糊、冻结帧 | 原始质检已计算亮度、极暗/极亮比例、Laplacian 方差和抽样哈希；上游曝光算子增加了全量三相机持续过亮/高光截断片段定位 | 仍缺少黑屏、欠曝、模糊和冻结帧的全量可配置验收，以及统一的自动风险等级。 |
| TCP 质量 | 已比较从机/示教平移量级，并提示欧拉角单步环绕 | 只有左臂比较，尚未统一坐标系/米毫米/角度表示，也没有可用于训练的 TCP 转换。 |
| 状态/动作范围 | 夹爪有 `[0, scale]` 检查；原始 MCAP 修复器提供单步/速度检测、孤立异常插值和人工复核报告；上游关节算子已支持 URDF 位置限位、速度、加速度、单帧跳点和可覆写 TOML 门限 | 仍缺已确认的当前 F1 本体 URDF/控制器逐关节速度与加速度规格，以及超限后的剔除、裁剪或人工复核准入策略。 |
| Episode 完整性 | 有公共时间区间、裁剪和输出帧/时长 | 没有 idle ratio、motion ratio、开始/结束事件和任务有效操作时长的验收规则。 |
| 任务标注一致性 | 要求 task，支持人工 success/failure | 没有 language↔video、language↔action、对象/目标位置的一致性检查。 |

### 14.3 TODO list

按数据方案的优先级维护；完成一项后应同时补代码、报告字段、README 示例和最小可复现验证。

#### P0：进入训练前必须补齐

- [ ] **统一 Topic/Schema 验收**：补充所有预期 Topic 的存在性、消息数量下限、中途停止检测，以及跨 Episode 的 schema、关节名、dtype、图像分辨率一致性报告。当前转换器只保证训练所需字段能够解析。
- [ ] **关节物理范围与状态连续性准入**：已完成孤立跳变的单步/速度检测、保守插值和人工复核报告；上游算子已纳入 URDF 位置限位、加速度和单帧跳点。仍需确认并接入当前 F1 14 个关节的真实 URDF/控制器规格，并明确超限时是剔除、裁剪还是人工复核。
- [ ] **动作时延分析**：利用 Header/publish/log time 和主从误差估计 action latency，明确是否补偿以及补偿后的验证方式。
- [ ] **Episode 验收表**：生成每条 Episode 的 duration、有效帧数、裁剪边界、success、failure_reason、任务文本和最终 PASS/WARNING/FAIL，支持失败/警告 Episode 清单。
- [ ] **训练集隔离规则**：增加按 Episode/采集 Session 划分 train/validation/test 的配置与检查，防止随机拆帧造成泄漏。
- [ ] **P0 自动化回归数据**：提交不含真实数据的最小合成 MCAP fixture，覆盖缺 Topic、schema/维度错误、NaN/Inf、非递增时间戳、夹爪越界、图像解码失败和对齐超限。

#### P1：质量闭环和规模化前补齐

- [ ] **Jitter 判定**：输出 mean/P95/P99/max jitter，按数据流配置阈值，并把大间隔、丢帧和时间戳异常区分为 warning/fail。
- [ ] **图像全量质量检查**：上游已完成三相机持续过亮/高光截断段定位；仍需增加黑屏/欠曝/模糊/冻结帧的阈值、连续段定位和验收等级，抽样不能替代最终验收。
- [ ] **State/Action 统计扩展**：上游算子已输出 velocity、acceleration 与步长的 P50/P95/P99/P99.9/max；仍需补 saturation、idle ratio、motion ratio、Q01/Q99 和常量维度是否从训练字段移除的规则。
- [ ] **数据集级统计产物**：生成 `summary.json`、`episode_quality.csv`、`failed_episodes.txt`、`warning_episodes.txt`、`dataset_statistics.json` 和可选 HTML 报告，统一当前分散的 JSON/Markdown/CSV。
- [ ] **TCP 统一方案**：定义左右臂坐标系、单位、角度/旋转表示和部署端逆变换；完成后再决定是否把 TCP 纳入 state/action。
- [ ] **Task/Subtask 标注格式**：定义 subtask 名称、起止帧/时间、非重叠约束和人工复核字段，并在转换和报告中保留版本信息。
- [ ] **标注一致性检查**：增加对象、目标位置、任务结果与视频/动作的人工抽检清单；VLM 自动标注必须经过结构校验和人工抽检后才能入库。

#### P2：数据规模扩大后引入

- [ ] **Episode 去重与相似度检测**：视频/轨迹哈希、近似重复 Episode 和初末状态相似度分析。
- [ ] **任务/对象/场景/轨迹分布**：输出类别平衡、场景多样性和长任务帧数占比，避免单一任务压倒数据集。
- [ ] **质量评分**：按方案中的时间、图像、State、Action、Episode、标注权重生成 0–100 分，同时保留具体 flags，不能只用总分删数据。
- [ ] **Embedding/VLM 辅助筛选**：在已有对象检测管线之外，增加可复现的模型版本、输入采样和人工复核记录。
- [ ] **Normalization stats 产出与训练接入**：基于最终 LeRobot 数据计算 F1 专用 mean/std/min/max/Q01/Q99，并在 OpenPI 配置中验证 state/action 维度和掩码。

### 14.4 当前维护约定

1. 新增质量指标时，先在 `DATA_QUALITY_GUIDE.md` 写清定义、单位、默认门限和风险解释，再实现代码。
2. 能修复的数据必须保留原始值、修复方法、修正量和风险；不能用插值或裁剪把异常静默变成通过。
3. 任何“已完成”状态都需要一个可定位的代码/报告证据和一次最小数据验证；只有文档描述不算完成。
4. 新的 LeRobot 输出必须保留转换参数、输入 Episode、任务/成功标注和质量报告，保证同事可以复现和追责。

### 14.5 GitHub 同步与本轮验证（2026-10-02）

本轮同步基准为 [GitHub 提交 `3b533c498ca3901b5ca9f6c4a55570827ced6412`](https://github.com/ang-ang-ang/f1_robot_demo/commit/3b533c498ca3901b5ca9f6c4a55570827ced6412)
（2026-09-28，`add_exposure_operator`）。工作目录原本没有 `.git`，采用下载上游快照后逐文件校验的方式同步。
新增的 7 个上游文件为 `analyze_f1_lerobot_operators.py`、`exposure_operator.py`、`joint_motion_operator.py`、
`quality_operator.py`、`quality_operator_config.py`、`operator_config.toml` 和 `QUALITY_OPERATORS.md`。
其余上游代码与本地一致；README 合并保留了本地跳变修复和清洗追踪章节。没有进行远程提交。

- [x] 上游文件逐项校验，无缺失，新增 7 个文件与上游内容一致。
- [x] `data_process/`、`sim_replay/` 与测试文件语法编译通过。
- [x] 2 项已有修复测试与 6 项绘图回归测试通过（`python -m pytest tests -q`）。
- [x] 新增绘图脚本和测试通过 Ruff 检查与格式检查。
- [x] 真机 `episode_000096_openbox` 重新解析并生成 14 关节 PNG、角度 CSV 和追踪 JSON；修复器 dry-run 为 9 个关节复核事件、0 个自动修复。
- [ ] 上游 LeRobot 质量算子：补齐环境、当前本体 URDF 和 LeRobot 数据后执行端到端验收。
