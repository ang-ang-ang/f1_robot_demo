# VLA机器人数据清洗与质量评估指标文档

## 1. 文档目的

本文档用于定义 VLA（Vision-Language-Action）机器人数据在采集、转换、清洗、标注和训练前的质量检查标准。

适用的数据流程为：

```text
机器人采集
↓
ROS2 / MCAP 原始数据
↓
数据检查与清洗
↓
时间同步与格式转换
↓
LeRobot 数据集
↓
任务 / 子任务标注
↓
训练前质量验证
↓
VLA模型训练（如 π0.5）
```

数据清洗的核心目标不是简单删除“异常数据”，而是确保最终进入训练的数据满足以下要求：

* 数据完整
* 时间同步可靠
* 图像质量可用
* 机器人状态连续
* 动作数据正确
* Episode 完整有效
* 语言标注与任务一致
* 数据分布合理
* 满足模型训练格式和归一化要求

---

# 2. 数据质量总体框架

VLA 数据质量主要分为以下 8 类：


| 类别           | 主要内容                         | 优先级 |
| -------------- | -------------------------------- | ------ |
| 数据完整性     | Topic、帧、字段、维度是否完整    | P0     |
| 时间质量       | FPS、时间戳、Jitter、丢帧        | P0     |
| 多模态同步     | Camera、State、Action 是否对齐   | P0     |
| 图像质量       | 黑屏、模糊、冻结、曝光           | P1     |
| State质量      | NaN、越界、跳变、异常速度        | P0     |
| Action质量     | 动作跳变、延迟、饱和、连续性     | P0     |
| Episode质量    | 成功、失败、空闲、完整性         | P0     |
| 标注与数据分布 | Task/Subtask、类别平衡、统计分布 | P1/P2  |

其中：

* **P0：必须检查，不通过原则上不得进入训练集**
* **P1：强烈建议检查，可根据实际数据设 Warning 阈值**
* **P2：数据规模扩大后逐步引入**

---

# 3. 原始数据完整性指标

## 3.1 Topic 完整性

检查每个 Episode 是否包含预期数据流。

典型数据包括：

```text
Camera
├── Head Camera
├── Left Wrist Camera
└── Right Wrist Camera

Robot State
├── Joint State
├── TCP Pose
└── Gripper State

Action
├── Joint Command
├── TCP Command
└── Gripper Command
```

### 检查指标

* 必要 Topic 是否存在
* Topic 消息数量是否正常
* Topic 是否中途停止
* 不同 Episode 的 Topic Schema 是否一致

### 建议判定

```text
必要 Topic 缺失：
FAIL

非关键辅助 Topic 缺失：
WARNING
```

---

## 3.2 数据 Schema 一致性

检查：

* State 维度
* Action 维度
* Joint 数量
* 图像分辨率
* 数据类型
* Camera 数量

例如：

```text
Episode 1: state dim = 16
Episode 2: state dim = 16
Episode 3: state dim = 14
```

Episode 3 应被标记为异常。

---

# 4. 时间质量指标

时间质量是机器人/VLA数据最重要的质量维度之一。

不同传感器允许采用不同采样频率，例如：

```text
Camera       30 Hz
Robot State  50 Hz
Action       25~30 Hz
```

不同频率本身不是问题，关键是时间戳必须可靠，并能够在后处理阶段准确对齐。

---

## 4.1 实际采样频率

实际频率计算：

```text
Actual FPS =
(消息数量 - 1) / (最后时间戳 - 第一时间戳)
```

需要分别统计：

* Camera FPS
* Joint State FPS
* TCP FPS
* Action FPS
* Gripper FPS

示例：

```text
Head Camera
Expected: 30 Hz
Actual:   29.97 Hz
```

---

## 4.2 帧间隔

计算连续消息时间间隔：

```text
Δt = timestamp[i] - timestamp[i-1]
```

统计：

* Mean
* Standard Deviation
* Min
* Max
* P95
* P99

对于 30 Hz 相机：

```text
理论帧间隔 ≈ 33.3 ms
```

如果出现：

```text
33 ms
33 ms
34 ms
120 ms
33 ms
```

则可能存在丢帧或系统阻塞。

---

## 4.3 Jitter

Jitter 用于反映采样周期的不稳定程度。

```text
Jitter =
|实际帧间隔 - 理论帧间隔|
```

建议统计：

* Mean Jitter
* P95 Jitter
* P99 Jitter
* Max Jitter

工程初始参考：


| Camera 30Hz      | 建议判定       |
| ---------------- | -------------- |
| P95 < 5 ms       | GOOD           |
| 5\~15 ms         | WARNING        |
| >15 ms           | CHECK          |
| 单次 Gap >100 ms | WARNING / FAIL |

以上阈值建议根据设备实际运行情况进一步校准。

---

## 4.4 时间戳连续性

必须检查：

* Timestamp 是否单调递增
* 是否存在重复 Timestamp
* 是否存在 Timestamp 回退
* 是否存在异常大间隔

原则上：

```text
Timestamp Regression > 0
→ FAIL
```

---

# 5. 多模态同步指标

VLA 数据通常同时包含：

```text
Image
Robot State
Action
Language
```

模型训练时需要将这些数据映射到统一时间轴。

---

## 5.1 Camera 同步误差

对于：

```text
Head Camera
Left Wrist Camera
Right Wrist Camera
```

计算：

```text
Camera Sync Error =
|Camera Timestamp - Reference Timestamp|
```

统计：

* Mean
* P95
* P99
* Max

如果三路相机具有硬件同步能力，应优先保证采集时刻同步。

---

## 5.2 State 同步误差

例如统一时间点：

```text
Dataset Timestamp = 10.000 s

Camera = 10.002
State  = 9.998
```

则：

```text
State Sync Error = 2 ms
```

需要统计整个 Episode 的同步误差分布。

---

## 5.3 Action 同步误差

Action 对齐尤其重要。

不能简单假设：

```text
Observation(t)
→ Action(t)
```

机器人遥操作通常存在：

```text
观察场景
↓
操作者产生动作
↓
控制命令生成
↓
通信与控制
↓
机器人执行
```

因此实际关系可能为：

```text
Observation(t)
→ Action(t + Δt)
```

建议分析 Action latency，并在数据转换阶段进行统一补偿。

---

# 6. 图像质量指标

## 6.1 图像解码成功率

检查所有图像是否能够正常解码。

指标：

```text
Decode Success Rate
```

建议：

```text
关键 Camera 解码失败
→ FAIL / Episode剔除
```

---

## 6.2 黑屏与欠曝

检测：

* 图像平均亮度
* 低亮度像素比例
* 全黑或近似全黑图像

用于发现：

* Camera 未启动
* 镜头遮挡
* 光照异常

---

## 6.3 过曝

统计高亮像素比例，例如：

```text
pixel > 250
```

用于检测：

* 强光
* 自动曝光异常
* 视觉信息丢失

---

## 6.4 模糊检测

可通过 Laplacian Variance 等指标检测严重运动模糊或失焦。

目标不是删除所有存在运动模糊的数据，而是识别：

```text
无法有效识别物体或场景
```

的异常帧。

---

## 6.5 Frozen Frame

需要检测：

```text
Timestamp 正常变化
但连续图像内容完全不变化
```

常见原因：

* Camera pipeline 卡死
* Driver 异常
* 数据重复写入

可使用：

* Frame MSE
* SSIM
* Perceptual Hash

等方法进行检测。

---

# 7. Robot State质量指标

## 7.1 NaN / Inf

检查所有机器人状态：

```text
Joint
TCP
Gripper
```

是否包含：

```text
NaN
Inf
```

判定建议：

```text
出现 NaN / Inf
→ FAIL
```

---

## 7.2 Joint Range

检查：

```text
joint_min <= joint_position <= joint_max
```

超过机器人实际关节限制的数据应被标记。

---

## 7.3 Joint Velocity

计算：

```text
velocity =
(q[t] - q[t-1]) / Δt
```

用于检测：

* Encoder异常
* 数据跳变
* 时间戳异常

---

## 7.4 Joint Acceleration

进一步计算：

```text
acceleration =
(v[t] - v[t-1]) / Δt
```

异常大的加速度通常意味着：

* State跳变
* Tracking异常
* 数据采集异常

---

# 8. Action质量指标

VLA最终学习的是：

```text
Observation
→ Action
```

因此 Action 数据质量优先级非常高。

---

## 8.1 Action完整性

检查：

* Action维度
* Action缺失
* NaN / Inf
* Action类型一致性

---

## 8.2 Action范围

检查：

```text
Action Min
Action Max
```

是否处于机器人允许范围。

---

## 8.3 Action Jump

连续 Action：

```text
Action[t]
Action[t+1]
```

变化不应出现明显非物理跳变。

可以计算：

```text
|Action[t] - Action[t-1]|
```

用于检测：

* 遥操作异常
* 数据损坏
* 单位错误
* 时间错位

---

## 8.4 Action Smoothness

可以从速度、加速度、Jerk 等角度分析。

用于发现：

```text
遥操作抖动
错误插值
控制命令异常
```

---

## 8.5 Action Saturation

如果 Action 有固定范围，例如：

```text
[-1, 1]
```

则需要统计：

```text
|action| > 0.99
```

的比例。

若大量 Action 长时间处于极限值，需要检查：

* 控制器是否饱和
* 数据归一化是否错误
* 动作空间是否定义不合理

---

# 9. Episode质量指标

Episode 是 VLA 数据管理的基本单位之一。

每条 Episode 建议至少包含：

```text
任务开始
↓
机器人执行
↓
任务成功 / 失败
↓
Episode结束
```

---

## 9.1 Episode Duration

统计：

* Duration
* Frame Count
* 有效操作时间

用于识别：

* Episode过短
* Episode异常长
* 数据录制未正确停止

---

## 9.2 Idle Ratio

大量无动作数据会降低训练数据有效密度。

可以计算：

```text
Idle Ratio =
静止帧数量 / 总帧数量
```

静止可通过以下指标定义：

* Action变化低于阈值
* Joint速度低于阈值

---

## 9.3 Motion Ratio

对应：

```text
Active Motion Ratio
```

用于判断 Episode 是否包含有效操作。

---

## 9.4 Task Success

每条 Episode 强烈建议增加：

```text
success = true / false
```

失败数据进一步增加：

```text
failure_reason
```

例如：

```text
grasp_failed
object_dropped
collision
timeout
teleop_abort
sensor_failure
```

---

# 10. Task语言标注

VLA需要语言与机器人行为建立对应关系。

Episode级至少需要：

```text
Task Instruction
```

例如：

```text
Pick up the red gear and place it into the tray.
```

建议语言中尽量包含：

```text
操作
+
目标物体
+
目标位置
```

避免过度模糊描述，例如：

```text
Move object.
```

---

# 11. Subtask标注

长任务建议进一步切分为 Subtask。

例如：

```text
0.0s      2.2s      3.4s      5.1s      8.0s

Reach     Grasp     Lift      Move      Place
```

每个 Subtask 建议包含：

```text
subtask_name
start_frame
end_frame
start_time
end_time
```

需要检查：

* Subtask 是否重叠
* 是否存在明显 Gap
* Boundary 是否合理
* Subtask 与视频内容是否一致

---

# 12. 标注质量指标

对于人工或VLM自动标注，需要检查：

```text
Language ↔ Video
Language ↔ Action
Task ↔ Object
Task ↔ Target Position
Task ↔ Success Result
```

例如：

```text
Video:
Blue Cube

Label:
Pick up the red cube
```

应判定为标注错误。

---

# 13. VLM自动标注策略

VLM可以用于：

* 自动生成Task描述
* 自动生成Subtask
* 自动寻找Subtask Boundary
* 自动识别场景与物体

推荐流程：

```text
VLM自动标注
↓
结构合法性检查
↓
人工抽检
↓
异常修正
↓
正式进入训练集
```

VLM自动标注不建议未经检查直接作为最终真值。

---

# 14. 数据分布指标

随着数据规模扩大，需要持续检查数据集分布。

建议统计：

## Task Distribution

例如：

```text
Pick       45%
Place      30%
Insert     15%
Assembly   10%
```

避免某些任务占比过高。

---

## Object Distribution

统计：

* 不同物体
* 不同颜色
* 不同尺寸
* 不同类别

---

## Scene Distribution

统计：

* 工作台
* 位置
* 背景
* 光照
* 相机视角

---

## Trajectory Distribution

避免大量重复或高度相似轨迹。

---

# 15. 重复数据检测

需要检测：

```text
完全重复Episode
近似重复Episode
```

可使用：

* Video Hash
* Trajectory Hash
* Initial State / Final State
* Visual Embedding
* Trajectory Embedding

用于降低重复数据占比，提高有效数据多样性。

---

# 16. 数据归一化统计

进入 π0.5 等模型训练前，需要统计：

```text
Mean
Std
Min
Max
Q01
Q99
```

主要针对：

```text
Observation State
Action
```

推荐重点关注：

```text
Q01
Q99
```

因为 Min/Max 容易受到极端异常数据影响。

---

## 16.1 低方差维度

如果：

```text
std ≈ 0
```

说明该维度几乎没有变化。

需要判断：

* 该自由度是否未使用
* 传感器是否失效
* 是否应该参与模型归一化

---

# 17. 推荐质量状态

建议每个 Episode 最终输出三级状态：

```text
PASS
WARNING
FAIL
```

## PASS

数据完整且满足训练要求。

## WARNING

存在一定异常，但经过确认后仍可使用。

例如：

```text
Camera jitter略高
Idle ratio偏高
少量模糊帧
```

## FAIL

原则上不得直接进入训练。

例如：

```text
Topic缺失
NaN / Inf
Timestamp回退
Camera长时间失效
State/Action维度错误
严重Action异常
```

---

# 18. Quality Score设计

除了状态之外，可以额外提供0\~100的数据质量评分。

参考权重：


| 类型        | 权重 |
| ----------- | ---: |
| 时间质量    |   25 |
| 图像质量    |   15 |
| State质量   |   15 |
| Action质量  |   20 |
| Episode质量 |   15 |
| 标注质量    |   10 |
| 合计        |  100 |

示例：

```json
{
  "episode": 137,
  "quality_score": 82.4,
  "status": "WARNING",
  "flags": [
    "camera_jitter_high",
    "idle_ratio_high"
  ]
}
```

质量评分主要用于：

* 数据排序
* 快速筛查
* 数据版本对比

不建议单独依赖 Quality Score 决定是否删除数据，应同时保留具体异常 Flags。

---

# 19. 第一阶段推荐执行指标

项目初期不建议一次性实现全部指标。

第一阶段优先实现：

## P0：必须完成

```text
Topic完整性
Schema一致性
Timestamp连续性
实际FPS
大间隔检测
Camera解码
NaN / Inf
Joint越界
Action越界
State/Action维度
Episode完整性
Task标签
Success/Failure
```

---

## P1：第二阶段完成

```text
Jitter
Camera同步
State同步
Action同步
Action latency
Frozen Frame
Action Jump
Idle Ratio
Mean/Std/Q01/Q99
```

---

## P2：规模化后引入

```text
图像模糊
曝光检测
重复Episode检测
Trajectory聚类
Scene Diversity
Task Balance
VLM自动Subtask标注
Embedding-based数据筛选
```

---

# 20. 推荐工程流程

最终建议项目建立如下数据处理流水线：

```text
                     Raw MCAP
                        │
                        ↓
                ① 数据完整性检查
              Topic / Schema / Count
                        │
                        ↓
                  ② 时间质量检查
            FPS / Timestamp / Jitter / Gap
                        │
                        ↓
                  ③ Sensor质量检查
        Camera Decode / Freeze / Blur / Exposure
                        │
                        ↓
                ④ State / Action检查
       NaN / Range / Jump / Velocity / Saturation
                        │
                        ↓
                    ⑤ 时间对齐
          Camera / State / Action Resampling
                        │
                        ↓
                  ⑥ Episode检查
        Success / Failure / Idle / Start / End
                        │
                        ↓
                ⑦ 转换为LeRobot
                        │
                        ↓
                ⑧ Task/Subtask标注
                  Human + VLM
                        │
                        ↓
                  ⑨ 标注质量检查
                        │
                        ↓
                 ⑩ 数据统计分析
          Distribution / Mean / Std / Q01 / Q99
                        │
                        ↓
                   VLA模型训练
```

---

# 21. 推荐输出文件

数据清洗工具建议为每批数据生成：

```text
dataset_qc/
├── summary.json
├── episode_quality.csv
├── failed_episodes.txt
├── warning_episodes.txt
├── dataset_statistics.json
└── report.html
```

其中：

## episode\_quality.csv

建议字段：

```text
episode_id
duration
camera_fps
state_fps
action_fps
camera_jitter
sync_error
nan_count
action_jump_count
idle_ratio
success
quality_score
status
flags
```

---

# 22. 项目阶段目标

第一阶段目标不是建立复杂的“智能数据清洗系统”，而是建立一条可靠、可重复执行的数据质量流水线：

```text
MCAP
↓
自动检查
↓
自动生成PASS/WARNING/FAIL
↓
异常Episode人工复核
↓
转换LeRobot
↓
标注
↓
训练
```

当数据规模从几十条增长到数百、数千条后，再逐步加入：

```text
自动视觉质量检测
自动数据去重
自动轨迹聚类
VLM自动标注
数据多样性评估
自动质量评分
```

最终形成可规模化运行的 VLA 数据生产与质量管理体系。
