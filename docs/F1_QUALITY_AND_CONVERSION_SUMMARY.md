# F1 开箱数据质量分析与 LeRobot 转换汇总

生成日期：2026-09-03

## 1. 处理范围

本次处理以下四条 MCAP Episode：

- `F1_data_SOP_openbox/episode_000004`
- `F1_data_SOP_openbox/episode_000006`
- `F1_data_test/episode_000001`
- `F1_data_test/episode_000002`

所有数据均转换为 20 Hz、16 维状态、16 维动作和三路 H.264 图像的 LeRobot 数据集。

## 2. Episode 配置

- SOP 配置：`F1_data_SOP_openbox/episodes.conversion.json`
- test 配置：`F1_data_test/episodes.conversion.json`
- 四条 Episode 的均匀抽帧均显示最终箱盖处于打开状态，因此暂时标记为 `success=true`。
- 成功标记属于基于联系表的初步视觉确认，正式训练前仍应按照项目的任务成功标准复核。
- `F1_data_test/episode_000001` 裁剪为 `3.0～83.5 s`，排除了约 `1.95 s` 和 `2.50 s` 的示教初始化跳变。
- 其余三条轨迹的运动信号几乎覆盖全程，保留完整公共有效时间范围。

## 3. 最终质量报告

### 3.1 F1_data_SOP_openbox

- 最终报告：`F1_data_SOP_openbox/quality_reports_final/20260903T065719011214Z_F1_data_SOP_openbox/quality_report.md`
- JSON：`F1_data_SOP_openbox/quality_reports_final/20260903T065719011214Z_F1_data_SOP_openbox/quality_report.json`
- 风险 CSV：`F1_data_SOP_openbox/quality_reports_final/20260903T065719011214Z_F1_data_SOP_openbox/risks.csv`
- 总体状态：`warning`
- 高风险失败项：`0`
- SOP 状态和夹爪状态源约为 `30.3 Hz`，因此使用 `20 ms` 最大状态重采样误差门限和 `45 ms` 源状态间隔门限。

| Episode | 帧数 | 状态最大误差 | 动作最大误差 | 最大动作步长 | 跟随全局 P95 | 最差关节 P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `episode_000004` | 2,033 | 16.757 ms | 47.819 ms | 25.507° | 2.319° | 3.739° |
| `episode_000006` | 2,102 | 17.110 ms | 43.783 ms | 23.032° | 2.536° | 3.548° |

主要告警：

- 两条数据的头相机时间戳存在重复或倒退，并已按帧序号重建。
- `episode_000004` 头相机非递增时间戳 338 次，最大修正 166.863 ms。
- `episode_000006` 头相机非递增时间戳 167 次，最大修正 133.490 ms。
- `episode_000004` 在输出帧 1911、相对裁剪起点约 95.55 秒处存在一次 25.507° 动作跳变，正式训练前建议检查该时刻视频。
- TCP 数据存在米/毫米量级和欧拉角环绕风险；当前关节空间数据集没有把 TCP 写入模型状态或动作。

### 3.2 F1_data_test

- 最终报告：`F1_data_test/quality_reports_final/20260903T064733549176Z_F1_data_test/quality_report.md`
- JSON：`F1_data_test/quality_reports_final/20260903T064733549176Z_F1_data_test/quality_report.json`
- 风险 CSV：`F1_data_test/quality_reports_final/20260903T064733549176Z_F1_data_test/risks.csv`
- 总体状态：`warning`
- 高风险失败项：`0`

| Episode | 裁剪范围 | 帧数 | 状态最大误差 | 动作最大误差 | 最大动作步长 | 跟随全局 P95 | 最差关节 P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `episode_000001` | 3.0～83.5 s | 1,611 | 10.855 ms | 43.351 ms | 21.658° | 2.085° | 3.877° |
| `episode_000002` | 公共完整范围 | 1,615 | 8.671 ms | 49.966 ms | 11.247° | 1.624° | 2.721° |

主要告警：

- `episode_000001` 头相机非递增时间戳 183 次，最大修正 133.492 ms。
- `episode_000002` 头相机非递增时间戳 127 次，最大修正 66.777 ms。
- 两条数据均存在 TCP 单位或欧拉角环绕风险，但不影响当前关节空间字段。
- 裁剪后的两条数据均没有超过 25° 的动作跳变。

## 4. LeRobot 转换结果

### 4.1 SOP 数据集

- 数据集目录：`converted/F1_data_SOP_openbox_lerobot`
- repo id：`local/f1_sop_openbox`
- Episode 数：2
- 总帧数：4,135
- Episode 帧数：2,033、2,102
- 转换状态：`completed`
- 转换报告：`converted/F1_data_SOP_openbox_lerobot/meta/f1_conversion.md`

### 4.2 test 数据集

- 数据集目录：`converted/F1_data_test_lerobot`
- repo id：`local/f1_test_openbox`
- Episode 数：2
- 总帧数：3,226
- Episode 帧数：1,611、1,615
- 转换状态：`completed`
- 转换报告：`converted/F1_data_test_lerobot/meta/f1_conversion.md`

## 5. 回读验证

两套数据均已通过 LeRobot `LeRobotDataset` 回读：

- FPS：20
- `observation.state`：16 维
- `action`：16 维
- `observation.images.head`：`3 × 720 × 1280`
- `observation.images.left_wrist`：`3 × 480 × 848`
- `observation.images.right_wrist`：`3 × 480 × 848`
- Episode index：均正确覆盖 0 和 1
- 12 个视频文件均通过 FFprobe 检查，编码为 H.264、20 FPS，分辨率正确。

## 6. 复现命令

SOP 最终质量评估：

```bash
uv run examples/f1/analyze_f1_data_quality.py \
  --raw-dir test_data/F1_data_SOP_openbox \
  --output-dir test_data/F1_data_SOP_openbox/quality_reports_final \
  --episode-config test_data/F1_data_SOP_openbox/episodes.conversion.json \
  --max-state-delta-ms 20 \
  --max-state-gap-ms 45
```

test 最终质量评估：

```bash
uv run examples/f1/analyze_f1_data_quality.py \
  --raw-dir test_data/F1_data_test \
  --output-dir test_data/F1_data_test/quality_reports_final \
  --episode-config test_data/F1_data_test/episodes.conversion.json
```

SOP 转换：

```bash
uv run examples/f1/convert_f1_mcap_to_lerobot.py \
  --raw-dir test_data/F1_data_SOP_openbox \
  --output-root test_data/converted/F1_data_SOP_openbox_lerobot \
  --repo-id local/f1_sop_openbox \
  --episode-config test_data/F1_data_SOP_openbox/episodes.conversion.json \
  --max-state-delta-ms 20 \
  --report-dir test_data/F1_data_SOP_openbox/conversion_reports
```

test 转换：

```bash
uv run examples/f1/convert_f1_mcap_to_lerobot.py \
  --raw-dir test_data/F1_data_test \
  --output-root test_data/converted/F1_data_test_lerobot \
  --repo-id local/f1_test_openbox \
  --episode-config test_data/F1_data_test/episodes.conversion.json \
  --report-dir test_data/F1_data_test/conversion_reports
```

如果重新执行转换且输出目录已经存在，需要先明确确认旧数据不再使用，再使用转换器的 `--overwrite` 参数。
