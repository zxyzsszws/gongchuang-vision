# 工创赛 · 智能搬运视觉系统（gongchuang-vision）

2026 年工程创新大赛「智能搬运」赛项的**视觉侧**软件：二维码扫码 → 任务解析 → 导航/校准/放置全流程状态机，经 UART 协议与机器人主控（**电控组负责，另一队伍**）通信。视觉不做底层运动控制，只输出任务、导航点与校准修正量。

- 协议文档（交电控组冻结）：[`docs/protocol-v0.1.md`](docs/protocol-v0.1.md)
- 设计规格：[`docs/superpowers/specs/2026-10-14-gongchuang-vision-design.md`](docs/superpowers/specs/2026-10-14-gongchuang-vision-design.md)

## 1. 环境安装

开发环境为 Windows + miniconda（Python 3.12，conda base 即可）：

```bash
conda activate base
pip install -r requirements.txt
```

依赖：`opencv-contrib-python`（**必须是 contrib 版**：扫码用 `cv2.wechat_qrcode_WeChatQRCode`，标准 opencv-python 没有该模块）、`numpy`、`pyyaml`、`pyserial`、`pytest`。

最终运行设备（Jetson Nano / x86）后定，所有硬件路径均已配置化，整个目录可拷贝迁移。

## 2. 运行测试

```bash
pytest -v
```

全部 45 项测试**无需任何硬件**：算法单测 + 校准精度回归（合成数据）+ 协议往返/粘包断包 + MockMcu 仿真端到端。`pytest.ini`/配置已设 `pythonpath=.`，在仓库根目录运行即可。

## 3. 三种运行模式

```bash
python -m src.main --config config/default.yaml --mode sim|replay|live [--max-seconds 120]
```

| 模式 | 链路 | 图像来源 | 需要的硬件 | 用途 |
|---|---|---|---|---|
| `sim` | MockMcu（内置 OPS 里程漂移 + 陀螺缓漂模型） | 合成感知（SimScene，基于 FieldMap 造观测） | **无** | 端到端跑通全流程状态机、验证校准闭环、CI |
| `replay` | SerialLink 真实串口 | `replay_path` 指定的视频文件（.mp4/.avi）或图片目录 | 需串口可打开（连主控或先占位）；无需相机 | 现场实拍素材回放调试识别与校准 |
| `live` | SerialLink 真实串口 | USBCamera（`camera.main_index`） | 相机 + 串口（主控） | 赛场实机运行与参数现调 |

`sim` 模式零硬件即可完整比赛流程直到 `DONE`；`live`/`replay` 的串口参数见 `serial` 配置节。运行结束打印 FSM 状态迁移轨迹，退出码 0=DONE。

## 4. 配置文件指南（`config/default.yaml`）

| 键 | 含义 |
|---|---|
| `mode` | 缺省运行模式（命令行 `--mode` 优先） |
| `camera` | `main_index` 主相机设备索引（live 用）、`qr_index`（预留）、`width/height/fps` |
| `serial` | `port`（如 COM3）、`baudrate`（115200，与协议一致勿改） |
| `timing` | 协议时序：`heartbeat_ms` 心跳周期、`heartbeat_timeout_s` 断链超时、`ack_timeout_ms`/`ack_retry` ACK 重发、`scan_timeout_s` 扫码超时（不可降级 → SAFESTOP） |
| `calib` | 校准：`stable_frames` 防抖帧数 K、`retry` 重试上限、`max_residual_mm/deg` 残差阈值（超限判误匹配丢弃）、`weight_middle/weight_pair_mid` 中环/左右环中点加权 |
| `color` | `min_area` 色块面积下限；`hsv_table` 1-6 号颜色的 HSV [下限,上限]（OpenCV H 范围 0-180）——**现场光照下必调** |
| `ring` | `geo` 半径分档→环编号映射；`hough.param1/param2` 霍夫阈值（param2 减小→更灵敏、伪检增多）；`spacing_*`/`collinear_tol_deg` 三色环分组几何约束 |
| `field_file` | FieldMap 场地图路径（**相对运行目录**，即在仓库根运行；赛前标定产物，见下节） |
| `log_dir` | 结构化运行日志目录，产出 `run_<日期_时间>_<毫秒>.jsonl` |
| `replay_path` / `sim_images` | replay 的视频/图片目录；sim 的循环图片源 |

## 5. 赛前现场标定 Checklist

校准精度直接决定搬运成败。赛前按以下顺序走一遍：

1. **打印样例场地图核对流程**：按 `config/field.example.yaml` 的结构（`tri.left/middle/right` 三色环理论圆心 mm + 半径 + `base_angle_deg`；`rings` 1-3 号数字环圆心/半径）打印一份样例场地示意图，确认量测口径（场地原点=扫码区角点，单位 mm）。
2. **拍摄实物**：在赛场对**三色环三联**与**数字环 1/2/3** 实拍照片/视频（覆盖比赛视角距离与光照），存到本地用作 `replay_path`。
3. **填实测坐标 + 现调参数**：复制 `field.example.yaml` 为赛场版本，把环的实测圆心/半径填进去，`field_file` 指向它；然后跑 `--mode live` 现场调参：
   - `color.hsv_table`：对着真实物料色块逐色调 HSV 上下限（光照影响最大）；
   - `ring.hough`：调 `param2`（灵敏度）与 `param1`（边缘阈值），必要时改 `ring.geo` 半径分档；
   - `calib.weight_*`：若中环刻度数字识别可靠则保持 0.7/0.3，否则调低 `weight_middle`。

## 6. 赛后日志复盘

每次运行在 `log_dir`（默认 `logs/`）生成 JSONL 结构化日志（时间/FSM 状态/事件）。复盘：

```bash
python tools/replay_log.py logs/run_xxx.jsonl
```

逐行打印 `t state event` 摘要，用于定位超时、SAFESTOP 触发点与校准残差异常。

## 7. 工程结构

```
config/            # default.yaml + field*.yaml（场地图）
src/algorithms/    # ① 纯函数层：qr_parse/color_detect/ring_detect/calibration（不碰硬件）
src/devices/       # ② 设备层：camera（USB/Video/ImageFolder）、link（SerialLink/MockMcu）
src/core/          # ③ 调度层：protocol（帧编解码）、kalman、fsm（比赛状态机）
src/main.py        # ④ 入口：--mode live|replay|sim
tests/             # pytest：算法单测 + 校准精度回归 + 协议往返 + sim 端到端
tools/             # replay_log.py 日志复盘等
docs/              # protocol-v0.1.md（电控冻结文档）+ specs/ 设计文档
```

两条硬性规则：算法层纯函数化（合成数据即可验证精度、不依赖真车）；MockMcu 是一等公民（电控代码就绪前即可跑通全链路，联调仅切换 `serial.port` 配置）。
