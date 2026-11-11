# 工创赛 · 智能搬运视觉系统（gongchuang-vision）

2026 年工程创新大赛「智能搬运」赛项的**视觉侧**软件：二维码扫码 → 任务解析 → 导航/校准/放置全流程状态机，经 UART 协议与机器人主控（**电控组负责，另一队伍**）通信。视觉不做底层运动控制，只输出任务、导航点与校准修正量。

- 协议文档（交电控组冻结）：[`docs/protocol-v2.0.0.md`](docs/protocol-v2.0.0.md)（视觉只输出测量结果，路径规划与运动控制全部由电控负责；帧格式与校验算法对齐《RoboMaster 2027 机甲大师高校系列赛通信协议 V1.0.0》）
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

全部 132 项测试**无需任何硬件**：算法单测 + 校准精度回归（合成数据）+ 协议往返/粘包断包/坏校验/文档示例对账 + CRC 数据手册对齐 + MockMcu 仿真端到端（含校准闭环收敛断言）+ **live 视觉链路合成图端到端**（投影三色环 → 霍夫检出 → 外参换算 → 校准修正量方向正确）。`pytest.ini`/配置已设 `pythonpath=.`，在仓库根目录运行即可（也从任意目录可运行，见 `tests/conftest.py`）。

> Windows 注意：若 `python` 是 Microsoft Store 的占位符，命令会静默失败（退出码 1、无输出）。用 `conda activate base` 后的 python，或直接写 `C:\Users\<user>\miniconda3\python.exe`。

## 3. 三种运行模式

```bash
python -m src.main --config config/default.yaml --mode sim|replay|live [--max-seconds 120]
```

| 模式 | 链路 | 图像来源 | 需要的硬件 | 用途 |
|---|---|---|---|---|
| `sim` | MockMcu（内置 OPS 里程漂移 + 陀螺缓漂模型，参数见 `sim` 配置节） | 合成感知（SimScene：FieldMap 理论环 × **当前主控位姿误差**刚体变换后的 field-as-seen + ±3mm 抖动） | **无** | 端到端跑通全流程状态机、验证校准闭环、CI |
| `replay` | SerialLink 真实串口 | `replay_path` 指定的视频文件（.mp4/.avi）或图片目录 | 需串口可打开（连主控或先占位）；无需相机 | 现场实拍素材回放调试识别与校准 |
| `live` | SerialLink 真实串口 | USBCamera（`camera.main_index` 环识别 + `camera.qr_index` 扫码） | 相机 + 串口（主控）+ **已标定相机外参** | 赛场实机运行与参数现调 |

`sim` 模式零硬件即可完整比赛流程直到 `DONE`；`live`/`replay` 的串口参数见 `serial` 配置节。运行结束打印 FSM 状态迁移轨迹，退出码 0=DONE。

导航目标是**真实的**：GOTO 坐标由二维码批次放置位置首位匹配 `field_file` 的 FieldMap 数字环心得出（无匹配环回退原点，RETURN 恒回起点 (0,0)）。`sim` 校准闭环为真闭环：SimScene 按 MockMcu「上报−标称」当前位姿误差合成观测，e2e 测试断言两次 CAL_APPLIED 的修正量达到漂移量级且应用后误差收缩过半、int16 钳位次数为 0（虚拟时钟驱动，完全确定可复现）。

## 4. 配置文件指南（`config/default.yaml`）

| 键 | 含义 |
|---|---|
| `mode` | 缺省运行模式（命令行 `--mode` 优先） |
| `camera` | `main_index` 主相机设备索引（live 用）、`qr_index` 扫码相机索引（与主摄相同则复用）、`qr_model_dir` WeChatQRCode 模型目录、`width/height/fps` |
| `serial` | `port`（如 COM3 / Linux 下 `/dev/ttyTHS1`）、`baudrate`（默认 115200，须与主控一致；协议本身与波特率无关） |
| `camera_calib_file` | 相机外参文件（像素→车体毫米），**live/replay 必填**，见下节标定流程 |
| `sim` | sim 模式 MockMcu 漂移模型：`drift_mm_per_m` 里程漂移、`drift_deg_per_m` 航向缓漂（默认 8mm/m 使整场标称误差留在 int16 里程量程内） |
| `timing` | 协议时序：`heartbeat_ms` 心跳周期、`heartbeat_timeout_s` 断链超时、`ack_timeout_ms`/`ack_retry` ACK 重发、`scan_timeout_s` 扫码超时（不可降级 → SAFESTOP） |
| `calib` | 校准：`stable_frames` 防抖帧数 K、`retry` 重试上限、`max_residual_mm/deg` 残差阈值（超限判误匹配丢弃）、`weight_middle/weight_pair_mid` 中环/左右环中点加权 |
| `color` | `min_area` 色块面积下限；`hsv_table` 1-6 号颜色的 HSV [下限,上限]（OpenCV H 范围 0-180）——**现场光照下必调** |
| `ring` | `geo` 半径分档→环编号映射（数字环用）；`hough.param1/param2` 霍夫阈值（param2 减小→更灵敏、伪检增多）；`search_radius` 霍夫搜索半径范围（像素，须覆盖三色环与数字环）；`spacing_px_*`/`max_spacing_ratio`/`radius_tol`/`collinear_tol_deg` 三色环分组几何约束 |
| `field_file` | FieldMap 场地图路径（**相对运行目录**，即在仓库根运行；赛前标定产物，见下节） |
| `log_dir` | 结构化运行日志目录，产出 `run_<日期_时间>_<毫秒>.jsonl` |
| `replay_path` / `sim_images` | replay 的视频/图片目录；sim 的循环图片源 |

## 5. 赛前现场标定 Checklist

校准精度直接决定搬运成败。赛前按以下顺序走一遍：

1. **打印样例场地图核对流程**：按 `config/field.example.yaml` 的结构（`tri.left/middle/right` 三色环理论圆心 mm + 半径 + `base_angle_deg`；`rings` 1-3 号数字环圆心/半径）打印一份样例场地示意图，确认量测口径（场地原点=扫码区角点，单位 mm）。
2. **拍摄实物**：在赛场对**三色环三联**与**数字环 1/2/3** 实拍照片/视频（覆盖比赛视角距离与光照），存到本地用作 `replay_path`。
3. **填实测坐标 + 现调参数**：复制 `field.example.yaml` 为赛场版本，把环的实测圆心/半径填进去，`field_file` 指向它；**先标相机外参**，再跑 `--mode live` 调参：
   - **相机外参（必做，否则 live 拒绝启动）**：量出车体坐标系下两个可辨认点的 mm 坐标（原点=相机光轴正下方地面点，车头 +y、右侧 +x），在图上点出对应像素坐标，跑：
     ```bash
     python tools/calibrate_camera.py \
         --pair 170,240,-150,0 --pair 470,240,150,0 \
         --out config/camera_calib.yaml
     ```
     工具打印每点残差，**>10mm 说明点配对错了或单位不是 mm**，别急着上车。远近都要看三色环时加 `--auto-scale-from-tri`（尺度自估，只需量准安装角与偏移）。
   - `color.hsv_table`：对着真实物料色块逐色调 HSV 上下限（光照影响最大）；
   - `ring.hough`：调 `param2`（灵敏度）与 `param1`（边缘阈值），必要时改 `ring.geo` 半径分档；
   - `ring.spacing_px_min/max`：**像素**间距门限，对着实拍画面量三联在图像里的间距（换相机/改安装必重标）；
   - `calib.weight_*`：若中环刻度数字识别可靠则保持 0.7/0.3，否则调低 `weight_middle`。
4. **扫码相机**：`camera.qr_index` 指向花雀广角相机索引（与主摄相同则复用主摄）；把 WeChatQRCode 的 4 个模型文件放好，`camera.qr_model_dir` 指向该目录——**留空会退化成 OpenCV 内建缺省模型，远距离小码识读率明显下降**。

## 6. 赛后日志复盘

每次运行在 `log_dir`（默认 `logs/`）生成 JSONL 结构化日志（时间/FSM 状态/事件）。复盘：

```bash
python tools/replay_log.py logs/run_xxx.jsonl
```

逐行打印 `t state event` 摘要，用于定位超时、SAFESTOP 触发点与校准残差异常。

## 7. 工程结构

```
config/            # default.yaml + field*.yaml（场地图）+ camera_calib*.yaml（相机外参）
src/algorithms/    # ① 纯函数层：qr_parse/color_detect/ring_detect/calibration/geometry（不碰硬件）
src/devices/       # ② 设备层：camera（USB/Video/ImageFolder/多摄）、vision（感知管线）、link（SerialLink/MockMcu）
src/core/          # ③ 调度层：crc（CRC8/CRC16）、protocol（帧编解码）、kalman、fsm（比赛状态机）
src/main.py        # ④ 入口：--mode live|replay|sim
tests/             # pytest：算法单测 + 校准精度回归 + 协议往返/文档对账 + sim 端到端 + live 合成图端到端
tools/             # calibrate_camera.py 外参标定、replay_log.py 日志复盘
docs/              # protocol-v2.0.0.md（电控冻结文档）+ superpowers/specs 设计文档
```

两条硬性规则：算法层纯函数化（合成数据即可验证精度、不依赖真车）；MockMcu 是一等公民（电控代码就绪前即可跑通全链路，联调仅切换 `serial.port` 配置）。

## 8. 已知限制（如实记录，联调/Phase-4 前必读）

- **相机外参必填，无外参拒绝启动**：live/replay 的观测必须换算到**场地毫米**。换算链路是「像素 →(相机安装外参)→ 车体毫米 →(主控上报位姿)→ 场地毫米」。缺 `camera_calib_file` 时 `main` 以退出码 2 明确报错，**不会**静默降级——因为此时发出去的修正量方向会错。标定只需量尺子（`tools/calibrate_camera.py`，两点即可），不需要棋盘格与相机内参；远近跨度大时开 `auto_scale_from_tri`，尺度由三色环半径实时自估，只需量准安装角与偏移。
- **三色环分组门限是像素量**：`ring.spacing_px_*` 指图像里的像素间距，随安装高度/焦距变化，**换相机或改安装必须重标**。数字环按半径带分编号（`ring.geo`），三色环靠「等半径 + 共线 + 间距」与之区分，两类不再串味。
- **相机读取为同步单线程**：live/replay 的 `CameraSource.read()` + 检测在 `fsm.step` 内联执行，未实现 spec §2.5 的相机线程→帧队列模型（控制端裁决：F7 有意推迟——线程化需真实相机才能验证，慢帧期间心跳/ACK 时序会顺延）。
- **结构化 JSONL 日志尚未接线**：`setup_logging()` 已装配 handler，但业务代码仍只往 `log_events` 里记事件，`logs/*.jsonl` 目前是空文件，`tools/replay_log.py` 因此无内容可读（Phase-4 B 项）。
- **颜色识别与 PICK 未接入流程**：`detect_colors` 与 `PICK(0x0006)` 已实现但 FSM 不使用，原料区转盘的颜色/顺序识别与取料动作待电控联调后接（Phase-4 C 项）。
- **逐件精细放置偏移属 Phase-4**：PLACE 的 `fine_x/fine_y` 当前恒发 0，导航只用批次位置首位对应的环心（`docs/protocol-v2.0.0.md` 表 1-10）。

