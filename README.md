# 工创赛 · 智能搬运视觉系统（gongchuang-vision）

2026 年工程创新大赛「智能搬运」赛项的**视觉侧**软件。分工是「**视觉只测量、电控全规划**」：

- 视觉自己用三个同心圆参考靶解出车身在场地中的位姿，通过串口把**测量结果**发给机器人主控；
- 视觉不规划路径、不下发目标点、不判断动作是否完成——那些全部由电控负责。

- 协议文档（交电控组冻结）：[`docs/protocol-v2.1.0.md`](docs/protocol-v2.1.0.md)
- 设计规格：[`docs/superpowers/specs/2026-10-14-gongchuang-vision-design.md`](docs/superpowers/specs/2026-10-14-gongchuang-vision-design.md)

## 1. 环境安装

开发环境为 Windows + Python 3.12（仓库内已带 `.venv`）：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖：`opencv-contrib-python`（**必须是 contrib 版**：扫码用 `cv2.wechat_qrcode_WeChatQRCode`，标准 opencv-python 没有该模块）、`numpy`、`pyyaml`、`pyserial`、`pytest`。

最终运行设备（Jetson Orin）见 [`docs/deploy-jetson.md`](docs/deploy-jetson.md)。所有硬件路径均已配置化，整个目录可拷贝迁移。

## 2. 运行测试

```powershell
.venv\Scripts\python.exe -m pytest -q
```

全部 226 项测试**无需任何硬件**。覆盖范围：

- 协议层：帧编解码往返、粘包/断包/坏校验、CRC 与数据手册对齐；
- 文档对账：`docs/protocol-v2.1.0.md` 的表 1-2~1-5、命令长度、字节级样例必须与实现逐项一致；
- 算法层：三点解位姿、180° 镜像歧义消解、同心圆检出（轮廓法 + 充满度 + 去重）；
- 任务计划：二维码四组数字的分组规则、按颜色配对算第二批暂存区靶号；
- 端到端：MockMcu 带里程漂移 → 视觉看见靶 → 电控估计误差清零；
- live 链路：按真值渲染靶 → 检出 → 外参换算 → 解位姿，结果等于真值。

`pytest.ini` 已设 `pythonpath=.`，在仓库根目录运行即可。

> Windows 注意：若 `python` 是 Microsoft Store 的占位符，命令会静默失败（退出码 1、无输出）。请直接写 `.venv\Scripts\python.exe`。

## 3. 三种运行模式

```bash
python -m src.main --config config/default.yaml --mode sim|replay|live [--max-seconds 120]
```

| 模式 | 链路 | 图像来源 | 需要的硬件 | 用途 |
|---|---|---|---|---|
| `sim` | MockMcu（带里程漂移，按场地图的航点连线行驶） | 合成感知（SimScene：按场景真位姿投影三靶 + 抖动量） | **无** | 端到端验证「测量→上报→误差清零」、CI |
| `replay` | SerialLink 真实串口 | `replay_path` 指定的视频文件或图片目录 | 需串口可打开；无需相机 | 现场实拍素材回放调试识别 |
| `live` | SerialLink 真实串口 | USBCamera（`camera.main_index` 靶识别 + `camera.qr_index` 扫码） | 相机 + 串口 + **已标定相机外参** | 赛场实机运行与参数现调 |

视觉单元是测量仪器，不判断比赛何时结束，因此没有「完成」这个终态。退出码只区分「运行到时限」与「进入安全停止」。

## 4. 通信要点（细节见协议文档）

视觉 → 电控共 4 条：

| 命令码 | 名称 | 长度 | 何时发 |
|---|---|---|---|
| `0x0002` | 任务计划 | 16B | 扫码成功后发一次，必须得到应答 |
| `0x0003` | 车体位姿 | 9B | 每 50ms，仅当当前区域三个靶都可见 |
| `0x0004` | 转盘模型 | 16B | 每 50ms，仅当转盘可见 |
| `0x0008` | 臂解算结果 | 10B | 收到 `0x0006` 后立即回 |

电控 → 视觉共 2 条：

| 命令码 | 名称 | 长度 | 何时发 |
|---|---|---|---|
| `0x0005` | 去向提示 | 1B | 目标区域改变时发一次 |
| `0x0006` | 臂状态 | 7B | 请求解算时发，发后必须静止 |

双向 1 条：`0x0007` 应答（2B，`ack_seq` + 结果字节），既是任务计划的应答，也是解算的 done。

**只有两类帧需要应答**：任务计划（等 `0x0007`）与臂解算（`0x0006` 由 `0x0008` 应答，`0x0008` 由 `0x0007` 应答）。位姿、转盘、去向提示一律不应答——它们每 50ms 一帧，每帧都回 ack 会让一半带宽变成纯应答帧。

**本版本没有心跳**。周期测量帧本身就是链路活跃的证据；而电控是事件驱动的，停着作业时可能十几秒不发一帧，把静默当断链只能得到误报。

`0x0001`（心跳）与 `0x0201`/`0x0202`/`0x0203`（位姿上报、校准修正、动作完成）均已删除，收到静默丢弃。

### 机械臂目标解算

相机装在机械臂末端、正对地面。电控先发 `0x0006`（伸出长度 l、相机高度 h、臂 yaw α、目标编号），视觉据此把像素换算到**电机坐标系**（原点在车身右侧边中点即臂转轴，+x 沿车头，+y 朝车身外），读出目标点 (x, y) 后解：

```
α = asin(y / l)          横向只能靠臂转出来
d = x − l·cos α          前后靠车沿自身 x 挪
```

`d` 是**相对当前位置**的移动量。伸出长度比赛中固定死，所以「够不着」只能靠挪车补；`|y| > l` 时明确报「横向够不着」（状态字节 bit0~1 = 1），不钳位——钳位会给出一个「看起来能到、实际差一截」的答案。

`0x0008` 带回 `echo_seq`（对应 `0x0006` 的 seq）。电控必须按它去重：同一个 `echo_seq` 只执行一次，否则视觉因等不到 done 而重发时，车会沿同一个 `d` 走两遍——这类错误不报错，只表现为位置错。

### 位姿是怎么解出来的

1. 相机看图，检出同心圆靶，聚成靶心（像素）；
2. 用**相机安装外参**把像素换算成**车身毫米**（只换一次，停在车身坐标系）；
3. 用当前区域的标称入场位姿预测三个靶应在车身坐标的什么位置，从所有候选里挑最贴合的三个（画面里可能同时混着另一组靶和转盘上的圆；挑出的靶离预测位置超过 `target.max_match_mm` 就判定「本区域靶不在画面里」，一帧都不发）；
4. 拿这三个靶的车身坐标与场地图里的三个靶心做相似变换拟合，解出车在场地中的位姿。

三点共线，观测顺序整体反向会得到「朝向差 180°、残差同样为 0」的另一个解，几何上无法区分。消歧优先级：①靶心数字识别；②上一帧已解出的位姿（时间连续）；③场地图里该区域的 `entry_pose`（赛前实测的标称入场位姿）。换区域时上一帧位姿必须丢弃。

## 5. 配置文件指南（`config/default.yaml`）

| 键 | 含义 |
|---|---|
| `mode` | 缺省运行模式（命令行 `--mode` 优先） |
| `camera` | `main_index` 主摄索引、`qr_index` 扫码相机索引（与主摄相同则复用）、`qr_model_dir` WeChatQRCode 模型目录、`width/height/fps` |
| `serial` | `port`（如 COM3 / `/dev/ttyTHS1`）、`baudrate`（默认 115200） |
| `camera_calib_file` | 相机外参（像素→车身毫米），**live/replay 必填**，见下节 |
| `sim` | `drift_mm_per_m` 里程漂移率、`drift_deg_per_m` 航向缓漂率、`jitter_mm` 观测抖动、`speed_mm_s` 行驶速度、`turntable_omega01` 转盘角速度（0.1°/s 为单位，600 = 60°/s 即 6 秒一圈） |
| `timing` | 与协议表 1-15 一一对应：`ack_timeout_ms`、`exec_timeout_ms`、`ack_retry`、`pose_interval_ms`、`turntable_interval_ms`、`scan_timeout_s` |
| `target` | 同心圆检出参数：`search_radius` 半径搜索范围（像素）、`cluster_tol_px` 聚类容差、`min_rings` 至少几圈才认作靶、`min_circularity` 圆度门限、`min_fill` 充满度门限、`same_radius_ratio`/`dedup_px` 去重、`max_match_mm` 选靶允许的预测偏差、`threshold_block`/`threshold_c` 自适应阈值 |
| `color` | `min_area` 色块面积下限；`hsv_table` 1-6 号颜色的 HSV [下限,上限]（OpenCV H 范围 0-180）——**现场光照下必调** |
| `field_file` | 场地图路径（**相对运行目录**，即在仓库根运行） |
| `log_dir` | 结构化运行日志目录，产出 `run_<日期_时间>_<毫秒>.jsonl` |
| `replay_path` / `sim_images` | replay 的视频/图片目录；sim 的循环图片源 |

### 场地图（`config/field.example.yaml`）

| 键 | 含义 |
|---|---|
| `reference_groups` | 两组（`rough`/`staging`）各三个同心圆靶的场地圆心、最外圈半径、圈数 |
| `entry_pose` | 该区域的标称入场位姿，**只用于消解 180° 歧义**，不写则默认 `(0,0,0)` |
| `turntable` | 转盘圆心、工位回转半径、工位数 |
| `start` | 开赛停车位（场地原点即定义在此处的机械臂底座中心投影） |
| `region_poses` | 各区域标称停车位姿。**只在 sim 里用来生成行驶路线**，真实系统不读 |
| `palletizing` | 码垛层高，供电控控制机械臂升降 |

赛前必须实测替换：三靶间距越大，朝向解算越准（间距 600mm 时约 0.06°，间距 300mm 时约 0.12°）。

## 6. 赛前现场标定 Checklist

1. **量三个靶**：把两组各三个靶的场地圆心、外圈半径、圈数填进 `field_file`。三个靶必须共线，加载时会校验，不共线直接报错。
2. **量标称入场位姿**：车正常停在该区域作业位置时，机械臂底座中心在地面的投影相对场地原点的 `x/y` 与车头朝向。写进 `entry_pose`——它只用于消歧，写错会导致整场朝向反 180°。
3. **标相机外参（必做，否则 live 拒绝启动）**：
   ```bash
   python tools/calibrate_camera.py \
       --pair 170,240,-150,0 --pair 470,240,150,0 \
       --out config/camera_calib.yaml
   ```
   原点取**机械臂底座中心在地面的投影**，不是相机光轴（协议附录二第四节）。工具打印每点残差，**>10mm 说明点配对错了或单位不是 mm**，别急着上车。
4. **现调检出参数**：对着实拍画面调 `target.search_radius`（必须覆盖最小圈到最外圈）、`threshold_c`（光照不均时调大）、`min_rings`（杂散圆多时调大）。`target.max_match_mm` 控制「实测靶心离预测位置多远还认」。两组靶相距约 1.8m，取 500mm 已很宽松；若现场停位偏差确实更大，再往上调。
5. **调颜色**：`color.hsv_table` 对着真实物料色块逐色调 HSV 上下限。
6. **扫码相机**：`camera.qr_index` 指向花雀广角相机索引；把 WeChatQRCode 的 4 个模型文件放好，`camera.qr_model_dir` 指向该目录——**留空会退化成 OpenCV 内建缺省模型，远距离小码识读率明显下降**。

## 7. 赛后日志复盘

每次运行在 `log_dir`（默认 `logs/`）生成 JSONL 结构化日志（时间/FSM 状态/事件）。复盘：

```bash
python tools/replay_log.py logs/run_xxx.jsonl
```

逐行打印 `t state event` 摘要，用于定位超时、SAFESTOP 触发点与位姿残差异常。

## 8. 工程结构

```
config/            # default.yaml + field*.yaml（场地图）+ camera_calib*.yaml（相机外参）
src/algorithms/    # ① 纯函数层：qr_parse/qr_decode/color_detect/ring_detect/localize/fieldmap/geometry
src/devices/       # ② 设备层：camera（USB/Video/ImageFolder/多摄）、vision（感知管线）、sim_scene、link
src/core/          # ③ 调度层：crc、protocol（帧编解码）、taskplan（二维码语义）、fsm（状态机）
src/main.py        # ④ 入口：--mode live|replay|sim
tests/             # pytest：算法单测 + 协议/文档对账 + 端到端（sim 与合成图 live 链路）
tools/             # calibrate_camera.py 外参标定、replay_log.py 日志复盘
docs/              # protocol-v2.1.0.md（电控冻结文档；v2.0.0 保留作历史）+ superpowers/specs 设计文档
```

两条硬性规则：算法层纯函数化（合成数据即可验证精度、不依赖真车）；MockMcu 是一等公民（电控代码就绪前即可跑通全链路，联调仅切换 `serial.port` 配置）。

## 9. 已知限制（如实记录）

- **相机外参必填，无外参拒绝启动**：缺 `camera_calib_file` 时 `main` 以退出码 2 明确报错，**不会**静默降级——像素当毫米发出去会让电控把车开歪。标定只需量尺子（两点即可），不需要棋盘格与相机内参。
- **真实转盘识别尚未实现**：`VisionPerceive.turntable()` 显式返回 `None`，FSM 因此不发 `0x0004`。sim 里的转盘数据来自 SimScene 的合成模型，不是识别结果，不能当联调依据。
- **靶心数字识别尚未实现**：`0x0003` 的质量字节里「编号来源」目前只会是「按入场方向约定推定」，不会是「读到数字」。三个靶结构完全一致且都印了数字，实现后能进一步提高消歧可靠性。
- **相机读取为同步单线程**：`CameraSource.read()` + 检测在 `fsm.step` 内联执行，未做相机线程→帧队列。慢帧期间心跳/ACK 时序会顺延。
- **结构化 JSONL 日志尚未接线**：`setup_logging()` 已装配 handler，但业务代码仍只往 `log_events` 里记事件，`logs/*.jsonl` 目前是空文件。
- **选靶依赖标称入场位姿**：`targets()` 用该区域 `entry_pose` 预测三个靶应落在车身坐标的什么位置，再从画面所有候选里挑最贴合的三个。若 `entry_pose` 写错太多（> `target.max_match_mm`），视觉会一直判定「本区域的靶不在画面里」而不发位姿。它**只影响挑哪三个**，不影响解算精度。
- **转盘不能当校准参考物**：转盘上的物料在动，无法作为固定参考。位姿校准只在粗加工区/暂存区做，靠三个共线圆；单个圆无法定出朝向。
