# 工创赛 · 智能搬运视觉系统（gongchuang-vision）

2026 年工程创新大赛「智能搬运」赛项的**视觉侧**软件。分工是「**视觉只测量、电控全规划**」，而且 V2.2.0 起改成**完全按需应答**：视觉平时一句话都不说，电控发一条请求才回一帧结果。

- 视觉只回答三个问题：任务计划是什么、转盘取料点此刻停着哪个颜色、把相机对准主圆需要怎么动；
- 视觉不规划路径、不下发目标点、不判断动作是否完成，也不主动上报任何东西——那些全部由电控负责。

- 协议文档（交电控组冻结）：[`docs/protocol-v2.2.0.md`](docs/protocol-v2.2.0.md)
- 通信全过程演示（浏览器直接打开）：[`docs/v2.2.0-comm-demo.html`](docs/v2.2.0-comm-demo.html)
- 上一版（已被取代，仅作历史）：[`docs/protocol-v2.1.0.md`](docs/protocol-v2.1.0.md)
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

全部 252 项测试**无需任何硬件**。覆盖范围：

- 协议层：帧编解码往返、粘包/断包/坏校验、CRC 与数据手册对齐；
- 文档对账：`docs/protocol-v2.2.0.md` 的表 1-4~1-12、任务长度、字节级样例必须与实现逐项一致；
- 算法层：主圆检出（轮廓法 + 充满度 + 去重）、按「离画面中心最近」挑主圆、对准解算的正解反查；
- 转盘停稳：两种策略（累计位移 / 边界圆）各自的判据与边界情况；
- 任务计划：二维码四组数字的分组规则、按颜色配对算第二批暂存区靶号；
- 端到端：MockMcu 按真实电控的方式问话（等不到结果就重发原帧）→ 视觉逐条作答；
- live 链路：合成图像 → 检出主圆/色块 → 外参换算 → 结果。

`pytest.ini` 已设 `pythonpath=.`，在仓库根目录运行即可。

> Windows 注意：若 `python` 是 Microsoft Store 的占位符，命令会静默失败（退出码 1、无输出）。请直接写 `.venv\Scripts\python.exe`。

## 3. 三种运行模式

```bash
python -m src.main --config config/default.yaml --mode sim|replay|live [--max-seconds 120]
```

| 模式 | 链路 | 图像来源 | 需要的硬件 | 用途 |
|---|---|---|---|---|
| `sim` | MockMcu（按顺序问三条任务） | 合成感知（SimScene：按「东西相对光轴在哪」直接合成三件测量） | **无** | 端到端验证「问→答→重发去重」、CI |
| `replay` | SerialLink 真实串口 | `replay_path` 指定的视频文件或图片目录 | 需串口可打开；无需相机 | 现场实拍素材回放调试识别 |
| `live` | SerialLink 真实串口 | USBCamera（`camera.main_index` 主圆 + `camera.qr_index` 扫码） | 相机 + 串口 + **已标定相机外参** | 赛场实机运行与参数现调 |

视觉单元是应答方，不判断比赛何时结束，因此没有「完成」这个终态。退出码只区分「运行到时限」与「进入安全停止」。

## 4. 通信要点（细节见协议文档）

只有两条命令，用任务码区分三件事：

| 命令码 | 方向 | data 长度 | 说明 |
|---|---|---|---|
| `0x0010` | 电控→视觉 | 1 / 1 / 5 | 请求，首字节是任务码，其余随任务而定 |
| `0x0011` | 视觉→电控 | 19 / 4 / 7 | 结果，前 3 字节固定为 `echo_seq`、任务码、状态码 |

三个任务码：

| 任务码 | 用途 | 请求 data | 结果 data |
|---|---|---|---|
| 1 | 扫码，要任务计划 | 任务码 | 3 + **16 字节计划** |
| 2 | 转盘取料点此刻的颜色 | 任务码 | 3 + 颜色 |
| 3 | 对准主圆，要 d 与 α | 任务码 + l(2) + α(2) | 3 + d(2) + α(2) |

结果长度对同一任务码是**固定**的：失败时缺的内容填 0，电控按任务码一次读满 N 字节即可，不必处理「正文短了」的分支。

状态码只有四个：`0 ST_OK`（成功）、`1 ST_NOT_READY`（没测到，可重发）、`2 ST_UNSOLVABLE`（测到了但无解，别沿用）、`3 ST_BAD_REQUEST`（实现不匹配，需要人查）。

**没有任何主动上报**：没有心跳、没有位姿、没有转盘模型、没有 done。V2.1.0 的 `0x0002`~`0x0008` 与更早的 `0x0001`、`0x0201`~`0x0203` 全部下架，收到时静默丢弃。

**只有一个重试者**：电控等不到结果就重发**原帧**（seq 与内容都不变），视觉见到同一个请求就把上次结果原样再发一遍。视觉从不主动重发。去重键是 `(seq, 任务码, 载荷)` 三元组——只看 seq 不够，seq 会回绕。

**视觉在电控重发时会继续等，不重置计时**。这一条是必需的：电控的重发间隔可能比视觉的处理时间短，若每次都重置计时，转盘永远等不到超时、扫码也永远等不到超时，表现为「视觉永远不回」。

### 转盘取料点为什么是「等待型」任务

电控把臂悬停在取料点上方之后发请求，此时转盘可能还在转，视觉要连续取帧、判定它停稳，才回颜色。停止判定完全由视觉看画面完成，电控**不需要也不应该**告知「现在已经停了」。

两种判定策略，用配置 `turntable.stop_strategy` 切换：

| 取值 | 判据 |
|---|---|
| `motion`（缺省） | 离取料点最近的那块物料，**相对计时起点的累计位移**持续小于 `tol_mm` 达到 `hold_s` 秒，且它离取料点不超过 `pick_tol_mm` |
| `containment` | 某块物料**完全落进**以取料点为圆心、半径 `物料半径 × containment_margin` 的边界圆内部 |

判据必须是「相对计时起点的累计位移」而不是「相邻两帧的位移」：转盘 6 秒一圈、工位半径 150mm 时，取料点附近线速度约 150mm/s，20ms 一帧只走 3mm。3mm 永远小于任何能容忍抖动的门限，转着的转盘会被判成停稳。

转盘上**没有已知尺寸的参考物**，所以这一条的像素到毫米换算只能靠相机外参的 `scale_mm_per_px`。没有有效外参时视觉一律回 `ST_NOT_READY`，不会用假设尺度硬算——那样颜色可能猜对，「离取料点多远」却全错，停稳判定也就不可信了。

### 对准主圆的解算

相机装在机械臂末端、正对地面。电控发任务码 3（伸出长度 l、臂 yaw α），视觉把像素偏移换毫米后，**先补上相机自身在电机坐标系里的位置**，再解：

```
目标 = (l·cos α, l·sin α) + R(α)·(画面量到的偏移)
α = asin(y / l)          横向只能靠臂转出来
d = x − l·cos α          前后靠车沿自身 x 挪
```

漏掉相机位置那一项是最容易犯的错：结果会稳定地少一整条臂长，看起来「有输出」，实际永远差那么远。

像素到毫米的**尺度不由电控上报**：主圆外圈直径是固定值，视觉量出它在画面里的像素直径即可反算（`mm/px = 外圈直径 ÷ 像素直径`）。量不到外圈（被画面裁掉、半径不足 2px）时退回相机外参里的 `scale_mm_per_px`，状态仍是 `ST_OK`、质量档降一级——量到主圆就算测到，只是尺度是估的。外圈与外参都没有时回 `ST_NOT_READY`：没有任何依据能把像素换成毫米，此时**不许**拿假设值硬算。

`d` 是**相对当前位置**的移动量。伸出长度比赛中固定死，所以「够不着」只能靠挪车补；`|y| > l` 时明确报 `ST_UNSOLVABLE`，不钳位——钳位会给出一个「看起来能到、实际差一截」的答案。

电控必须按 `echo_seq` 去重：同一个 `echo_seq` 只执行一次，否则视觉原样重发时，车会沿同一个 `d` 走两遍——这类错误不报错，只表现为位置错。

### 放置时只量一个主圆

V2.2.0 取消位姿纠正。放置原料时视觉只量画面中心附近**那一个主同心圆**，回「把相机对准它需要 d 与 α」；其余几个圆在哪、车身当前位姿如何，由电控用自己已知的场地几何与本次测量的主圆位置推算。

画面里可能有多个同心圆候选（相邻靶位的圆也会进画面），视觉的挑法是「**离画面中心最近的那个**」：电控已经把臂伸到目标上方，主圆必然靠近光轴。距离相同时取圈数多的那个。

## 5. 配置文件指南（`config/default.yaml`）

| 键 | 含义 |
|---|---|
| `mode` | 缺省运行模式（命令行 `--mode` 优先） |
| `camera` | `main_index` 主摄索引、`qr_index` 扫码相机索引（与主摄相同则复用）、`qr_model_dir` WeChatQRCode 模型目录、`width/height/fps` |
| `serial` | `port`（如 COM3 / `/dev/ttyTHS1`）、`baudrate`（默认 115200） |
| `camera_calib_file` | 相机外参（像素→毫米）。**live/replay 应当有效**：缺了程序照常启动，但任务 2（转盘颜色）一律回「没准备好」；任务 3（对准）仍靠主圆外圈自量尺度，见下节 |
| `sim` | `jitter_mm` 观测抖动、`turntable_omega01` 转盘角速度（0.1°/s 为单位，600 = 60°/s 即 6 秒一圈）、`turntable_stop_after_s` 多少秒后停住（留空 = 一直转） |
| `timing` | 与协议表 1-12 一一对应：`frame_interval_ms` 取帧间隔、`scan_timeout_s` 扫码超时、`turntable_timeout_s` 转盘超时、`max_bad_requests` 连续不合法上限 |
| `target` | 主圆检出参数：`search_radius` 半径搜索范围（像素）、`cluster_tol_px` 聚类容差、`min_rings` 至少几圈才认作靶、`min_circularity` 圆度门限、`min_fill` 充满度门限、`same_radius_ratio`/`dedup_px` 去重、**`ring_outer_mm` 外圈实际直径**（尺度唯一依据）、`threshold_block`/`threshold_c` 自适应阈值 |
| `turntable` | `stop_strategy` 停稳策略、`pick_point_mm` 取料点相对光轴的位置、`material_radius_mm` 物料半径、`tol_mm`/`hold_s`/`pick_tol_mm` motion 策略参数、`containment_margin` 边界圆系数 |
| `color` | `min_area` 色块面积下限；`hsv_table` 1-6 号颜色的 HSV [下限,上限]（OpenCV H 范围 0-180）——**现场光照下必调** |
| `log_dir` | 结构化运行日志目录，产出 `run_<日期_时间>_<毫秒>.jsonl` |
| `replay_path` / `sim_images` | replay 的视频/图片目录；sim 的循环图片源 |

## 6. 赛前现场标定 Checklist

1. **量主圆外圈直径**：填进 `target.ring_outer_mm`。**这是对准精度的唯一尺度来源**，量错了 d 就整体按比例偏。同时确认主圆在画面里的像素半径不低于 2px——太小尺度的相对误差会被放大。
2. **量物料直径**：填进 `turntable.material_radius_mm`。用 `containment` 策略时它直接决定边界圆半径；用 `motion` 策略时只用于日志。
3. **确定取料点在画面里的位置**：填进 `turntable.pick_point_mm`（相对光轴，mm）。缺省 `(0, 0)` 表示画面中心。
4. **量转盘停机时长**：据此确认 `timing.turntable_timeout_s` 是否盖得住一次完整等待。缺省 3s，转盘 6 秒一圈时偏短，现场要实测。
5. **标相机外参（必做，否则任务 2 一直答不出来）**：
   ```bash
   python tools/calibrate_camera.py \
       --pair 170,240,-150,0 --pair 470,240,150,0 \
       --out config/camera_calib.yaml
   ```
   给的 `mm_x, mm_y` 是**两个标定点在电机坐标系里的位置**（x 沿车头、y 朝车身外，mm）。V2.2.0 只用外参的**尺度**一项，所以真正要紧的是你量的两点间距与像素间距之比；原点取哪里不影响尺度，但两点必须量准且单位是 mm。工具打印每点残差，**>10mm 说明点配对错了或单位不是 mm**，别急着上车。标完把 `camera_calib_file` 指向这个文件——任务 2（转盘颜色）没有外参就永远答不出来，而程序不会因为缺外参而拒绝启动，现场很难分辨是「没标定」还是「没看到东西」。
6. **现调检出参数**：对着实拍画面调 `target.search_radius`（必须覆盖最小圈到最外圈）、`threshold_c`（光照不均时调大）、`min_rings`（杂散圆多时调大）。
7. **调颜色**：`color.hsv_table` 对着真实物料色块逐色调 HSV 上下限。
8. **扫码相机**：`camera.qr_index` 指向花雀广角相机索引；把 WeChatQRCode 的 4 个模型文件放好，`camera.qr_model_dir` 指向该目录——**留空会退化成 OpenCV 内建缺省模型，远距离小码识读率明显下降**。

## 7. 赛后日志复盘

运行时在 `log_dir`（默认 `logs/`）按 `run_<日期_时间>_<毫秒>.jsonl` 记录结构化日志。复盘：

```bash
python tools/replay_log.py logs/run_xxx.jsonl
```

逐行打印 `t state event` 摘要。V2.2.0 里值得盯的事件：

| 事件 | 含义 |
|---|---|
| `RESULT` | 回了一帧结果（任务码、状态码、`echo_seq`） |
| `RESULT_RESEND` | 同一个请求又来了，原样重发上次的答案 |
| `DUPLICATE_WHILE_BUSY` | 同一个请求在**还在处理时**又来了。出现得多说明电控重发间隔比视觉处理时间短，但不影响结果——计时没有被重置 |
| `PICK_COLOR` | 转盘停稳、认出颜色 |
| `TURNTABLE_TIMEOUT` / `SCAN_TIMEOUT` | 等到上限也没测到，回了 `ST_NOT_READY` |
| `ALIGN` | 对准解算完成（d、α、质量档） |
| `ALIGN_NO_RING` | 画面里没找到主圆，回了 `ST_NOT_READY` |
| `ALIGN_NO_SCALE` | 主圆找到了，但外圈量不出尺度、也没有外参兜底，回了 `ST_NOT_READY` |
| `PLAN_REJECTED` | 二维码读到了但内容不自洽，拒绝下发计划 |
| `PLAN_UNUSABLE` | 计划内容自洽但校验字节标了「不可用」，拒绝下发 |
| `BAD_REQUEST` / `PROTOCOL_MISMATCH` | 收到不合法请求；连续 5 个就进安全停止 |

## 8. 工程结构

```
config/            # default.yaml（全部参数）+ camera_calib*.yaml（相机外参）
src/algorithms/    # ① 纯函数层：qr_parse/qr_decode/color_detect/ring_detect/turntable/arm/geometry
src/devices/       # ② 设备层：camera（USB/Video/ImageFolder/多摄）、vision（感知管线）、sim_scene、link
src/core/          # ③ 调度层：crc、protocol（帧编解码与任务码）、taskplan（二维码语义）、fsm（状态机）
src/main.py        # ④ 入口：--mode live|replay|sim
tests/             # pytest：算法单测 + 协议/文档对账 + 端到端（sim 与合成图 live 链路）
tools/             # calibrate_camera.py 外参标定、replay_log.py 日志复盘
docs/              # protocol-v2.2.0.md（电控冻结文档；v2.1.0/v2.0.0 保留作历史）
                   # + v2.2.0-comm-demo.html（通信全过程演示）
                   # + arm-comm-demo.html / arm-motion-demo.html（V2.1.0 几何演示，含作废提示）
                   # + superpowers/specs 设计文档
```

两条硬性规则：算法层纯函数化（合成数据即可验证精度、不依赖真车）；MockMcu 是一等公民（电控代码就绪前即可跑通全链路，联调仅切换 `serial.port` 配置）。

`src/algorithms/arm.py` 只算 `d` 与 `α`，`src/algorithms/turntable.py` 只判「停稳了没有、是什么颜色」。位姿解算相关的 `localize.py`、`fieldmap.py` 与场地图配置已在 V2.2.0 全部删除——视觉不再需要知道自己在场地哪里。

## 9. 已知限制（如实记录）

- **相机外参要标定，但缺了也能启动**：缺 `camera_calib_file` 时程序照常跑起来。此时 **任务 2（转盘颜色）一律回 `ST_NOT_READY`**——转盘上没有已知尺寸物，尺度只能来自外参。**任务 3（对准主圆）照常工作**，因为它的尺度是从主圆外圈实测反算的（`target.ring_outer_mm`），只有外圈也量不到时才回 `ST_NOT_READY`（日志 `ALIGN_NO_SCALE`）。**不会**静默拿 1.0mm/px 去算——那样像素当毫米发出去会让电控把车开歪，而且不报错。标定只需量尺子（两点即可），不需要棋盘格与相机内参。
- **所有真实尺寸尚未实测**：`target.ring_outer_mm`（缺省 120mm）、`turntable.material_radius_mm`（缺省 25mm）、`turntable.pick_point_mm`（缺省画面中心）目前都是占位值。**主圆外圈直径是对准精度的唯一尺度来源**，填错会让 `d` 整体按比例偏；物料半径与取料点直接影响转盘颜色判定。这三项必须在场地实测后替换。
- **转盘识别未在真实转盘上验证**：链路是通的（色块检出 → 换毫米 → 停稳判定 → 选颜色），但只在合成场景里跑过。真实工况的两个已知风险：物料侧面反光会让 HSV 阈值不稳；转盘停机后可能仍有极慢的爬行，`tol_mm` 与 `hold_s` 需要现场调。
- **未在 Jetson 上复测**：目标机当时不可用，本地 Windows 全绿（252 项）不等于目标机全绿。下次可连时先跑 `pytest -q` 与 `--mode sim`。
- **相机读取为同步单线程**：`CameraSource.read()` + 检测在 `fsm.step` 内联执行，未做相机线程→帧队列。慢帧期间取帧间隔会顺延，转盘停稳判定的实际帧率低于 `frame_interval_ms` 的名义值。
- **结构化 JSONL 日志尚未接线**：`setup_logging()` 已装配 handler，但业务代码仍只往 `log_events` 里记事件，`logs/*.jsonl` 目前是空文件——复盘工具读它只能读到空。
- **尺度只在主圆外圈可见时是实测值**：外圈被画面裁掉或像素半径不足 2px 时退回 `camera_calib` 里的固定 `scale_mm_per_px`，此时状态仍是 `ST_OK`、质量档降一级；外参与外圈都没有时回 `ST_NOT_READY`。电控看不到「实测还是估的」这一差别（V2.2.0 的结果体只有 d 与 α），只在日志的 `ALIGN` 事件（第 5 项是质量档）里体现。
- **转盘颜色与对准不能同时做**：一次只处理一条请求，新请求会丢掉还没完成的旧作业。电控要连续问两件事必须等前一条答完。
