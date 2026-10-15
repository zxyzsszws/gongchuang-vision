# 工创赛智能搬运 · 视觉系统设计（2026-10-14）

状态：已与视觉组逐节确认。需求分析见飞书文档《工创赛视觉》(SO76dPyh9oHOJtxVYY7cwPGbnXf)。

## 0. 背景与前提

- 无 2025 完整代码可复用，按 2025 开源方案思路重建（CSDN 三篇博文仅作参考素材）。
- 开发环境：本机 Windows；最终运行设备（Jetson Nano / x86）后定 → 所有硬件路径配置化，整体目录可拷贝迁移。
- 本轮范围：完整视觉系统（扫码、颜色识别、圆环识别、视觉校准、串口通信、状态机），分阶段实现。
- 相机在手可实时调试，同时保留图片/视频回放通道。
- 串口协议由视觉侧起草 v0.1，联调前与电控组冻结。
- 架构选型：方案 A「单进程多线程 + 四层模块」（对比过多进程组件化=过度设计、单体脚本=不可维护）。

## 1. 工程结构与分层

```
gongchuang-vision/
├── config/
│   └── default.yaml          # 相机索引/分辨率、串口路径/波特率、HSV阈值、环几何参数、模式选择
├── src/
│   ├── algorithms/           # ① 纯函数层：不 import 硬件相关库，只进图像/坐标、只出 dataclass
│   │   ├── qr_parse.py       #   4组三位数解析 → 任务结构体
│   │   ├── color_detect.py   #   HSV 阈值 + 面积过滤 → 物料色块（6色映射 1-6）
│   │   ├── ring_detect.py    #   CLAHE + 形态学梯度 + 霍夫圆 → 圆环(圆心/半径/编号)
│   │   └── calibration.py    #   校准数学：左右环相减→Z轴；中环vs左右环中点加权→OPS位置
│   ├── devices/              # ② 设备层：接口 + 真实/模拟双实现
│   │   ├── camera.py         #   CameraSource 接口；USBCamera / VideoFile / ImageFolder
│   │   └── link.py           #   McuLink 接口；SerialLink（pyserial）/ MockMcu（OPS漂移模拟器）
│   ├── core/                 # ③ 调度层
│   │   ├── protocol.py       #   帧编解码 + 流式解析
│   │   ├── kalman.py         #   坐标卡尔曼滤波
│   │   └── fsm.py            #   比赛状态机
│   └── main.py               # ④ 入口：--mode live|replay|sim
├── tests/                    # pytest：算法单测 + 校准精度回归 + 协议往返
├── tools/                    # 调试：标注回放、校准曲线绘图、协议抓包打印、日志复盘
└── requirements.txt          # opencv-contrib-python(含WeChatQRCode), numpy, pyyaml, pyserial, pytest
```

两条硬性规则：

1. `algorithms/` 禁止触碰硬件与全局状态（纯函数），使校准精度可在单机用合成数据验证，不依赖真车。
2. `MockMcu` 是一等公民：按协议帧模拟主控，内置「OPS 随里程线性漂移 + 陀螺仪缓漂」模型；电控代码就绪前即可跑通状态机与校准闭环，联调时仅配置切换 Serial/Mock 实现。

## 2. 核心数据结构与模块接口

### 2.1 数据结构（dataclass，尽量 frozen）

- `TaskPlan`: batch1[Color], batch2[Color], rough_pos1, staging_pos, rough_pos2（放置环编号 1-3）
- `RingObs`: center(x,y), radius, label|None
- `TriRingObs`: left/middle/right: RingObs（三色环三联观测）
- `Pose`: x, y, heading（场地坐标，mm / °）
- `CalibResult`: pose_before, pose_after, residual
- `Frame`: seq, cmd, payload, checksum

### 2.2 算法层接口

| 函数 | 签名与行为 |
|---|---|
| `qr_parse(text) -> TaskPlan` | `"452+321+254+312"` → 任务结构；非法格式抛 `QrFormatError` |
| `detect_colors(img, hsv_table, min_area) -> list[Blob]` | BGR → 质心+面积+颜色号 |
| `detect_rings(img, geo_cfg) -> list[RingObs]` | BGR → 霍夫圆列表，按半径分档映射编号 |
| `detect_tri_ring(img) -> TriRingObs\|None` | 三连环齐全才返回，否则 None（宁缺勿错） |
| `calibrate(pose, obs: TriRingObs, field: FieldMap) -> Pose` | 见 2.3 |

### 2.3 校准算法（重点剩余功能）

- **Z 轴（朝向）**：`heading' = atan2(right.y-left.y, right.x-left.x)`，与 `FieldMap` 理论基准角之差作为修正量。
- **OPS 位置**：`pos' = w1·(中环实测↔理论修正) + w2·(左右环中点实测↔理论修正)`，权重 w1>w2（中环刻数字、几何最可靠），来自场地标定先验。
- `FieldMap`（config/field.yaml）：三色环与数字环理论圆心/半径，赛前标定一次。
- 校准时机：全场比赛 2 次（CAL1、CAL2 状态）。

### 2.4 设备层接口

```python
class CameraSource(Protocol):   # USBCamera / VideoFile / ImageFolder 同一契约
    def read() -> (ok, frame)
class McuLink(Protocol):        # SerialLink / MockMcu
    def send(frame); def poll() -> Frame|None; def recv_pose() -> Pose|None
```

### 2.5 线程模型与数据流

- 相机线程 → 帧队列（maxsize=1，丢旧帧）→ 主循环 FSM。
- 串口线程：收帧解析入 rx 队列；发送加锁。
- FSM 每帧按当前状态调用算法：SCAN→qr_parse；NAVIGATE→Pose 接收+卡尔曼平滑；CALIBRATE→detect_tri_ring→calibrate→下发修正帧；PLACE→detect_rings→微调坐标下发。

### 2.6 状态机

`IDLE → SCANNING → PLAN_READY → NAV1 → CAL1 → PLACE1 → NAV2 → CAL2 → PLACE2 → RETURN → DONE`

- 校准触发防抖：进入 CAL 状态后连续 K 帧稳定检出 `TriRingObs` 才执行（K=5，config 可调），取中位数结果，防单帧误检污染位姿。
- 任意状态超时/异常 → `RETRY(n)`（n=2，config 可调）→ 超限 → `SAFESTOP`。

## 3. 串口协议 v0.1（待与电控组冻结）

物理层：UART 115200-8-N-1，小端。

帧格式：`AA 55 | len(1B) | seq(1B) | cmd(1B) | payload(N=len-7) | xor(1B) | 0D`；len 为含头尾的总字节数（固定开销 7 字节），xor 覆盖 len..payload。

| 命令 | 方向 | 载荷 | 说明 |
|---|---|---|---|
| 0x10 HEARTBEAT | 双向 | 空 | 500ms 周期，超时 2s → SAFESTOP |
| 0x11 TASK_PLAN | 视→控 | batch1[3] batch2[3] pos1 pos2 posS | 扫码结果，开赛只发一次 |
| 0x12 GOTO | 视→控 | x,y:int16(mm) heading:int16(0.1°) | 请求导航 |
| 0x13 CALIBRATE | 视→控 | dx,dy:int16 dhead:int16 | **修正量**（非绝对坐标），主控叠加到其推算 |
| 0x14 PLACE | 视→控 | ring_id:1B fine_x,fine_y:int16 | 目标环+视觉微调 |
| 0x15 PICK | 视→控 | color:1B offset:int16×2 | 取料请求+相对偏移 |
| 0x16 ACK/NAK | 双向 | 被确认帧 seq | 100ms 未 ACK 重发，3 次失败 SAFESTOP |
| 0x20 POSE | 控→视 | x y heading + 里程 uint16 | 10Hz 主动上报，校准输入 |
| 0x21 STATE | 控→视 | state:1B | 主控状态，FSM 推进依据 |
| 0x22 DONE | 控→视 | task_id:1B | 单动作完成事件 |

设计要点：校准走增量、主控保留推算主权；场地原点=扫码区角点，双方共用 FieldMap；协议实现集中在 `core/protocol.py`（`encode` / `StreamDecoder.feed`，流式抗粘包断包）。

冻结清单（联调前与电控组确认）：①帧格式与命令表 ②场地原点与单位 ③POSE 频率 ④ACK 超时参数 ⑤SAFESTOP 复位责任。

## 4. 错误处理与日志

原则：宁停不错，所有异常收敛到 SAFESTOP。

| 故障 | 检测 | 处置 |
|---|---|---|
| 扫码失败/格式非法 | SCANNING 超时 15s / QrFormatError | 重试→GOTO 微调姿态→超时 SAFESTOP（不可降级） |
| 三色环检不出 | CAL 连续 K 帧不满足 | GOTO 小范围移动重找；2 次失败**跳过校准降级完赛** |
| 单帧误检 | 中位数+面积/半径合理性过滤 | 丢弃 |
| 串口断链 | HEARTBEAT 2s / ACK×3 | SAFESTOP + 状态快照 |
| 相机掉线 | read() 连续失败 | 重开 1 次→SAFESTOP |
| 校准残差过大 | residual > 15mm/3° | 判为误匹配，丢弃并告警 |

关键取舍：校准失败可跳过（保完赛），扫码失败不可（无任务依据）。

日志：`logging` 双输出（控制台 + `logs/run-<ts>.jsonl` 结构化：时间/FSM状态/帧/算法输入摘要/结果）；`tools/replay_log.py` 赛后复盘；调试窗口仅 live/replay 模式，比赛配置默认关。

## 5. 测试策略

1. **单测（pytest，无硬件）**：qr_parse 边界；协议 encode→decode 往返 + 粘包/断包/坏校验注入；`calibrate` 合成数据回归——真实位姿+漂移模型+环坐标噪声，断言校准后误差 < 未校准误差 1/5，2 次校准下全程位置误差有界（量化验收「2 次够不够」）。
2. **回放测试**：`tests/fixtures/` 现场实拍视频（扫码/三色环/数字环）+ 期望结果 JSON，`--mode replay` 断言 FSM 轨迹。
3. **仿真集成**：`--mode sim` = MockMcu + 合成图像源端到端跑全图，CI 全自动零硬件。
4. 实机项（真相机、真串口联调）不编自动化测试，列入阶段 4 人工验收清单。

## 6. 实现阶段

1. 骨架 + 扫码：工程脚手架、config、qr_parse + WeChatQRCode 封装、回放通道。
2. 识别：color_detect、ring_detect、HSV/霍夫参数标定测试集。
3. 校准：calibration.py + kalman + MockMcu 漂移模型 + 合成数据精度回归。
4. 通信与整机：protocol.py、fsm.py 全流程、sim 端到端、协议初稿交电控组、实机联调接口。
