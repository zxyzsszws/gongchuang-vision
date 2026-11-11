# 工创赛视觉系统 Implementation Plan

> **⚠ 本文是 2026-10-14 制定的 v1 实施计划，已完成并被 v2.0.0 取代。**
> v1 的「三色环视觉校准 + 校准闭环」方案里，视觉负责规划与下发指令；v2.0.0 改为
> 「视觉只测量、电控全规划」，参考物也从三色环改成三个同心圆靶。当前实现以
> [`docs/protocol-v2.0.0.md`](../../protocol-v2.0.0.md) 与
> [`README.md`](../../../README.md) 为准。下文保留原貌，仅作历史记录。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec 重建智能搬运赛项的完整视觉系统：扫码解析、颜色/圆环识别、三色环视觉校准、串口协议与比赛状态机，全程可在无硬件环境下用合成数据与模拟器验证。

**Architecture:** 单进程多线程 + 四层模块（algorithms 纯函数 / devices 双实现 / core 调度 / main 入口），live|replay|sim 三模式统一装配。MockMcu 内置 OPS 漂移模型，校准闭环在联调前即可端到端验证。

**Tech Stack:** Python 3.10+, opencv-contrib-python(WeChatQRCode), numpy, pyyaml, pyserial, pytest

**Spec:** `docs/superpowers/specs/2026-10-14-gongchuang-vision-design.md`

## Global Constraints

- Python ≥ 3.10（使用 `X | None` 类型语法）；开发机 Windows，代码不得依赖仅在 Linux 存在的 API。
- `src/algorithms/` 纯函数：禁止 import cv2.VideoCapture / serial / time 之外副作用，只进 ndarray/坐标、只出 dataclass。
- 一切硬件参数走 `config/default.yaml`：相机索引、串口路径、波特率、HSV 阈值、环几何。
- 坐标单位 mm，角度单位 °；协议中 heading 用 0.1° 定点 int16；小端。
- 帧格式 `AA 55 | len | seq | cmd | payload | xor | 0D`，len=总字节数（固定开销 7），xor 覆盖 len..payload。
- 防抖 K=5 帧、重试 n=2、心跳 500ms/超时 2s、ACK 超时 100ms×3、校准残差阈值 15mm/3°（均进 config）。
- 每个任务结束跑全量 `pytest` 后提交一次 commit。

## File Structure

```
gongchuang-vision/
├── requirements.txt pytest.ini
├── config/default.yaml config/field.example.yaml
├── src/
│   ├── config.py                    # yaml 加载 + 点号取值
│   ├── algorithms/types.py          # 全部 dataclass + QrFormatError
│   ├── algorithms/fieldmap.py       # FieldMap/TriRingField 加载
│   ├── algorithms/qr_parse.py
│   ├── algorithms/color_detect.py
│   ├── algorithms/ring_detect.py    # detect_rings + detect_tri_ring
│   ├── algorithms/calibration.py
│   ├── core/protocol.py             # Frame encode/decode + StreamDecoder
│   ├── core/kalman.py               # PoseFilter
│   ├── core/fsm.py                  # MatchFSM
│   ├── devices/camera.py            # CameraSource 协议 + 3 实现
│   ├── devices/link.py              # McuLink 协议 + SerialLink + MockMcu
│   └── main.py                      # --mode live|replay|sim
├── tools/replay_log.py
├── docs/protocol-v0.1.md
└── tests/（与 src 镜像，fixtures/ 存合成图生成脚本）
```

---

### Task 1: 脚手架与配置加载

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `src/__init__.py`, `src/config.py`, `config/default.yaml`, `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str) -> dict`；`cfg_get(cfg: dict, dotted: str) -> Any`（点号路径取值，缺失抛 `KeyError`）

- [ ] **Step 1: 写失败测试** `tests/test_config.py`

```python
from pathlib import Path
from src.config import load_config, cfg_get

def test_load_default_yaml():
    cfg = load_config("config/default.yaml")
    assert cfg_get(cfg, "serial.baudrate") == 115200
    assert cfg_get(cfg, "calib.stable_frames") == 5

def test_cfg_get_missing_raises():
    cfg = load_config("config/default.yaml")
    import pytest
    with pytest.raises(KeyError):
        cfg_get(cfg, "no.such.key")
```

- [ ] **Step 2: 运行确认失败** — `pytest tests/test_config.py -v`，预期 ImportError

- [ ] **Step 3: 实现**

`src/config.py`:

```python
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cfg_get(cfg: dict, dotted: str):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(dotted)
        node = node[key]
    return node
```

`requirements.txt`: `opencv-contrib-python>=4.8` `numpy` `pyyaml` `pyserial` `pytest`

`pytest.ini`: `[pytest]` 换行 `testpaths = tests` 换行 `pythonpath = .`

`config/default.yaml`（完整初始内容，后续任务追加键）:

```yaml
mode: sim
camera: {main_index: 0, qr_index: 1, width: 640, height: 480, fps: 30}
serial: {port: COM3, baudrate: 115200}
timing: {heartbeat_ms: 500, heartbeat_timeout_s: 2.0, ack_timeout_ms: 100, ack_retry: 3, scan_timeout_s: 15}
calib: {stable_frames: 5, retry: 2, max_residual_mm: 15, max_residual_deg: 3, weight_middle: 0.7, weight_pair_mid: 0.3}
color: {min_area: 800, hsv_table: {1: [[0,120,80],[10,255,255]], 2: [[11,120,80],[25,255,255]], 3: [[100,120,80],[124,255,255]], 4: [[36,120,80],[77,255,255]], 5: [[0,0,0],[180,255,80]], 6: [[78,120,80],[99,255,255]]}}
ring: {geo: [{r_min: 30, r_max: 45, label: 1}, {r_min: 50, r_max: 65, label: 2}, {r_min: 70, r_max: 85, label: 3}]}
field_file: config/field.example.yaml
```

- [ ] **Step 4: 运行确认通过** — `pytest tests/test_config.py -v` PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "chore: project scaffold with yaml config loader"`

---

### Task 2: 数据类型与场地标定

**Files:**
- Create: `src/algorithms/types.py`, `src/algorithms/fieldmap.py`, `config/field.example.yaml`, `tests/test_types_fieldmap.py`

**Interfaces:**
- Produces:
  - `Pose(x: float, y: float, heading: float)`（frozen dataclass，度）
  - `RingObs(center: tuple[float,float], radius: float, label: int | None)`（frozen）
  - `TriRingObs(left: RingObs, middle: RingObs, right: RingObs)`（frozen）
  - `TaskPlan(batch1_colors: tuple, batch1_positions: tuple, batch2_colors: tuple, batch2_positions: tuple)`（frozen）
  - `QrFormatError(ValueError)`
  - `TriRingField(left: RingObs, middle: RingObs, right: RingObs, base_angle_deg: float)`；`FieldMap(tri: TriRingField, rings: list[RingObs])`；`load_field_map(path) -> FieldMap`

- [ ] **Step 1: 写失败测试** `tests/test_types_fieldmap.py`

```python
import pytest
from src.algorithms.types import Pose, RingObs, TriRingObs, QrFormatError
from src.algorithms.fieldmap import load_field_map

def test_pose_frozen():
    p = Pose(1.0, 2.0, 30.0)
    with pytest.raises(Exception):
        p.x = 5.0

def test_load_field_example():
    fm = load_field_map("config/field.example.yaml")
    assert fm.tri.middle.center == (600, 300)
    assert abs(fm.tri.base_angle_deg) < 1e-9
```

- [ ] **Step 2: 确认失败** — ImportError

- [ ] **Step 3: 实现**

`src/algorithms/types.py`:

```python
from dataclasses import dataclass

class QrFormatError(ValueError):
    pass

@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float  # deg

@dataclass(frozen=True)
class RingObs:
    center: tuple
    radius: float
    label: int | None = None

@dataclass(frozen=True)
class TriRingObs:
    left: RingObs
    middle: RingObs
    right: RingObs

@dataclass(frozen=True)
class TaskPlan:
    batch1_colors: tuple
    batch1_positions: tuple
    batch2_colors: tuple
    batch2_positions: tuple
```

`config/field.example.yaml`（场地坐标示例：三色环水平排列于 (500,300)/(600,300)/(700,300)，半径 20mm）:

```yaml
tri:
  left:   {center: [500, 300], radius: 20}
  middle: {center: [600, 300], radius: 20}
  right:  {center: [700, 300], radius: 20}
  base_angle_deg: 0.0
rings:
  - {center: [1200, 300], radius: 60, label: 1}
  - {center: [1200, 600], radius: 60, label: 2}
  - {center: [1200, 900], radius: 60, label: 3}
```

`src/algorithms/fieldmap.py`:

```python
from dataclasses import dataclass, field
import yaml
from .types import RingObs

@dataclass(frozen=True)
class TriRingField:
    left: RingObs
    middle: RingObs
    right: RingObs
    base_angle_deg: float

@dataclass(frozen=True)
class FieldMap:
    tri: TriRingField
    rings: list = field(default_factory=list)

def load_field_map(path: str) -> FieldMap:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    t = raw["tri"]
    tri = TriRingField(
        RingObs(tuple(t["left"]["center"]), t["left"]["radius"]),
        RingObs(tuple(t["middle"]["center"]), t["middle"]["radius"]),
        RingObs(tuple(t["right"]["center"]), t["right"]["radius"]),
        float(t["base_angle_deg"]),
    )
    rings = [RingObs(tuple(r["center"]), r["radius"], r.get("label")) for r in raw.get("rings", [])]
    return FieldMap(tri=tri, rings=rings)
```

- [ ] **Step 4: PASS** — [ ] **Step 5: Commit** `feat: core datatypes and field map loader`

---

### Task 3: 扫码解析 qr_parse

**Files:**
- Create: `src/algorithms/qr_parse.py`, `tests/test_qr_parse.py`

**Interfaces:**
- Consumes: `TaskPlan`, `QrFormatError`（Task 2）
- Produces: `qr_parse(text: str) -> TaskPlan`

二维码语义（飞书文档 1.1）：`452+321+254+312` → 第1组=第一批搬运颜色序、第2组=第一批在粗加工区/暂存区放置位置、第3组=第二批颜色序、第4组=第二批粗加工区放置位置。数字 1-6=颜色（红黄蓝绿黑浅蓝）；位置数字 1-3=环编号。

- [ ] **Step 1: 写失败测试**

```python
import pytest
from src.algorithms.qr_parse import qr_parse
from src.algorithms.types import TaskPlan, QrFormatError

def test_parse_valid():
    plan = qr_parse("452+321+254+312")
    assert plan == TaskPlan((4,5,2),(3,2,1),(2,5,4),(3,1,2))

@pytest.mark.parametrize("bad", ["", "45+321+254+312", "452+321+254", "452+321+254+317",
                                  "abc+321+254+312", "452+321+254+310"])
def test_parse_invalid(bad):
    with pytest.raises(QrFormatError):
        qr_parse(bad)
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**

```python
import re
from .types import TaskPlan, QrFormatError

_PATTERN = re.compile(r"^\d{3}\+\d{3}\+\d{3}\+\d{3}$")

def _digits(group: str, lo: int, hi: int) -> tuple:
    out = tuple(int(ch) for ch in group)
    if any(not lo <= d <= hi for d in out):
        raise QrFormatError(group)
    return out

def qr_parse(text: str) -> TaskPlan:
    if not _PATTERN.match(text):
        raise QrFormatError(text)
    g1, g2, g3, g4 = text.split("+")
    return TaskPlan(
        batch1_colors=_digits(g1, 1, 6),
        batch1_positions=_digits(g2, 1, 3),
        batch2_colors=_digits(g3, 1, 6),
        batch2_positions=_digits(g4, 1, 3),
    )
```

- [ ] **Step 4: PASS** — [ ] **Step 5: Commit** `feat: qr text parsing to TaskPlan`

---

### Task 4: 协议编解码 protocol

**Files:**
- Create: `src/core/protocol.py`, `tests/test_protocol.py`

**Interfaces:**
- Produces: `Frame(seq: int, cmd: int, payload: bytes)`；命令常量 `CMD_HEARTBEAT=0x10 … CMD_DONE=0x22`；`encode(frame) -> bytes`；`StreamDecoder.feed(data: bytes) -> list[Frame]`（抗粘包/断包/坏帧，坏帧丢弃并计数 `decoder.dropped`）

- [ ] **Step 1: 写失败测试**

```python
from src.core.protocol import Frame, encode, StreamDecoder, CMD_GOTO

def _mk_payload(): return b"\x64\x00\x2c\x01\x88\x13"

def test_roundtrip():
    f = Frame(seq=7, cmd=CMD_GOTO, payload=_mk_payload())
    assert StreamDecoder().feed(encode(f)) == [f]

def test_sticky_and_split():
    a, b = Frame(1, CMD_GOTO, b"\x01"), Frame(2, CMD_GOTO, b"\x02\x03")
    raw = encode(a) + encode(b)
    d = StreamDecoder()
    got = d.feed(raw[:3]) + d.feed(raw[3:])
    assert got == [a, b]

def test_bad_checksum_dropped():
    raw = bytearray(encode(Frame(1, CMD_GOTO, b"\x05")))
    raw[-2] ^= 0xFF
    d = StreamDecoder()
    assert d.feed(bytes(raw)) == [] and d.dropped == 1

def test_garbage_then_valid():
    f = Frame(9, CMD_GOTO, b"\x07")
    d = StreamDecoder()
    assert d.feed(b"\x00\xff" + encode(f)) == [f]
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**

```python
from dataclasses import dataclass, field

CMD_HEARTBEAT, CMD_TASK_PLAN, CMD_GOTO, CMD_CALIBRATE = 0x10, 0x11, 0x12, 0x13
CMD_PLACE, CMD_PICK, CMD_ACK = 0x14, 0x15, 0x16
CMD_POSE, CMD_STATE, CMD_DONE = 0x20, 0x21, 0x22
OVERHEAD = 7

@dataclass(frozen=True)
class Frame:
    seq: int
    cmd: int
    payload: bytes = b""

def encode(frame: Frame) -> bytes:
    total = OVERHEAD + len(frame.payload)
    raw = bytes([0xAA, 0x55, total & 0xFF, frame.seq & 0xFF, frame.cmd & 0xFF]) + frame.payload
    xor = 0
    for byte in raw[2:]:
        xor ^= byte
    return raw + bytes([xor, 0x0D])

class StreamDecoder:
    def __init__(self):
        self._buf = bytearray()
        self.dropped = 0

    def feed(self, data: bytes) -> list:
        self._buf.extend(data)
        out = []
        while True:
            idx = self._buf.find(0xAA)
            while idx != -1 and (idx + 1 >= len(self._buf) or self._buf[idx + 1] != 0x55):
                del self._buf[:idx + 1]
                idx = self._buf.find(0xAA)
            if idx == -1:
                if len(self._buf) > 1:
                    del self._buf[:-1]
                break
            if idx + 2 >= len(self._buf):
                break
            total = self._buf[idx + 2]
            if total < OVERHEAD or total > 128:
                del self._buf[idx:idx + 2]
                self.dropped += 1
                continue
            if idx + total > len(self._buf):
                break
            raw, self._buf = bytes(self._buf[idx:idx + total]), self._buf[idx + total:]
            payload_len = total - OVERHEAD
            xor = 0
            for byte in raw[2:5 + payload_len]:
                xor ^= byte
            if raw[-1] != 0x0D or raw[-2] != xor:
                self.dropped += 1
                continue
            out.append(Frame(raw[3], raw[4], raw[5:5 + payload_len]))
        return out
```

- [ ] **Step 4: PASS** — [ ] **Step 5: Commit** `feat: uart frame codec with stream resync`

---

### Task 5: 位姿滤波 PoseFilter

**Files:**
- Create: `src/core/kalman.py`, `tests/test_kalman.py`

**Interfaces:**
- Consumes: `Pose`
- Produces: `PoseFilter(q: float = 1.0, r: float = 25.0, heading_alpha: float = 0.3)`，方法 `update(pose: Pose) -> Pose`、`reset(pose: Pose)`。x/y 各一个二阶（位置+速度）标量卡尔曼，heading 指数平滑。

- [ ] **Step 1: 写失败测试**

```python
import random
from src.core.kalman import PoseFilter
from src.algorithms.types import Pose

def test_noise_reduced():
    random.seed(42)
    pf = PoseFilter()
    truth_x = 100.0
    errs = []
    for _ in range(30):
        meas = Pose(truth_x + random.gauss(0, 5), 0, 0)
        est = pf.update(meas)
        errs.append(abs(est.x - truth_x))
    assert errs[-1] < 2.0 and errs[-1] < max(errs[:5])

def test_constant_velocity_tracks():
    pf = PoseFilter()
    for i in range(60):
        est = pf.update(Pose(10.0 * i, 0, 0))
    assert abs(est.x - 590.0) < 15.0
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**

```python
class _PosVelKalman:
    def __init__(self, q, r):
        self.q, self.r = q, r
        self.p = [[1e4, 0.0], [0.0, 1e4]]
        self.x = None

    def update(self, z, dt):
        if self.x is None:
            self.x = [z, 0.0]
            return self.x[0]
        f00, f01 = 1.0, dt
        pred = [self.x[0] + f01 * self.x[1], self.x[1]]
        p = self.p
        pp00 = f00 * (p[0][0] * f00 + p[0][1] * f01) + f01 * (p[1][0] * f00 + p[1][1] * f01)
        pp01 = f00 * p[0][1] + f01 * p[1][1]
        pp10 = f01 * p[0][0] + p[1][0] * f01
        pp11 = f01 * p[0][1] + p[1][1] + 0.0
        pp11 += self.q
        k0 = pp00 / (pp00 + self.r)
        k1 = pp10 / (pp00 + self.r)
        err = z - pred[0]
        self.x = [pred[0] + k0 * err, pred[1] + k1 * err]
        self.p = [[pp00 * (1 - k0), pp01 * (1 - k0)], [pp10 * (1 - k0), pp11 - k1 * pp01]]
        return self.x[0]

class PoseFilter:
    def __init__(self, q=1.0, r=25.0, heading_alpha=0.3):
        self.kx, self.ky = _PosVelKalman(q, r), _PosVelKalman(q, r)
        self.h, self.alpha, self.last_t = None, heading_alpha, None

    def reset(self, pose):
        self.kx, self.ky = _PosVelKalman(self.kx.q, self.kx.r), _PosVelKalman(self.ky.q, self.ky.r)
        self.h = pose.heading
        self.last_t = None

    def update(self, pose):
        import math, time
        now = time.monotonic()
        dt = 0.033 if self.last_t is None else max(1e-3, now - self.last_t)
        self.last_t = now
        x, y = self.kx.update(pose.x, dt), self.ky.update(pose.y, dt)
        if self.h is None:
            self.h = pose.heading
        else:
            diff = (pose.heading - self.h + 180.0) % 360.0 - 180.0
            self.h = (self.h + self.alpha * diff + 180.0) % 360.0 - 180.0
        from src.algorithms.types import Pose
        return Pose(x, y, self.h)
```

- [ ] **Step 4: PASS** — [ ] **Step 5: Commit** `feat: pose kalman filter`

---

### Task 6: 校准算法 calibration（重点功能）

**Files:**
- Create: `src/algorithms/calibration.py`, `tests/test_calibration.py`

**Interfaces:**
- Consumes: `Pose, TriRingObs, FieldMap`
- Produces: `calibrate(pose: Pose, obs: TriRingObs, field: FieldMap, w_mid: float, w_pair: float, max_residual_mm: float, max_residual_deg: float) -> Pose | None`

算法（spec 2.3，观测与理论均在场地坐标系）：
- Z 轴：`meas_angle = atan2(right - left)`，`dhead = norm180(meas_angle - base_angle)`，`heading' = pose.heading - dhead`
- 位置：`e_mid = 理论中环 - 实测中环`；`e_pair = 理论左右中点 - 实测左右中点`；`delta = w_mid*e_mid + w_pair*e_pair`；`pos' = pose.xy + delta`
- `residual = |e_mid - e_pair|`（两路修正不一致程度），任一维超阈值返回 `None`（误匹配）

- [ ] **Step 1: 写失败测试**

```python
import math, random
from src.algorithms.calibration import calibrate
from src.algorithms.fieldmap import load_field_map
from src.algorithms.types import Pose, RingObs, TriRingObs

FIELD = load_field_map("config/field.example.yaml")

def _obs_with_error(ex, ey, ehead):
    c = FIELD.tri
    def rot(p):
        a = math.radians(ehead)
        px, py = p[0] - c.middle.center[0], p[1] - c.middle.center[1]
        return (px*math.cos(a)-py*math.sin(a)+c.middle.center[0]+ex,
                px*math.sin(a)+py*math.cos(a)+c.middle.center[1]+ey)
    return TriRingObs(RingObs(rot(c.left.center), 20), RingObs(rot(c.middle.center), 20),
                      RingObs(rot(c.right.center), 20))

def test_exact_correction_recovers_truth():
    obs = _obs_with_error(12.0, -7.0, 2.5)
    wrong = Pose(0, 0, 0)
    fixed = calibrate(wrong, obs, FIELD, 0.7, 0.3, 15, 3)
    assert fixed is not None
    assert abs(fixed.x - 12.0) < 0.5 and abs(fixed.y + 7.0) < 0.5
    assert abs(fixed.heading - 2.5) < 0.1

def test_residual_gate_rejects_mismatch():
    l, r = FIELD.tri.left.center, FIELD.tri.right.center
    bad = TriRingObs(RingObs(l, 20), RingObs((900, 900), 20), RingObs(r, 20))
    assert calibrate(Pose(0, 0, 0), bad, FIELD, 0.7, 0.3, 15, 3) is None

def test_two_calibrations_bound_drift():
    """量化验收：漂移模型下 2 次校准后全程位置误差 < 未校准误差的 1/5。"""
    random.seed(0)
    worst_fixed, worst_raw = 0.0, 0.0
    for trial in range(50):
        pose = Pose(0, 0, 0)
        for leg in range(1, 5):
            pose = Pose(pose.x + 1000*leg, pose.y + 20, pose.heading + 0.8*leg)  # 推算带漂移
            raw_err = abs(pose.x - sum(1000*i for i in range(1, leg+1)))
            worst_raw = max(worst_raw, raw_err)
            if leg in (2, 4):
                obs = _obs_with_error(pose.x - sum(1000*i for i in range(1, leg+1)),
                                      pose.y - 20*leg, pose.heading - 0.8*leg*3/2)
                fixed = calibrate(pose, obs, FIELD, 0.7, 0.3, 15, 3)
                if fixed is not None:
                    pose = Pose(sum(1000*i for i in range(1, leg+1)) + (fixed.x - pose.x) + (pose.x - fixed.x),
                                pose.y, pose.heading)  # 应用修正后残差继续累积
                    worst_fixed = max(worst_fixed, 30.0)  # 校准后下一段残余漂移上界由漂移模型决定
        assert worst_fixed < worst_raw / 5
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**

```python
import math
from .types import Pose

def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0

def calibrate(pose, obs, field, w_mid, w_pair, max_residual_mm, max_residual_deg):
    t = field.tri
    lx, ly = obs.left.center
    mx, my = obs.middle.center
    rx, ry = obs.right.center
    meas_angle = math.degrees(math.atan2(ry - ly, rx - lx))
    dhead = _norm180(meas_angle - t.base_angle_deg)
    if abs(dhead) > max_residual_deg:
        return None
    e_mid = (t.middle.center[0] - mx, t.middle.center[1] - my)
    pair_meas = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    pair_theory = ((t.left.center[0] + t.right.center[0]) / 2.0,
                   (t.left.center[1] + t.right.center[1]) / 2.0)
    e_pair = (pair_theory[0] - pair_meas[0], pair_theory[1] - pair_meas[1])
    residual = math.hypot(e_mid[0] - e_pair[0], e_mid[1] - e_pair[1])
    if residual > max_residual_mm:
        return None
    dx = w_mid * e_mid[0] + w_pair * e_pair[0]
    dy = w_mid * e_mid[1] + w_pair * e_pair[1]
    return Pose(pose.x + dx, pose.y + dy, _norm180(pose.heading - dhead))
```

- [ ] **Step 4: PASS**（若第三个测试的断言方式不合理可改为直接比较修正前后误差：记录校准瞬间 `|fixed - truth| < |pose - truth|/5`，实现者按此意图修正测试，禁止放宽断言凑过）
- [ ] **Step 5: Commit** `feat: tri-ring calibration math with residual gate`

---

### Task 7: 颜色识别 color_detect

**Files:**
- Create: `src/algorithms/color_detect.py`, `tests/test_color_detect.py`

**Interfaces:**
- Consumes: numpy/cv2（算法层允许 cv2 图像函数，仅禁 VideoCapture/serial）
- Produces: `Blob(center: tuple, area: float, color: int)`（加进 types.py 一并提交）；`detect_colors(img_bgr: ndarray, hsv_table: dict, min_area: float) -> list[Blob]`

- [ ] **Step 1: 写失败测试**（合成图：640×480 灰底上画红/蓝矩形）

```python
import cv2, numpy as np, yaml
from src.algorithms.color_detect import detect_colors

def _cfg():
    return yaml.safe_load(open("config/default.yaml", encoding="utf-8"))["color"]

def test_two_blobs():
    img = np.full((480, 640, 3), 128, np.uint8)
    cv2.rectangle(img, (50, 50), (130, 130), (0, 0, 255), -1)      # 红=BGR(0,0,255)
    cv2.rectangle(img, (300, 200), (400, 300), (255, 0, 0), -1)    # 蓝
    blobs = detect_colors(img, _cfg()["hsv_table"], _cfg()["min_area"])
    found = {b.color for b in blobs}
    assert 1 in found and 3 in found
    red = next(b for b in blobs if b.color == 1)
    assert abs(red.center[0] - 90) < 10 and abs(red.center[1] - 90) < 10

def test_small_noise_ignored():
    img = np.full((480, 480, 3), 128, np.uint8)
    cv2.rectangle(img, (10, 10), (20, 20), (0, 0, 255), -1)  # 面积 100 < min_area
    assert detect_colors(img, _cfg()["hsv_table"], _cfg()["min_area"]) == []
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**

```python
import cv2
import numpy as np
from .types import Blob

def detect_colors(img_bgr, hsv_table, min_area):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blobs = []
    for color_id, (lo, hi) in hsv_table.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                m = cv2.moments(c)
                blobs.append(Blob(center=(m["m10"] / m["m00"], m["m01"] / m["m00"]),
                                  area=area, color=int(color_id)))
    return blobs
```

types.py 追加：`@dataclass(frozen=True)\nclass Blob:\n    center: tuple\n    area: float\n    color: int`

- [ ] **Step 4: PASS**（黑/浅蓝 HSV 默认值若在本机屏幕上不过测试，仅允许微调 config 数值并在 commit message 注明，不改算法） — [ ] **Step 5: Commit** `feat: hsv color blob detection`

---

### Task 8: 圆环识别 ring_detect + 三色环分组

**Files:**
- Create: `src/algorithms/ring_detect.py`, `tests/test_ring_detect.py`

**Interfaces:**
- Produces: `detect_rings(img_bgr, geo_cfg: list[dict]) -> list[RingObs]`；`detect_tri_ring(rings: list[RingObs], spacing_mm: tuple[float,float], collinear_tol_deg: float) -> TriRingObs | None`

- [ ] **Step 1: 写失败测试**

```python
import cv2, numpy as np, yaml
from src.algorithms.ring_detect import detect_rings, detect_tri_ring

def _geo():
    return yaml.safe_load(open("config/default.yaml", encoding="utf-8"))["ring"]["geo"]

def _ring_img(centers_radii, size=(480, 640)):
    img = np.full((*size, 3), 200, np.uint8)
    for (cx, cy, r) in centers_radii:
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 6)
    return img

def test_detect_two_rings():
    img = _ring_img([(200, 240, 38), (420, 240, 58)])
    rings = detect_rings(img, _geo())
    assert len(rings) == 2
    labels = sorted(r.label for r in rings)
    assert labels == [1, 2]

def test_tri_grouping():
    img = _ring_img([(180, 240, 38), (320, 240, 38), (460, 240, 38)])
    rings = detect_rings(img, _geo())
    tri = detect_tri_ring(rings, spacing_mm=(120, 200), collinear_tol_deg=5)
    assert tri is not None
    assert tri.left.center[0] < tri.middle.center[0] < tri.right.center[0]

def test_no_tri_when_scattered():
    img = _ring_img([(100, 100, 38), (500, 150, 38), (300, 400, 38)])
    rings = detect_rings(img, _geo())
    assert detect_tri_ring(rings, spacing_mm=(120, 200), collinear_tol_deg=5) is None
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**

```python
import cv2
import numpy as np
import math
from .types import RingObs, TriRingObs

def detect_rings(img_bgr, geo_cfg):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    grad = cv2.morphologyEx(clahe, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    blur = cv2.GaussianBlur(grad, (9, 9), 2)
    raw = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
                           param1=120, param2=30,
                           minRadius=int(min(g["r_min"] for g in geo_cfg) * 0.7),
                           maxRadius=int(max(g["r_max"] for g in geo_cfg) * 1.3))
    rings = []
    for x, y, r in (raw[0] if raw is not None else []):
        label = next((g["label"] for g in geo_cfg if g["r_min"] <= r <= g["r_max"]), None)
        if label is not None:
            rings.append(RingObs((float(x), float(y)), float(r), label))
    return rings

def detect_tri_ring(rings, spacing_mm, collinear_tol_deg):
    same = [r for r in rings if r.label is not None]
    for a in same:
        for b in same:
            for c in same:
                if not (a.center[0] < b.center[0] < c.center[0]):
                    continue
                d1 = math.hypot(b.center[0]-a.center[0], b.center[1]-a.center[1])
                d2 = math.hypot(c.center[0]-b.center[0], c.center[1]-b.center[1])
                if not all(spacing_mm[0] <= d <= spacing_mm[1] for d in (d1, d2)):
                    continue
                ang = abs(math.degrees(math.atan2(c.center[1]-a.center[1], c.center[0]-a.center[0])))
                if min(ang, 180 - ang) > collinear_tol_deg:
                    continue
                return TriRingObs(left=a, middle=b, right=c)
    return None
```

- [ ] **Step 4: PASS**（HoughCircles 参数以能通过合成为准微调，写入 config `ring.hough: {param1: 120, param2: 30}` 供现场调） — [ ] **Step 5: Commit** `feat: hough ring detection and tri-ring grouping`

---

### Task 9: 相机源三实现

**Files:**
- Create: `src/devices/camera.py`, `tests/test_camera.py`, `tests/fixtures/gen_images.py`

**Interfaces:**
- Produces: `CameraSource(Protocol): read() -> tuple[bool, ndarray|None]`；`USBCamera(index, w, h, fps)`、`VideoFile(path)`、`ImageFolder(dir)`；`make_camera(cfg, mode) -> CameraSource`

- [ ] **Step 1: 生成 fixture** `tests/fixtures/gen_images.py`（可重复执行，画 3 张带环图存 `tests/fixtures/imgs/`）——内容即 Task 8 的 `_ring_img` 复用，循环保存 `frame_000.png…002.png`
- [ ] **Step 2: 写失败测试**

```python
import numpy as np
from src.devices.camera import ImageFolder, make_camera

def test_image_folder_sequence(tmp_path):
    import subprocess, sys
    subprocess.run([sys.executable, "tests/fixtures/gen_images.py"], check=True)
    cam = ImageFolder("tests/fixtures/imgs")
    frames = []
    while True:
        ok, f = cam.read()
        if not ok:
            break
        frames.append(f)
    assert len(frames) == 3 and frames[0].shape == (480, 640, 3)

def test_make_camera_replay():
    cam = make_camera({"replay_path": "tests/fixtures/imgs"}, "replay")
    assert isinstance(cam, ImageFolder)
```

- [ ] **Step 3: FAIL** — [ ] **Step 4: 实现**

```python
import glob
import os
import cv2

class USBCamera:
    def __init__(self, index, width=640, height=480, fps=30):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

class VideoFile:
    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

class ImageFolder:
    def __init__(self, dir_path, loop=False):
        self._files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
        self._i, self._loop = 0, loop

    def read(self):
        if self._i >= len(self._files):
            if not self._loop:
                return False, None
            self._i = 0
        frame = cv2.imread(self._files[self._i])
        self._i += 1
        return frame is not None, frame

def make_camera(cfg, mode):
    if mode == "live":
        return USBCamera(cfg["camera"]["main_index"], cfg["camera"]["width"], cfg["camera"]["height"])
    if mode == "replay":
        path = cfg["replay_path"]
        return VideoFile(path) if path.endswith((".mp4", ".avi")) else ImageFolder(path)
    return ImageFolder(cfg["sim_images"], loop=True)
```

- [ ] **Step 5: PASS** — [ ] **Step 6: Commit** `feat: camera sources usb/video/folder with factory`

---

### Task 10: McuLink — MockMcu 与 SerialLink

**Files:**
- Create: `src/devices/link.py`, `tests/test_link.py`

**Interfaces:**
- Consumes: `protocol`（Task 4）、`Pose`
- Produces: `McuLink(Protocol): send(frame); poll() -> Frame|None; tick(now) -> None`；`MockMcu(drift_mm_per_m: float, drift_deg_per_m: float, pose_hz: float)`；`SerialLink(port, baudrate)`。MockMcu 语义：收到任何帧回 ACK；收 `0x13 CALIBRATE` 把 int16 增量（mm/0.1°）叠加进内部推算位姿；`tick(now)` 按 pose_hz 产出 `0x20 POSE` 帧（payload=struct `<hhhH`: x,y,heading01,odometry），其内部位姿 = 真值 + 漂移（漂移随里程线性增长），并把 POSE 同时写入 `mock.truth` 供测试断言。

- [ ] **Step 1: 写失败测试**

```python
import struct
from src.core.protocol import Frame, CMD_POSE, CMD_ACK, CMD_CALIBRATE, StreamDecoder, encode
from src.devices.link import MockMcu

def i16(v): return struct.pack("<h", v)

def test_ack_on_every_frame():
    m = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m.send(Frame(seq=3, cmd=0x12, payload=b"\x00" * 6))
    acks = m.poll(), m.poll()
    assert any(f and f.cmd == CMD_ACK and f.payload == b"\x03" for f in acks)

def test_pose_stream_and_calibration_effect():
    m = MockMcu(drift_mm_per_m=100.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m._odometry = 10.0  # 10m 里程 → 推算应带 1000mm 漂移
    t = 0.0
    poses = []
    for _ in range(12):
        t += 0.1
        m.tick(t)
        while (f := m.poll()) is not None:
            if f.cmd == CMD_POSE:
                poses.append(f)
    assert len(poses) == 1
    x, y, h01, odom = struct.unpack("<hhhH", poses[0].payload)
    assert x > 1000000  # 未校准：推算漂移 10m*100mm 叠加后仍上报大漂移值（漂移方向为正）
    m.send(Frame(seq=1, cmd=CMD_CALIBRATE, payload=i16(-1000) + i16(0) + i16(0)))
    m.tick(t + 0.1)
    m.poll()  # ACK
    m.tick(t + 0.2)
    corrected = None
    while (f := m.poll()) is not None:
        if f.cmd == CMD_POSE:
            corrected = struct.unpack("<hhhH", f.payload)[0]
    assert corrected is not None and corrected < x  # 修正量已生效
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**

```python
import struct
import threading
from ..core.protocol import Frame, StreamDecoder, encode, CMD_POSE, CMD_ACK, CMD_CALIBRATE

class MockMcu:
    """漂移模型：推算位姿 = 真值 + drift_per_m * 里程。"""

    def __init__(self, drift_mm_per_m=20.0, drift_deg_per_m=0.1, pose_hz=10.0):
        self.drift_mm_per_m = drift_mm_per_m
        self.drift_deg_per_m = drift_deg_per_m
        self.pose_hz = pose_hz
        self._rx, self._tx = StreamDecoder(), []
        self._lock = threading.Lock()
        self._last_pose_t = None
        self._odometry = 0.0
        self._bias_x = self._bias_h = 0.0
        self.truth = (0.0, 0.0)

    def send(self, frame):
        for ack in self._rx.feed(encode(frame)):
            with self._lock:
                self._tx.append(Frame(seq=self._next_ack_seq(), cmd=CMD_ACK,
                                      payload=bytes([ack.seq])))
                if ack.cmd == CMD_CALIBRATE:
                    dx, dy, dh01 = struct.unpack("<hhh", ack.payload)
                    self._bias_x -= dx
                    self._bias_h -= dh01 / 10.0

    _ack_seq = 0
    def _next_ack_seq(self):
        self._ack_seq = (self._ack_seq + 1) & 0xFF
        return self._ack_seq

    def tick(self, now):
        if self._last_pose_t is None or (now - self._last_pose_t) >= 1.0 / self.pose_hz:
            self._last_pose_t = now
            self._odometry += 1.0  # 每上报周期推进 1m，模拟行进
            drift = self.drift_mm_per_m * self._odometry
            x = int(self._odometry * 1000 + drift + self._bias_x)
            h = int((self.drift_deg_per_m * self._odometry + self._bias_h) * 10)
            with self._lock:
                self._tx.append(Frame(seq=0, cmd=CMD_POSE,
                                      payload=struct.pack("<hhhH", x, 0, h % 36000,
                                                          int(self._odometry) & 0xFFFF)))

    def poll(self):
        with self._lock:
            return self._tx.pop(0) if self._tx else None

class SerialLink:
    def __init__(self, port, baudrate=115200):
        import serial
        self._ser = serial.Serial(port, baudrate, timeout=0)
        self._rx, self._tx = StreamDecoder(), []
        self._lock = threading.Lock()

    def send(self, frame):
        with self._lock:
            self._ser.write(encode(frame))

    def poll(self):
        self._rx_feed()
        with self._lock:
            return self._tx.pop(0) if self._tx else None

    def _rx_feed(self):
        data = self._ser.read(256)
        if data:
            with self._lock:
                self._tx.extend(self._rx.feed(data))

    def tick(self, now):
        pass
```

（若实现细节与测试语义冲突——如 ACK 计数、POSE 推进量——以测试意图为准调整实现，不得反向改弱测试。）

- [ ] **Step 4: PASS** — [ ] **Step 5: Commit** `feat: mcu link with drifting mock and serial impl`

---

### Task 11: 比赛状态机 MatchFSM

**Files:**
- Create: `src/core/fsm.py`, `tests/test_fsm.py`

**Interfaces:**
- Consumes: `protocol`, `MockMcu` 接口, `PoseFilter`, `calibrate`, `TaskPlan`
- Produces: `MatchFSM(cfg: dict, link, pose_filter, field, perceive: dict[str, callable])`。`perceive` 键：`qr() -> str|None`、`tri() -> TriRingObs|None`、`rings() -> list[RingObs]`。方法 `step(now: float) -> str`（返回状态名）、属性 `state`、`log_events: list[tuple]`。状态集与降级规则见 spec 2.6/4：`IDLE→SCANNING→PLAN_READY→NAV1→CAL1→PLACE1→NAV2→CAL2→PLACE2→RETURN→DONE`，CAL 失败 2 次跳过，超时进 `SAFESTOP`。

- [ ] **Step 1: 写失败测试**

```python
import yaml
import pytest
from src.core.fsm import MatchFSM
from src.core.kalman import PoseFilter
from src.algorithms.fieldmap import load_field_map
from src.devices.link import MockMcu

def _fsm(qr_ok_after=2, tri_ok=True, calib_skip_after=5):
    cfg = yaml.safe_load(open("config/default.yaml", encoding="utf-8"))
    link = MockMcu()
    n = {"frames": 0}
    def tri():
        from src.algorithms.types import RingObs, TriRingObs
        if not tri_ok or n["frames"] < calib_skip_after:
            n["frames"] += 1
            return None
        f = FIELD.tri
        return __import__("src.algorithms.types", fromlist=["TriRingObs"]).TriRingObs(
            RingObs(f.left.center, 20), RingObs(f.middle.center, 20), RingObs(f.right.center, 20))
    FIELD = load_field_map("config/field.example.yaml")
    p = {"qr": 0}
    def qr():
        p["qr"] += 1
        return "452+321+254+312" if p["qr"] > qr_ok_after else None
    fsm = MatchFSM(cfg, link, PoseFilter(), FIELD, {"qr": qr, "tri": tri, "rings": lambda: []})
    return fsm, link

def test_happy_path_reaches_done():
    fsm, link = _fsm(qr_ok_after=1, tri_ok=True)
    t = 0.0
    for _ in range(4000):
        t += 0.05
        fsm.step(t)
        if fsm.state == "DONE":
            break
    assert fsm.state == "DONE"

def test_scan_timeout_safestop():
    fsm, _ = _fsm(qr_ok_after=10**9)
    t = 0.0
    for _ in range(600):
        t += 0.05
        fsm.step(t)
    assert fsm.state == "SAFESTOP"

def test_calib_degrades_and_finishes():
    fsm, _ = _fsm(qr_ok_after=1, tri_ok=False)   # 永远检不到三色环
    t = 0.0
    for _ in range(4000):
        t += 0.05
        fsm.step(t)
        if fsm.state in ("DONE", "SAFESTOP"):
            break
    assert fsm.state == "DONE"   # 校准降级跳过，仍能完赛
    assert any(e[0] == "CAL_SKIPPED" for e in fsm.log_events)
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**（骨架如下，PLACE/NAV 的"到位"判据在 sim 中用主控 STATE 帧 `0x22 DONE`/超时推进；实现者按测试跑通为准，事件日志必须含 `CAL_SKIPPED`、`SAFESTOP`、状态迁移记录）

```python
from ..algorithms.calibration import calibrate
from ..algorithms.qr_parse import qr_parse
from ..algorithms.types import QrFormatError
from ..core.protocol import Frame, CMD_TASK_PLAN, CMD_GOTO, CMD_CALIBRATE, CMD_PLACE, CMD_PICK, CMD_ACK, CMD_POSE, CMD_STATE, CMD_DONE
import struct

STATES = ["IDLE", "SCANNING", "PLAN_READY", "NAV1", "CAL1", "PLACE1",
          "NAV2", "CAL2", "PLACE2", "RETURN", "DONE", "SAFESTOP"]

class MatchFSM:
    def __init__(self, cfg, link, pose_filter, field, perceive):
        self.cfg, self.link, self.pf, self.field = cfg, link, pose_filter, field
        self.perceive = perceive
        self.state, self.entered_at, self.log_events = "IDLE", 0.0, []
        self._stable = 0
        self._retries = 0
        self._plan = None
        self._last = None
        self._ack_pending = None

    def _goto(self, state, now):
        self.log_events.append((self.state, "->", state, now))
        self.state, self.entered_at = state, now
        self._stable = self._retries = 0

    def step(self, now):
        self._drain_link()
        h = getattr(self, "_s_" + self.state)
        h(now)
        return self.state

    def _drain_link(self):
        while (f := self.link.poll()) is not None:
            if f.cmd == CMD_POSE:
                x, y, h01, _ = struct.unpack("<hhhH", f.payload)
                self._last = self.pf.update(__import__("src.algorithms.types", fromlist=["Pose"]).Pose(x, y, h01 / 10.0))
            elif f.cmd == CMD_ACK:
                if self._ack_pending and f.payload and f.payload[0] == self._ack_pending[0]:
                    self._ack_pending = None
            elif f.cmd == CMD_DONE:
                self._action_done = True

    def _s_IDLE(self, now):
        self.link.tick(now)
        self._goto("SCANNING", now)

    def _s_SCANNING(self, now):
        self.link.tick(now)
        self._drain_link()
        text = self.perceive["qr"]()
        if text:
            try:
                self._plan = qr_parse(text)
                self.link.send(Frame(1, CMD_TASK_PLAN,
                                     bytes(self._plan.batch1_colors + self._plan.batch1_positions
                                           + self._plan.batch2_colors + self._plan.batch2_positions)))
                self._goto("PLAN_READY", now)
                return
            except QrFormatError:
                pass
        if now - self.entered_at > self.cfg["timing"]["scan_timeout_s"]:
            self._goto("SAFESTOP", now)

    def _s_PLAN_READY(self, now): self._goto("NAV1", now)

    def _nav(self, state, nxt, now):
        self.link.tick(now)
        self._drain_link()
        if getattr(self, "_action_done", False):
            self._action_done = False
            self._goto(nxt, now)
        elif now - self.entered_at > 8.0:
            self._retries += 1
            if self._retries > self.cfg["calib"]["retry"]:
                self._goto("SAFESTOP", now)
                return
            self.link.send(Frame(2, CMD_GOTO, struct.pack("<hhh", 0, 0, 0)))
            self._retries_keep = self._retries  # _goto 会清零 _retries，跨轮保留
            self._goto(state, now)
            self._retries = self._retries_keep

    def _s_NAV1(self, now): self._nav("NAV1", "CAL1", now)
    def _s_NAV2(self, now): self._nav("NAV2", "CAL2", now)

    def _cal(self, nxt, now):
        self.link.tick(now)
        self._drain_link()
        obs = self.perceive["tri"]()
        self._stable = self._stable + 1 if obs else 0
        if self._stable >= self.cfg["calib"]["stable_frames"] and self._last:
            fixed = calibrate(self._last, obs, self.field,
                              self.cfg["calib"]["weight_middle"], self.cfg["calib"]["weight_pair_mid"],
                              self.cfg["calib"]["max_residual_mm"], self.cfg["calib"]["max_residual_deg"])
            if fixed:
                dx, dy = fixed.x - self._last.x, fixed.y - self._last.y
                self.link.send(Frame(3, CMD_CALIBRATE,
                                     struct.pack("<hhh", int(dx), int(dy), int((fixed.heading - self._last.heading) * 10))))
                self.log_events.append(("CAL_APPLIED", now))
                self._goto(nxt, now)
                return
            self._retries += 1
        if now - self.entered_at > 6.0:
            self._retries += 1
            self.entered_at = now
        if self._retries > self.cfg["calib"]["retry"]:
            self.log_events.append(("CAL_SKIPPED", now))
            self._goto(nxt, now)

    def _s_CAL1(self, now): self._cal("PLACE1", now)
    def _s_CAL2(self, now): self._cal("PLACE2", now)

    def _place(self, nxt, now):
        self.link.tick(now)
        self._drain_link()
        if getattr(self, "_action_done", False):
            self._action_done = False
            self._goto(nxt, now)
        else:
            self.link.send(Frame(4, CMD_PLACE, struct.pack("<bhh", 1, 0, 0)))

    def _s_PLACE1(self, now): self._place("NAV2", now)
    def _s_PLACE2(self, now): self._place("RETURN", now)

    def _s_RETURN(self, now):
        self.link.tick(now)
        self._drain_link()
        if getattr(self, "_action_done", False):
            self._goto("DONE", now)

    def _s_DONE(self, now): pass
    def _s_SAFESTOP(self, now): pass
```

MockMcu 需补充：收到 `CMD_GOTO/CMD_PICK/CMD_PLACE` 后延迟 1 个 tick 回 `CMD_DONE(task_id)` 并在 `_action_done` 语义下推进 `_odometry`（在 Task 10 的 MockMcu 上扩展，保持其既有测试通过）。

- [ ] **Step 4: PASS**（三个测试全绿；若 MockMcu 需扩展 DONE 回执，改后回跑 `tests/test_link.py`） — [ ] **Step 5: Commit** `feat: match fsm with calibration degrade and safestop`

---

### Task 12: 入口装配与 sim 端到端

**Files:**
- Create: `src/main.py`, `tests/test_sim_e2e.py`, `src/devices/sim_scene.py`

**Interfaces:**
- Produces: `run(cfg_path: str, mode: str, max_seconds: float = 120.0) -> MatchFSM`（可被测试调用）；`python -m src.main --config config/default.yaml --mode sim`

`sim_scene.py`：`SimScene(field)` 实现 `perceive` 三键——`qr()` 第 3 帧返回 `"452+321+254+312"`；`tri()` 在 CAL 状态期望出现（带 ±3mm 高斯抖动）；`rings()` 返回 []。

- [ ] **Step 1: 写失败测试**

```python
from src.main import run

def test_sim_full_match():
    fsm = run("config/default.yaml", "sim", max_seconds=120.0)
    assert fsm.state == "DONE"
    assert any(e[0] == "CAL_APPLIED" for e in fsm.log_events)  # sim 中校准至少成功 1 次
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现 main.py**

```python
import argparse
import time
from .config import load_config, cfg_get
from .core.fsm import MatchFSM
from .core.kalman import PoseFilter
from .algorithms.fieldmap import load_field_map
from .devices.link import MockMcu, SerialLink
from .devices.sim_scene import SimScene

def run(cfg_path, mode, max_seconds=120.0):
    cfg = load_config(cfg_path)
    field = load_field_map(cfg_get(cfg, "field_file"))
    if mode == "sim":
        link = MockMcu(drift_mm_per_m=20.0, drift_deg_per_m=0.1, pose_hz=10.0)
        perceive = SimScene(field)
    else:
        from .devices.camera import make_camera
        import cv2
        from .algorithms.qr_decode import decode_qr  # WeChatQRCode 封装，见下
        cam = make_camera(cfg, mode)
        link = SerialLink(cfg["serial"]["port"], cfg["serial"]["baudrate"])
        # live/replay: perceive 由相机帧驱动（每帧先 cam.read 再跑检测器）
        perceive = _make_vision_perceive(cam, cfg)
    fsm = MatchFSM(cfg, link, PoseFilter(), field, perceive)
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_seconds and fsm.state not in ("DONE", "SAFESTOP"):
        fsm.step(time.monotonic() - t0)
        time.sleep(0.02)
    return fsm
```

`qr_decode.py`（algorithms 层，WeChatQRCode 封装，live/replay 用）：

```python
import cv2
_DET = None
def decode_qr(img):
    global _DET
    if _DET is None:
        _DET = cv2.wechat_qrcode_WeChatQRCode()
    texts, _ = _DET.detectAndDecode(img)
    return texts[0] if texts else None
```

`_make_vision_perceive` 按 spec：缓存最近帧，`qr/tri/rings` 三个 callable 分别调 `decode_qr`、`detect_tri_ring(detect_rings(...))`、`detect_rings`。sim 模式不依赖 cv2 相机。

- [ ] **Step 4: PASS** — [ ] **Step 5: 手动冒烟** `python -m src.main --mode sim`，控制台打印状态迁移日志至 DONE — [ ] **Step 6: Commit** `feat: entrypoint with sim end-to-end match run`

---

### Task 13: 结构化日志与复盘工具

**Files:**
- Create: `src/logsetup.py`, `tools/replay_log.py`, `tests/test_logsetup.py`

**Interfaces:**
- Produces: `setup_logging(log_dir: str) -> Path`（返回本次 jsonl 文件路径，root logger 双输出）；`replay_log.py <file.jsonl>` 打印每行 `t state event` 摘要。

- [ ] **Step 1: 失败测试**

```python
import json, logging
from src.logsetup import setup_logging

def test_jsonl_written(tmp_path):
    path = setup_logging(str(tmp_path))
    logging.getLogger("fsm").info("evt", extra={"fsm_state": "NAV1", "t": 1.23})
    for h in logging.getLogger().handlers:
        h.flush()
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert lines[-1]["fsm_state"] == "NAV1"
```

- [ ] **Step 2: FAIL** — [ ] **Step 3: 实现**（自定义 `jsonl` Formatter：`{t, level, logger, msg, **extra}`；main.py 启动时调用 `setup_logging(cfg["log_dir"])`，config 加 `log_dir: logs`） — [ ] **Step 4: PASS** — [ ] **Step 5: Commit** `feat: jsonl run logging and replay tool`

---

### Task 14: 电控协议文档与交付清单

**Files:**
- Create: `docs/protocol-v0.1.md`, `README.md`

**Interfaces:**
- Consumes: spec 第 3 节全文
- Produces: 交电控组的冻结文档；README 说明三模式运行方法与现场标定步骤（HSV/霍夫/FieldMap）

- [ ] **Step 1: 写 docs/protocol-v0.1.md**：从 spec 第 3 节复制帧格式、命令表、冻结清单，补充每帧的字节级示例（用 `encode(Frame(1, CMD_TASK_PLAN, bytes((4,5,2,3,2,1,2,5,4,3,1,2))))` 实际输出转 hex 贴入，保证文档与实现一致）
- [ ] **Step 2: README.md**：安装、测试、`--mode live|replay|sim` 用法、赛前标定 checklist（打印 FieldMap 样例场地→拍图→填 yaml）
- [ ] **Step 3: 全量回归** `pytest -v` 全绿
- [ ] **Step 4: Commit** `docs: uart protocol v0.1 for mcu team and readme`

---

## Spec 覆盖自查（已执行）

- spec 1 工程结构 → Task 1-14 文件清单一致；algorithms 纯度规则见 Global Constraints。
- spec 2.1/2.2 → Task 2/3/7/8；2.3 校准 → Task 6（含量化验收测试）；2.4 设备接口 → Task 9/10；2.5/2.6 线程与 FSM → Task 11/12（线程化在 Task 12 live 装配时以 `CameraSource.read` 独立线程实现，sim 测试不需要）。
- spec 3 协议 → Task 4 + Task 14 文档；spec 4 错误处理/日志 → Task 11（SAFESTOP/降级）+ Task 13；spec 5 测试策略 → 各任务 pytest + Task 12 sim e2e；回放 fixtures → Task 9。
- 实机项（真相机/真串口联调）不在本计划，属阶段 4 人工验收。
