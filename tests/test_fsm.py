import struct
import yaml
import pytest
from src.core.fsm import MatchFSM
from src.core.kalman import PoseFilter
from src.core.protocol import (CMD_ACK, CMD_DONE, CMD_GOTO, CMD_HEARTBEAT,
                               CMD_PLACE, CMD_POSE, CMD_TASK_PLAN, Frame)
from src.algorithms.fieldmap import FieldMap, load_field_map
from src.devices.link import MockMcu
from tests.conftest import repo_path

# tri() 闭包引用模块级 FIELD（_fsm 内的同名赋值仅是局部变量，不创建全局），
# 故此处显式定义，测试函数本体保持计划简报逐字不变。
FIELD = load_field_map(repo_path("config", "field.example.yaml"))


def _fsm(qr_ok_after=2, tri_ok=True, calib_skip_after=5):
    cfg = yaml.safe_load(open(repo_path("config", "default.yaml"), encoding="utf-8"))
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
    FIELD = load_field_map(repo_path("config", "field.example.yaml"))
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


class _DeadLink:
    """完全不回的链路：poll() 永远 None，用于验证心跳超时 → SAFESTOP。"""
    def __init__(self):
        self.sent = []

    def send(self, frame):
        self.sent.append(frame)

    def poll(self):
        return None

    def tick(self, now):
        pass


def test_link_timeout_safestop():
    # 裁决 2：HEARTBEAT 500ms 周期发送；2s 无来流 → SAFESTOP，记录 LINK_TIMEOUT。
    cfg = yaml.safe_load(open(repo_path("config", "default.yaml"), encoding="utf-8"))
    dead = _DeadLink()
    fsm = MatchFSM(cfg, dead, PoseFilter(), FIELD,
                   {"qr": lambda: None, "tri": lambda: None, "rings": lambda: []})
    t = 0.0
    for _ in range(100):
        t += 0.05
        fsm.step(t)
        if fsm.state == "SAFESTOP":
            break
    assert fsm.state == "SAFESTOP"
    assert any(e[0] == "LINK_TIMEOUT" for e in fsm.log_events)
    assert any(f.cmd == CMD_HEARTBEAT for f in dead.sent)  # 心跳确实在发


# ---------- 最终评审修复波（F1–F4）：注入式链路 + 回归测试 ----------


class _FakeLink:
    """可编程 fake 链路：记录全部外发帧；rx 队列供 poll 消费。"""

    def __init__(self, ack=True, done_on=()):
        self.sent = []
        self.rx = []
        self._ack = ack
        self._done_on = done_on  # 触发 CMD_DONE 的命令集合

    def send(self, frame):
        self.sent.append(frame)
        if self._ack:
            self.rx.append(Frame(seq=9, cmd=CMD_ACK, data=bytes([frame.seq])))
        if frame.cmd in self._done_on:
            self.rx.append(Frame(seq=0, cmd=CMD_DONE, data=bytes([frame.seq & 0xFF])))

    def poll(self):
        return self.rx.pop(0) if self.rx else None

    def tick(self, now):
        pass


def _fsm_on(link, qr_text="452+321+254+312", tri=lambda: None, field=None):
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return MatchFSM(cfg, link, PoseFilter(), field or FIELD,
                    {"qr": (lambda: qr_text) if qr_text else (lambda: None),
                     "tri": tri, "rings": lambda: []})


def test_retry_resets_action_timeout_window():
    # F1 回归：链路全 ACK 但永不 DONE——重试必须重开超时窗口，
    # SAFESTOP 只允许在完整 retry×window 预算后触发（修复前 3 次 GOTO 挤在 180ms 内）。
    link = _FakeLink(ack=True)
    fsm = _fsm_on(link)
    nav_timeout = fsm.cfg["timing"].get("nav_timeout_s", 8.0)
    t = 0.0
    for _ in range(700):
        t += 0.05
        fsm.step(t)
    timeouts = [e for e in fsm.log_events if e[0] == "ACTION_TIMEOUT"]
    assert fsm.state == "SAFESTOP" and len(timeouts) == 1
    goto_cmds = [f.cmd for f in link.sent if f.cmd == CMD_GOTO]
    # 首发 + calib.retry(=2) 次重发，全部真实落在各自窗口末尾
    assert len(goto_cmds) == 3
    assert timeouts[0][2] >= nav_timeout * (len(goto_cmds) - 1) + nav_timeout - 0.5


def test_ack_gate_holds_control_frames_until_resolved():
    # F2 回归：链路从不 ACK——前一条控制帧重发预算耗尽前，
    # 新的控制帧（NAV1 的 GOTO）必须被闸住不得发出；预算耗尽 → ACK_TIMEOUT → SAFESTOP。
    # 心跳是周期帧，不受闸（证明闸只作用于控制帧）。
    link = _FakeLink(ack=False)
    fsm = _fsm_on(link)
    t = 0.0
    for _ in range(30):
        t += 0.05
        fsm.step(t)
        if fsm.state == "SAFESTOP":
            break
    assert any(e[0] == "ACK_TIMEOUT" for e in fsm.log_events)
    assert fsm.state == "SAFESTOP"
    sent_cmds = [f.cmd for f in link.sent]
    assert CMD_TASK_PLAN in sent_cmds                      # 第一条控制帧已发出并重发
    assert sent_cmds.count(CMD_TASK_PLAN) == 3             # 100ms × 3 次重发预算
    assert CMD_GOTO not in sent_cmds                       # 未解决前第二条控制帧被闸住
    assert any(f.cmd == CMD_HEARTBEAT for f in link.sent)  # 心跳不受闸


def test_bad_pose_length_does_not_crash_fsm():
    # 校验通过但 POSE data 长度不符：记录事件、丢弃该帧、状态机继续跑
    link = _FakeLink(ack=True, done_on=(CMD_GOTO, CMD_PLACE))
    link.rx.insert(0, Frame(seq=0, cmd=CMD_POSE, data=b"\x01\x02\x03"))
    fsm = _fsm_on(link)
    t = 0.0
    for _ in range(10):
        t += 0.05
        fsm.step(t)
    assert any(e[0] == "BAD_POSE_LEN" for e in fsm.log_events)
    assert fsm.state not in ("SAFESTOP",)   # 短帧不足以判断链（仍在收流）


def test_nav_and_place_data_use_plan_targets():
    # F3 回归：GOTO 目标取自 FieldMap 数字环（plan 位置首位→环心），
    # PLACE 第一字节 = 批次首个颜色号；RING 3→(1200,900)，RING 1→(1200,300)。
    # QR "452+321+254+312"：batch1 首位=3、batch2 首位=3 → NAV1/NAV2 都去 (1200,900)。
    link = _FakeLink(ack=True, done_on=(CMD_GOTO, CMD_PLACE))
    fsm = _fsm_on(link)
    t = 0.0
    for _ in range(3000):
        t += 0.05
        fsm.step(t)
        if fsm.state in ("DONE", "SAFESTOP"):
            break
    assert fsm.state == "DONE"
    gotos = [struct.unpack("<hhh", f.data) for f in link.sent if f.cmd == CMD_GOTO]
    assert gotos[0] == (1200, 900, 0)  # NAV1：batch1_positions[0]=3 → 3 号环心
    assert gotos[1] == (1200, 900, 0)  # NAV2：batch2_positions[0]=3 → 3 号环心
    assert gotos[-1] == (0, 0, 0)      # RETURN：回起点
    places = [struct.unpack("<bhh", f.data) for f in link.sent if f.cmd == CMD_PLACE]
    assert [p[0] for p in places] == [4, 2]  # batch1_colors[0], batch2_colors[0]
    assert all(p[1] == 0 and p[2] == 0 for p in places)  # 精细偏移属 Phase-4，保持 0


def test_nav_target_falls_back_to_origin_without_matching_ring():
    # F3 兜底：FieldMap 无匹配编号环 → GOTO(0,0,0)
    empty_field = FieldMap(tri=FIELD.tri, rings=[])
    link = _FakeLink(ack=True, done_on=(CMD_GOTO, CMD_PLACE))
    fsm = _fsm_on(link, field=empty_field)
    t = 0.0
    for _ in range(40):
        t += 0.05
        fsm.step(t)
    gotos = [struct.unpack("<hhh", f.data) for f in link.sent if f.cmd == CMD_GOTO]
    assert gotos and gotos[0] == (0, 0, 0)


def test_calibrate_uses_median_of_stable_frames():
    # F4 回归（spec §2.6 中位数防抖）：stable_frames 帧中混入一帧尖峰，
    #  applied 修正量必须等于中位数修正（+40mm），而不是末帧（尖峰）修正（−80mm）。
    from src.algorithms.types import RingObs, TriRingObs

    class _PoseLink(_FakeLink):
        """额外预置一条 POSE：滤波后位姿恒为 (1000,200,0)，修正量可精确预测。"""

        def __init__(self):
            super().__init__(ack=True, done_on=(CMD_GOTO, CMD_PLACE))
            self.rx.insert(0, Frame(seq=0, cmd=CMD_POSE,
                                    data=struct.pack("<hhhH", 1000, 200, 0, 1)))

    tri_n = {"n": 0}

    def tri():
        tri_n["n"] += 1
        ex, ey = -40.0, -10.0            # 观测 = 理论 + ε，ε = (−40,−10)
        if tri_n["n"] == 5:              # 第 5 稳定帧混入 +120mm x 尖峰
            ex += 120.0
        f = FIELD.tri
        return TriRingObs(RingObs((f.left.center[0] + ex, f.left.center[1] + ey), 20),
                          RingObs((f.middle.center[0] + ex, f.middle.center[1] + ey), 20),
                          RingObs((f.right.center[0] + ex, f.right.center[1] + ey), 20))

    link = _PoseLink()
    fsm = _fsm_on(link, tri=tri)
    now = 0.0
    for _ in range(200):
        now += 0.05
        fsm.step(now)
        if fsm.state != "NAV1" and any(e[0] == "CAL_APPLIED" for e in fsm.log_events):
            break
    cals = [e for e in fsm.log_events if e[0] == "CAL_APPLIED"]
    assert len(cals) == 1  # 进入 CAL1 后第 5 稳定帧即触发
    dx, dy = cals[0][1], cals[0][2]
    assert abs(dx - 40.0) < 5.0 and abs(dy - 10.0) < 5.0  # 中位数 → 尖峰被抑制
    assert abs(dx + 80.0) > 50.0                         # 绝不得是末帧（尖峰）修正值
