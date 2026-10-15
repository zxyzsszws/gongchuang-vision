import yaml
import pytest
from src.core.fsm import MatchFSM
from src.core.kalman import PoseFilter
from src.core.protocol import CMD_HEARTBEAT
from src.algorithms.fieldmap import load_field_map
from src.devices.link import MockMcu

# tri() 闭包引用模块级 FIELD（_fsm 内的同名赋值仅是局部变量，不创建全局），
# 故此处显式定义，测试函数本体保持计划简报逐字不变。
FIELD = load_field_map("config/field.example.yaml")


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
    cfg = yaml.safe_load(open("config/default.yaml", encoding="utf-8"))
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
