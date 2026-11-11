"""状态机测试（v2.0.0 语义）。

v2.0.0 的视觉是**测量仪器**：只有 SCANNING（读码发计划）和 SERVICE（持续测量）
两个工作状态。它不规划路径、不下发目标点、不判断动作完成，所以 v1.0.0 里
那些「导航超时」「动作完成推进」「校准修正量」的测试都已不存在。这里测的是
新契约里真正重要的东西：

- 扫码阶段的容错（不自洽的码不发计划、继续扫，最终超时安全停止）；
- 任务计划一定要得到应答（重发原帧，超限安全停止）；
- 断链超时；
- 位姿/转盘上报的 50ms 节流；
- 区域切换时丢弃上一帧位姿先验。
"""
import yaml

import pytest

from src.algorithms.fieldmap import load_field_map
from src.core.fsm import MatchFSM
from src.core.protocol import (CMD_ACK, CMD_HEARTBEAT, CMD_POSE, CMD_REGION,
                               CMD_TASK_PLAN, CMD_TURNTABLE,
                               REGION_MATERIAL, REGION_ROUGH, REGION_STAGING,
                               Frame)
from src.devices.sim_scene import SimScene
from tests.conftest import repo_path

FIELD = load_field_map(repo_path("config", "field.example.yaml"))


def _cfg():
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class FakeLink:
    """可编程链路：记录所有外发帧，``poll`` 消费 rx 队列。"""

    def __init__(self, ack=True):
        self.sent = []
        self.rx = []
        self._ack = ack

    def send(self, frame):
        self.sent.append(frame)
        if self._ack:
            self.rx.append(Frame(seq=0, cmd_id=CMD_ACK, data=bytes([frame.seq])))

    def poll(self):
        return self.rx.pop(0) if self.rx else None

    def tick(self, now):
        pass

    def cmd_ids(self):
        return [f.cmd_id for f in self.sent]


class SilentLink(FakeLink):
    """完全不回的链路：用于验证断链超时。"""

    def __init__(self):
        super().__init__(ack=False)


def _fsm(link, qr=None, scene=None):
    perceive = scene or SimScene(FIELD)
    if qr is not None:
        perceive.qr = qr
    return MatchFSM(_cfg(), link, FIELD, perceive)


def _run(fsm, seconds=10.0, dt=0.02):
    now = 0.0
    while now < seconds:
        fsm.step(now)
        if fsm.state == "SAFESTOP":
            break
        now += dt
    return fsm


# ---------- 扫码阶段 ----------

def test_scanning_enters_service_after_plan_acked():
    link = FakeLink()
    fsm = _run(_fsm(link), seconds=2.0)
    assert fsm.state == "SERVICE"
    assert CMD_TASK_PLAN in link.cmd_ids()


def test_bad_qr_is_rejected_and_scanning_continues():
    """格式非法或内容不自洽都不发计划；连扫不到就超时安全停止。"""
    link = FakeLink()
    fsm = _fsm(link, qr=lambda: "452+321+254+612")
    _run(fsm, seconds=1.0)
    assert fsm.state == "SCANNING", "不自洽的码不得推进到 SERVICE"
    assert CMD_TASK_PLAN not in link.cmd_ids()
    assert any(e[0] == "PLAN_REJECTED" for e in fsm.log_events)


def test_scan_timeout_goes_safestop():
    link = FakeLink()
    fsm = _fsm(link, qr=lambda: None)
    _run(fsm, seconds=20.0)
    assert fsm.state == "SAFESTOP"
    assert any(e[0] == "SCAN_TIMEOUT" for e in fsm.log_events)


def test_unacked_plan_is_resent_as_same_frame_then_safestop():
    """文档表 1-14：应答超时重发，重发 3 次仍无应答则安全停止。"""
    link = SilentLink()
    fsm = _fsm(link)
    _run(fsm, seconds=10.0)
    assert fsm.state == "SAFESTOP"
    plans = [f for f in link.sent if f.cmd_id == CMD_TASK_PLAN]
    assert len(plans) == 4, f"首发 + 重发 3 次 = 4 帧，实际 {len(plans)}"
    assert len({f.seq for f in plans}) == 1, "重发必须是原帧（seq 不变）"
    assert len({bytes(f.data) for f in plans}) == 1


# ---------- 断链与心跳 ----------

def test_heartbeat_is_sent_periodically():
    """心跳是周期帧，与是否有测量结果、是否收到应答都无关。"""
    link = FakeLink()
    fsm = _fsm(link)
    _run(fsm, seconds=1.9)
    beats = [f for f in link.sent if f.cmd_id == CMD_HEARTBEAT]
    assert len(beats) >= 3, f"1.9 秒内应有至少 3 次心跳，实际 {len(beats)}"


def test_link_timeout_goes_safestop_after_silence():
    """通信建立后对方突然静默超过 2s → 安全停止（文档表 1-14）。

    注意「建立后」这个前提：开机起就一帧未收时不能按断链判（可能只是主控还没
    启动），那条路径由任务计划的应答超时兜底。故这里先让链路正常回一帧，再静默。
    """
    link = FakeLink()
    fsm = _fsm(link)
    _run(fsm, seconds=1.0)                 # 正常通信，进入 SERVICE
    assert fsm.state == "SERVICE"
    assert fsm._last_rx_at is not None

    link._ack = False                      # 此后彻底静默
    link.rx.clear()
    _run(fsm, seconds=10.0)
    assert fsm.state == "SAFESTOP"
    assert any(e[0] == "LINK_TIMEOUT" for e in fsm.log_events)


def test_receiving_frames_keeps_link_alive():
    """持续收到来帧时不得误判断链。"""
    link = FakeLink()
    fsm = _fsm(link)
    now = 0.0
    for _ in range(300):            # 6 秒
        fsm.step(now)
        now += 0.02
    assert fsm.state == "SERVICE"
    assert not any(e[0] == "LINK_TIMEOUT" for e in fsm.log_events)


# ---------- SERVICE 的节流 ----------

def _service_fsm(region=REGION_ROUGH, truth=None):
    """走完扫码进 SERVICE，返回 (fsm, link, 当前虚拟时刻)。

    返回时刻很重要：主循环的 ``now`` 必须单调递增。测试若从头再传 0.0，
    FSM 内部 ``_last_pose_at`` 已经约 1.0，节流判断会被时间倒退搅乱。
    """
    from src.algorithms.types import Pose
    link = FakeLink()
    scene = SimScene(FIELD, jitter_mm=0.0)
    scene.pose_truth = truth or Pose(1500.0, 1000.0, 270.0)
    fsm = _fsm(link, scene=scene)
    now = 0.0
    while now < 1.0:
        fsm.step(now)
        now += 0.02
    assert fsm.state == "SERVICE"
    fsm._region = region
    fsm._last_rx_at = None          # 让断链检测不因测试里没来帧而触发
    return fsm, link, now


def test_pose_and_turntable_are_throttled_to_50ms():
    """上报周期是 50ms：主循环比它快时，实际间隔按 50ms 向上取整到下一拍。

    这里用 10ms 的最小拍、并以整数毫秒推进时钟（``k * 0.01``），避免浮点累加
    误差把 2 秒算成 1999ms 而少一帧。
    """
    fsm, link, now = _service_fsm()
    before = link.cmd_ids().count(CMD_POSE)
    before_tt = link.cmd_ids().count(CMD_TURNTABLE)
    k0 = int(round(now * 100))
    for k in range(k0, k0 + 200):       # 2 秒，每拍 10ms
        fsm._last_rx_at = None
        fsm.step(k * 0.01)
    poses = link.cmd_ids().count(CMD_POSE) - before
    assert 39 <= poses <= 41, f"2 秒内应约 40 帧位姿，实际 {poses}"
    tt = link.cmd_ids().count(CMD_TURNTABLE) - before_tt
    assert 39 <= tt <= 41, f"转盘也应约 40 帧，实际 {tt}"


def test_pose_is_not_sent_faster_than_50ms():
    """真正的门控断言：两次上报的最小间隔不得小于 50ms。"""
    from src.core.protocol import CMD_POSE as POSE_CMD

    fsm, link, now = _service_fsm()
    stamps = []
    end = now + 0.5
    while now < end:
        fsm._last_rx_at = None
        n_before = len(link.sent)
        fsm.step(now)
        if any(f.cmd_id == POSE_CMD for f in link.sent[n_before:]):
            stamps.append(now)
        now += 0.005                # 5ms 一拍，远快于上报周期
    assert len(stamps) >= 8
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 0.049, f"上报间隔不得小于 50ms，实测最小 {min(gaps)*1000:.1f}ms"


def test_no_pose_before_first_region_hint():
    fsm, link, now = _service_fsm(region=None)
    end = now + 1.0
    while now < end:
        fsm._last_rx_at = None
        fsm.step(now)
        now += 0.02
    assert link.cmd_ids().count(CMD_POSE) == 0


def test_switching_region_drops_stale_pose_prior():
    """两组靶相距很远，跨区域后上一帧位姿不再是有效先验，必须清掉。"""
    fsm, link, now = _service_fsm()
    fsm.step(now)
    assert fsm._last_pose is not None

    link.rx.append(Frame(seq=0, cmd_id=CMD_REGION,
                         data=bytes((REGION_STAGING,))))
    fsm.step(now + 0.02)
    assert fsm._region == REGION_STAGING
    assert fsm._last_pose is None, "换区域后必须丢弃上一帧位姿"


def test_same_region_hint_keeps_pose_prior():
    fsm, link, now = _service_fsm()
    fsm.step(now)
    kept = fsm._last_pose
    assert kept is not None
    link.rx.append(Frame(seq=0, cmd_id=CMD_REGION,
                         data=bytes((REGION_ROUGH,))))
    fsm.step(now + 0.02)
    assert fsm._last_pose == kept, "同一区域重复提示不应丢掉先验"


def test_bad_region_length_is_reported_not_fatal():
    fsm, link, now = _service_fsm()
    link.rx.append(Frame(seq=0, cmd_id=CMD_REGION, data=b"\x02\x03"))
    fsm.step(now)
    assert any(e[0] == "BAD_REGION_LEN" for e in fsm.log_events)
    assert fsm.state == "SERVICE"


def test_retired_commands_are_silently_ignored():
    """0x0201/0x0202/0x0203 已删除：收到不得报错，也不得改变状态。"""
    fsm, link, now = _service_fsm()
    for cmd in (0x0201, 0x0202, 0x0203):
        link.rx.append(Frame(seq=0, cmd_id=cmd, data=b"\x01\x02"))
    fsm.step(now)
    assert fsm.state == "SERVICE"
    assert not any(e[0] in ("BAD_POSE_LEN", "BAD_REGION_LEN")
                   for e in fsm.log_events)
