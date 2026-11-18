"""状态机测试（V2.1.0 语义）。

V2.1.0 的视觉是**测量仪器**：只有 SCANNING（读码发计划）和 SERVICE（持续测量）
两个工作状态。它不规划路径、不下发目标点、不判断动作是否完成。

这里测的是新契约里真正重要的东西：

- 扫码阶段的容错（不自洽的码不发计划、继续扫，最终超时安全停止）；
- 任务计划一定要得到应答（重发原帧，超限安全停止）；
- **没有心跳**：周期性测量帧本身就是活跃信号，静默不等于断链；
- 位姿/转盘上报的 50ms 节流；
- 区域切换时丢弃上一帧位姿先验；
- 臂往返：0x0006 → 0x0008 → done，丢帧靠重发补齐。
"""
import struct

import pytest
import yaml

from src.algorithms.fieldmap import load_field_map
from src.core.fsm import MatchFSM
from src.core.protocol import (ACK_DUPLICATE, ACK_FAILED, ACK_OK, CMD_ACK,
                               CMD_ARM_SOLUTION, CMD_ARM_STATE, CMD_POSE,
                               CMD_REGION, CMD_TASK_PLAN, CMD_TURNTABLE,
                               REGION_MATERIAL, REGION_ROUGH, REGION_STAGING,
                               Frame)
from src.devices.sim_scene import SimScene
from tests.conftest import repo_path

FIELD = load_field_map(repo_path("config", "field.example.yaml"))


def _cfg():
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class FakeLink:
    """可编程链路：记录所有外发帧，``poll`` 消费 rx 队列。

    ``ack_plan`` 控制是否对任务计划回应答；``ack_solution`` 控制是否对解算帧
    回 done。两者分开才能构造「计划通了但臂往返丢帧」这类场景。
    """

    def __init__(self, ack_plan=True, ack_solution=True):
        self.sent = []
        self.rx = []
        self._ack_plan = ack_plan
        self._ack_solution = ack_solution
        self.done_results = []

    def send(self, frame):
        self.sent.append(frame)
        if frame.cmd_id == CMD_TASK_PLAN and self._ack_plan:
            self.rx.append(Frame(seq=0, cmd_id=CMD_ACK,
                                 data=bytes((frame.seq, ACK_OK))))
        elif frame.cmd_id == CMD_ARM_SOLUTION and self._ack_solution:
            self.done_results.append((frame.seq, ACK_OK))
            self.rx.append(Frame(seq=0, cmd_id=CMD_ACK,
                                 data=bytes((frame.seq, ACK_OK))))

    def poll(self):
        return self.rx.pop(0) if self.rx else None

    def tick(self, now):
        pass

    def cmd_ids(self):
        return [f.cmd_id for f in self.sent]

    def of(self, cmd_id):
        return [f for f in self.sent if f.cmd_id == cmd_id]


def _fsm(link, qr=None, scene=None):
    perceive = scene or SimScene(FIELD)
    if qr is not None:
        perceive.qr = qr
    return MatchFSM(_cfg(), link, FIELD, perceive)


def _run(fsm, seconds=10.0, dt=0.02, start=0.0):
    now = start
    end = start + seconds
    while now < end:
        fsm.step(now)
        if fsm.state == "SAFESTOP":
            break
        now += dt
    return now


# ---------- 扫码阶段 ----------

def test_scanning_enters_service_after_plan_acked():
    link = FakeLink()
    fsm = _fsm(link)
    _run(fsm, seconds=2.0)
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
    """文档表 1-15：应答超时重发，重发 3 次仍无应答则安全停止。"""
    link = FakeLink(ack_plan=False)
    fsm = _fsm(link)
    _run(fsm, seconds=10.0)
    assert fsm.state == "SAFESTOP"
    plans = link.of(CMD_TASK_PLAN)
    assert len(plans) == 4, f"首发 + 重发 3 次 = 4 帧，实际 {len(plans)}"
    assert len({f.seq for f in plans}) == 1, "重发必须是原帧（seq 不变）"
    assert len({bytes(f.data) for f in plans}) == 1


def test_plan_failure_reply_stops_immediately():
    """电控明确回「失败」时不必等超时：两边对不上，再扫也扫不出别的码。"""
    class Refusing(FakeLink):
        def send(self, frame):
            self.sent.append(frame)
            if frame.cmd_id == CMD_TASK_PLAN:
                self.rx.append(Frame(seq=0, cmd_id=CMD_ACK,
                                     data=bytes((frame.seq, ACK_FAILED))))

    link = Refusing()
    fsm = _fsm(link)
    _run(fsm, seconds=1.0)
    assert fsm.state == "SAFESTOP"
    assert any(e[0] == "NACK" for e in fsm.log_events)
    assert len(link.of(CMD_TASK_PLAN)) == 1, "已明确拒绝就不该再重发"


# ---------- 没有心跳，静默不等于断链 ----------

def test_there_is_no_heartbeat_frame():
    """V2.1.0 取消心跳：一条 0x0001 都不该出现。"""
    link = FakeLink()
    fsm = _fsm(link)
    _run(fsm, seconds=3.0)
    assert link.of(0x0001) == []


def test_long_silence_from_mcu_does_not_stop_the_vision():
    """电控停着作业十几秒不发一帧是正常的，视觉不得据此安全停止。

    这条是取消心跳后的关键性质：把静默当断链只能得到误报。
    """
    link = FakeLink()
    fsm = _fsm(link)
    now = _run(fsm, seconds=1.0)
    assert fsm.state == "SERVICE"

    link.rx.clear()
    now = _run(fsm, seconds=30.0, start=now)    # 半分钟没有任何来帧
    assert fsm.state == "SERVICE", "静默不得触发安全停止"


def test_periodic_measurement_frames_keep_flowing_during_silence():
    """静默期间测量帧照发：链路是活的这件事由这些帧证明，不靠心跳。"""
    link = FakeLink()
    scene = SimScene(FIELD, jitter_mm=0.0)
    scene.pose_truth = __import__("src.algorithms.types", fromlist=["Pose"]).Pose(
        1500.0, 1000.0, 270.0)
    fsm = _fsm(link, scene=scene)
    now = _run(fsm, seconds=1.0)
    fsm._region = REGION_ROUGH
    link.rx.clear()
    _run(fsm, seconds=2.0, start=now)
    assert link.of(CMD_POSE), "静默期间位姿必须继续发"


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
        fsm.step(k * 0.01)
    poses = link.cmd_ids().count(CMD_POSE) - before
    assert 39 <= poses <= 41, f"2 秒内应约 40 帧位姿，实际 {poses}"
    tt = link.cmd_ids().count(CMD_TURNTABLE) - before_tt
    assert 39 <= tt <= 41, f"转盘也应约 40 帧，实际 {tt}"


def test_pose_is_not_sent_faster_than_50ms():
    """真正的门控断言：两次上报的最小间隔不得小于 50ms。"""
    fsm, link, now = _service_fsm()
    stamps = []
    end = now + 0.5
    while now < end:
        n_before = len(link.sent)
        fsm.step(now)
        if any(f.cmd_id == CMD_POSE for f in link.sent[n_before:]):
            stamps.append(now)
        now += 0.005                # 5ms 一拍，远快于上报周期
    assert len(stamps) >= 8
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 0.049, f"上报间隔不得小于 50ms，实测最小 {min(gaps)*1000:.1f}ms"


def test_no_pose_before_first_region_hint():
    fsm, link, now = _service_fsm(region=None)
    end = now + 1.0
    while now < end:
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
    """0x0001/0x0201/0x0202/0x0203 已删除：收到不得报错，也不得改变状态。"""
    fsm, link, now = _service_fsm()
    for cmd in (0x0001, 0x0201, 0x0202, 0x0203):
        link.rx.append(Frame(seq=0, cmd_id=cmd, data=b"\x01\x02"))
    fsm.step(now)
    assert fsm.state == "SERVICE"
    assert not any(e[0].startswith("BAD_")
                   for e in fsm.log_events), fsm.log_events


# ---------- 臂解算往返 ----------

def _arm_frame(seq=1, l_mm=600, h_mm=300, alpha01=0, target_id=0):
    return Frame(seq=seq, cmd_id=CMD_ARM_STATE,
                 data=struct.pack("<HHHb", l_mm, h_mm, alpha01, target_id))


def _arm_ready(**scene_kw):
    """进 SERVICE 并让画面里有一个已知目标点。"""
    from src.algorithms.types import Pose
    link = FakeLink()
    scene = SimScene(FIELD, jitter_mm=0.0, arm_target_body=(500.0, 100.0),
                     **scene_kw)
    scene.pose_truth = Pose(1500.0, 1000.0, 270.0)
    fsm = _fsm(link, scene=scene)
    now = _run(fsm, seconds=1.0)
    fsm._region = REGION_ROUGH
    return fsm, link, scene, now


def test_arm_state_triggers_solution_within_one_step():
    fsm, link, scene, now = _arm_ready()
    link.rx.append(_arm_frame(seq=5))
    fsm.step(now + 0.02)

    sols = link.of(CMD_ARM_SOLUTION)
    assert len(sols) == 1, "收到臂状态必须立刻解算，不该等到下一个上报周期"
    assert sols[0].seq != 5, "解算帧用它自己的 seq（电控靠 echo_seq 配对）"

    d_mm, alpha01, status, echo_seq = struct.unpack("<hhBB", sols[0].data[:6])
    assert echo_seq == 5, "echo_seq 必须带回请求的 seq"
    assert status & 0x80, "有目标时应置可用位"


def test_solution_echoes_seq_not_reuses_it():
    """解算帧的 seq 来自视觉自己的计数器，不能直接抄请求的 seq。

    两侧各有一套 seq 空间，混用会让双方的应答匹配错位。
    """
    fsm, link, scene, now = _arm_ready()
    for k, s in enumerate((11, 12, 13)):
        link.rx.append(_arm_frame(seq=s))
        fsm.step(now + 0.02 + k * 0.02)
    echoed = [struct.unpack("<hhBB", f.data[:6])[3]
              for f in link.of(CMD_ARM_SOLUTION)]
    assert echoed == [11, 12, 13]


def test_solution_matches_arm_geometry():
    """端到端：解出的 d、α 代回正解必须落在目标点上。"""
    import math
    fsm, link, scene, now = _arm_ready()
    link.rx.append(_arm_frame(seq=1, l_mm=600))
    fsm.step(now + 0.02)

    d_mm, alpha01, _status, _echo = struct.unpack(
        "<hhBB", link.of(CMD_ARM_SOLUTION)[0].data[:6])
    l = 600.0
    a = math.radians(alpha01 / 10.0)
    # 车身 (500,100) → 电机系是 (500, 100 + 620/2 = 410)
    assert abs(d_mm + l * math.cos(a) - 500.0) < 1.5, "d 反查 x 不符"
    assert abs(l * math.sin(a) - 410.0) < 1.5, "α 反查 y 不符"


def test_no_target_reports_explicitly():
    """画面里没有目标时，必须报「没测到」而不是发一个假坐标。"""
    fsm, link, scene, now = _arm_ready()
    scene.arm_target_body = None
    link.rx.append(_arm_frame(seq=1))
    fsm.step(now + 0.02)

    sol = link.of(CMD_ARM_SOLUTION)[0]
    assert sol.data[4] & 0x03 == 2, "结果字节应为「没测到目标」"
    assert not sol.data[4] & 0x80, "没测到目标时不得置可用位"
    assert any(e[0] == "ARM_NO_TARGET" for e in fsm.log_events)


def test_unreachable_target_reports_no_solution():
    """横向超出臂长：明确报无解，电控据此决定挪车。"""
    fsm, link, scene, now = _arm_ready()
    scene.arm_target_body = (500.0, 310.0)   # 电机系 y = 310 + 310 = 620 > 600
    link.rx.append(_arm_frame(seq=1, l_mm=600))
    fsm.step(now + 0.02)

    status = link.of(CMD_ARM_SOLUTION)[0].data[4]
    assert status & 0x03 == 1, "结果字节应为「够不着」"
    assert not status & 0x80


def test_bad_arm_length_is_reported():
    fsm, link, scene, now = _arm_ready()
    link.rx.append(Frame(seq=1, cmd_id=CMD_ARM_STATE, data=b"\x01\x02"))
    fsm.step(now + 0.02)
    assert any(e[0] == "BAD_ARM_LEN" for e in fsm.log_events)
    assert link.of(CMD_ARM_SOLUTION) == []


def test_duplicate_arm_request_resends_same_frame():
    """电控没收到解算帧而重发同一个 seq：原样重发，不重新解算。

    重发必须逐字节一致——电控靠「同 seq 的解算帧」识别重复请求，若内容变了
    它就分不清这是新请求还是重发。
    """
    fsm, link, scene, now = _arm_ready()
    link._ack_solution = False          # done 丢了，解算帧还挂在待确认槽
    link.rx.append(_arm_frame(seq=7))
    fsm.step(now + 0.02)
    first = link.of(CMD_ARM_SOLUTION)[0]

    link.rx.append(_arm_frame(seq=7))   # 电控重发同一个请求
    fsm.step(now + 0.04)
    sols = link.of(CMD_ARM_SOLUTION)
    assert len(sols) == 2
    assert sols[1].seq == first.seq, "重发必须用同一个 seq"
    assert sols[1].data == first.data, "重发必须逐字节一致"
    assert any(e[0] == "ARM_RESEND" for e in fsm.log_events)


def test_new_arm_request_after_done_is_solved_again():
    """上一轮收到 done 后，新的请求（seq 不同）必须重新解算。"""
    fsm, link, scene, now = _arm_ready()
    link.rx.append(_arm_frame(seq=7))
    fsm.step(now + 0.02)
    fsm.step(now + 0.04)                # 收 done，清待确认槽

    scene.arm_target_body = (600.0, 0.0)
    link.rx.append(_arm_frame(seq=8))
    fsm.step(now + 0.06)
    sols = link.of(CMD_ARM_SOLUTION)
    assert len(sols) == 2
    assert sols[0].data != sols[1].data, "目标变了，结果也应不同"


def test_solution_waits_for_done_with_longer_timeout():
    """等 done 用的是执行超时，比纯应答的 100ms 长得多。

    done 在动作**执行完**才发，用 100ms 判超时会一直误判、疯狂重发。
    """
    fsm, link, scene, now = _arm_ready()
    link._ack_solution = False
    link.rx.append(_arm_frame(seq=1))
    fsm.step(now + 0.02)
    assert len(link.of(CMD_ARM_SOLUTION)) == 1

    # 100ms 后还没超时（exec_timeout_ms = 2000）
    fsm.step(now + 0.12)
    assert len(link.of(CMD_ARM_SOLUTION)) == 1, "不应按 100ms 就重发"

    # 超过 2s 才重发
    fsm.step(now + 2.5)
    assert len(link.of(CMD_ARM_SOLUTION)) == 2, "超过执行超时应重发"


def test_done_timeout_eventually_stops():
    """done 一直不来：重发 3 次后安全停止，不硬撑。"""
    fsm, link, scene, now = _arm_ready()
    link._ack_solution = False
    link.rx.append(_arm_frame(seq=1))
    now += 0.02
    fsm.step(now)
    for _ in range(6):
        now += 2.5
        fsm.step(now)
        if fsm.state == "SAFESTOP":
            break
    assert fsm.state == "SAFESTOP"
    assert any(e[0] == "ACK_TIMEOUT" and e[1] == CMD_ARM_SOLUTION
               for e in fsm.log_events)


def test_duplicate_result_ack_is_not_an_error():
    """电控认出重复请求后回 ACK_DUPLICATE，视觉应视为正常结束。"""
    class Dup(FakeLink):
        def send(self, frame):
            self.sent.append(frame)
            if frame.cmd_id == CMD_TASK_PLAN:
                self.rx.append(Frame(0, CMD_ACK, bytes((frame.seq, ACK_OK))))
            elif frame.cmd_id == CMD_ARM_SOLUTION:
                self.rx.append(Frame(0, CMD_ACK,
                                     bytes((frame.seq, ACK_DUPLICATE))))

    from src.algorithms.types import Pose
    link = Dup()
    scene = SimScene(FIELD, jitter_mm=0.0, arm_target_body=(500.0, 100.0))
    scene.pose_truth = Pose(1500.0, 1000.0, 270.0)
    fsm = _fsm(link, scene=scene)
    now = _run(fsm, seconds=1.0)
    link.rx.append(_arm_frame(seq=1))
    fsm.step(now + 0.02)
    fsm.step(now + 0.04)
    assert fsm.state == "SERVICE"
    assert CMD_ARM_SOLUTION not in fsm._pending
