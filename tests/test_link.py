"""MockMcu / 链路层测试（V2.1.0 语义）。

V2.1.0 里机器人主控做的事：按自己规划行驶、发去向提示、收位姿后整体替换估计、
主动索取臂解算。它**不向视觉上报位姿**，**不发心跳**。

这里测重点是「哪些帧该回应答、哪些不该」以及臂往返的 `seq` 语义——这两处写错
在真机上表现为「莫名其妙地重发」或「同一个动作执行两遍」，都不容易现场定位。
"""
import math
import struct
import sys

import pytest

from src.core.protocol import (ACK_DUPLICATE, ACK_FAILED, ACK_OK,
                               CMD_ACK, CMD_ARM_SOLUTION, CMD_ARM_STATE,
                               CMD_POSE, CMD_REGION, CMD_TASK_PLAN,
                               CMD_TURNTABLE, REGION_MATERIAL, REGION_ROUGH,
                               REGION_STAGING, REGION_START, Frame)
from src.devices.link import MockMcu


def _drain(m):
    frames = []
    while (f := m.poll()) is not None:
        frames.append(f)
    return frames


def _ack(m):
    return [f for f in _drain(m) if f.cmd_id == CMD_ACK]


def _plan(seq=1):
    return Frame(seq=seq, cmd_id=CMD_TASK_PLAN, data=b"\x00" * 16)


def _solution(seq=1, echo=1, d_mm=100, alpha01=0, status=0x20 | 0x80):
    return Frame(seq=seq, cmd_id=CMD_ARM_SOLUTION,
                 data=struct.pack("<hhBBhh", d_mm, alpha01, status, echo,
                                  700, 100))


# ---------- 应答只给该给的 ----------

def test_plan_is_acked():
    m = MockMcu(route=[])
    m.send(_plan(seq=3))
    acks = _ack(m)
    assert len(acks) == 1
    assert acks[0].data == bytes((3, ACK_OK)), "data = (被确认的 seq, 结果)"


def test_periodic_measurement_frames_are_not_acked():
    """位姿与转盘是周期测量帧，每帧都回应答会让一半带宽变成纯应答。"""
    m = MockMcu(route=[])
    m.send(Frame(1, CMD_POSE, b"\x00" * 9))
    m.send(Frame(2, CMD_TURNTABLE, b"\x00" * 16))
    m.send(Frame(3, CMD_REGION, b"\x02"))
    assert _ack(m) == []


def test_arm_state_is_answered_by_solution_not_ack():
    """0x0006 的应答是 0x0008（业务帧即应答），不应另回一个 ack。"""
    m = MockMcu(route=[])
    m.request_arm()
    frames = _drain(m)
    assert [f.cmd_id for f in frames] == [CMD_ARM_STATE]


def test_bad_plan_length_is_rejected():
    m = MockMcu(route=[])
    m.send(Frame(1, CMD_TASK_PLAN, b"\x00" * 8))
    assert _ack(m)[0].data == bytes((1, ACK_FAILED))


# ---------- 行驶与漂移 ----------

def test_route_emits_region_hints_in_order():
    route = (REGION_MATERIAL, REGION_ROUGH, REGION_STAGING, REGION_START)
    m = MockMcu(route=route, leg_mm=100.0, speed_mm_s=100.0)
    now = 0.0
    for _ in range(200):
        m.tick(now)
        now += 0.02
    assert m.regions_sent == list(route)


def test_regions_sent_as_frames():
    m = MockMcu(route=(REGION_ROUGH,), leg_mm=10.0, speed_mm_s=1000.0)
    m.tick(0.0)
    m.tick(0.1)
    hints = [f for f in _drain(m) if f.cmd_id == CMD_REGION]
    assert hints and hints[0].data == bytes((REGION_ROUGH,))


def test_travel_is_time_driven_not_call_driven():
    """行驶距离由经过的时间决定，与 tick 调用次数无关。

    若按调用次数推进，主循环频率（20Hz/50Hz）会直接变成车速，这是很容易
    写错且很难察觉的一类 bug。
    """
    a = MockMcu(route=(REGION_ROUGH,), leg_mm=10000.0, speed_mm_s=1000.0)
    b = MockMcu(route=(REGION_ROUGH,), leg_mm=10000.0, speed_mm_s=1000.0)
    a.tick(0.0)
    a.tick(1.0)                     # 一次 1 秒
    b.tick(0.0)
    for k in range(1, 21):          # 二十次 0.05 秒
        b.tick(k * 0.05)
    assert abs(a.true_x - b.true_x) < 1e-6
    assert abs(a.true_x - 1000.0) < 1e-6


def test_drift_accumulates_with_distance():
    m = MockMcu(route=(), drift_mm_per_m=20.0, drift_deg_per_m=1.0)
    m.tick(0.0)
    m.tick(1.0)                     # speed 默认 1200mm/s → 1.2m
    assert abs(m.true_x - 1200.0) < 1e-6
    assert abs(m.est_x - 1200.0 * 1.02) < 1e-6
    assert m.position_error_mm > 20.0
    assert abs(m.heading_error_deg - 1.2) < 1e-9


def test_apply_correction_replaces_not_adds():
    """视觉位姿是绝对量，必须整体替换自身估计（文档附录二第五节）。"""
    m = MockMcu(route=(), drift_mm_per_m=50.0)
    m.tick(0.0)
    m.tick(1.0)
    assert m.position_error_mm > 10.0

    from src.algorithms.types import Pose
    m.apply_correction(Pose(m.true_x, m.true_y, m.true_h))
    assert m.position_error_mm < 1e-9
    assert m.corrections == 1
    m.tick(2.0)
    assert m.position_error_mm > 0.0


def test_no_pose_reports_are_produced():
    """主控不再上报位姿：tick 只产出去向提示。"""
    m = MockMcu(route=(REGION_ROUGH, REGION_STAGING),
                leg_mm=10.0, speed_mm_s=1000.0)
    now = 0.0
    for _ in range(100):
        m.tick(now)
        now += 0.02
    ids = {f.cmd_id for f in _drain(m)}
    assert ids <= {CMD_REGION}, f"主控不应产出位姿帧，实际 {ids}"


def test_no_heartbeat_is_ever_produced():
    m = MockMcu(route=(REGION_ROUGH,), leg_mm=10.0, speed_mm_s=1000.0)
    now = 0.0
    for _ in range(200):
        m.tick(now)
        now += 0.02
    assert 0x0001 not in {f.cmd_id for f in _drain(m)}


def test_apply_pose_replaces_estimate_on_pose_frame():
    m = MockMcu(route=(REGION_ROUGH,), leg_mm=1000.0, speed_mm_s=1000.0,
                drift_mm_per_m=50.0, apply_pose=True)
    m.tick(0.0)
    m.tick(1.0)
    assert m.position_error_mm > 10.0

    data = struct.pack("<hhh", int(m.true_x), int(m.true_y),
                       int(round(m.true_h * 10))) + bytes(3)
    m.send(Frame(seq=1, cmd_id=CMD_POSE, data=data))
    assert m.corrections == 1
    assert m.position_error_mm < 2.0


def test_apply_pose_off_by_default():
    m = MockMcu(route=())
    m.send(Frame(seq=1, cmd_id=CMD_POSE, data=b"\x00" * 9))
    assert m.corrections == 0


def test_malformed_pose_frame_is_ignored():
    m = MockMcu(route=(), apply_pose=True)
    m.send(Frame(seq=1, cmd_id=CMD_POSE, data=b"\x01\x02"))
    assert m.corrections == 0


# ---------- 臂往返 ----------

def test_request_arm_encodes_l_alpha():
    """0x0006 只带 l、α 与目标编号，共 5 字节；相机高度不再上报。"""
    m = MockMcu(route=(), arm_l_mm=580.0)
    m.est_h = 27.5
    seq = m.request_arm(target_id=2)
    frame = _drain(m)[0]
    assert frame.cmd_id == CMD_ARM_STATE and frame.seq == seq
    assert len(frame.data) == 5
    l_mm, alpha01, tid = struct.unpack("<HHb", frame.data)
    assert (l_mm, tid) == (580, 2)
    assert alpha01 == 275, "α 以 0.1° 传，27.5° → 275"


def test_arm_seq_increments_and_skips_zero():
    m = MockMcu(route=[])
    seqs = [m.request_arm() for _ in range(3)]
    assert seqs == [1, 2, 3]


def test_solution_is_executed_after_exec_time():
    m = MockMcu(route=(), exec_s=0.2, use_done=False)
    m.request_arm()
    m.send(_solution(seq=1, echo=1, d_mm=120))
    assert m.solutions[-1]["d_mm"] == 120
    assert m.executed == [], "还没到执行时刻，不应记为已执行"

    m.tick(0.0)
    m.tick(0.3)
    assert len(m.executed) == 1 and m.executed[0]["d_mm"] == 120


def test_done_is_sent_after_execution():
    m = MockMcu(route=(), exec_s=0.2)
    m.request_arm()
    m.send(_solution(seq=1, echo=1))
    m.tick(0.0)
    assert m.done_sent == [], "动作没做完不该回 done"
    m.tick(0.3)
    assert m.done_sent == [(1, ACK_OK)]


def test_duplicate_solution_is_recognized_by_echo_seq():
    """视觉等不到 done 而重发解算帧（seq 不变）时，只能补 done，不能再执行一次。

    否则车会沿同一个 d 再走一遍——这是最危险的错误之一：不报错，只是位置错了。
    """
    m = MockMcu(route=(), exec_s=0.2)
    m.request_arm()
    m.send(_solution(seq=5, echo=1, d_mm=120))
    m.tick(0.0)
    m.tick(0.3)
    assert len(m.executed) == 1

    m.send(_solution(seq=5, echo=1, d_mm=120))     # 同 seq 重发
    _drain(m)
    assert len(m.executed) == 1, "重复请求不得重复执行"
    assert m.done_sent[-1] == (5, ACK_DUPLICATE), "应回「重复请求」而不是成功"


def test_unusable_solution_is_not_executed():
    """够不着/没测到目标的结果不得执行，但仍然要回 done（否则视觉一直重发）。"""
    m = MockMcu(route=(), exec_s=0.0)
    m.request_arm()
    m.send(_solution(seq=1, echo=1, status=0x01))   # OUTCOME_UNREACHABLE，无可用位
    m.tick(0.0)
    m.tick(0.1)
    assert m.executed == [], "无解不该被执行"
    assert m.done_sent, "但必须回 done，否则视觉会一直重发"


# ---------- 航点行驶 ----------

def _run_route(wps, route, seconds=20.0, dt=0.02):
    m = MockMcu(route=route, waypoints=wps, speed_mm_s=1200.0)
    now = 0.0
    track = []
    while now < seconds:
        m.tick(now)
        track.append((m.true_x, m.true_y))
        now += dt
    return m, track


def test_waypoints_visit_each_nominal_pose():
    from src.algorithms.types import Pose

    wps = [Pose(0.0, 0.0, 0.0), Pose(600.0, 1600.0, 270.0),
           Pose(1500.0, 1000.0, 270.0)]
    m, _ = _run_route(wps, (REGION_START, REGION_MATERIAL, REGION_ROUGH))
    assert m.regions_sent == [REGION_START, REGION_MATERIAL, REGION_ROUGH]
    assert abs(m.true_x - 1500.0) < 1e-6
    assert abs(m.true_y - 1000.0) < 1e-6
    assert abs(m.true_h - 270.0) < 1e-6


def test_waypoints_keep_the_car_inside_the_field():
    from src.algorithms.types import Pose

    wps = [Pose(0.0, 0.0, 0.0), Pose(600.0, 1600.0, 270.0),
           Pose(1500.0, 1000.0, 270.0), Pose(1500.0, 2000.0, 90.0)]
    _, track = _run_route(wps, (REGION_START, REGION_MATERIAL,
                                REGION_ROUGH, REGION_STAGING))
    xs = [p[0] for p in track]
    ys = [p[1] for p in track]
    assert 0.0 <= min(xs) and max(xs) <= 1500.0 + 1e-6
    assert 0.0 <= min(ys) and max(ys) <= 2000.0 + 1e-6


def test_waypoints_length_must_match_route():
    from src.algorithms.types import Pose

    with pytest.raises(ValueError, match="waypoints"):
        MockMcu(route=(REGION_ROUGH, REGION_STAGING),
                waypoints=[Pose(0.0, 0.0, 0.0)])


def test_waypoint_travel_is_also_time_driven():
    from src.algorithms.types import Pose

    wps = [Pose(0.0, 0.0, 0.0), Pose(1200.0, 0.0, 0.0)]
    route = (REGION_START, REGION_ROUGH)

    a = MockMcu(route=route, waypoints=wps, speed_mm_s=1200.0)
    a.tick(0.0)
    a.tick(0.5)

    b = MockMcu(route=route, waypoints=wps, speed_mm_s=1200.0)
    b.tick(0.0)
    for k in range(1, 11):          # 十次 0.05 秒
        b.tick(k * 0.05)
    assert abs(a.true_x - b.true_x) < 1e-6
    assert abs(a.true_x - 600.0) < 1e-6


def test_seriallink_structural_and_serial_import_is_lazy():
    import src.devices.link as link

    assert hasattr(link, "SerialLink")
    assert hasattr(link, "McuLink")
    assert "serial" not in sys.modules
