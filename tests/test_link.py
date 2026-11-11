"""MockMcu / 链路层测试（v2.0.0 语义）。

v2.0.0 里机器人主控只做三件事：回应答、发去向提示、按自己的里程推算行驶。
它**不向视觉上报位姿**，因此这里测的是「ACK 正确」「区域提示按路线发出」
「里程漂移真的会累积」「应用视觉位姿后误差归零」。
"""
import math
import struct
import sys

import pytest

from src.core.protocol import (CMD_ACK, CMD_POSE, CMD_REGION, CMD_TASK_PLAN,
                               REGION_MATERIAL, REGION_ROUGH, REGION_STAGING,
                               REGION_START, Frame)
from src.devices.link import MockMcu


def _drain(m):
    frames = []
    while (f := m.poll()) is not None:
        frames.append(f)
    return frames


def test_ack_on_every_frame():
    m = MockMcu(route=[])
    m.send(Frame(seq=3, cmd_id=CMD_TASK_PLAN, data=b"\x00" * 16))
    acks = [f for f in _drain(m) if f.cmd_id == CMD_ACK]
    assert len(acks) == 1
    assert acks[0].data == b"\x03", "应答的 data 必须是被确认帧的 seq"


def test_ack_seq_increments():
    m = MockMcu(route=[])
    for seq in (1, 2, 3):
        m.send(Frame(seq=seq, cmd_id=CMD_TASK_PLAN, data=b"\x00" * 16))
    acks = [f for f in _drain(m) if f.cmd_id == CMD_ACK]
    assert [a.data[0] for a in acks] == [1, 2, 3]


def test_received_records_cmd_and_data():
    m = MockMcu(route=[])
    m.send(Frame(seq=1, cmd_id=CMD_POSE, data=bytes(range(9))))
    assert (CMD_POSE, bytes(range(9))) in m.received


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
    # 推算值 = 真值 × (1 + 20/1000)
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
    # 再走一段，误差重新累积（说明是替换而非把偏移永久抹掉）
    m.tick(2.0)
    assert m.position_error_mm > 0.0


def test_no_pose_reports_are_produced():
    """v2.0.0 主控不再上报位姿：tick 只产出去向提示。"""
    m = MockMcu(route=(REGION_ROUGH, REGION_STAGING),
                leg_mm=10.0, speed_mm_s=1000.0)
    now = 0.0
    for _ in range(100):
        m.tick(now)
        now += 0.02
    ids = {f.cmd_id for f in _drain(m)}
    assert ids <= {CMD_REGION}, f"主控不应产出位姿帧，实际 {ids}"


def test_apply_pose_replaces_estimate_on_pose_frame():
    """``apply_pose=True`` 时，收到 0x0003 就整体替换估计（真实电控行为）。"""
    from src.algorithms.types import Pose

    m = MockMcu(route=(REGION_ROUGH,), leg_mm=1000.0, speed_mm_s=1000.0,
                drift_mm_per_m=50.0, apply_pose=True)
    m.tick(0.0)
    m.tick(1.0)
    assert m.position_error_mm > 10.0

    import struct
    from src.core.protocol import CMD_POSE as POSE_CMD
    data = struct.pack("<hhh", int(m.true_x), int(m.true_y),
                       int(round(m.true_h * 10))) + bytes(3)
    m.send(Frame(seq=1, cmd_id=POSE_CMD, data=data))
    assert m.corrections == 1
    assert m.position_error_mm < 2.0


def test_apply_pose_off_by_default():
    """默认不应用：单元测试要能显式区分「帧发出去了」与「误差归零了」。"""
    from src.core.protocol import CMD_POSE as POSE_CMD

    m = MockMcu(route=())
    m.send(Frame(seq=1, cmd_id=POSE_CMD, data=b"\x00" * 9))
    assert m.corrections == 0


def test_malformed_pose_frame_is_ignored():
    """长度不是 9 的位姿帧不得改动估计（真实电控也会丢弃）。"""
    from src.core.protocol import CMD_POSE as POSE_CMD

    m = MockMcu(route=(), apply_pose=True)
    m.send(Frame(seq=1, cmd_id=POSE_CMD, data=b"\x01\x02"))
    assert m.corrections == 0


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
    """按航点走时，每个区域到点时真值必须落在该航点的标称停位上。"""
    from src.algorithms.types import Pose

    wps = [Pose(0.0, 0.0, 0.0), Pose(600.0, 1600.0, 270.0),
           Pose(1500.0, 1000.0, 270.0)]
    m, _ = _run_route(wps, (REGION_START, REGION_MATERIAL, REGION_ROUGH))
    assert m.regions_sent == [REGION_START, REGION_MATERIAL, REGION_ROUGH]
    assert abs(m.true_x - 1500.0) < 1e-6
    assert abs(m.true_y - 1000.0) < 1e-6
    assert abs(m.true_h - 270.0) < 1e-6


def test_waypoints_keep_the_car_inside_the_field():
    """没有航点时车会沿 +x 一直开到场外（x 冲到上万毫米），这正是要修的问题。"""
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
    """航点路径同样由时钟驱动：分多次 tick 与一次 tick 走到同一位置。"""
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
    # 模块级不 import pyserial（SerialLink.__init__ 内惰性导入）。
    assert "serial" not in sys.modules
