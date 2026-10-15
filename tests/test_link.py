import struct
import sys

from src.core.protocol import Frame, CMD_POSE, CMD_ACK, CMD_CALIBRATE
from src.devices.link import MockMcu


def i16(v):
    return struct.pack("<h", v)


def test_ack_on_every_frame():
    m = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m.send(Frame(seq=3, cmd=0x12, payload=b"\x00" * 6))
    acks = m.poll(), m.poll()
    assert any(f and f.cmd == CMD_ACK and f.payload == b"\x03" for f in acks)


def test_pose_stream_and_calibration_effect():
    # 模型: 上报位姿 = 真值 + drift×里程 + 已叠加修正。
    # 里程初值 10m, drift=100mm/m, tick 周期 = 1/pose_hz = 0.1s → 每 tick 出一条 POSE,
    # 每出一条 POSE 里程推进 1m。
    m = MockMcu(drift_mm_per_m=100.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m._odometry = 10.0  # 10m 里程
    m.tick(0.1)
    poses = []
    while (f := m.poll()) is not None:
        if f.cmd == CMD_POSE:
            poses.append(f)
    assert len(poses) == 1
    x, y, h01, odom = struct.unpack("<hhhH", poses[0].payload)
    # odom=11 → 真值 11*1000mm, 漂移 +100*11=+1100mm → 精确 12100
    assert (odom, x, y, h01) == (11, 11 * 1000 + 100 * 11, 0, 0)
    err_before = x - odom * 1000  # +1100mm

    # 视觉侧下发修正量 (fixed−last) = −1100mm，主控应“叠加”使上报靠近真值
    m.send(Frame(seq=1, cmd=CMD_CALIBRATE, payload=i16(-1100) + i16(0) + i16(0)))
    m.tick(0.2)
    drained = []
    while (f := m.poll()) is not None:
        drained.append(f)
    acks = [f for f in drained if f.cmd == CMD_ACK]
    assert len(acks) == 1 and acks[0].payload == b"\x01"
    corrected = [f for f in drained if f.cmd == CMD_POSE]
    assert len(corrected) == 1
    cx, _, _, codom = struct.unpack("<hhhH", corrected[0].payload)
    assert codom == 12
    # odom=12 → 12*1000 + 100*12 + (-1100) = 12100, 精确命中模型值
    assert cx == 12 * 1000 + 100 * 12 - 1100
    err_after = cx - codom * 1000  # +100mm
    assert abs(err_after) < abs(err_before) / 5


def test_heading_correction_is_added_not_subtracted():
    # drift_deg_per_m=1.0: odom=11 时航向漂移 +11.0° → h01=+110 (0.1° 单位)
    m = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=1.0, pose_hz=10.0)
    m._odometry = 10.0
    m.tick(0.1)
    while (f := m.poll()) is not None:
        if f.cmd == CMD_POSE:
            assert struct.unpack("<hhhH", f.payload)[2] == int(1.0 * 11 * 10)
    # 修正 −11.0° (dh01=−110) 叠加后: odom=12 → (12.0 − 11.0)*10 = 10
    m.send(Frame(seq=2, cmd=CMD_CALIBRATE, payload=i16(0) + i16(0) + i16(-110)))
    m.tick(0.2)
    while (f := m.poll()) is not None:
        if f.cmd == CMD_POSE:
            assert struct.unpack("<hhhH", f.payload)[2] == int((1.0 * 12 - 11.0) * 10)


def test_serlink_structural_and_serial_import_is_lazy():
    import src.devices.link as link

    assert hasattr(link, "SerialLink")
    assert hasattr(link, "McuLink")
    # 模块级不得 import pyserial（SerialLink.__init__ 内惰性导入），
    # 否则本行断言在 link 顶部 import serial 时会失败。
    assert "serial" not in sys.modules
