import struct
import sys

from src.core.protocol import Frame, CMD_POSE, CMD_ACK, CMD_CALIBRATE, CMD_GOTO
from src.devices.link import MockMcu


def i16(v):
    return struct.pack("<h", v)


def test_ack_on_every_frame():
    m = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m.send(Frame(seq=3, cmd=CMD_GOTO, data=b"\x00" * 6))
    acks = m.poll(), m.poll()
    assert any(f and f.cmd == CMD_ACK and f.data == b"\x03" for f in acks)


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
    x, y, h01, odom = struct.unpack("<hhhH", poses[0].data)
    # odom=11 → 真值 11*1000mm, 漂移 +100*11=+1100mm → 精确 12100
    assert (odom, x, y, h01) == (11, 11 * 1000 + 100 * 11, 0, 0)
    err_before = x - odom * 1000  # +1100mm

    # 视觉侧下发修正量 (fixed−last) = −1100mm，主控应“叠加”使上报靠近真值
    m.send(Frame(seq=1, cmd=CMD_CALIBRATE, data=i16(-1100) + i16(0) + i16(0)))
    m.tick(0.2)
    drained = []
    while (f := m.poll()) is not None:
        drained.append(f)
    acks = [f for f in drained if f.cmd == CMD_ACK]
    assert len(acks) == 1 and acks[0].data == b"\x01"
    corrected = [f for f in drained if f.cmd == CMD_POSE]
    assert len(corrected) == 1
    cx, _, _, codom = struct.unpack("<hhhH", corrected[0].data)
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
            assert struct.unpack("<hhhH", f.data)[2] == int(1.0 * 11 * 10)
    # 修正 −11.0° (dh01=−110) 叠加后: odom=12 → (12.0 − 11.0)*10 = 10
    m.send(Frame(seq=2, cmd=CMD_CALIBRATE, data=i16(0) + i16(0) + i16(-110)))
    m.tick(0.2)
    while (f := m.poll()) is not None:
        if f.cmd == CMD_POSE:
            assert struct.unpack("<hhhH", f.data)[2] == int((1.0 * 12 - 11.0) * 10)


def test_negative_heading_reports_signed_value():
    # 回归：dh01=-5 叠加后净航向 h=-5(0.1°)。旧实现 h % 36000 → 35995，
    # 超出 int16 上界，struct.pack('<h') 直接抛 struct.error（轻微过修后为常态）。
    # 冻结格式 <hhhH 不变，应改为有符号回绕到 [-18000,18000)。
    m = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m.send(Frame(seq=1, cmd=CMD_CALIBRATE, data=i16(0) + i16(0) + i16(-5)))
    m.tick(0.1)  # 修复前此处 struct.error 崩溃
    poses = [f for f in _drain(m) if f.cmd == CMD_POSE]
    assert len(poses) == 1
    assert struct.unpack("<hhhH", poses[0].data)[2] == -5


def test_int16_saturation_and_clamp_counter():
    # x = odom*1000 + drift*odom 随里程线性增长，odom≈30 起必然越过 32767；
    # 长仿真不得崩溃：打包时饱和钳位，并以公开计数器 out_of_range_clamps 记录。
    m = MockMcu(drift_mm_per_m=100.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m._odometry = 33.0  # 本次 tick 推进到 34 → x = 34*1000 + 100*34 = 37400
    m.tick(0.1)
    poses = [f for f in _drain(m) if f.cmd == CMD_POSE]
    assert len(poses) == 1
    x, y, h01, odom = struct.unpack("<hhhH", poses[0].data)
    assert (odom, x) == (34, 32767)  # 饱和到 int16 上界
    assert m.out_of_range_clamps == 1
    # 负方向同样饱和（y 轴：真值偏移超界）
    m2 = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m2.truth = (0.0, -50000.0)
    m2.tick(0.1)
    poses2 = [f for f in _drain(m2) if f.cmd == CMD_POSE]
    assert struct.unpack("<hhhH", poses2[0].data)[1] == -32768
    assert m2.out_of_range_clamps == 1
    # 未超界的普通 pack 不增加计数（前面的精确值测试隐式覆盖，这里显式断言）
    m3 = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m3.tick(0.1)
    assert m3.out_of_range_clamps == 0


def test_calibrate_with_bad_length_is_ignored():
    # 校验通过但 data 长度不符（主控实现 bug / 版本不一致）不得崩，只忽略该帧
    m = MockMcu(drift_mm_per_m=0.0, drift_deg_per_m=0.0, pose_hz=10.0)
    m.send(Frame(seq=1, cmd=CMD_CALIBRATE, data=i16(100)))   # 只有 2 字节
    m.send(Frame(seq=2, cmd=CMD_CALIBRATE, data=i16(0) + i16(0) + i16(-5)))
    poses = [f for f in _drain(m) if f.cmd == CMD_POSE]
    assert poses == []            # 还没 tick，先确认没提前产帧
    m.tick(0.1)
    poses = [f for f in _drain(m) if f.cmd == CMD_POSE]
    assert len(poses) == 1
    # 第二条（合法）生效：航向 −5；若第一条被误解析成修正会明显不同
    assert struct.unpack("<hhhH", poses[0].data)[2] == -5


def _drain(m):
    frames = []
    while (f := m.poll()) is not None:
        frames.append(f)
    return frames


def test_seriallink_structural_and_serial_import_is_lazy():
    import src.devices.link as link

    assert hasattr(link, "SerialLink")
    assert hasattr(link, "McuLink")
    # 模块级不得 import pyserial（SerialLink.__init__ 内惰性导入），
    # 否则本行断言在 link 顶部 import serial 时会失败。
    assert "serial" not in sys.modules
