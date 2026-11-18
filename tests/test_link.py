"""MockMcu / 链路层测试（V2.2.0 语义）。

V2.2.0 里电控只做一件事：**需要测量时发请求，等结果**。它不再发周期帧、不发
心跳、不收位姿。这里测三个容易写错且现场难定位的点：

- 收发方向：电控发的请求必须到视觉手里，而不是被自己收掉；
- 重发：等不到结果时重发**原帧**（seq 与内容都不变），视觉靠它认出重复；
- 收到结果后按 ``echo_seq`` 清掉待重试项，不能一直重发同一个请求。
"""
import struct
import sys

from src.core.protocol import (CMD_REQUEST, CMD_RESULT, ST_BAD_REQUEST, ST_OK,
                               TASK_ALIGN, TASK_SCAN, TASK_TURNTABLE, Frame)
from src.devices.link import MockMcu


def _drain(m):
    frames = []
    while (f := m.poll()) is not None:
        frames.append(f)
    return frames


def _result(echo=1, task=TASK_SCAN, status=ST_OK, body=b""):
    return Frame(seq=1, cmd_id=CMD_RESULT,
                 data=bytes((echo, task, status)) + body)


# ---------- 收发方向 ----------

def test_request_goes_to_the_vision_side():
    """电控发出的请求必须能被视觉 poll 到，不能留在电控自己的收件箱里。"""
    m = MockMcu()
    seq = m.request_scan()
    frames = _drain(m)
    assert len(frames) == 1
    assert frames[0].cmd_id == CMD_REQUEST
    assert frames[0].seq == seq
    assert frames[0].data == bytes((TASK_SCAN,))
    assert m.received == [], "自己发的请求不该被自己当成收到的帧"


def test_result_sent_by_vision_is_recorded():
    """视觉发来的结果走 send，电控在 tick 里解析。"""
    m = MockMcu()
    seq = m.request_scan()
    m.tick(0.0)
    _drain(m)

    plan = bytes(range(16))
    m.send(_result(echo=seq, task=TASK_SCAN, body=plan))
    m.tick(0.01)
    assert m.plan == plan
    assert m.statuses == [(TASK_SCAN, ST_OK)]
    assert m.pending_count == 0, "收到结果后不该再重发这个请求"


def test_request_bytes_start_with_task_code():
    m = MockMcu(arm_l_mm=580.0, arm_alpha_deg=27.5)
    m.request_align(now=0.0)
    f = _drain(m)[0]
    assert f.data[0] == TASK_ALIGN
    l_mm, alpha01 = struct.unpack("<HH", f.data[1:])
    assert (l_mm, alpha01) == (580, 275), "α 以 0.1° 传，27.5° → 275"


def test_turntable_request_carries_only_the_task_code():
    m = MockMcu()
    m.request_turntable()
    assert _drain(m)[0].data == bytes((TASK_TURNTABLE,))


# ---------- seq ----------

def test_seq_increments_and_skips_zero():
    m = MockMcu()
    seqs = [m.request_scan(now=0.0) for _ in range(3)]
    assert seqs == [1, 2, 3]
    rest = [m.request_scan(now=0.0) for _ in range(253)]
    assert rest[:252] == list(range(4, 256)), "4..255 依次递增"
    assert rest[252] == 1, "255 之后回到 1，0 不使用"


# ---------- 重发 ----------

def test_retry_resends_the_same_frame():
    """seq 与内容都不变——视觉靠「同一个 seq」认出这是重复请求。"""
    m = MockMcu(retry=3, retry_s=0.1)
    seq = m.request_scan(now=0.0)
    first = _drain(m)[0]

    m.tick(0.05)
    assert _drain(m) == [], "还没到重试时刻"
    m.tick(0.12)
    again = _drain(m)
    assert len(again) == 1
    assert again[0].seq == first.seq == seq
    assert again[0].data == first.data


def test_gives_up_after_the_retry_limit():
    m = MockMcu(retry=2, retry_s=0.1)
    m.request_scan(now=0.0)
    _drain(m)
    for k in range(1, 10):
        m.tick(k * 0.1)
        _drain(m)
    assert m.gave_up == [TASK_SCAN]
    assert m.pending_count == 0
    assert len(m.sent) == 3, "首发 + 重发 2 次"


def test_retry_stops_as_soon_as_the_result_arrives():
    m = MockMcu(retry=5, retry_s=0.1)
    seq = m.request_scan(now=0.0)
    _drain(m)
    m.send(_result(echo=seq, task=TASK_SCAN, body=bytes(16)))
    m.tick(0.05)
    n = len(m.sent)
    for k in range(2, 20):
        m.tick(k * 0.1)
        _drain(m)
    assert len(m.sent) == n, "结果已到，不得继续重发"
    assert m.gave_up == []


def test_retry_is_per_request_not_global():
    """同时挂着两个请求时，各自的超时与重试互不干扰。"""
    m = MockMcu(retry=1, retry_s=0.1)
    a = m.request_scan(now=0.0)
    b = m.request_turntable(now=0.0)
    _drain(m)

    m.send(_result(echo=a, task=TASK_SCAN, body=bytes(16)))
    m.tick(0.05)
    _drain(m)
    m.tick(0.12)
    resent = _drain(m)
    assert [f.seq for f in resent] == [b], "只该重发还没收到结果的那个"
    assert resent[0].data[0] == TASK_TURNTABLE


# ---------- 结果解析 ----------

def test_failed_result_also_clears_the_pending_request():
    """「没准备好」也是结果：电控不该再重发同一个请求，而应由它自己决定下一步。"""
    m = MockMcu(retry=3, retry_s=0.1)
    seq = m.request_turntable(now=0.0)
    _drain(m)
    m.send(_result(echo=seq, task=TASK_TURNTABLE, status=ST_BAD_REQUEST))
    m.tick(0.05)
    assert m.pending_count == 0
    assert m.colors == [], "非成功结果不得参与业务判断"
    assert m.statuses == [(TASK_TURNTABLE, ST_BAD_REQUEST)]


def test_align_result_is_parsed_as_d_and_alpha():
    m = MockMcu()
    seq = m.request_align(now=0.0, l_mm=600.0)
    _drain(m)
    m.send(_result(echo=seq, task=TASK_ALIGN,
                   body=struct.pack("<hh", 120, -300)))
    m.tick(0.05)
    assert m.solutions[-1] == {"d_mm": 120, "alpha_deg": -30.0, "echo": seq}


def test_turntable_colour_is_recorded():
    m = MockMcu()
    seq = m.request_turntable(now=0.0)
    _drain(m)
    m.send(_result(echo=seq, task=TASK_TURNTABLE, body=bytes((5,))))
    m.tick(0.05)
    assert m.colors == [5]


def test_truncated_result_is_ignored_not_crashed():
    """半截帧在真机上会出现在断包瞬间，解析必须容忍而不是抛异常。"""
    m = MockMcu()
    m.send(Frame(seq=1, cmd_id=CMD_RESULT, data=b"\x01\x02"))
    m.tick(0.05)
    assert m.statuses == []
    assert m.plan is None


def test_unknown_command_is_ignored():
    m = MockMcu()
    m.send(Frame(seq=1, cmd_id=0x0003, data=b"\x00" * 9))   # V2.1.0 的位姿帧，已下架
    m.tick(0.05)
    assert m.statuses == []
    assert m.pending_count == 0


# ---------- 结构 ----------

def test_seriallink_structural_and_serial_import_is_lazy():
    import src.devices.link as link

    assert hasattr(link, "SerialLink")
    assert hasattr(link, "McuLink")
    assert "serial" not in sys.modules
