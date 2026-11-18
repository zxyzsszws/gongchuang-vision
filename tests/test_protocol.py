"""协议帧编解码测试：帧格式（0xA5 帧头 + 2B cmd_id + CRC16 整包）与命令集。

帧布局是 v1.0.0 冻结的部分，V2.1.0 只改了命令语义与命令集，这里继续强绑定。
"""
import struct

import pytest

from src.core import protocol as P
from src.core.crc import crc8, crc16
from src.core.protocol import (CMD_ACK, CMD_ARM_SOLUTION, CMD_ARM_STATE,
                               CMD_POSE, CMD_REGION, CMD_TASK_PLAN,
                               CMD_TURNTABLE, HEADER_LEN, OVERHEAD, SOF, Frame,
                               StreamDecoder, decode_frame, encode)


# ---------- 帧布局 ----------

def test_roundtrip():
    f = Frame(seq=7, cmd_id=CMD_TASK_PLAN, data=b"\x01" * 16)
    assert StreamDecoder().feed(encode(f)) == [f]


def test_frame_layout_bytes():
    # a5 | data_length(2, = len(data)+2) | seq | crc8 | cmd_id(2) | data | crc16(2)
    f = Frame(seq=1, cmd_id=CMD_REGION, data=bytes((2,)))
    raw = encode(f)
    assert raw[0] == SOF == 0xA5
    assert struct.unpack("<H", raw[1:3])[0] == 2 + 1
    assert raw[3] == 1
    assert raw[4] == crc8(raw[:4])                 # CRC8 覆盖 SOF+len+seq
    assert struct.unpack("<H", raw[5:7])[0] == CMD_REGION
    assert raw[7:-2] == f.data
    assert struct.unpack("<H", raw[-2:])[0] == crc16(raw[:-2])  # CRC16 整包


def test_overhead_is_nine_bytes():
    assert OVERHEAD == 9 and HEADER_LEN == 5
    assert len(encode(Frame(1, CMD_ARM_STATE, b"\x00" * 7))) == OVERHEAD + 7


def test_empty_data_roundtrip():
    for cmd_id in (0xFFFF,):
        f = Frame(seq=255, cmd_id=cmd_id, data=b"")
        assert StreamDecoder().feed(encode(f)) == [f]


def test_sticky_and_split():
    a = Frame(1, CMD_REGION, b"\x02")
    b = Frame(2, CMD_REGION, b"\x03")
    d = StreamDecoder()
    raw = encode(a) + encode(b)
    assert d.feed(raw[:3]) + d.feed(raw[3:]) == [a, b]


def test_split_at_every_offset():
    """断包：任意切分点都必须恢复同步（含切在 0xA5 / data_length / CRC 中间）。"""
    f = Frame(3, CMD_POSE, b"\x01\x02")
    raw = encode(f)
    for i in range(1, len(raw)):
        d = StreamDecoder()
        assert d.feed(raw[:i]) + d.feed(raw[i:]) == [f], f"lost sync at offset {i}"
        assert d.dropped == 0


def test_bad_crc_drops_one_byte_only():
    """坏帧只丢 1 字节后重新同步，粘包时不得连带丢掉其后的好帧。"""
    good = Frame(4, CMD_REGION, b"\x02")
    bad = bytearray(encode(good))
    bad[-1] ^= 0xFF                     # 破坏 CRC16
    d = StreamDecoder()
    got = d.feed(bytes(bad) + encode(good))
    assert got == [good]
    assert d.dropped >= 1


def test_decode_frame_rejects_truncated_and_bad_sof():
    raw = encode(Frame(1, CMD_REGION, b"\x02"))
    assert decode_frame(raw[:-1]) is None
    assert decode_frame(b"\x00" + raw[1:]) is None
    assert decode_frame(raw) == Frame(1, CMD_REGION, b"\x02")


# ---------- 命令集（表 1-5） ----------

def test_command_ids_match_doc_table():
    """命令码与数据段长度必须与文档表 1-5 完全一致。"""
    assert P.CMD_TASK_PLAN == 0x0002
    assert P.CMD_POSE == 0x0003
    assert P.CMD_TURNTABLE == 0x0004
    assert P.CMD_REGION == 0x0005
    assert P.CMD_ARM_STATE == 0x0006
    assert P.CMD_ACK == 0x0007
    assert P.CMD_ARM_SOLUTION == 0x0008
    assert P.DATA_LEN == {
        P.CMD_TASK_PLAN: 16,
        P.CMD_POSE: 9,
        P.CMD_TURNTABLE: 16,
        P.CMD_REGION: 1,
        P.CMD_ARM_STATE: 5,
        P.CMD_ACK: 2,
        P.CMD_ARM_SOLUTION: 10,
    }


def test_heartbeat_is_gone():
    """V2.1.0 取消心跳：0x0001 不得再是模块常量，也不得在命令表里。"""
    assert 0x0001 not in P.DATA_LEN
    assert 0x0001 in P.RETIRED_CMDS
    assert not any(getattr(P, n) == 0x0001 for n in dir(P)
                   if n.startswith("CMD_"))


def test_retired_commands_are_not_in_the_command_set():
    """已删除的命令不得出现在 DATA_LEN 里，也不得是模块常量。"""
    for cmd in P.RETIRED_CMDS:
        assert cmd not in P.DATA_LEN
        assert not any(getattr(P, n) == cmd for n in dir(P)
                       if n.startswith("CMD_")), f"0x{cmd:04X} 仍是模块常量"


def test_only_two_commands_expect_ack():
    """周期测量帧与单向提示不应答，否则一半带宽是纯应答。

    0x0006 的应答是 0x0008（业务帧即应答），所以它也不在待应答集合里。
    """
    assert P.EXPECTS_ACK == {CMD_TASK_PLAN, CMD_ARM_SOLUTION}
    for cmd in (CMD_POSE, CMD_TURNTABLE, CMD_REGION, CMD_ARM_STATE, CMD_ACK):
        assert cmd not in P.EXPECTS_ACK


def test_ack_result_codes():
    assert (P.ACK_OK, P.ACK_DUPLICATE, P.ACK_FAILED) == (0, 1, 2)


def test_region_codes():
    assert (P.REGION_MATERIAL, P.REGION_ROUGH, P.REGION_STAGING,
            P.REGION_START) == (1, 2, 3, 4)


def test_seq_must_be_a_byte():
    with pytest.raises(ValueError):
        encode(Frame(seq=256, cmd_id=CMD_ARM_STATE))
    with pytest.raises(ValueError):
        encode(Frame(seq=-1, cmd_id=CMD_ARM_STATE))
