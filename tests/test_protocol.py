"""协议帧编解码测试：帧格式（0xA5 帧头 + 2B cmd_id + CRC16 整包）与命令集。

帧布局是 v1.0.0 冻结的部分，v2.0.0 只改了命令语义，这里继续强绑定。
"""
import struct

import pytest

from src.core import protocol as P
from src.core.crc import crc8, crc16
from src.core.protocol import (CMD_ACK, CMD_HEARTBEAT, CMD_POSE, CMD_REGION,
                               CMD_TASK_PLAN, CMD_TURNTABLE, HEADER_LEN,
                               OVERHEAD, SOF, Frame, StreamDecoder, decode_frame,
                               encode)


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
    assert len(encode(Frame(1, CMD_HEARTBEAT))) == OVERHEAD


def test_empty_data_roundtrip():
    for cmd_id in (CMD_HEARTBEAT, 0xFFFF):
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
    assert P.CMD_HEARTBEAT == 0x0001
    assert P.CMD_TASK_PLAN == 0x0002
    assert P.CMD_POSE == 0x0003
    assert P.CMD_TURNTABLE == 0x0004
    assert P.CMD_REGION == 0x0005
    assert P.CMD_ACK == 0x0007
    assert P.DATA_LEN == {
        P.CMD_HEARTBEAT: 0,
        P.CMD_TASK_PLAN: 16,
        P.CMD_POSE: 9,
        P.CMD_TURNTABLE: 16,
        P.CMD_REGION: 1,
        P.CMD_ACK: 1,
    }


def test_retired_commands_are_not_in_the_command_set():
    """0x0201/0x0202/0x0203 已删除，不得出现在 DATA_LEN 里，也不得是模块常量。"""
    for cmd in P.RETIRED_CMDS:
        assert cmd not in P.DATA_LEN
        assert not any(getattr(P, n) == cmd for n in dir(P)
                       if n.startswith("CMD_")), f"0x{cmd:04X} 仍是模块常量"


def test_region_codes():
    assert (P.REGION_MATERIAL, P.REGION_ROUGH, P.REGION_STAGING,
            P.REGION_START) == (1, 2, 3, 4)


def test_seq_must_be_a_byte():
    with pytest.raises(ValueError):
        encode(Frame(seq=256, cmd_id=CMD_HEARTBEAT))
    with pytest.raises(ValueError):
        encode(Frame(seq=-1, cmd_id=CMD_HEARTBEAT))
