"""协议帧编解码测试：帧格式（0xA5 帧头 + 2B cmd_id + CRC16 整包）与 V2.2.0 命令集。

帧布局是 v1.0.0 就冻结的部分，V2.2.0 只改了命令码与语义，这里继续强绑定。
"""
import struct

import pytest

from src.core import protocol as P
from src.core.crc import crc8, crc16
from src.core.protocol import (CMD_REQUEST, CMD_RESULT, HEADER_LEN, OVERHEAD, SOF,
                               Frame, StreamDecoder, decode_frame, encode)


# ---------- 帧布局 ----------

def test_roundtrip():
    f = Frame(seq=7, cmd_id=CMD_REQUEST, data=b"\x01")
    assert StreamDecoder().feed(encode(f)) == [f]


def test_frame_layout_bytes():
    # a5 | data_length(2, = len(data)+2) | seq | crc8 | cmd_id(2) | data | crc16(2)
    f = Frame(seq=1, cmd_id=CMD_REQUEST, data=bytes((3,)))
    raw = encode(f)
    assert raw[0] == SOF == 0xA5
    assert struct.unpack("<H", raw[1:3])[0] == 2 + 1
    assert raw[3] == 1
    assert raw[4] == crc8(raw[:4])                 # CRC8 覆盖 SOF+len+seq
    assert struct.unpack("<H", raw[5:7])[0] == CMD_REQUEST
    assert raw[7:-2] == f.data
    assert struct.unpack("<H", raw[-2:])[0] == crc16(raw[:-2])  # CRC16 整包


def test_overhead_is_nine_bytes():
    assert OVERHEAD == 9 and HEADER_LEN == 5
    assert len(encode(Frame(1, CMD_RESULT, b"\x00" * 7))) == OVERHEAD + 7


def test_empty_data_roundtrip():
    for cmd_id in (0xFFFF,):
        f = Frame(seq=255, cmd_id=cmd_id, data=b"")
        assert StreamDecoder().feed(encode(f)) == [f]


def test_sticky_and_split():
    a = Frame(1, CMD_REQUEST, b"\x01")
    b = Frame(2, CMD_REQUEST, b"\x02")
    d = StreamDecoder()
    raw = encode(a) + encode(b)
    assert d.feed(raw[:3]) + d.feed(raw[3:]) == [a, b]


def test_split_at_every_offset():
    """断包：任意切分点都必须恢复同步（含切在 0xA5 / data_length / CRC 中间）。"""
    f = Frame(3, CMD_RESULT, b"\x01\x02\x03\x04")
    raw = encode(f)
    for i in range(1, len(raw)):
        d = StreamDecoder()
        assert d.feed(raw[:i]) + d.feed(raw[i:]) == [f], f"lost sync at offset {i}"
        assert d.dropped == 0


def test_bad_crc_drops_one_byte_only():
    """坏帧只丢 1 字节后重新同步，粘包时不得连带丢掉其后的好帧。"""
    good = Frame(4, CMD_REQUEST, b"\x02")
    bad = bytearray(encode(good))
    bad[-1] ^= 0xFF                     # 破坏 CRC16
    d = StreamDecoder()
    got = d.feed(bytes(bad) + encode(good))
    assert got == [good]
    assert d.dropped >= 1


def test_decode_frame_rejects_truncated_and_bad_sof():
    raw = encode(Frame(1, CMD_REQUEST, b"\x02"))
    assert decode_frame(raw[:-1]) is None
    assert decode_frame(b"\x00" + raw[1:]) is None
    assert decode_frame(raw) == Frame(1, CMD_REQUEST, b"\x02")


# ---------- 命令集（表 1-5） ----------

def test_command_ids_match_doc_table():
    """V2.2.0 只有两条命令：请求与结果。"""
    assert P.CMD_REQUEST == 0x0010
    assert P.CMD_RESULT == 0x0011
    assert P.DATA_LEN == {
        P.CMD_REQUEST: 1,
        P.CMD_RESULT: 4,
    }


def test_task_codes_and_lengths():
    """任务码与长度必须与文档「表 1-6」一致。"""
    assert P.TASK_SCAN == 1
    assert P.TASK_TURNTABLE == 2
    assert P.TASK_ALIGN == 3
    assert P.TASKS == (1, 2, 3)
    assert P.REQ_LEN == {P.TASK_SCAN: 1, P.TASK_TURNTABLE: 1, P.TASK_ALIGN: 5}
    assert P.RES_LEN == {P.TASK_SCAN: 19, P.TASK_TURNTABLE: 4, P.TASK_ALIGN: 7}


def test_result_lengths_cover_the_three_byte_header():
    """结果帧正文前面固定三字节：echo_seq + task + status。"""
    for task, n in P.RES_LEN.items():
        assert n >= 3, f"任务 {task} 的结果长度 {n} 装不下 echo_seq/task/status"


def test_all_v21_commands_are_retired():
    """V2.1.0 的七条命令全部下架，并且不是模块常量。"""
    for cmd in (0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008):
        assert cmd in P.RETIRED_CMDS
        assert cmd not in P.DATA_LEN
        assert not any(getattr(P, n) == cmd for n in dir(P)
                       if n.startswith("CMD_")), f"0x{cmd:04X} 仍是模块常量"


def test_heartbeat_is_gone():
    assert 0x0001 in P.RETIRED_CMDS
    assert 0x0001 not in P.DATA_LEN
    assert not any(getattr(P, n) == 0x0001 for n in dir(P)
                   if n.startswith("CMD_"))


def test_only_two_commands_exist():
    names = {n for n in dir(P) if n.startswith("CMD_")}
    assert names == {"CMD_REQUEST", "CMD_RESULT"}


# ---------- 状态码（表 1-8） ----------

def test_status_codes():
    assert (P.ST_OK, P.ST_NOT_READY, P.ST_UNSOLVABLE,
            P.ST_BAD_REQUEST) == (0, 1, 2, 3)


def test_every_status_has_a_name():
    for code in (P.ST_OK, P.ST_NOT_READY, P.ST_UNSOLVABLE, P.ST_BAD_REQUEST):
        assert code in P.STATUS_NAMES


def test_color_unknown_is_zero():
    assert P.COLOR_UNKNOWN == 0


def test_seq_must_be_a_byte():
    with pytest.raises(ValueError):
        encode(Frame(seq=256, cmd_id=CMD_REQUEST))
    with pytest.raises(ValueError):
        encode(Frame(seq=-1, cmd_id=CMD_REQUEST))
