"""协议帧编解码测试：新帧格式（0xA5 帧头 + 2B cmd_id + CRC16 整包）。"""
import struct

import pytest

from src.core.crc import crc8, crc16
from src.core.protocol import (CMD_GOTO, CMD_POSE, HEADER_LEN, OVERHEAD, SOF,
                               Frame, StreamDecoder, decode_frame, encode)


def _mk_data():
    return b"\x64\x00\x2c\x01\x88\x13"


def test_roundtrip():
    f = Frame(seq=7, cmd_id=CMD_GOTO, data=_mk_data())
    assert StreamDecoder().feed(encode(f)) == [f]


def test_frame_layout_bytes():
    # a5 | data_length(2, = len(data)+2) | seq | crc8 | cmd_id(2) | data | crc16(2)
    f = Frame(seq=1, cmd_id=0x0002, data=bytes((4, 5, 2)))
    raw = encode(f)
    assert raw[0] == SOF == 0xA5
    assert struct.unpack("<H", raw[1:3])[0] == 2 + 3
    assert raw[3] == 1
    assert raw[4] == crc8(raw[:4])                 # CRC8 覆盖 SOF+len+seq
    assert struct.unpack("<H", raw[5:7])[0] == 0x0002
    assert raw[7:-2] == f.data
    assert struct.unpack("<H", raw[-2:])[0] == crc16(raw[:-2])  # CRC16 整包


def test_overhead_is_nine_bytes():
    assert OVERHEAD == 9 and HEADER_LEN == 5
    assert len(encode(Frame(0, CMD_POSE))) == OVERHEAD


def test_empty_data_roundtrip():
    for cmd_id in (0x0001, 0xFFFF):
        f = Frame(seq=255, cmd_id=cmd_id, data=b"")
        assert StreamDecoder().feed(encode(f)) == [f]


def test_sticky_and_split():
    a, b = Frame(1, CMD_GOTO, b"\x01"), Frame(2, CMD_GOTO, b"\x02\x03")
    raw = encode(a) + encode(b)
    d = StreamDecoder()
    got = d.feed(raw[:3]) + d.feed(raw[3:])
    assert got == [a, b]


def test_split_at_every_offset():
    # 断包：任意切分点都必须恢复同步（含 0xA5 / data_length / CRC 中间）
    f = Frame(3, CMD_GOTO, b"\x01\x02")
    raw = encode(f)
    for i in range(1, len(raw)):
        d = StreamDecoder()
        assert d.feed(raw[:i]) + d.feed(raw[i:]) == [f], f"lost sync at offset {i}"
        assert d.dropped == 0


def test_byte_by_byte_feed():
    f = Frame(9, CMD_GOTO, b"\x07\x08\x09")
    d, got = StreamDecoder(), []
    for byte in encode(f):
        got += d.feed(bytes([byte]))
    assert got == [f]


def test_bad_crc16_dropped():
    raw = bytearray(encode(Frame(1, CMD_GOTO, b"\x05")))
    raw[-1] ^= 0xFF
    d = StreamDecoder()
    assert d.feed(bytes(raw)) == [] and d.dropped == 1


def test_bad_crc8_dropped():
    raw = bytearray(encode(Frame(1, CMD_GOTO, b"\x05")))
    raw[4] ^= 0xFF          # 破坏帧头 CRC8
    d = StreamDecoder()
    assert d.feed(bytes(raw)) == [] and d.dropped == 1


def test_bad_frame_does_not_eat_next_good_frame():
    # 粘包场景：坏帧在好帧前面，好帧必须仍能解出（丢 1 字节重同步）
    bad = bytearray(encode(Frame(1, CMD_GOTO, b"\x05")))
    bad[-1] ^= 0xFF
    good = Frame(2, CMD_GOTO, b"\x06")
    d = StreamDecoder()
    assert d.feed(bytes(bad) + encode(good)) == [good]


def test_data_length_out_of_range_dropped():
    # data_length < 2（装不下 cmd_id）应判坏帧
    raw = bytearray(encode(Frame(1, CMD_GOTO, b"\x00")))
    raw[1:3] = struct.pack("<H", 1)
    raw[4] = crc8(bytes(raw[:4]))       # 帧头 CRC8 仍合法，只有长度非法
    d = StreamDecoder()
    assert d.feed(bytes(raw)) == [] and d.dropped >= 1


def test_garbage_then_valid():
    f = Frame(9, CMD_GOTO, b"\x07")
    d = StreamDecoder()
    assert d.feed(b"\x00\xff\xa5\x13" + encode(f)) == [f]


def test_sof_inside_data_does_not_break_sync():
    # data 里塞入 0xA5（SOF）不得被误判为帧头
    f = Frame(5, CMD_GOTO, bytes([0xA5, 0xA5, 0x02, 0xA5]))
    assert StreamDecoder().feed(encode(f)) == [f]


def test_decode_frame_matches_stream_decoder():
    f = Frame(4, CMD_GOTO, _mk_data())
    assert decode_frame(encode(f)) == f
    # 长度不符 / 非 SOF / 坏校验 → None
    assert decode_frame(encode(f) + b"\x00") is None
    assert decode_frame(b"\x00" + encode(f)[1:]) is None
    assert decode_frame(bytes([*encode(f)[:-1], encode(f)[-1] ^ 0xFF])) is None


def test_encode_rejects_out_of_range():
    with pytest.raises(ValueError):
        encode(Frame(seq=256, cmd_id=CMD_GOTO))
    with pytest.raises(ValueError):
        encode(Frame(seq=0, cmd_id=0x10000))
