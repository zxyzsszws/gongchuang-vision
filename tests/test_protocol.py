from src.core.protocol import Frame, encode, StreamDecoder, CMD_GOTO


def _mk_payload(): return b"\x64\x00\x2c\x01\x88\x13"


def test_roundtrip():
    f = Frame(seq=7, cmd=CMD_GOTO, payload=_mk_payload())
    assert StreamDecoder().feed(encode(f)) == [f]


def test_sticky_and_split():
    a, b = Frame(1, CMD_GOTO, b"\x01"), Frame(2, CMD_GOTO, b"\x02\x03")
    raw = encode(a) + encode(b)
    d = StreamDecoder()
    got = d.feed(raw[:3]) + d.feed(raw[3:])
    assert got == [a, b]


def test_bad_checksum_dropped():
    raw = bytearray(encode(Frame(1, CMD_GOTO, b"\x05")))
    raw[-2] ^= 0xFF
    d = StreamDecoder()
    assert d.feed(bytes(raw)) == [] and d.dropped == 1


def test_garbage_then_valid():
    f = Frame(9, CMD_GOTO, b"\x07")
    d = StreamDecoder()
    assert d.feed(b"\x00\xff" + encode(f)) == [f]


def test_split_at_every_offset():
    # 断包：任意切分点（含 0xAA 之后）都必须恢复同步（接口要求抗粘包/断包）
    f = Frame(3, CMD_GOTO, b"\x01\x02")
    raw = encode(f)
    for i in range(1, len(raw)):
        d = StreamDecoder()
        assert d.feed(raw[:i]) + d.feed(raw[i:]) == [f], f"lost sync at offset {i}"
        assert d.dropped == 0
