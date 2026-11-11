"""CRC 表与算法校验：用协议文档（数据手册）给的常量固化，防止查表实现漂移。

CRC8 表由算法生成（`src/core/crc.py`）而非手抄 256 个常量——手抄易错且不可
审查。这里用文档表中可公开对账的 17 个值（首行 16 个 + 末值）做交叉验证：
若生成参数（poly/init/反射）被改动，这些断言会立刻失败。
"""
from src.core.crc import CRC8_TAB, CRC16_TAB, crc8, crc16

# 文档 CRC8 表首行（索引 0x00..0x0F）与最后一个值（索引 0xFF）
DOC_CRC8_FIRST_ROW = (
    0x00, 0x5E, 0xBC, 0xE2, 0x61, 0x3F, 0xDD, 0x83,
    0xC2, 0x9C, 0x7E, 0x20, 0xA3, 0xFD, 0x1F, 0x41,
)
DOC_CRC8_LAST = 0x35


def test_crc8_table_matches_datasheet_row():
    assert CRC8_TAB[:16] == DOC_CRC8_FIRST_ROW
    assert CRC8_TAB[255] == DOC_CRC8_LAST


def test_crc8_table_shape():
    assert len(CRC8_TAB) == 256
    assert all(0 <= v <= 0xFF for v in CRC8_TAB)
    assert len(set(CRC8_TAB)) == 256  # 双射：任一单字节错误必被检出


def test_crc8_init_and_empty():
    # init 0xFF、无结果异或：空数据返回初值本身
    assert crc8(b"") == 0xFF
    assert crc8(b"\x00") == CRC8_TAB[0xFF]
    assert crc8(b"\xa5") == CRC8_TAB[0xFF ^ 0xA5]


def test_crc8_is_linear_in_prefix():
    # 分段累计与一次性计算必须等价（StreamDecoder 依赖这一点做增量校验）
    data = bytes([0xA5, 0x0E, 0x00, 0x01])
    assert crc8(data[1:], crc8(data[:1])) == crc8(data)
    assert crc8(data[2:], crc8(data[:2])) == crc8(data)


def test_crc16_table_shape_and_poly():
    assert len(CRC16_TAB) == 256
    assert all(0 <= v <= 0xFFFF for v in CRC16_TAB)
    # 反射多项式 0x8408（0x1021 的位反转）：索引 1 的展开结果
    assert CRC16_TAB[1] == 0x1189


def test_crc16_init_and_empty():
    assert crc16(b"") == 0xFFFF
    assert crc16(b"\x00") == (0xFFFF >> 8) ^ CRC16_TAB[0xFF]


def test_crc16_is_linear_in_prefix():
    data = bytes([0xA5, 0x0E, 0x00, 0x01, 0x02, 0x00])
    assert crc16(data[3:], crc16(data[:3], 0xFFFF)) == crc16(data)
    assert crc16(data, 0xFFFF) == crc16(data)


def test_known_vectors():
    # CRC-16/X-25 的公开 check 值 0x906E 带结果异或；本协议不用 xorout，
    # 故期望值为 0x906E ^ 0xFFFF = 0x6F91。
    assert crc16(b"123456789") == 0x6F91
