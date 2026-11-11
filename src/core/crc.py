"""CRC8 / CRC16 校验（纯函数，无硬件依赖）。

参数取自协议文档（`docs/protocol-v0.2.md`）：
- **CRC8**：多项式 0x31 的反射形式（0x8C，LSB-first），初值 0xFF，无结果异或。
  用于帧头 5 字节中的最后 1 字节，覆盖 SOF / data_length / seq 共 4 字节。
- **CRC16**：多项式 0x1021 的反射形式（0x8408，LSB-first），初值 0xFFFF，
  无结果异或（即 CRC-16/X-25 去掉最终异或）。用于帧尾 2 字节，**整包**校验。

CRC8 表采用「算法生成 + 单测对照数据手册」的方式而不是手抄 256 个常量：
手抄极易出错且难以审查，生成式与查表式逐位等价（见 `tests/test_crc.py`
里对数据手册首行 16 个值与末值 0x35 的固化断言）。
"""

CRC8_INIT = 0xFF
CRC16_INIT = 0xFFFF

_CRC8_POLY_REFLECTED = 0x8C      # 0x31 的位反转
_CRC16_POLY_REFLECTED = 0x8408   # 0x1021 的位反转


def _build_crc8_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC8_POLY_REFLECTED if crc & 1 else crc >> 1
        table.append(crc & 0xFF)
    return tuple(table)


def _build_crc16_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC16_POLY_REFLECTED if crc & 1 else crc >> 1
        table.append(crc & 0xFFFF)
    return tuple(table)


CRC8_TAB = _build_crc8_table()
CRC16_TAB = _build_crc16_table()


def crc8(data, crc=CRC8_INIT):
    """逐字节查表 CRC8；crc 为初值（缺省 0xFF），便于分段累计。"""
    for byte in data:
        crc = CRC8_TAB[(crc ^ byte) & 0xFF]
    return crc


def crc16(data, crc=CRC16_INIT):
    """逐字节查表 CRC16；crc 为初值（缺省 0xFFFF），便于分段累计。"""
    for byte in data:
        crc = (crc >> 8) ^ CRC16_TAB[(crc ^ byte) & 0xFF]
    return crc
