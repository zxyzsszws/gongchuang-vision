"""串口帧编解码（实现 `docs/protocol-v1.0.0.md`）。

帧结构（对应文档「表 1-2 通信协议格式」「表 1-4 帧头详细定义」）::

    +-------------+--------+--------+--------------+
    | frame_header| cmd_id |  data  |  frame_tail  |
    |   5-byte    | 2-byte | n-byte | 2-byte CRC16 |
    +-------------+--------+--------+--------------+
      frame_header = SOF(1) + data_length(2) + seq(1) + CRC8(1)

- ``data_length`` = 2 + n（含 cmd_id 的 2 字节），小端 uint16。
- ``CRC8`` 覆盖 frame_header 的前 4 字节（偏移 0~3），校验值本身位于偏移 4。
- ``CRC16`` 覆盖 frame_header（含 CRC8）+ cmd_id + data，即**除 frame_tail 外的
  全部字节**，也就是文档所说的「整包校验」。
- 多字节字段一律小端。

流式解析三条约定：
1. 帧头没收全 / data 没收全 → 保留缓冲等下一批数据，不报错。
2. 坏帧（data_length 越界、CRC8 不符、CRC16 不符）→ **只丢 1 字节**后从下一个
   0xA5 重新找帧，绝不整段清空。这样粘包时一帧损坏不会连带丢掉其后的好帧。
3. data 内部出现 0xA5 不会误判为帧头（靠 CRC8 + CRC16 双重校验排除）。
"""
import struct
from dataclasses import dataclass

from .crc import crc8, crc16

SOF = 0xA5
HEADER_LEN = 5       # SOF(1) + data_length(2) + seq(1) + CRC8(1)
CMD_ID_LEN = 2       # cmd_id
TAIL_LEN = 2         # frame_tail（CRC16）
OVERHEAD = HEADER_LEN + CMD_ID_LEN + TAIL_LEN   # 9

# data_length 的合法区间：至少容纳 cmd_id，上限取 1024 以便在乱流里早放弃同步。
MIN_DATA_LENGTH = CMD_ID_LEN
MAX_DATA_LEN = 1024
MAX_DATA_LENGTH = MIN_DATA_LENGTH + MAX_DATA_LEN

# ---------- 命令码（对应文档「表 1-6 命令码 ID 一览」） ----------
# 0x00xx：视觉单元 → 机器人主控；0x02xx：机器人主控 → 视觉单元。
CMD_HEARTBEAT = 0x0001
CMD_TASK_PLAN = 0x0002
CMD_GOTO = 0x0003
CMD_CALIBRATE = 0x0004
CMD_PLACE = 0x0005
CMD_PICK = 0x0006
CMD_ACK = 0x0007
CMD_POSE = 0x0201
CMD_STATE = 0x0202
CMD_DONE = 0x0203

# 各命令的 data 长度（字节）。编码器不强制检查（便于抓包/造异常帧做测试），
# 由语义层按需比对；文档「表 1-6」的「数据段长度」列与此一致，有单测对账。
DATA_LEN = {
    CMD_HEARTBEAT: 0,
    CMD_TASK_PLAN: 12,
    CMD_GOTO: 6,
    CMD_CALIBRATE: 6,
    CMD_PLACE: 5,
    CMD_PICK: 5,
    CMD_ACK: 1,
    CMD_POSE: 8,
    CMD_STATE: 1,
    CMD_DONE: 1,
}


@dataclass(frozen=True)
class Frame:
    """一帧应用层数据，字段名与文档「表 1-4」一致。"""

    seq: int
    cmd_id: int
    data: bytes = b""


def encode(frame: Frame) -> bytes:
    """把 ``Frame`` 编成完整字节序列（自动填 CRC8 帧头校验与 CRC16 帧尾）。"""
    if not 0 <= frame.seq <= 0xFF:
        raise ValueError(f"seq 必须在 0..255，收到 {frame.seq}")
    if not 0 <= frame.cmd_id <= 0xFFFF:
        raise ValueError(f"cmd_id 必须在 0..65535，收到 {frame.cmd_id}")
    data = bytes(frame.data)
    if len(data) > MAX_DATA_LEN:
        raise ValueError(f"data 最长 {MAX_DATA_LEN} 字节，收到 {len(data)}")

    data_length = CMD_ID_LEN + len(data)
    header = struct.pack("<BHB", SOF, data_length, frame.seq)
    header += bytes([crc8(header)])                     # CRC8 在偏移 4
    body = header + struct.pack("<H", frame.cmd_id) + data
    return body + struct.pack("<H", crc16(body))         # 整包 CRC16


def decode_frame(raw: bytes):
    """校验并解析**一整帧**（长度必须精确）；不合法的帧返回 None。

    供离线抓包工具与文档示例对账使用；流式场景请用 ``StreamDecoder``。
    """
    if len(raw) < OVERHEAD or raw[0] != SOF:
        return None
    data_length = raw[1] | (raw[2] << 8)
    if not MIN_DATA_LENGTH <= data_length <= MAX_DATA_LENGTH:
        return None
    if len(raw) != HEADER_LEN + data_length + TAIL_LEN:
        return None
    if crc8(raw[:4]) != raw[4]:
        return None
    if crc16(raw[:-2]) != (raw[-2] | (raw[-1] << 8)):
        return None
    return Frame(raw[3], raw[5] | (raw[6] << 8), raw[7:-2])


class StreamDecoder:
    """流式解帧：抗粘包/断包，坏帧只丢 1 字节后重新同步。"""

    def __init__(self):
        self._buf = bytearray()
        self.dropped = 0        # 判为坏帧的次数（length/CRC8/CRC16 任一不过）
        self.dropped_bytes = 0  # 为重新同步而跳过的字节数（诊断用）

    def feed(self, data: bytes) -> list:
        self._buf.extend(data)
        out = []
        while True:
            idx = self._buf.find(SOF)
            if idx < 0:
                self.dropped_bytes += len(self._buf)
                self._buf.clear()
                return out
            if idx:
                self.dropped_bytes += idx
                del self._buf[:idx]
            if len(self._buf) < HEADER_LEN:
                return out  # 帧头未收全
            data_length = self._buf[1] | (self._buf[2] << 8)
            if not MIN_DATA_LENGTH <= data_length <= MAX_DATA_LENGTH:
                self._skip_one()
                continue
            if crc8(self._buf[:4]) != self._buf[4]:
                self._skip_one()
                continue
            total = HEADER_LEN + data_length + TAIL_LEN
            if len(self._buf) < total:
                return out  # data 未收全
            if crc16(self._buf[:total - TAIL_LEN]) != (self._buf[total - 2]
                                                       | (self._buf[total - 1] << 8)):
                self._skip_one()
                continue
            raw = bytes(self._buf[:total])
            del self._buf[:total]
            out.append(Frame(raw[3],
                             raw[5] | (raw[6] << 8),
                             raw[7:total - TAIL_LEN]))

    def _skip_one(self):
        """丢掉当前 SOF 字节，从下一个 0xA5 重新找帧（坏帧不牵连后续好帧）。"""
        del self._buf[:1]
        self.dropped += 1
        self.dropped_bytes += 1
