"""串口帧编解码：RoboMaster 风格帧（`docs/protocol-v0.2.md`）。

帧结构（多字节字段一律 little-endian）::

    frame_header(5B) | cmd_id(2B) | data(nB) | frame_tail(2B)
    ├─ SOF          1B   固定 0xA5
    ├─ data_length  2B   = len(data) + 2，**含 cmd_id 的 2 字节**
    ├─ seq          1B   发送方自增序号（0 保留给周期帧）
    └─ CRC8         1B   覆盖 SOF + data_length + seq 共 4 字节
    frame_tail 为 CRC16，覆盖「header + cmd_id + data」整包。

设计约束（与电控组冻结口径一致）：
- 固定开销 = 5 + 2 + 2 = 9 字节；单帧 data 上限 ``MAX_DATA_LEN``。
- 坏帧一律「丢 1 字节后重新找 SOF」，绝不清空整个缓冲——这样粘包场景下
  一帧损坏不会连带丢弃它后面的好帧。
- 兼容粘包/断包/帧内出现 0xA5（靠 CRC8 头校验 + CRC16 整包校验区分）。
"""
import struct
from dataclasses import dataclass

from .crc import crc8, crc16

SOF = 0xA5
HEADER_LEN = 5      # SOF(1) + data_length(2) + seq(1) + CRC8(1)
CMD_LEN = 2         # cmd_id
CRC_LEN = 2         # frame_tail
OVERHEAD = HEADER_LEN + CMD_LEN + CRC_LEN   # 9

# data_length 的合法区间：至少容纳 cmd_id；上限给小到足以在乱流里快速放弃同步，
# 又给调试留足余量（本协议最大帧 TASK_PLAN 仅 12 字节）。
MIN_DATA_LENGTH = CMD_LEN
MAX_DATA_LEN = 1024
MAX_DATA_LENGTH = MIN_DATA_LENGTH + MAX_DATA_LEN

# ---------- 命令码（cmd_id，2 字节） ----------
# 0x0001-0x000F：视觉 → 主控；0x0011-0x001F：主控 → 视觉。
# 注：图片里的 0x0001/0x0002/0x0003/0x0101（比赛状态/结果/血量/场地事件）只作
# 编号风格参考，本协议不复用其语义。
CMD_HEARTBEAT = 0x0001
CMD_TASK_PLAN = 0x0002
CMD_GOTO = 0x0003
CMD_CALIBRATE = 0x0004
CMD_PLACE = 0x0005
CMD_PICK = 0x0006
CMD_ACK = 0x0007
CMD_POSE = 0x0011
CMD_STATE = 0x0012
CMD_DONE = 0x0013


@dataclass(frozen=True)
class Frame:
    """一帧应用层数据。``cmd`` 即文中的 cmd_id（2 字节）。"""

    seq: int
    cmd: int
    data: bytes = b""


def encode(frame: Frame) -> bytes:
    """把 ``Frame`` 编成完整字节序列（含帧头 CRC8 与帧尾 CRC16）。"""
    if not 0 <= frame.seq <= 0xFF:
        raise ValueError(f"seq 必须在 0..255，收到 {frame.seq}")
    if not 0 <= frame.cmd <= 0xFFFF:
        raise ValueError(f"cmd 必须在 0..65535，收到 {frame.cmd}")
    data = bytes(frame.data)
    if len(data) > MAX_DATA_LEN:
        raise ValueError(f"data 最长 {MAX_DATA_LEN} 字节，收到 {len(data)}")

    data_length = CMD_LEN + len(data)
    header = struct.pack("<BHB", SOF, data_length, frame.seq)
    header += bytes([crc8(header)])
    body = header + struct.pack("<H", frame.cmd) + data
    return body + struct.pack("<H", crc16(body))


def decode_frame(raw: bytes):
    """校验并解析**一整帧**（长度精确）；坏帧返回 None。供离线抓包工具使用。"""
    if len(raw) < OVERHEAD or raw[0] != SOF:
        return None
    data_length = raw[1] | (raw[2] << 8)
    if not MIN_DATA_LENGTH <= data_length <= MAX_DATA_LENGTH:
        return None
    if len(raw) != HEADER_LEN + data_length + CRC_LEN:
        return None
    if crc8(raw[:4]) != raw[4]:
        return None
    if crc16(raw[:-2]) != (raw[-2] | (raw[-1] << 8)):
        return None
    return Frame(raw[3], raw[5] | (raw[6] << 8), raw[7:-2])


class StreamDecoder:
    """流式解帧：抗粘包/断包，坏帧丢 1 字节后重新同步。"""

    def __init__(self):
        self._buf = bytearray()
        self.dropped = 0        # 丢弃的坏帧数（CRC8/CRC16/length 任一不过计 1）
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
                return out  # 帧头未收全，等下批数据
            data_length = self._buf[1] | (self._buf[2] << 8)
            if not MIN_DATA_LENGTH <= data_length <= MAX_DATA_LENGTH:
                self._skip_one()
                continue
            if crc8(self._buf[:4]) != self._buf[4]:
                self._skip_one()
                continue
            total = HEADER_LEN + data_length + CRC_LEN
            if len(self._buf) < total:
                return out  # data 未收全
            if crc16(self._buf[:total - CRC_LEN]) != (self._buf[total - 2]
                                                      | (self._buf[total - 1] << 8)):
                self._skip_one()
                continue
            raw = bytes(self._buf[:total])
            del self._buf[:total]
            out.append(Frame(raw[3], raw[5] | (raw[6] << 8), raw[7:total - CRC_LEN]))

    def _skip_one(self):
        """丢掉当前 SOF 字节，从下一个 0xA5 重新找帧（坏帧不牵连后续好帧）。"""
        del self._buf[:1]
        self.dropped += 1
        self.dropped_bytes += 1
