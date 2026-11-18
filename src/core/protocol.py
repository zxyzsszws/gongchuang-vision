"""串口帧编解码 + V2.2.0 的命令与任务码（文档 `docs/protocol-v2.2.0.md`）。

帧结构：

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

## V2.2.0 只有两条命令

    CMD_REQUEST  电控→视觉   我要一件测量结果，任务码说明要哪一件
    CMD_RESULT   视觉→电控   这是结果，带回请求的 seq

没有心跳，没有独立的应答帧，也没有 done。**结果帧本身就是应答**：电控等不到
结果就重发原请求，视觉见到同一个 seq 就把上次的结果原样再发一遍。一条往返
说清一件事，加任务时只加任务码、不加命令码。
"""
import struct
from dataclasses import dataclass

from .crc import crc8, crc16

SOF = 0xA5
HEADER_LEN = 5       # SOF(1) + data_length(2) + seq(1) + CRC8(1)
ID_FIELD_LEN = 2     # cmd_id 占 2 字节（注意不是命令码常量，命名刻意避开 CMD_ 前缀）
TAIL_LEN = 2         # frame_tail（CRC16）
OVERHEAD = HEADER_LEN + ID_FIELD_LEN + TAIL_LEN   # 9

# data_length 的合法区间：至少容纳 cmd_id，上限取 1024 以便在乱流里早放弃同步。
MIN_DATA_LENGTH = ID_FIELD_LEN
MAX_DATA_LEN = 1024
MAX_DATA_LENGTH = MIN_DATA_LENGTH + MAX_DATA_LEN

# ---------- 命令码（对应文档「表 1-5 命令码 ID 一览」） ----------
CMD_REQUEST = 0x0010       # 电控→视觉：请求一次测量
CMD_RESULT = 0x0011        # 视觉→电控：测量结果（即应答）

# 不再使用：收到应静默丢弃并计数，不得视为错误。
#   0x0001 —— V2.1.0 取消心跳
#   0x0002/0x0003/0x0004 —— V2.2.0 改为按需请求，不再主动上报；任务计划并入
#                            CMD_RESULT 任务码 1，位姿上报整体取消
#   0x0005 —— V2.2.0 取消区域提示（视觉已不需要知道自己在哪个区域）
#   0x0006/0x0007/0x0008 —— V2.2.0 并入 CMD_REQUEST/CMD_RESULT 的任务码 3
#   0x0201/0x0202/0x0203 —— V2.0.0 已删除（主控上报位姿 / 主控状态 / 动作完成）
RETIRED_CMDS = frozenset({0x0001, 0x0002, 0x0003, 0x0004, 0x0005,
                          0x0006, 0x0007, 0x0008,
                          0x0201, 0x0202, 0x0203})

# ---------- 任务码（对应文档「表 1-6 任务码一览」） ----------
TASK_SCAN = 1           # 扫码，取 16 字节任务计划
TASK_TURNTABLE = 2      # 转盘上此刻停在取料点的是哪个颜色
TASK_ALIGN = 3          # 把相机对准主圆，回 d 与 α

TASKS = (TASK_SCAN, TASK_TURNTABLE, TASK_ALIGN)

# 请求的数据段长度（含任务码那 1 字节）。文档「表 1-6」的「请求长度」列。
REQ_LEN = {
    TASK_SCAN: 1,           # 任务码
    TASK_TURNTABLE: 1,      # 任务码
    TASK_ALIGN: 5,          # 任务码 + l(2) + α(2)
}

# 结果的数据段长度（含 echo_seq、任务码、状态那 3 字节）。文档「表 1-6」。
RES_LEN = {
    TASK_SCAN: 19,          # 3 + 16 字节任务计划
    TASK_TURNTABLE: 4,      # 3 + 颜色 1 字节
    TASK_ALIGN: 7,          # 3 + d(2) + α(2)
}

# 两条命令的 data 长度不固定（随任务码变），所以 DATA_LEN 只记「最短合法长度」，
# 供解析时做下限检查；逐任务的精确长度看 REQ_LEN / RES_LEN。
DATA_LEN = {
    CMD_REQUEST: min(REQ_LEN.values()),
    CMD_RESULT: min(RES_LEN.values()),
}

# ---------- 状态码（结果帧第 3 字节，对应文档「表 1-8」） ----------
ST_OK = 0               # 成功，正文可直接使用
ST_NOT_READY = 1        # 还没准备好 / 没测到：稍后重发同一个请求
ST_UNSOLVABLE = 2       # 测到了但无解或不可信：换个姿态再来，别沿用
ST_BAD_REQUEST = 3      # 任务码不认识、载荷长度不对：实现不匹配，需要人查

STATUS_NAMES = {
    ST_OK: "成功",
    ST_NOT_READY: "没准备好",
    ST_UNSOLVABLE: "无解或不可信",
    ST_BAD_REQUEST: "请求不合法",
}

# ---------- 颜色编号（与任务计划里的颜色一致，0 表示认不出） ----------
COLOR_UNKNOWN = 0


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

    data_length = ID_FIELD_LEN + len(data)
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
