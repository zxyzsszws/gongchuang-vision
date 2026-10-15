from dataclasses import dataclass, field

CMD_HEARTBEAT, CMD_TASK_PLAN, CMD_GOTO, CMD_CALIBRATE = 0x10, 0x11, 0x12, 0x13
CMD_PLACE, CMD_PICK, CMD_ACK = 0x14, 0x15, 0x16
CMD_POSE, CMD_STATE, CMD_DONE = 0x20, 0x21, 0x22
OVERHEAD = 7


@dataclass(frozen=True)
class Frame:
    seq: int
    cmd: int
    payload: bytes = b""


def encode(frame: Frame) -> bytes:
    total = OVERHEAD + len(frame.payload)
    raw = bytes([0xAA, 0x55, total & 0xFF, frame.seq & 0xFF, frame.cmd & 0xFF]) + frame.payload
    xor = 0
    for byte in raw[2:]:
        xor ^= byte
    return raw + bytes([xor, 0x0D])


class StreamDecoder:
    def __init__(self):
        self._buf = bytearray()
        self.dropped = 0

    def feed(self, data: bytes) -> list:
        self._buf.extend(data)
        out = []
        while True:
            idx = self._buf.find(0xAA)
            while idx != -1 and idx + 1 < len(self._buf) and self._buf[idx + 1] != 0x55:
                del self._buf[:idx + 1]
                idx = self._buf.find(0xAA)
            if idx == -1:
                if len(self._buf) > 1:
                    del self._buf[:-1]
                break
            if idx + 2 >= len(self._buf):
                break
            total = self._buf[idx + 2]
            if total < OVERHEAD or total > 128:
                del self._buf[idx:idx + 2]
                self.dropped += 1
                continue
            if idx + total > len(self._buf):
                break
            raw, self._buf = bytes(self._buf[idx:idx + total]), self._buf[idx + total:]
            payload_len = total - OVERHEAD
            xor = 0
            for byte in raw[2:5 + payload_len]:
                xor ^= byte
            if raw[-1] != 0x0D or raw[-2] != xor:
                self.dropped += 1
                continue
            out.append(Frame(raw[3], raw[4], raw[5:5 + payload_len]))
        return out
