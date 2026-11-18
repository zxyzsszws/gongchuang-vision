"""MockMcu：模拟机器人主控（V2.2.0 语义）。

V2.2.0 里电控向视觉要三样东西，都要**主动发请求**，视觉才作答：

1. 扫码（任务 1）—— 开赛时问一次，拿到 16 字节任务计划；
2. 转盘取料点的颜色（任务 2）—— 到原料区把臂悬停好之后问，视觉等转盘停稳再答；
3. 对准主圆（任务 3）—— 到放置点把臂伸到主圆附近之后问，拿到 d 与 α 去执行。

没有周期上报，也没有 done：电控收到结果就完事，还需要就再发一次请求。

## 收发方向

这个模拟器在同一个进程里跟视觉对插，所以方向得说清楚：

- ``send(frame)`` —— 视觉发来一帧（视觉调用它），这里当作收到；
- ``poll()`` —— 交给视觉收的队列（视觉调用它），电控自己的请求排在这里。

真实串口上这两个方向是两个独立的字节流，这里只是把它们拆成两个队列，语义一致。

## 这个模拟器做什么

- 收到结果就记下来（任务 1 存计划、任务 2 存颜色、任务 3 存 d 与 α）；
- 等不到结果就重发**原请求**（seq 与内容都不变），超过重试上限记为放弃。

**里程推算已删除。** V2.2.0 的视觉不再上报位姿，电控里也就不存在「被视觉
清零」这件事。电控的定位精度归电控自己负责。
"""
import struct
import threading
from typing import Optional, Protocol

from ..core.protocol import (CMD_REQUEST, CMD_RESULT, ST_OK, TASK_ALIGN,
                             TASK_SCAN, TASK_TURNTABLE, Frame, StreamDecoder,
                             encode)

# 等结果等多久才重发（秒）。真实电控用几十毫秒级；这里留宽一点，避免
# 「视觉还在等转盘停」时就被误判超时而反复重发。
DEFAULT_RETRY_S = 1.0


class McuLink(Protocol):
    """链路的最小接口：发一帧、取一帧、按时间推进。"""

    def send(self, frame: Frame) -> None: ...

    def poll(self) -> Optional[Frame]: ...

    def tick(self, now: float) -> None: ...


class MockMcu:
    """按需问话的机器人主控模拟器。

    - ``arm_l_mm`` / ``arm_alpha_deg``：问对准时上报的臂状态；
    - ``retry``：等不到结果时重发原请求的次数上限；
    - ``retry_s``：每次重发前等多久。
    """

    def __init__(self, arm_l_mm=600.0, arm_alpha_deg=0.0,
                 retry=3, retry_s=DEFAULT_RETRY_S):
        self.arm_l_mm = float(arm_l_mm)
        self.arm_alpha_deg = float(arm_alpha_deg)
        self.retry = int(retry)
        self.retry_s = float(retry_s)

        self._lock = threading.Lock()
        self._inbox = []          # 视觉发来的帧（send 进来、tick 里消费）
        self._outbox = []         # 待视觉收取的帧（poll 出去）
        self.sent = []            # 电控发出的所有帧（含重发），诊断与测试用
        self.requests = []        # 已发出的任务码（按发送顺序，含重发）
        self._seq = 0
        self._pending = {}        # seq -> {task, payload, deadline, retries}

        # 收到的结果
        self.received = []        # [(cmd_id, data), ...]
        self.plan = None          # 任务 1 的 16 字节
        self.colors = []          # 任务 2 的颜色（按到达顺序）
        self.solutions = []       # 任务 3 的 {d_mm, alpha_deg, echo}
        self.statuses = []        # [(task, status), ...] 含失败回执
        self.gave_up = []         # 重试用尽的任务码

    # ---------- 链路接口 ----------

    def send(self, frame: Frame) -> None:
        """视觉发来一帧：先进收件箱，等 ``tick`` 时再解析，避免在调用栈里回调。"""
        with self._lock:
            self._inbox.append(frame)

    def poll(self) -> Optional[Frame]:
        """电控要发给视觉的下一帧。"""
        with self._lock:
            return self._outbox.pop(0) if self._outbox else None

    def tick(self, now: float) -> None:
        """按时间推进：处理收到的结果、推进重试。"""
        self._drain()
        self._tick_retry(now)

    def _drain(self):
        with self._lock:
            frames = list(self._inbox)
            self._inbox.clear()
        for f in frames:
            self._on_frame(f)

    def _on_frame(self, f):
        self.received.append((f.cmd_id, f.data))
        if f.cmd_id != CMD_RESULT or len(f.data) < 3:
            return
        echo_seq, task, status = f.data[0], f.data[1], f.data[2]
        body = f.data[3:]
        self.statuses.append((task, status))
        self._pending.pop(echo_seq, None)
        if status != ST_OK:
            return
        if task == TASK_SCAN:
            self.plan = body
        elif task == TASK_TURNTABLE and body:
            self.colors.append(body[0])
        elif task == TASK_ALIGN and len(body) >= 4:
            d_mm, alpha01 = struct.unpack("<hh", body[:4])
            self.solutions.append({"d_mm": d_mm,
                                   "alpha_deg": alpha01 / 10.0,
                                   "echo": echo_seq})

    # ---------- 请求 ----------

    def _next_seq(self):
        self._seq = self._seq % 255 + 1
        return self._seq

    def _emit(self, frame):
        """把一帧放进发件箱（视觉下一次 poll 就能拿到），同时记进 ``sent``。"""
        with self._lock:
            self._outbox.append(frame)
            self.sent.append(frame)

    def _request(self, task, payload, now):
        seq = self._next_seq()
        self._emit(Frame(seq, CMD_REQUEST, bytes((task,)) + bytes(payload)))
        self.requests.append(task)
        self._pending[seq] = {"task": task, "payload": bytes(payload),
                              "deadline": now + self.retry_s, "retries": 0}
        return seq

    def _tick_retry(self, now):
        for seq in list(self._pending):
            job = self._pending[seq]
            if now < job["deadline"]:
                continue
            if job["retries"] >= self.retry:
                del self._pending[seq]
                self.gave_up.append(job["task"])
                continue
            # 重发原帧：seq 与内容都不变，视觉据此认出「同一个请求」
            self._emit(Frame(seq, CMD_REQUEST,
                             bytes((job["task"],)) + job["payload"]))
            job["retries"] += 1
            job["deadline"] = now + self.retry_s

    def request_scan(self, now=0.0):
        """任务 1：扫码要任务计划。"""
        return self._request(TASK_SCAN, b"", now)

    def request_turntable(self, now=0.0):
        """任务 2：问取料点是哪个颜色。"""
        return self._request(TASK_TURNTABLE, b"", now)

    def request_align(self, now=0.0, l_mm=None, alpha_deg=None):
        """任务 3：上报臂状态，要 d 与 α 把相机对准主圆。"""
        l = self.arm_l_mm if l_mm is None else l_mm
        a = self.arm_alpha_deg if alpha_deg is None else alpha_deg
        alpha01 = int(round(a * 10.0)) & 0xFFFF
        return self._request(TASK_ALIGN,
                             struct.pack("<HH", int(round(l)), alpha01), now)

    def sent_cmd_ids(self):
        """电控发出过的帧的命令码（含重发）。"""
        with self._lock:
            return [f.cmd_id for f in self.sent]

    @property
    def pending_count(self):
        """还在等结果、且没到重试上限的请求数。"""
        return len(self._pending)


class SerialLink:
    """真实串口链路：pyserial 读线程 + 流式解帧。"""

    def __init__(self, port, baudrate):
        import serial      # pyserial 只在 live/replay 下需要

        self._ser = serial.Serial(port, baudrate, timeout=0)
        self._decoder = StreamDecoder()
        self._lock = threading.Lock()
        self._rx = []
        self.dropped = 0

    def tick(self, now):
        data = self._ser.read(4096)
        if not data:
            return
        frames = self._decoder.feed(data)
        with self._lock:
            self._rx.extend(frames)

    def send(self, frame):
        with self._lock:
            self._ser.write(encode(frame))

    def poll(self):
        with self._lock:
            return self._rx.pop(0) if self._rx else None
