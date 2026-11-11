"""McuLink 设备层：协议化串口链路 + 带漂移推算模拟的 MockMcu。

帧格式见 `docs/protocol-v0.2.md`（0xA5 帧头 + 2 字节 cmd_id + CRC16 整包校验）。

MockMcu 语义：
- 收到任何帧都回 ACK（data = 被确认帧的 seq）。
- CMD_CALIBRATE data `<hhh` 为修正量（fixed − last），**叠加**进推算位姿，
  使上报值靠近真值（规格「修正量，主控叠加到其推算」）。
- tick(now) 按 pose_hz 产出 CMD_POSE（data `<hhhH`: x, y, heading01, odometry），
  上报位姿 = 真值 + 漂移×里程 + 已应用的修正量；每上报一次里程 +1m。
- 收到 CMD_GOTO/CMD_PICK/CMD_PLACE 视为接受动作：**延迟 1 个 tick** 回
  CMD_DONE(task_id)，并在触发 DONE 时推进 `_odometry`（动作行进的里程增量）。
"""
import struct
import threading
from typing import Optional, Protocol

from ..core.protocol import (CMD_ACK, CMD_CALIBRATE, CMD_DONE, CMD_GOTO,
                             CMD_PICK, CMD_PLACE, CMD_POSE, Frame,
                             StreamDecoder, encode)

# 触发“延迟 1 tick DONE”主控动作类命令
_ACTION_CMDS = (CMD_GOTO, CMD_PICK, CMD_PLACE)


class McuLink(Protocol):
    def send(self, frame: Frame) -> None: ...

    def poll(self) -> Optional[Frame]: ...

    def tick(self, now: float) -> None: ...


def _clamp_i16(v: int):
    """int16 饱和钳位，返回 (钳位后值, 是否发生钳位)。"""
    if v < -32768:
        return -32768, True
    if v > 32767:
        return 32767, True
    return v, False


class MockMcu:
    """漂移模型：推算位姿 = 真值 + drift_per_m * 里程（修正量叠加后可靠近真值）。"""

    def __init__(self, drift_mm_per_m=20.0, drift_deg_per_m=0.1, pose_hz=10.0):
        self.drift_mm_per_m = drift_mm_per_m
        self.drift_deg_per_m = drift_deg_per_m
        self.pose_hz = pose_hz
        self._rx, self._tx = StreamDecoder(), []
        self._lock = threading.Lock()
        self._last_pose_t = None
        self._odometry = 0.0
        self._bias_x = self._bias_y = self._bias_h = 0.0
        self._ack_seq = 0
        self.truth = (0.0, 0.0)
        # 打包时触发 int16 饱和钳位的帧计数（长仿真可观测/可断言）
        self.out_of_range_clamps = 0
        # 最近一次 POSE 快照（上报值/标称真值），供只读访问器 get_pose_report 使用
        self._pose_snap = (0, 0, 0, 0.0, 0.0)  # reported_x, reported_y, head01, nominal_x, nominal_y
        # 动作命令 → 延迟 1 tick 的 CMD_DONE 队列（task_id, 剩余 tick 数）
        self._pending_done = []
        self._task_id = 0

    def send(self, frame):
        for acked in self._rx.feed(encode(frame)):
            with self._lock:
                self._ack_seq = (self._ack_seq + 1) & 0xFF
                self._tx.append(Frame(seq=self._ack_seq, cmd=CMD_ACK,
                                      data=bytes([acked.seq])))
                if acked.cmd == CMD_CALIBRATE:
                    if len(acked.data) != 6:      # 防御：长度不符不崩，仅忽略
                        continue
                    dx, dy, dh01 = struct.unpack("<hhh", acked.data)
                    # 修正量叠加到推算位姿（不是相减：那会把上报推离真值）
                    self._bias_x += dx
                    self._bias_y += dy
                    self._bias_h += dh01 / 10.0
                if acked.cmd in _ACTION_CMDS:
                    # 接受动作：下一个 tick 触发 DONE（单线程驱动约定）
                    self._task_id = (self._task_id + 1) & 0xFF
                    self._pending_done.append([self._task_id, 1])

    def _fire_due_done(self):
        """弹出到期 pending，推进里程并产出 CMD_DONE(task_id)。持锁调用。"""
        still, fired = [], []
        for item in self._pending_done:
            item[1] -= 1
            if item[1] <= 0:
                fired.append(item[0])
            else:
                still.append(item)
        self._pending_done = still
        for task_id in fired:
            self._odometry += 1.0  # 动作执行亦产里程（GOTO/PLACE 行进）
            self._tx.append(Frame(seq=0, cmd=CMD_DONE,
                                  data=bytes([task_id])))

    def tick(self, now):
        with self._lock:
            if self._pending_done:
                self._fire_due_done()
        if self._last_pose_t is None or (now - self._last_pose_t) >= 1.0 / self.pose_hz:
            self._last_pose_t = now
            self._odometry += 1.0  # 每上报周期推进 1m，模拟行进
            x, clamped_x = _clamp_i16(
                int(self._odometry * 1000 + self.drift_mm_per_m * self._odometry + self._bias_x))
            y, clamped_y = _clamp_i16(int(self.truth[1] + self._bias_y))
            # 航向回绕到有符号区间 [-18000, 18000)：Python 的 % 恒非负，
            # 负净航向会被映射到 [1,35999]，≥32768 时 struct '<h' 直接抛
            # struct.error（轻微过修后为常态）。格式 <hhhH 由规格冻结，不改布局。
            h = ((int((self.drift_deg_per_m * self._odometry + self._bias_h) * 10)
                  + 18000) % 36000) - 18000
            # 记录本次上报快照（reported vs nominal 标称真值），供只读访问器取用；
            # nominal 为纯里程位姿（odom×1000mm），不含漂移/修正。
            self._pose_snap = (x, y, h, self._odometry * 1000.0, self.truth[1])
            if clamped_x or clamped_y:
                self.out_of_range_clamps += 1  # 每个被钳位的 pack 计一次
            with self._lock:
                self._tx.append(Frame(seq=0, cmd=CMD_POSE,
                                      data=struct.pack("<hhhH", x, y, h,
                                                       int(self._odometry) & 0xFFFF)))

    def poll(self):
        with self._lock:
            return self._tx.pop(0) if self._tx else None

    def get_pose_report(self) -> dict:
        """只读访问器（供 sim 场景按「当前位姿误差」合成 field-as-seen，不改任何模型行为）：

        - reported_x/y/head01：最近一次 POSE 上报的 int16 值（含钳位）。
        - nominal_x/y：模型标称真值（里程×1000mm / truth_y），不含漂移与修正。
        - err_x_mm / err_y_mm / err_head_deg：实时「上报 − 标称」位姿误差
          （= 标称漂移×里程 + 已叠加修正；bias 随 CALIBRATE 同步更新，立即可见）。
        - odom：当前里程（m）。上报位姿 = nominal + err，场景侧据此合成观测。
        """
        rx, ry, rh01, nx, ny = self._pose_snap
        return {
            "reported_x": rx, "reported_y": ry, "reported_head01": rh01,
            "nominal_x": nx, "nominal_y": ny,
            "err_x_mm": self.drift_mm_per_m * self._odometry + self._bias_x,
            "err_y_mm": self._bias_y,
            "err_head_deg": self.drift_deg_per_m * self._odometry + self._bias_h,
            "odom": self._odometry,
        }


class SerialLink:
    def __init__(self, port, baudrate=115200):
        import serial  # 惰性导入：测试环境不打开端口
        self._ser = serial.Serial(port, baudrate, timeout=0)
        self._rx, self._tx = StreamDecoder(), []
        self._lock = threading.Lock()

    def send(self, frame):
        with self._lock:
            self._ser.write(encode(frame))

    def poll(self):
        self._rx_feed()
        with self._lock:
            return self._tx.pop(0) if self._tx else None

    def _rx_feed(self):
        data = self._ser.read(256)  # timeout=0 → 非阻塞
        if data:
            with self._lock:
                self._tx.extend(self._rx.feed(data))

    def tick(self, now):
        pass
