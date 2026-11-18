"""MockMcu：模拟机器人主控（V2.1.0 语义）。

在「视觉只测量、电控全规划」的分工里，电控向视觉索取的东西只有两类：

1. **位姿**（0x0003）—— 被动接收，收到就整体替换自身估计；
2. **臂解算**（0x0006 → 0x0008 → 0x0007 done）—— 主动发起，一次请求一次应答。

这里的模拟器按真实电控的行为做这几件事：

- 收到 ``0x0002``（任务计划）回一帧 ``0x0007``；
- 按自己的规划在区域之间行驶，进入新区域时发 ``0x0005``；
- 想伸手取放料时发 ``0x0006`` 臂状态，收到 ``0x0008`` 后执行，执行完回 done；
- 自己不算位姿，里程推算带累积误差（``drift_mm_per_m``），用来验证「视觉每次
  看见靶就把误差清零」这一核心性质。

**不发心跳。** 这个协议里电控是事件驱动的，停着作业时可能十几秒不发一帧，那是
正常状态；把静默当断链只能得到误报。
"""
import math
import struct
import threading
from typing import Optional, Protocol

from ..algorithms.types import Pose
from ..core.protocol import (ACK_DUPLICATE, ACK_FAILED, ACK_OK, CMD_ACK,
                             CMD_ARM_SOLUTION, CMD_ARM_STATE, CMD_POSE,
                             CMD_REGION, CMD_TASK_PLAN, REGION_MATERIAL,
                             REGION_ROUGH, REGION_STAGING, REGION_START, Frame,
                             StreamDecoder, encode)


class McuLink(Protocol):
    """链路的最小接口：发一帧、取一帧、按时间推进。"""

    def send(self, frame: Frame) -> None: ...

    def poll(self) -> Optional[Frame]: ...

    def tick(self, now: float) -> None: ...


def _norm180(deg):
    """角度归一到 (−180, 180]：比较两个朝向时必须走最短方向。"""
    return (deg + 180.0) % 360.0 - 180.0


# 默认行驶路线：起点 → 原料区 → 粗加工区 → 暂存区 → 再走一遍 → 回起点。
# 第一批和第二批流程相同，因此路线是两轮。第一项是开机所在区域。
DEFAULT_ROUTE = (REGION_START, REGION_MATERIAL, REGION_ROUGH,
                 REGION_STAGING, REGION_MATERIAL, REGION_ROUGH,
                 REGION_STAGING, REGION_START)


class MockMcu:
    """带里程漂移的机器人主控模拟器。

    - ``route``：车依次所处的区域码列表，第一项是开机所在区域（起点）。
    - ``waypoints``：与 ``route`` 等长的标称停位（``Pose``）。给了就按航点走
      （真值在区域之间连线移动、到点后转向该区域的车头朝向）；不给则沿 +x
      直走，仅用于只关心漂移量的单元测试。
    - ``drift_mm_per_m`` / ``drift_deg_per_m``：里程推算的累积误差率。
    - ``apply_pose``：收到 0x0003 是否像真实电控那样**整体替换**自身估计。
    - ``arm_l_mm`` / ``exec_s``：臂的水平伸出长度与一次动作的耗时。
    """

    def __init__(self, drift_mm_per_m=8.0, drift_deg_per_m=0.1,
                 route=DEFAULT_ROUTE,
                 leg_mm=1200.0, speed_mm_s=1200.0, waypoints=None,
                 apply_pose=False, arm_l_mm=600.0, arm_h_mm=300.0,
                 exec_s=0.2, use_done=True):
        self.drift_mm_per_m = drift_mm_per_m
        self.drift_deg_per_m = drift_deg_per_m
        self.route = list(route)
        self.leg_mm = float(leg_mm)
        self.speed_mm_s = float(speed_mm_s)
        # 收到 0x0003 就整体替换自身估计（真实电控的做法）。单元测试默认关掉，
        # 这样「发一帧有没有把误差清零」是个显式动作，不会被隐式应用掩盖。
        self.apply_pose = bool(apply_pose)
        if waypoints is not None and len(waypoints) != len(self.route):
            raise ValueError(
                f"waypoints 长度 {len(waypoints)} 必须与 route 长度 "
                f"{len(self.route)} 一致，否则不知道每个区域停在哪")
        self.waypoints = list(waypoints) if waypoints else None

        self.arm_l_mm = float(arm_l_mm)
        self.arm_h_mm = float(arm_h_mm)
        self.exec_s = float(exec_s)
        self.use_done = bool(use_done)

        self._rx = StreamDecoder()
        self._tx = []
        self._lock = threading.Lock()
        self._ack_seq = 0
        self._arm_seq = 0

        # 电控自己以为的位置（带累积误差），以及真值（mock 内部知道，仅供测试观察）
        self.est_x = self.est_y = self.est_h = 0.0
        self.true_x = self.true_y = self.true_h = 0.0
        self._odometry_m = 0.0

        self._leg_index = 0
        self._leg_progress = 0.0
        self._finished = False
        self._last_now = None
        self.received = []        # 收到过的 (cmd_id, data)，供测试观察
        self.corrections = 0      # 应用过多少次视觉位姿
        self.regions_sent = []    # 发过的去向提示序列

        # 臂解算往返的状态
        self.arm_requests = []    # 发过的 0x0006（data, seq）
        self.solutions = []       # 收到过的 0x0008（已解码 dict）
        self.done_sent = []       # 发过的 done（ack_seq, result）
        self.executed = []        # 真正执行过的解算（按 seq 去重后的结果）
        self._pending_solution = None   # 待执行的解算
        self._done_at = None            # 该在什么时刻回 done
        self._executed_seqs = set()     # 已执行过的 0x0006 seq，用于识别重复请求

    # ---------- 链路接口 ----------

    def send(self, frame):
        """视觉发来一帧：按命令分别处理，只有该回应的才回。"""
        for got in self._rx.feed(encode(frame)):
            with self._lock:
                self.received.append((got.cmd_id, bytes(got.data)))
            if got.cmd_id == CMD_TASK_PLAN:
                self._reply_ack(got.seq, self._plan_result(got.data))
            elif got.cmd_id == CMD_ARM_SOLUTION:
                self._on_solution(got)
            elif got.cmd_id == CMD_POSE:
                if self.apply_pose:
                    self._apply_pose_bytes(bytes(got.data))
            # 0x0004 转盘 / 0x0005 去向提示：只收，不应答

    def _plan_result(self, data):
        return ACK_OK if len(data) == 16 else ACK_FAILED

    def _on_solution(self, frame):
        """收到解算结果：登记执行，到点回 done。

        同一个 0x0006 seq 只执行一次。视觉因为等不到 done 而重发解算帧时，seq
        不变，这里认出重复就只补一个 done，不再执行——否则车会沿同一个 d 走两遍。
        """
        if len(frame.data) != 10:
            return
        d_mm, alpha01, status, echo_seq = struct.unpack("<hhBB", frame.data[:6])
        sol = {
            "d_mm": d_mm,
            "alpha_deg": alpha01 / 10.0,
            "status": status,
            "echo_seq": echo_seq,
            "usable": bool(status & 0x80),
        }
        self.solutions.append(sol)

        if echo_seq in self._executed_seqs:
            # 重复请求：只补 done，不重复动作
            self._reply_ack(frame.seq, ACK_DUPLICATE)
            return
        self._executed_seqs.add(echo_seq)
        # 真正「执行完」是在 tick 里到点之后才记进 ``executed``，
        # 这样测试能区分「收到了指令」与「动作已经做完」。
        self._pending_solution = (frame.seq, sol)

    def _reply_ack(self, ack_seq, result):
        with self._lock:
            self._ack_seq = (self._ack_seq + 1) & 0xFF
            self._tx.append(Frame(seq=self._ack_seq, cmd_id=CMD_ACK,
                                  data=bytes((ack_seq & 0xFF, result & 0xFF))))
            self.done_sent.append((ack_seq & 0xFF, result & 0xFF))

    def _apply_pose_bytes(self, data):
        """解码 0x0003 并整体替换估计；长度不对就忽略（真实电控也会这么做）。"""
        if len(data) != 9:
            return
        x, y, h01 = struct.unpack("<hhh", data[:6])
        self.apply_correction(Pose(x, y, h01 / 10.0))

    def poll(self):
        with self._lock:
            return self._tx.pop(0) if self._tx else None

    def tick(self, now):
        """按虚拟时钟行驶与执行动作；进入新区域时发一次去向提示。"""
        if self._pending_solution is not None and self._done_at is None:
            self._done_at = now + self.exec_s
        if self._done_at is not None and now >= self._done_at:
            frame_seq, sol = self._pending_solution
            self._done_at = None
            self._pending_solution = None
            if sol["usable"]:
                self.executed.append(sol)
            if self.use_done:
                self._reply_ack(frame_seq, ACK_OK)

        with self._lock:
            if self._last_now is None:
                self._last_now = now
                if self.route:
                    self._emit(self.route[0])
                    self._leg_index = 1
                    if self.waypoints:
                        self._snap_to_waypoint(0)
                else:
                    self._leg_index = 0
            dt = max(0.0, now - self._last_now)
            self._last_now = now

            remaining = self.speed_mm_s * dt
            if not self.route:
                # 没有路线（``route=()``）：只做漂移台架，一直沿 +x 走，永不到站。
                if remaining > 0.0:
                    self._travel(remaining)
                return
            while remaining > 0.0 and not self._finished:
                leg_len = self._leg_length()
                need = leg_len - self._leg_progress
                if remaining < need:
                    done = self._leg_progress + remaining
                    self._step_leg(done, leg_len, at_end=False)
                    self._leg_progress = done
                    return
                remaining -= need
                self._step_leg(leg_len, leg_len, at_end=True)
                self._leg_progress = 0.0
                if self._leg_index >= len(self.route):
                    self._finished = True
                    return
                self._emit(self.route[self._leg_index])
                self._leg_index += 1
                if self._leg_index >= len(self.route):
                    self._finished = True
                    return
                self._begin_leg()

    # ---------- 臂状态请求 ----------

    def request_arm(self, region=None, target_id=0):
        """发一帧 0x0006，向视觉索取一次臂解算。返回这次请求的 seq。

        真实电控在发这一帧之前要把车停稳，并在等到解算结果之前保持静止；
        这里只模拟「发出请求」这个动作。
        """
        self._arm_seq = self._arm_seq % 255 + 1
        alpha01 = int(round(self.est_h * 10.0)) & 0xFFFF
        data = struct.pack("<HHHb", int(round(self.arm_l_mm)),
                           int(round(self.arm_h_mm)), alpha01,
                           int(target_id) & 0xFF)
        with self._lock:
            self._tx.append(Frame(seq=self._arm_seq, cmd_id=CMD_ARM_STATE,
                                  data=data))
        self.arm_requests.append((data, self._arm_seq))
        return self._arm_seq

    # ---------- 航点支持 ----------

    def _leg_length(self):
        if not self.waypoints:
            return self.leg_mm
        return self._leg_cache

    def _begin_leg(self):
        if not self.waypoints:
            return
        dst = self.waypoints[self._leg_index]
        self._leg_from = (self.true_x, self.true_y)
        self._leg_to = (dst.x, dst.y)
        self._leg_cache = math.hypot(dst.x - self.true_x, dst.y - self.true_y)
        if self._leg_cache < 1e-6:
            self._leg_cache = 1e-6      # 原地转向也算一段，避免死循环

    def _snap_to_waypoint(self, i):
        wp = self.waypoints[i]
        self.true_x, self.true_y, self.true_h = wp.x, wp.y, wp.heading
        self.est_x, self.est_y, self.est_h = wp.x, wp.y, wp.heading
        self._leg_index = i + 1
        self._leg_progress = 0.0
        if self._leg_index >= len(self.route):
            self._finished = True
            return
        self._begin_leg()

    def _step_leg(self, done, leg_len, at_end):
        """把车沿当前段推进到「本段已走 ``done`` 毫米」的位置。

        传的是**本段累计**距离而不是本次增量：用增量的话每拍都从段首重新插值，
        多拍之后车会停在原地附近，看起来像「开不动」。
        """
        frac = done / leg_len if leg_len else 1.0
        prev = (self.true_x, self.true_y)
        if not self.waypoints:
            self.true_x += done - self._leg_progress
        else:
            x0, y0 = self._leg_from
            x1, y1 = self._leg_to
            self.true_x = x0 + (x1 - x0) * frac
            self.true_y = y0 + (y1 - y0) * frac
        self._drift(self.true_x - prev[0], self.true_y - prev[1])

        if at_end and self.waypoints:
            idx = min(self._leg_index, len(self.waypoints) - 1)
            want = self.waypoints[idx].heading
            delta = _norm180(want - self.true_h)
            # 推算朝向要归一化到 (−180, 180]：不归一的话 0°→270° 会记成 −90°，
            # 下一段再转又变成 −360°、−720°，看起来像「航向偏了好几圈」。
            self.true_h = want
            self.est_h = _norm180(self.est_h + delta)

    def _drift(self, dx, dy):
        """按位移向量累积里程误差（``est`` 相对真值的偏离）。"""
        dist = math.hypot(dx, dy)
        bias = self.drift_mm_per_m / 1000.0
        self.est_x += dx * (1.0 + bias)
        self.est_y += dy * (1.0 + bias)
        self.est_h = _norm180(
            self.est_h + self.drift_deg_per_m * (dist / 1000.0))
        self._odometry_m += dist / 1000.0

    def _emit(self, region):
        self.regions_sent.append(region)
        self._tx.append(Frame(seq=0, cmd_id=CMD_REGION,
                              data=bytes([region & 0xFF])))

    def _travel(self, distance_mm):
        self.true_x += distance_mm
        self._drift(distance_mm, 0.0)

    # ---------- 供测试/场景使用的观察口 ----------

    def apply_correction(self, pose):
        """模拟电控收到视觉位姿：**整体替换**自身估计（不是叠加）。"""
        self.est_x, self.est_y, self.est_h = pose.x, pose.y, pose.heading
        self.corrections += 1

    @property
    def position_error_mm(self):
        return math.hypot(self.est_x - self.true_x, self.est_y - self.true_y)

    @property
    def heading_error_deg(self):
        """「电控估计 − 真值」的朝向误差，归一到 (−180, 180]。

        270° 与 −90° 是同一个朝向，直接相减会得到 360°。
        """
        return _norm180(self.est_h - self.true_h)

    def sent_cmd_ids(self):
        return [cid for cid, _ in self.received]


class SerialLink:
    def __init__(self, port, baudrate=115200):
        import serial  # 惰性导入：测试环境不打开端口
        self._ser = serial.Serial(port, baudrate, timeout=0)
        self._rx = StreamDecoder()
        self._tx = []
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
