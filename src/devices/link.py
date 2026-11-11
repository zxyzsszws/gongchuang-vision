"""MockMcu：模拟机器人主控（v2.0.0 语义）。

在 v2.0.0 的分工里，机器人主控**不向视觉上报任何位置信息**，因此这里的模拟器
只做电控真正会做的事：

1. 收到任意帧回 ``0x0007`` 应答（data = 被确认帧的 seq）。
2. 按自己的规划在区域之间行驶，行驶到某区域时发 ``0x0005`` 去向提示。
3. 自身里程推算带累积误差（``drift_mm_per_m``），用于验证「视觉每次看见靶
   就能把误差清零」这一核心性质。
4. ``apply_pose=True`` 时，收到 ``0x0003`` 会像真实电控那样**整体替换**自身估计。

与旧实现的区别：不再产出 ``0x0201`` 位姿上报、不再消费 ``0x0004`` 修正量、
不再产出 ``0x0203`` 动作完成事件——这些命令在 v2.0.0 已删除。
"""
import math
import struct
import threading
from typing import Optional, Protocol

from ..algorithms.types import Pose
from ..core.protocol import (CMD_ACK, CMD_POSE, CMD_REGION, REGION_MATERIAL,
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
    - ``apply_correction(pose)``：模拟「电控收到视觉位姿后整体替换自身估计」。
    """

    def __init__(self, drift_mm_per_m=8.0, drift_deg_per_m=0.1,
                 route=DEFAULT_ROUTE,
                 leg_mm=1200.0, speed_mm_s=1200.0, waypoints=None,
                 apply_pose=False):
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

        self._rx = StreamDecoder()
        self._tx = []
        self._lock = threading.Lock()
        self._ack_seq = 0

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

    # ---------- 链路接口 ----------

    def send(self, frame):
        """视觉发来一帧：回 ACK，并记录内容。

        ``apply_pose=True`` 时，收到 ``0x0003`` 会像真实电控那样**整体替换**自身
        估计（协议附录二第五节）。这样 sim 里能看到「每看见一次靶，里程误差归零」
        这条核心性质，而不只是「帧发出去了」。
        """
        for acked in self._rx.feed(encode(frame)):
            with self._lock:
                self.received.append((acked.cmd_id, bytes(acked.data)))
                self._ack_seq = (self._ack_seq + 1) & 0xFF
                self._tx.append(Frame(seq=self._ack_seq, cmd_id=CMD_ACK,
                                      data=bytes([acked.seq])))
            if self.apply_pose and acked.cmd_id == CMD_POSE:
                self._apply_pose_bytes(bytes(acked.data))

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
        """按虚拟时钟沿路线行驶；每进入一个新区域发一次去向提示。

        行驶是**时间驱动**的（``speed_mm_s`` 决定每段耗时），不是一个调用一次就
        走完一段路段——否则调用频率（即主循环频率）会变成行驶速度。

        ``route`` 是「车依次所处的区域」，第一项即开机所在区域（起点），在第一次
        tick 时发出；此后每走完一段路进入下一项。路线走完后车停住不再移动。

        给了 ``waypoints`` 时，每段路的长度取两个航点之间的实际距离，真值沿直线
        插值到下一个航点；航点里的朝向在到点瞬间一次性生效（真实底盘原地转向的
        简化）。没给航点时降级为沿 +x 走固定 ``leg_mm``。
        """
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
                # 这是单元测试用来单独观察漂移累积的用法。
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
                # 走完这一段：到点 → 发该区域的去向提示 → 准备下一段
                remaining -= need
                self._step_leg(leg_len, leg_len, at_end=True)
                self._leg_progress = 0.0
                if self._leg_index >= len(self.route):
                    self._finished = True       # 路线走完，车停住
                    return
                self._emit(self.route[self._leg_index])
                self._leg_index += 1
                if self._leg_index >= len(self.route):
                    self._finished = True       # 已经到最后一个区域
                    return
                self._begin_leg()

    # ---------- 航点支持 ----------

    def _leg_length(self):
        """当前这一段的长度（mm）。"""
        if not self.waypoints:
            return self.leg_mm
        return self._leg_cache

    def _begin_leg(self):
        """进入新的一段：算出这一段的长度，并记录起点真值。"""
        if not self.waypoints:
            return
        dst = self.waypoints[self._leg_index]
        self._leg_from = (self.true_x, self.true_y)
        self._leg_to = (dst.x, dst.y)
        self._leg_cache = math.hypot(dst.x - self.true_x, dst.y - self.true_y)
        if self._leg_cache < 1e-6:
            self._leg_cache = 1e-6      # 原地转向也算一段，避免死循环

    def _snap_to_waypoint(self, i):
        """开局直接站在第 i 个航点上（起点位姿是已知的）。"""
        wp = self.waypoints[i]
        self.true_x, self.true_y, self.true_h = wp.x, wp.y, wp.heading
        self.est_x, self.est_y, self.est_h = wp.x, wp.y, wp.heading
        self._leg_index = i + 1
        self._leg_progress = 0.0
        if self._leg_index >= len(self.route):
            self._finished = True       # 路线只有一项：原地不动
            return
        self._begin_leg()

    def _step_leg(self, done, leg_len, at_end):
        """把车沿当前段推进到「本段已走 ``done`` 毫米」的位置。

        传的是**本段累计**距离而不是本次增量：用增量的话每拍都从段首重新插值，
        多拍之后车会停在原地附近，看起来像「开不动」。

        到达段末（``at_end``）时，车头朝向切到该航点的标称朝向。推算值也同步
        跟着转（电控知道「我该转多少」），只保留按里程累积的小误差——若让推算
        朝向原地不动，误差会变成几十度，失真到没有参考价值。
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
            # 到点后车头转到该航点的标称朝向。``_leg_index`` 在这里仍指向本段终点；
            # 越界只在「路线只有一项」这种退化配置下出现，夹一下避免抛错。
            idx = min(self._leg_index, len(self.waypoints) - 1)
            want = self.waypoints[idx].heading
            delta = _norm180(want - self.true_h)
            # 推算朝向要归一化到 (−180, 180]：不归一的话 0°→270° 会记成 −90°，
            # 下一段再转又变成 −360°、−720°，看起来像「航向偏了好几圈」。
            self.true_h = want
            self.est_h = _norm180(self.est_h + delta)

    def _drift(self, dx, dy):
        """按位移向量累积里程误差（``est`` 相对真值的偏离）。

        ``est`` 是电控自己推算的位置。它按里程累加，但里程计有比例误差
        （``drift_mm_per_m``）与朝向缓漂（``drift_deg_per_m``），所以推算出的
        位置会沿实际位移方向按比例偏出去——**两个轴都要偏**。只给 x 加误差会让
        电控的 y 估计永远停在 0，任何东西向的航线都测不出真实的横向偏差。

        朝向按走过的路程缓漂；到点转向由 ``_step_leg`` 单独处理。
        """
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
        """沿 +x 前进一段（无航点时的降级路径）：真值精确，推算值按漂移率偏离。"""
        self.true_x += distance_mm
        self._drift(distance_mm, 0.0)

    # ---------- 供测试/场景使用的观察口 ----------

    def apply_correction(self, pose):
        """模拟电控收到视觉位姿：**整体替换**自身估计（不是叠加）。"""
        self.est_x, self.est_y, self.est_h = pose.x, pose.y, pose.heading
        self.corrections += 1

    @property
    def position_error_mm(self):
        """当前「电控估计 − 真值」的位移误差（mm）。"""
        return math.hypot(self.est_x - self.true_x, self.est_y - self.true_y)

    @property
    def heading_error_deg(self):
        """「电控估计 − 真值」的朝向误差，归一到 (−180, 180]。

        270° 与 −90° 是同一个朝向，直接相减会得到 360°，看起来像「偏了一整圈」。
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
