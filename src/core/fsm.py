"""MatchFSM：视觉单元的主状态机（v2.0.0 语义）。

## 与 v1.0.0 的根本区别

v1.0.0 的视觉是「指挥官」：它规划导航、下发放置指令、用 ACK/DONE 推进流程，
共十一个状态。v2.0.0 的视觉是**测量仪器**，它不推进任何流程、不判断任何动作
是否完成，因此状态机收缩为两个工作状态：

    SCANNING  —— 读二维码，自检任务计划，发一次 0x0002
    SERVICE   —— 持续测量：能看见参考靶就发位姿，能看见转盘就发转盘模型

流程推进（取料、放料、码垛、回起点）全部由机器人主控负责，视觉不参与。

## SERVICE 状态下每拍做的事

1. 按心跳周期发 ``0x0001``（无论有无测量结果）。
2. 处理来帧：``0x0005`` 切区域、``0x0007`` 清待确认槽，其余一律静默丢弃。
3. 每 50ms：若当前区域三个参考靶都在，解位姿并发 ``0x0003``。
4. 每 50ms：若转盘可见，发 ``0x0004``。

## 为什么「三个靶不全就不发」而不是发一个降级结果

只看见 1 个或 2 个靶时朝向不可观（见 ``algorithms/localize.py`` 的推导）；发出去
的半成品位姿会让电控把车开歪，比不发更糟。宁可让电控继续用自己的里程推算。

## 180° 镜像歧义由谁消解

三个靶共线，观测顺序整体反向会得到「朝向差 180°、残差同样为 0」的另一个解，
几何上无法区分。v2.0.0 里电控不上报位姿，所以先验只能来自视觉自己：

1. 靶心数字识别结果（最可靠）；
2. 上一帧解出的位姿（时间连续，车不会在一帧之间掉头）；
3. 场地图里该区域的**标称入场位姿**（赛前实测，见 ``TargetGroup.entry_pose``）。
"""
import struct

from ..algorithms.localize import DegenerateInput, resolve_order
from ..core.protocol import (CMD_ACK, CMD_HEARTBEAT, CMD_POSE, CMD_REGION,
                             CMD_TASK_PLAN, CMD_TURNTABLE, Frame)
from ..core.taskplan import TaskPlanError, build_task_plan, describe

STATES = ["SCANNING", "SERVICE", "SAFESTOP"]

# 位姿质量字节各位（文档「表 1-8」）
Q_TARGETS_MASK = 0x03       # bit0~1：参与解算的靶数
Q_SOURCE_SHIFT = 2          # bit2~3：编号来源
Q_SCALE_AUTO = 0x10         # bit4：尺度由三点自估
Q_GRADE_SHIFT = 5           # bit5~6：质量等级
Q_USABLE = 0x80             # bit7：本帧可用

# 编号来源（文档表 1-8 的 bit2~3）
SRC_DIGITS = 0       # 读到靶心上的数字
SRC_ENTRY = 1        # 按入场方向约定推定（标称入场位姿或上一帧）
SRC_PLAN = 2         # 与任务计划交叉核对（当前未实现，仅保留位值）
SRC_UNKNOWN = 3      # 无法区分

GRADE_BAD, GRADE_MID, GRADE_GOOD, GRADE_BEST = 0, 1, 2, 3


def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0


def _i16(v):
    """int16 饱和钳位（文档附录二第三节：超量程饱和，不回绕）。

    取整用四舍五入而不是截断：截断会让所有坐标系统性偏小 0.5mm 左右，
    两轴叠加约 0.7mm 的固定偏差，而这是完全可以避免的。
    """
    return max(-32768, min(32767, int(round(v))))


def _angle01(deg):
    """角度归一到 0.1° 的有符号区间（文档附录二第二节）。"""
    return int(((deg * 10.0 + 1800) % 3600) - 1800)


class MatchFSM:
    """外部契约：``state``（当前状态名）与 ``log_events``（事件流）。"""

    def __init__(self, cfg, link, field, perceive):
        self.cfg, self.link, self.field = cfg, link, field
        self.perceive = perceive
        self.state = "SCANNING"
        self.entered_at = 0.0
        self.log_events = []

        self._seq = 0
        self._last_hb = None
        self._last_pose_at = None
        self._last_tt_at = None
        self._last_rx_at = None        # 上次收到任何来帧的时刻（断链检测）
        self._deadline = None          # 待确认帧的重发截止时刻
        self._pending = None           # (seq, frame) 等待 ACK 的那一帧
        self._retries = 0              # 当前待确认帧已重发次数
        self._region = None            # 当前去向提示；None 表示尚未收到
        self._last_pose = None         # 上一帧解出的位姿（消解 180° 歧义用）
        self._plan = None              # TaskPlan
        self._pose_sent = 0
        self._turntable_sent = 0
        self._plan_rejected = 0

    # ---------- 基础设施 ----------

    def _next_seq(self):
        self._seq = self._seq % 255 + 1     # 1..255，0 不使用（文档表 1-4）
        return self._seq

    def _send(self, cmd_id, data=b"", now=0.0, expect_ack=True):
        """发一帧。``expect_ack`` 为真且此前有待确认帧时返回 False（不覆盖）。"""
        if expect_ack and self._pending is not None:
            return False
        frame = Frame(self._next_seq(), cmd_id, data)
        self.link.send(frame)
        if expect_ack:
            self._pending = (frame.seq, frame)
            self._deadline = now + self.cfg["timing"]["ack_timeout_ms"] / 1000.0
        return True

    def _heartbeat(self, now):
        interval = self.cfg["timing"]["heartbeat_ms"] / 1000.0
        if self._last_hb is None or now - self._last_hb >= interval:
            self._last_hb = now
            self.link.send(Frame(self._next_seq(), CMD_HEARTBEAT))
            # 心跳不占用待确认槽：它是周期帧，重发会与下一拍心跳撞车

    def _check_ack(self, now):
        """待确认帧的应答超时处理：重发原帧（seq 不变），超限进安全停止。"""
        if self._pending is None or self._deadline is None:
            return
        if now < self._deadline:
            return
        timeout = self.cfg["timing"]["ack_timeout_ms"] / 1000.0
        limit = self.cfg["timing"]["ack_retry"]
        _seq, frame = self._pending
        if self._retries >= limit:
            self.log_events.append(("ACK_TIMEOUT", frame.cmd_id, now))
            self._pending = self._deadline = None
            self._goto("SAFESTOP", now)
            return
        self.link.send(frame)       # 重发原帧，seq 不变
        self._retries += 1
        self._deadline = now + timeout

    def _check_link(self, now):
        """断链检测（文档表 1-14）：超过 2s 未收到任何来帧则安全停止。"""
        if self._last_rx_at is None:
            return
        limit = self.cfg["timing"]["link_timeout_s"]
        if now - self._last_rx_at > limit:
            self.log_events.append(("LINK_TIMEOUT", round(now - self._last_rx_at, 2),
                                    now))
            self._goto("SAFESTOP", now)

    def _goto(self, state, now):
        self.log_events.append((self.state, "->", state, now))
        self.state = state
        self.entered_at = now

    def _drain(self, now):
        while (f := self.link.poll()) is not None:
            self._last_rx_at = now
            if f.cmd_id == CMD_ACK:
                if self._pending is not None and f.data \
                        and f.data[0] == self._pending[0]:
                    self._pending = None
                    self._retries = 0
            elif f.cmd_id == CMD_REGION:
                if len(f.data) != 1:
                    self.log_events.append(("BAD_REGION_LEN", len(f.data), now))
                    continue
                if f.data[0] != self._region:
                    # 换区域后上一帧的位姿不再是有效的先验（两组靶相距很远），
                    # 必须丢掉，改用新区域的标称入场位姿消解 180° 歧义。
                    self._last_pose = None
                self._region = f.data[0]
                self.log_events.append(("REGION", self._region, now))
            # 其余命令（含已废弃的 0x0201/0x0202/0x0203）静默丢弃

    def _due(self, last, now, key):
        """周期门控：距上次发送是否已到间隔。

        比较时留 1µs 的容差。浮点数上 ``1.2 - 1.15`` 等于
        ``0.04999999999999982``，比 0.05 略小，于是本该在 1.20 发的帧被推迟到
        1.21 甚至 1.30——表现为「周期偶尔翻倍」，而且只与具体时间戳有关，
        极难复现。容差远大于浮点表示误差（1e-16 量级），又远小于 50ms 周期，
        不会影响节流本身。
        """
        interval = self.cfg["timing"][key] / 1000.0
        return last is None or now - last >= interval - 1e-6

    # ---------- 主循环 ----------

    def step(self, now):
        if self.state in ("SAFESTOP",):
            return self.state
        self._heartbeat(now)
        self.link.tick(now)
        self._drain(now)
        self._check_link(now)
        if self.state in ("SAFESTOP",):
            return self.state
        getattr(self, "_s_" + self.state)(now)
        return self.state

    # ---------- SCANNING ----------

    def _s_SCANNING(self, now):
        text = self.perceive["qr"]()
        if text:
            try:
                plan = build_task_plan(text)
            except (TaskPlanError, ValueError) as e:
                # 格式非法或内容不自洽：不发计划，继续扫（文档 §1.6 容错规则）
                self._plan_rejected += 1
                self.log_events.append(("PLAN_REJECTED", str(e), now))
            else:
                if not plan.usable:
                    self._plan_rejected += 1
                    self.log_events.append(("PLAN_UNUSABLE", describe(plan), now))
                elif self._send(CMD_TASK_PLAN, plan.to_bytes(), now):
                    self._plan = plan
                    self.log_events.append(("PLAN_SENT", describe(plan), now))
                    self._goto("SERVICE", now)
                    return
        if now - self.entered_at > self.cfg["timing"]["scan_timeout_s"]:
            self.log_events.append(("SCAN_TIMEOUT", now))
            self._goto("SAFESTOP", now)

    # ---------- SERVICE ----------

    def _s_SERVICE(self, now):
        self._check_ack(now)
        if self.state == "SAFESTOP":
            return
        if self._due(self._last_pose_at, now, "pose_interval_ms"):
            self._send_pose(now)
        if self._due(self._last_tt_at, now, "turntable_interval_ms"):
            self._send_turntable(now)

    def _send_pose(self, now):
        region = self._region
        group = self.field.group_for_region(region) if region is not None else None
        if group is None:
            return                       # 该区域没有参考靶（原料区/起点）
        obs = self.perceive["targets"](region)
        if obs is None or len(obs) != len(group):
            return                       # 三靶不全：不发（朝向不可观）

        theory = [(t.center[0], t.center[1]) for t in group.targets]
        points = [p.center for p in obs]
        digits = tuple(p.label for p in obs) if all(
            p.label is not None for p in obs) else None

        # 共线三点的 180° 镜像歧义：数字最可靠；其次是上一帧（时间连续）；
        # 首帧用赛前实测的标称入场位姿。
        prior = self._last_pose or group.entry_pose

        try:
            cand = resolve_order(points, theory, prior=prior, digits=digits)
        except (DegenerateInput, ValueError) as e:
            self.log_events.append(("POSE_DEGENERATE", str(e), now))
            return

        sol = cand.solution
        # 编号来源只有两种实际情况：读到数字，或按入场方向约定推定（先验来自上一帧
        # 或标称入场位姿——两者都属于「约定」，协议里没有更细的位值）。
        source = SRC_DIGITS if digits is not None else SRC_ENTRY
        if not obs.collinear:
            source = SRC_UNKNOWN

        grade = self._grade(sol.residual_mm, len(points), source)
        quality = ((len(points) & Q_TARGETS_MASK)
                   | ((source & 0x03) << Q_SOURCE_SHIFT)
                   | ((grade & 0x03) << Q_GRADE_SHIFT))
        if grade >= GRADE_MID:
            quality |= Q_USABLE

        data = (struct.pack("<hhh", _i16(sol.pose.x), _i16(sol.pose.y),
                            _angle01(sol.pose.heading))
                + bytes((min(127, int(round(sol.residual_mm * 2.0))), quality,
                         region & 0xFF)))
        self._send(CMD_POSE, data, now, expect_ack=False)
        self._last_pose_at = now
        self._pose_sent += 1
        self._last_pose = sol.pose
        self.log_events.append(("POSE", round(sol.pose.x, 2),
                                round(sol.pose.y, 2),
                                round(sol.pose.heading, 3),
                                round(sol.residual_mm, 3), now))

    def _grade(self, residual_mm, n_targets, source):
        """质量分级：残差为主，靶数与编号来源为辅。

        「来源」只影响同残差下的一档：读到靶心数字最可靠；按入场方向约定推定
        也可用（两个候选解相差 180°，位置精度不受影响，只有朝向依赖这个判断）。
        """
        if n_targets < 3 or source == SRC_UNKNOWN:
            return GRADE_BAD
        digital = source == SRC_DIGITS
        if residual_mm <= 2.0:
            return GRADE_BEST if digital else GRADE_GOOD
        if residual_mm <= 8.0:
            return GRADE_GOOD if digital else GRADE_MID
        if residual_mm <= 20.0:
            return GRADE_MID
        return GRADE_BAD

    def _send_turntable(self, now):
        getter = getattr(self.perceive, "turntable", None)
        if getter is None:
            return
        tt = getter()
        if tt is None:
            return
        colors = [0, 0, 0]
        for i, st in enumerate(tt.stations[:3]):
            colors[i] = st.color & 0x0F
        data = (struct.pack("<hhHhh", _i16(tt.center[0]), _i16(tt.center[1]),
                            max(0, min(65535, int(round(tt.radius)))),
                            _angle01(tt.angle01 / 10.0),
                            _i16(tt.omega01))
                + bytes((len(tt.stations) & 0xFF, tt.status & 0xFF,
                         colors[0], colors[1], colors[2], 0)))
        self._send(CMD_TURNTABLE, data, now, expect_ack=False)
        self._last_tt_at = now
        self._turntable_sent += 1

    # ---------- 终态 ----------

    def _s_SAFESTOP(self, now):
        pass
