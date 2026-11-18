"""MatchFSM：视觉单元的主状态机（V2.1.0 语义）。

## 分工

视觉是**测量仪器**，不推进流程、不判断动作是否完成：

    SCANNING  —— 读二维码，自检任务计划，发一次 0x0002
    SERVICE   —— 持续测量：能看见参考靶就发位姿，能看见转盘就发转盘模型，
                 收到臂状态就回解算结果

取料、放料、码垛、回起点全部由机器人主控负责。

## 为什么没有心跳

周期性帧（位姿 20Hz、转盘 20Hz）本身就是链路活跃的证据，再加一层心跳是重复的。
真正需要「确认对方收到」的只有两处往返：任务计划、臂解算。它们各有应答超时，
超时重发、重试超限就安全停止——链路死掉一定会在这两处之一暴露出来。

反过来说，**不能靠「多久没收到任何来帧」判断断链**：这个协议里电控是事件驱动的，
停着作业时可能十几秒不发一帧，那是正常的。把静默当断链只能得到误报。

## 臂解算的三段往返

    电控 --0x0006 臂状态--> 视觉        电控此时必须静止
    视觉 --0x0008 解算结果--> 电控      这是对 0x0006 的应答，不是独立周期帧
    电控 --0x0007 done--> 视觉          动作执行完之后才发

任一侧没等到回应就重发**原帧（seq 不变）**。视觉重发 0x0008 时 seq 也不变，
电控据此认出「这一帧我执行过了」，只补一个 done，不再动一次——否则车会沿着
同一个 d 走两遍。

## 180° 镜像歧义由谁消解

三个靶共线，观测顺序整体反向会得到「朝向差 180°、残差同样为 0」的另一个解，
几何上无法区分。先验只能来自视觉自己：

1. 靶心数字识别结果（最可靠）；
2. 上一帧解出的位姿（时间连续，车不会在一帧之间掉头）；
3. 场地图里该区域的**标称入场位姿**（赛前实测，见 ``TargetGroup.entry_pose``）。
"""
import struct

from ..algorithms.arm import ArmState, no_target_solution, solve_arm
from ..algorithms.localize import DegenerateInput, resolve_order
from ..core.protocol import (ACK_DUPLICATE, ACK_FAILED, ACK_OK, CMD_ACK,
                             CMD_ARM_SOLUTION, CMD_ARM_STATE, CMD_POSE,
                             CMD_REGION, CMD_TASK_PLAN, CMD_TURNTABLE,
                             EXPECTS_ACK, Frame)
from ..core.taskplan import TaskPlanError, build_task_plan, describe

STATES = ["SCANNING", "SERVICE", "SAFESTOP"]

# 位姿质量字节各位（文档「表 1-9」）
Q_TARGETS_MASK = 0x03       # bit0~1：参与解算的靶数
Q_SOURCE_SHIFT = 2          # bit2~3：编号来源
Q_SCALE_AUTO = 0x10         # bit4：尺度由三点自估
Q_GRADE_SHIFT = 5           # bit5~6：质量等级
Q_USABLE = 0x80             # bit7：本帧可用

# 编号来源（文档表 1-9 的 bit2~3）
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


def _u16(v):
    return max(0, min(65535, int(round(v))))


def _angle01(deg):
    """角度归一到 0.1° 的有符号区间（文档附录二第二节）。"""
    return int(((deg * 10.0 + 1800) % 3600) - 1800)


class _Pending:
    """一个等待应答的帧。"""

    __slots__ = ("frame", "deadline", "retries", "timeout")

    def __init__(self, frame, deadline, timeout):
        self.frame = frame
        self.deadline = deadline
        self.timeout = timeout
        self.retries = 0


class MatchFSM:
    """外部契约：``state``（当前状态名）与 ``log_events``（事件流）。"""

    def __init__(self, cfg, link, field, perceive):
        self.cfg, self.link, self.field = cfg, link, field
        self.perceive = perceive
        self.state = "SCANNING"
        self.entered_at = 0.0
        self.log_events = []

        self._seq = 0
        self._last_pose_at = None
        self._last_tt_at = None
        self._pending = {}              # cmd_id -> _Pending
        self._region = None             # 当前去向提示；None 表示尚未收到
        self._last_pose = None           # 上一帧解出的位姿（消解 180° 歧义用）
        self._plan = None                # TaskPlan
        self._pose_sent = 0
        self._turntable_sent = 0
        self._solution_sent = 0
        self._plan_rejected = 0
        self._arm_seen_seq = None        # 上一次处理过的 0x0006 的 seq
        self._arm_state = None           # 最近一次收到的臂状态

    # ---------- 基础设施 ----------

    def _next_seq(self):
        self._seq = self._seq % 255 + 1     # 1..255，0 不使用（文档表 1-4）
        return self._seq

    def _send(self, cmd_id, data=b"", now=0.0, expect_ack=False,
              timeout_key="ack_timeout_ms"):
        """发一帧。``expect_ack`` 为真时登记待确认槽（同一命令不重复登记）。"""
        if expect_ack and cmd_id in self._pending:
            return False
        frame = Frame(self._next_seq(), cmd_id, data)
        self.link.send(frame)
        if expect_ack:
            timeout = self.cfg["timing"][timeout_key] / 1000.0
            self._pending[cmd_id] = _Pending(frame, now + timeout, timeout)
        return True

    def _resend(self, cmd_id, now):
        """原样重发待确认帧（seq 不变），并顺延截止时刻。"""
        p = self._pending.get(cmd_id)
        if p is None:
            return False
        self.link.send(p.frame)
        p.deadline = now + p.timeout
        p.retries += 1
        return True

    def _check_acks(self, now):
        """应答超时：重发原帧，超过重试上限则安全停止（文档表 1-15）。

        ``CMD_ARM_SOLUTION`` 等的是 done，而 done 在动作**执行完**才发，所以它
        用自己的超时（``exec_timeout_ms``），比纯收帧的应答超时长得多。
        """
        limit = int(self.cfg["timing"]["ack_retry"])
        for cmd_id in list(self._pending):
            p = self._pending[cmd_id]
            if now < p.deadline:
                continue
            if p.retries >= limit:
                self.log_events.append(("ACK_TIMEOUT", cmd_id, now))
                del self._pending[cmd_id]
                self._goto("SAFESTOP", now)
                return
            self.link.send(p.frame)
            p.retries += 1
            p.deadline = now + p.timeout
        # 循环里可能已进 SAFESTOP

    def _goto(self, state, now):
        self.log_events.append((self.state, "->", state, now))
        self.state = state
        self.entered_at = now

    def _drain(self, now):
        while (f := self.link.poll()) is not None:
            if f.cmd_id == CMD_ACK:
                self._on_ack(f, now)
            elif f.cmd_id == CMD_REGION:
                self._on_region(f, now)
            elif f.cmd_id == CMD_ARM_STATE:
                self._on_arm_state(f, now)
            # 其余命令（含已废弃的 0x0001/0x0201/0x0202/0x0203）静默丢弃

    def _on_region(self, f, now):
        if len(f.data) != 1:
            self.log_events.append(("BAD_REGION_LEN", len(f.data), now))
            return
        if f.data[0] != self._region:
            # 换区域后上一帧的位姿不再是有效的先验（两组靶相距很远），
            # 必须丢掉，改用新区域的标称入场位姿消解 180° 歧义。
            self._last_pose = None
        self._region = f.data[0]
        self.log_events.append(("REGION", self._region, now))

    def _on_ack(self, f, now):
        """0x0007：ack_seq + result 两字节。"""
        if len(f.data) != 2:
            self.log_events.append(("BAD_ACK_LEN", len(f.data), now))
            return
        ack_seq, result = f.data
        for cmd_id in list(self._pending):
            p = self._pending[cmd_id]
            if p.frame.seq != ack_seq:
                continue
            del self._pending[cmd_id]
            if result == ACK_OK or result == ACK_DUPLICATE:
                self.log_events.append(("ACK", cmd_id, result, now))
            elif result == ACK_FAILED:
                self.log_events.append(("NACK", cmd_id, now))
                # 任务计划被拒说明双方对不上，继续扫也扫不出别的码，直接停。
                if cmd_id == CMD_TASK_PLAN:
                    self._goto("SAFESTOP", now)
            else:
                self.log_events.append(("ACK_UNKNOWN_RESULT", result, now))
            return

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
        if self.state == "SAFESTOP":
            return self.state
        self.link.tick(now)
        self._drain(now)
        if self.state == "SAFESTOP":
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
                elif self._send(CMD_TASK_PLAN, plan.to_bytes(), now,
                                expect_ack=True):
                    self._plan = plan
                    self.log_events.append(("PLAN_SENT", describe(plan), now))
                    self._goto("SERVICE", now)
                    return
        if now - self.entered_at > self.cfg["timing"]["scan_timeout_s"]:
            self.log_events.append(("SCAN_TIMEOUT", now))
            self._goto("SAFESTOP", now)

    # ---------- SERVICE ----------

    def _s_SERVICE(self, now):
        self._check_acks(now)
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

    # ---------- 臂解算 ----------

    def _arm_target_getter(self):
        """取 perceive 里的臂目标测量入口（sim 与 live 都按这个键提供）。

        ``perceive`` 是 dict 或带 ``__getitem__`` 的对象，两种都要支持：
        ``SimScene`` 用 ``__getitem__``，单元测试里常用裸 dict。
        """
        try:
            return self.perceive["arm_target"]
        except (KeyError, TypeError, IndexError):
            return None

    def _on_arm_state(self, f, now):
        """收到 0x0006：解算并回 0x0008。

        电控重发同一个 seq 说明它没收到上一帧解算结果（或者等 done 超时了），
        此时**原样重发**上一帧 0x0008 而不是重新解算：电控必须静止，重算的结果
        应当相同，但重发原帧能保证逐字节一致，电控的重复识别也更简单。
        """
        if len(f.data) != 5:
            self.log_events.append(("BAD_ARM_LEN", len(f.data), now))
            return

        if self._arm_seen_seq == f.seq and CMD_ARM_SOLUTION in self._pending:
            self._resend(CMD_ARM_SOLUTION, now)
            self.log_events.append(("ARM_RESEND", f.seq, now))
            return

        l_mm, alpha01, target_id = struct.unpack("<HHb", f.data)
        state = ArmState(l_mm=float(l_mm), alpha_deg=alpha01 / 10.0, seq=f.seq)
        self._arm_state = state
        self._arm_seen_seq = f.seq

        getter = self._arm_target_getter()
        measured = getter(target_id) if getter is not None else None

        if measured is None:
            sol = no_target_solution(echo_seq=f.seq)
            self.log_events.append(("ARM_NO_TARGET", f.seq, now))
        else:
            sol = solve_arm(measured, state.l_mm, echo_seq=f.seq,
                            grade=GRADE_GOOD)
            self.log_events.append(("ARM_SOLVED", round(sol.d_mm, 2),
                                    round(sol.alpha_deg, 3), f.seq, now))

        data = struct.pack("<hhBBhh", _i16(sol.d_mm),
                           _angle01(sol.alpha_deg) & 0xFFFF, sol.status,
                           f.seq & 0xFF, _i16(sol.target[0]),
                           _i16(sol.target[1]))
        # 电控可能正忙（上一帧解算还在等 done）。此时不覆盖，让它先把上一轮走完；
        # 电控等不到 done 会重发，届时重新解算即可。
        if self._send(CMD_ARM_SOLUTION, data, now, expect_ack=True,
                      timeout_key="exec_timeout_ms"):
            self._solution_sent += 1

    # ---------- 终态 ----------

    def _s_SAFESTOP(self, now):
        pass
