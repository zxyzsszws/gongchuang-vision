"""MatchFSM：比赛主状态机（spec 2.6 / 4）。

IDLE→SCANNING→PLAN_READY→NAV1→CAL1→PLACE1→NAV2→CAL2→PLACE2→RETURN→DONE，
任意超时/断链收敛到 SAFESTOP：
- 扫码超时（scan_timeout_s）不可降级 → SAFESTOP。
- 校准连续 retry 次失败 → CAL_SKIPPED 降级完赛（保完赛原则）。
- HEARTBEAT 每 heartbeat_ms 发送；超过 heartbeat_timeout_s 无来流 →
  LINK_TIMEOUT → SAFESTOP（spec §3/§4）。
- 控制帧 100ms 未 ACK 重发，ack_retry 次失败 → SAFESTOP。
NAV/PLACE/RETURN 均为“进入即发指令、主控 CMD_DONE 推进”的同一模式
（RETURN 也发 GOTO，否则无外发触发会卡死——控制端裁决 1）。
单线程驱动：step(now) 由外层循环按仿真/真实时间调用，link 为 McuLink 协议对象。
"""
import math
import struct

from ..algorithms.calibration import calibrate
from ..algorithms.qr_parse import qr_parse
from ..algorithms.types import Pose, QrFormatError
from ..core.protocol import (CMD_ACK, CMD_CALIBRATE, CMD_DONE, CMD_GOTO,
                             CMD_HEARTBEAT, CMD_PICK, CMD_PLACE, CMD_POSE,
                             CMD_STATE, CMD_TASK_PLAN, Frame)

STATES = ["IDLE", "SCANNING", "PLAN_READY", "NAV1", "CAL1", "PLACE1",
          "NAV2", "CAL2", "PLACE2", "RETURN", "DONE", "SAFESTOP"]


def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0


def _i16(v):
    """int16 饱和钳位：链路协议字段为有符号 16 位，宁可丢精度不抛异常。"""
    return max(-32768, min(32767, int(v)))


class MatchFSM:
    def __init__(self, cfg, link, pose_filter, field, perceive):
        self.cfg, self.link, self.pf, self.field = cfg, link, pose_filter, field
        self.perceive = perceive
        self.state, self.entered_at, self.log_events = "IDLE", 0.0, []
        self._stable = 0          # CAL 防抖：连续稳定检出帧数
        self._retries = 0         # 当前状态内重试计数（跨 _goto 清零）
        self._plan = None
        self._last = None         # 最近一次滤波后 Pose（CMD_POSE 驱动）
        self._ack_pending = None  # [cmd, seq, deadline, attempts, frame]
        self._action_done = False # 收到 CMD_DONE 的闩锁
        self._sent = False        # 当前状态是否已外发动作指令
        self._last_in = None      # 最近一次收到来流的仿真时刻
        self._last_hb = None      # 最近一次发心跳的时刻
        self._seq = 0

    # ---------- 基础设施 ----------

    def _goto(self, state, now):
        self.log_events.append((self.state, "->", state, now))
        self.state, self.entered_at = state, now
        self._stable = self._retries = 0
        self._sent = False
        self._action_done = False  # 防上一动作的迟到 DONE 泄漏到下一状态

    def _next_seq(self):
        self._seq = self._seq % 255 + 1  # 1..255，0 保留给链路自增帧（POSE 等）
        return self._seq

    def _send(self, cmd, payload, now):
        frame = Frame(self._next_seq(), cmd, payload)
        self.link.send(frame)
        self._ack_pending = [cmd, frame.seq, now, 1, frame]

    def _drain_link(self, now):
        while (f := self.link.poll()) is not None:
            self._last_in = now  # 任何来流都刷新心跳超时基准
            if f.cmd == CMD_POSE:
                x, y, h01, _ = struct.unpack("<hhhH", f.payload)
                # POSE 的 seq 恒为 0（Task 10 约定），不得按 seq 去重丢帧
                self._last = self.pf.update(Pose(x, y, h01 / 10.0))
            elif f.cmd == CMD_ACK:
                if (self._ack_pending and f.payload
                        and f.payload[0] == self._ack_pending[1]):
                    self._ack_pending = None
            elif f.cmd == CMD_DONE:
                self._action_done = True
            # CMD_STATE 预留：当前推进只依赖 CMD_DONE

    def _heartbeat(self, now):
        interval = self.cfg["timing"]["heartbeat_ms"] / 1000.0
        if self._last_hb is None or now - self._last_hb >= interval:
            self._last_hb = now
            self.link.send(Frame(self._next_seq(), CMD_HEARTBEAT))
            # 心跳是探活帧，不占用 _ack_pending（MockMcu 对任何帧回 ACK）

    def _check_ack(self, now):
        p = self._ack_pending
        if p is None:
            return
        timeout = self.cfg["timing"]["ack_timeout_ms"] / 1000.0
        limit = self.cfg["timing"]["ack_retry"]
        if now - p[2] < timeout:
            return
        if p[3] >= limit:
            self._ack_pending = None
            self.log_events.append(("ACK_TIMEOUT", p[0], now))
            self._goto("SAFESTOP", now)
        else:
            self.link.send(p[4])
            p[2], p[3] = now, p[3] + 1

    # ---------- 主循环 ----------

    def step(self, now):
        if self._last_in is None:
            self._last_in = now  # 开赛基准：给链路第一帧的宽限
        active = self.state not in ("DONE", "SAFESTOP")
        if active:
            self._heartbeat(now)
        self.link.tick(now)
        self._drain_link(now)
        if not active:
            return self.state
        self._check_ack(now)
        if self.state == "SAFESTOP":
            return self.state
        if now - self._last_in > self.cfg["timing"]["heartbeat_timeout_s"]:
            self.log_events.append(("LINK_TIMEOUT", now))
            self._goto("SAFESTOP", now)
            return self.state
        getattr(self, "_s_" + self.state)(now)
        return self.state

    # ---------- 状态处理器 ----------

    def _s_IDLE(self, now):
        self._goto("SCANNING", now)

    def _s_SCANNING(self, now):
        text = self.perceive["qr"]()
        if text:
            try:
                self._plan = qr_parse(text)
            except QrFormatError:
                self._plan = None  # 格式非法：继续扫描，超时兜底 SAFESTOP
            else:
                # TASK_PLAN：batch1[3] batch2[3] 颜色 + 位置，开赛只发一次
                self._send(CMD_TASK_PLAN, bytes(self._plan.batch1_colors
                                                + self._plan.batch1_positions
                                                + self._plan.batch2_colors
                                                + self._plan.batch2_positions), now)
                self.log_events.append(("PLAN_PARSED", self._plan, now))
                self._goto("PLAN_READY", now)
                return
        if now - self.entered_at > self.cfg["timing"]["scan_timeout_s"]:
            self.log_events.append(("SCAN_TIMEOUT", now))
            self._goto("SAFESTOP", now)

    def _s_PLAN_READY(self, now):
        self._goto("NAV1", now)

    def _action(self, cmd, payload, nxt, now):
        """NAV/PLACE/RETURN 通用模式：进入即发，CMD_DONE 推进，超时重发。"""
        if self._action_done:
            self._goto(nxt, now)
            return
        if not self._sent:
            self._send(cmd, payload, now)
            self._sent = True
            return
        if now - self.entered_at > self.cfg["timing"].get("nav_timeout_s", 8.0):
            self._retries += 1
            if self._retries > self.cfg["calib"]["retry"]:
                self.log_events.append(("ACTION_TIMEOUT", cmd, now))
                self._goto("SAFESTOP", now)
            else:
                # 留在原地保留 _retries（裁决 3：重试上限 → SAFESTOP），
                # 下一拍 _sent=False 触发重发
                self._sent = False

    def _s_NAV1(self, now):
        # 目标点推导属场地任务层（Task 12 装配时替换）；协议占位 (0,0,0)
        self._action(CMD_GOTO, struct.pack("<hhh", 0, 0, 0), "CAL1", now)

    def _s_NAV2(self, now):
        self._action(CMD_GOTO, struct.pack("<hhh", 0, 0, 0), "CAL2", now)

    def _s_RETURN(self, now):
        # 裁决 1：RETURN 必须外发 GOTO（回起点），以 DONE 推进，否则永不会 DONE
        self._action(CMD_GOTO, struct.pack("<hhh", 0, 0, 0), "DONE", now)

    def _cal(self, nxt, now):
        obs = self.perceive["tri"]()
        self._stable = self._stable + 1 if obs else 0
        if self._stable >= self.cfg["calib"]["stable_frames"] and self._last:
            fixed = calibrate(self._last, obs, self.field,
                              self.cfg["calib"]["weight_middle"],
                              self.cfg["calib"]["weight_pair_mid"],
                              self.cfg["calib"]["max_residual_mm"],
                              self.cfg["calib"]["max_residual_deg"])
            if fixed:
                dx = fixed.x - self._last.x
                dy = fixed.y - self._last.y
                dh01 = _norm180(fixed.heading - self._last.heading) * 10.0
                self._send(CMD_CALIBRATE,
                           struct.pack("<hhh", _i16(dx), _i16(dy), _i16(dh01)), now)
                self.log_events.append(("CAL_APPLIED", dx, dy, now))
                self._goto(nxt, now)
                return
            self._retries += 1  # 残差超限判误匹配，计一次失败
        if now - self.entered_at > self.cfg["timing"].get("cal_timeout_s", 6.0):
            self._retries += 1
            self.entered_at = now  # 重开窗口，每 6s 计一次
        if self._retries > self.cfg["calib"]["retry"]:
            self.log_events.append(("CAL_SKIPPED", now))  # 降级：保完赛
            self._goto(nxt, now)

    def _s_CAL1(self, now):
        self._cal("PLACE1", now)

    def _s_CAL2(self, now):
        self._cal("PLACE2", now)

    def _s_PLACE1(self, now):
        color = self._plan.batch1_colors[0] if self._plan else 1
        self._action(CMD_PLACE, struct.pack("<bhh", color, 0, 0), "NAV2", now)

    def _s_PLACE2(self, now):
        color = self._plan.batch2_colors[0] if self._plan else 1
        self._action(CMD_PLACE, struct.pack("<bhh", color, 0, 0), "RETURN", now)

    def _s_DONE(self, now):
        pass

    def _s_SAFESTOP(self, now):
        pass
