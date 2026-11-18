"""MatchFSM：视觉单元的主状态机（V2.2.0 语义）。

## 分工

视觉是**按需应答的测量仪器**：

    电控 --0x0010 请求--> 视觉
    视觉 --0x0011 结果--> 电控

没有心跳、没有周期上报、没有 done。视觉不发任何电控没要过的帧，也从不主动
推进流程。唯一的状态是 SERVICE（随时待命）；SAFESTOP 只在双方实现明显对不上
时进入——那种情况继续答下去只会产生垃圾数据。

## 三条任务

| 任务码 | 要什么 | 视觉怎么做 |
|---|---|---|
| 1 | 任务计划 | 读二维码，自检后回 16 字节 |
| 2 | 转盘取料点的颜色 | 连续取帧，判定停稳后回颜色 |
| 3 | 对准主圆 | 量一帧主圆，回 d 与 α |

任务 2 与 3 的差别在于**要不要等**：对准是「电控已停稳，量一帧就行」，
转盘取料是「电控已经把臂悬停住，等转盘停到取料点」——所以后者要在多帧之间
维持判定状态，直到停稳或者超时。

## 重发与去重

电控等不到结果就重发**原请求（seq 不变）**。视觉见到同一个 seq 就把上次的
结果**原样再发一遍**（数据段逐字节相同，换了新的帧 seq）。电控按 ``echo_seq``
去重，因此同一个 seq 只会被执行一次。

为什么现在不需要「视觉重发」：视觉是应答方，它一发完就没事了；要不要重发由
电控决定。这样整条链路里只有**一个**重试者，不会出现两边同时重发、互相看错。
"""
import math
import struct

from ..algorithms.arm import (GRADE_BAD, GRADE_GOOD, GRADE_MID, AlignSolution,
                              mm_per_px_from_ring, no_target_solution,
                              scale_offset, solve_align)
from ..algorithms.turntable import make_strategy
from ..core.protocol import (CMD_REQUEST, CMD_RESULT, REQ_LEN, RES_LEN,
                             ST_BAD_REQUEST, ST_NOT_READY, ST_OK, TASK_ALIGN,
                             TASK_SCAN, TASK_TURNTABLE, TASKS, Frame)
from ..core.taskplan import TaskPlanError, build_task_plan, describe

STATES = ["SERVICE", "SAFESTOP"]

# 默认取料点在画面光学中心；配置里可用 turntable.pick_point_mm 覆盖。
_DEFAULT_PICK_POINT = (0.0, 0.0)
_DEFAULT_MATERIAL_RADIUS_MM = 25.0
# 主圆外圈直径的缺省值（mm）。必须与赛场上量到的实物一致，否则尺度整体偏。
_DEFAULT_RING_OUTER_MM = 120.0


class MatchFSM:
    """外部契约：``state``（当前状态名）与 ``log_events``（事件流）。"""

    def __init__(self, cfg, link, perceive):
        self.cfg, self.link = cfg, link
        self.perceive = perceive
        self.state = "SERVICE"
        self.entered_at = 0.0
        self.log_events = []

        self._seq = 0
        self._bad_requests = 0
        # 正在处理、还没到条件的请求：只允许一个（电控一次只等一件事）
        self._job = None
        # 当前在处理的请求的 (seq, 载荷)，用于给结果构造去重键
        self._cur_req = None
        # 上一次已经答完的请求：同 seq 且内容相同再来就原样重发
        self._done_key = None
        self._done_data = None

        # 计数（日志与测试用）
        self.counts = {TASK_SCAN: 0, TASK_TURNTABLE: 0, TASK_ALIGN: 0}
        self.bad_requests = 0
        self.resends = 0

        # 转盘停稳判定器：按配置造，延迟到第一次收到任务 2 时再建，
        # 这样「取料点」可以用第一次取帧时的画面尺寸来定。
        self._settle = None

    # ---------- 基础设施 ----------

    def _next_seq(self):
        self._seq = self._seq % 255 + 1     # 1..255，0 不使用（文档表 1-4）
        return self._seq

    def _send_result(self, task, status, payload, echo_seq, now, req_payload=None,
                     remember=True):
        """发一帧结果，并记下「这条请求的最终答案」，供同请求重发时原样再发。

        结果正文按任务码**补齐到固定长度**（``RES_LEN``）：失败时没有内容就填 0。
        固定长度对电控很重要——它按任务码一次读满 N 字节，不必再处理「这次正文
        短了」这种分支，也不会因为少读一个字节而把下一帧的帧头当数据。

        去重键必须是**请求**的三元组（seq、任务码、载荷），不能拿答案的载荷来比
        ——那是两回事。``req_payload`` 省略时取 ``self._cur_req``（当前在处理的
        那条请求），因为「一次只等一件事」，任何时刻至多一条。

        ``remember=False`` 用于不合法请求：那种帧不该覆盖上一条好请求的缓存答案，
        否则一条残帧就能让「原样重发上次结果」失效，电控白等一轮。
        """
        if req_payload is None:
            req_payload = self._cur_req[1] if self._cur_req else b""
        want = RES_LEN.get(task)
        body = bytes(payload)
        if want is not None:
            body = body[:want - 3].ljust(want - 3, b"\x00")
        data = bytes((echo_seq & 0xFF, task & 0xFF, status & 0xFF)) + body
        self.link.send(Frame(self._next_seq(), CMD_RESULT, data))
        if remember:
            self._done_key = (echo_seq, task, bytes(req_payload))
            self._done_data = data
        self.log_events.append(("RESULT", task, status, echo_seq, now))
        return data

    def _resend_last(self, now):
        self.link.send(Frame(self._next_seq(), CMD_RESULT, self._done_data))
        self.resends += 1
        self.log_events.append(("RESULT_RESEND", self._done_key[0], now))

    def _goto(self, state, now):
        self.log_events.append((self.state, "->", state, now))
        self.state = state
        self.entered_at = now

    def _due(self, last, now, key):
        """周期门控：距上次动作是否已到间隔（留 1µs 浮点容差）。"""
        interval = self._timeout_s(key, 0.02)
        return last is None or now - last >= interval - 1e-6

    def _timeout_s(self, key, default_s):
        """读一个以秒为单位的超时。

        配置里超时有两种单位并存：``frame_interval_ms`` 是毫秒，而
        ``scan_timeout_s`` / ``turntable_timeout_s`` 本身就是秒。按后缀区分，
        再按各自单位换算——否则 3s 的扫码超时会被当成 3ms，请求立刻失败。
        """
        raw = self.cfg["timing"].get(key, default_s)
        if key.endswith("_ms"):
            return float(raw) / 1000.0
        return float(raw)

    # ---------- 主循环 ----------

    def step(self, now):
        if self.state == "SAFESTOP":
            return self.state
        self.link.tick(now)
        self._drain(now)
        if self.state == "SAFESTOP":
            return self.state
        self._advance_job(now)
        return self.state

    def _drain(self, now):
        while (f := self.link.poll()) is not None:
            if f.cmd_id == CMD_REQUEST:
                self._on_request(f, now)
            # 其余命令（含 V2.1.0 及更早的命令码）静默丢弃，不得视为错误

    # ---------- 请求分发 ----------

    def _on_request(self, f, now):
        """收到 0x0010：校验 → 去重 → 起一个新作业。"""
        if not f.data:
            return self._bad(f, "空请求", now)

        task = f.data[0]
        if task not in TASKS:
            return self._bad(f, f"未知任务码 {task}", now)

        want = REQ_LEN[task]
        if len(f.data) != want:
            return self._bad(f, f"任务 {task} 的载荷应为 {want} 字节，"
                                f"收到 {len(f.data)}", now)

        # 同一个 seq、同样的内容再来一次。
        # ① 上一次已经答完了：原样重发，不重新测；
        # ② 还在处理中（正在等转盘停、正在扫码）：**不要重新计时**。
        #    这一条很关键：电控的重发间隔可能比视觉的处理时间短，若每次都重置
        #    计时，转盘永远等不到超时，扫码也永远等不到超时——表现为「永远不回」。
        key = (f.seq, task, bytes(f.data[1:]))
        if key == self._done_key and self._done_data is not None:
            self._resend_last(now)
            return
        job = self._job
        if job is not None and (job.msg.seq, job.msg.task,
                                bytes(job.msg.payload)) == key:
            self.log_events.append(("DUPLICATE_WHILE_BUSY", task, f.seq, now))
            return

        # 新请求：丢掉还没完成的旧作业（电控已经不等它了）
        self._job = None
        self._bad_requests = 0        # 只统计「连续」不合法
        self._cur_req = (f.seq, bytes(f.data[1:]))
        self._job_started(Message(task, f.seq, f.data[1:]), now)

    def _bad(self, f, why, now):
        self._bad_requests += 1
        self.bad_requests += 1
        self.log_events.append(("BAD_REQUEST", why, now))
        task = f.data[0] if f.data else 0
        # 不合法请求**不进缓存**：它不该覆盖上一条好请求的答案（否则一条残帧
        # 就能让电控的「原样重发」失效）。同一帧坏请求再来一次就再回一次错误，
        # 反正它会推高不合法计数、最终进 SAFESTOP。
        self._send_result(task, ST_BAD_REQUEST, b"", f.seq, now,
                          req_payload=bytes(f.data[1:]), remember=False)
        limit = int(self.cfg["timing"].get("max_bad_requests", 5))
        if self._bad_requests >= limit:
            # 连续不合法说明两边的实现已经对不上，继续答只会产生垃圾数据。
            self.log_events.append(("PROTOCOL_MISMATCH", self._bad_requests, now))
            self._goto("SAFESTOP", now)

    # ---------- 作业：一帧能答完的直接答，要等的挂起来 ----------

    def _job_started(self, msg, now):
        task = msg.task
        if task == TASK_ALIGN:
            return self._do_align(msg, now)
        if task == TASK_SCAN:
            return self._start_scan(msg, now)
        if task == TASK_TURNTABLE:
            return self._start_turntable(msg, now)

    def _advance_job(self, now):
        """推进需要等待的作业（扫码重试、转盘停稳）。"""
        job = self._job
        if job is None:
            return
        due = self._due(job.last_poll, now, "frame_interval_ms")
        if not due:
            return
        job.last_poll = now

        if job.msg.task == TASK_SCAN:
            return self._poll_scan(job, now)
        if job.msg.task == TASK_TURNTABLE:
            return self._poll_turntable(job, now)

    # ---------- 任务 1：扫码 ----------

    def _start_scan(self, msg, now):
        self._job = _Job(msg, now, self._timeout_s("scan_timeout_s", 15.0))
        self._job.last_poll = now
        self._poll_scan(self._job, now)

    def _poll_scan(self, job, now):
        reader = self._perceive("qr")
        text = reader() if callable(reader) else None
        if text:
            try:
                plan = build_task_plan(text)
            except (TaskPlanError, ValueError) as e:
                # 格式非法或内容不自洽：回「没准备好」，让电控换个时机再问。
                # 绝不能回一份错的计划——那会让车开到错的靶位，而且不报错。
                self.log_events.append(("PLAN_REJECTED", str(e), now))
                self.counts[TASK_SCAN] += 1
                self._job = None
                return self._send_result(TASK_SCAN, ST_NOT_READY, b"",
                                         job.msg.seq, now)
            if not plan.usable:
                self.log_events.append(("PLAN_UNUSABLE", describe(plan), now))
                self.counts[TASK_SCAN] += 1
                self._job = None
                return self._send_result(TASK_SCAN, ST_NOT_READY, b"",
                                         job.msg.seq, now)

            self.counts[TASK_SCAN] += 1
            self._job = None
            self.log_events.append(("PLAN", describe(plan), now))
            return self._send_result(TASK_SCAN, ST_OK, plan.to_bytes(),
                                     job.msg.seq, now)

        if now - job.started >= job.timeout_s:
            self.log_events.append(("SCAN_TIMEOUT", now))
            self.counts[TASK_SCAN] += 1
            self._job = None
            return self._send_result(TASK_SCAN, ST_NOT_READY, b"",
                                     job.msg.seq, now)

    # ---------- 任务 2：转盘取料点 ----------

    def _start_turntable(self, msg, now):
        if self._settle is None:
            self._settle = self._make_settle()
        self._settle.reset()
        self._job = _Job(msg, now, self._timeout_s("turntable_timeout_s", 3.0))
        self._job.last_poll = now
        self._poll_turntable(self._job, now)

    def _make_settle(self):
        """按配置造停稳判定器。取料点默认在光学中心，物料半径从配置读。"""
        cfg = self.cfg.get("turntable") or {}
        pick = tuple(cfg.get("pick_point_mm", _DEFAULT_PICK_POINT))
        radius = float(cfg.get("material_radius_mm",
                               _DEFAULT_MATERIAL_RADIUS_MM))
        return make_strategy(cfg, pick, radius)

    def _poll_turntable(self, job, now):
        reader = self._perceive("turntable")
        got = reader() if callable(reader) else None
        blobs = got[0] if isinstance(got, tuple) else got
        if blobs is not None:
            from ..algorithms.types import TurntableFrame
            color = self._settle.feed(TurntableFrame(t=now, blobs=blobs))
            if color:
                self.counts[TASK_TURNTABLE] += 1
                self._job = None
                self.log_events.append(("PICK_COLOR", color, now))
                return self._send_result(TASK_TURNTABLE, ST_OK,
                                         bytes((color,)), job.msg.seq, now)

        if now - job.started >= job.timeout_s:
            self.log_events.append(("TURNTABLE_TIMEOUT", now))
            self.counts[TASK_TURNTABLE] += 1
            self._job = None
            return self._send_result(TASK_TURNTABLE, ST_NOT_READY, b"",
                                     job.msg.seq, now)

    # ---------- 任务 3：对准主圆 ----------

    def _do_align(self, msg, now):
        """量一帧主圆，立刻回 d 与 α。电控此时已静止，等结果就行。

        注意「相机在哪」：相机装在臂末端，所以画面量到的偏移是**相对相机**的，
        要加上相机自身在电机坐标系里的位置才能得到主圆的绝对位置：

            相机位置 = (l·cos α, l·sin α)
            主圆位置 = 相机位置 + R(α)·(测得偏移)

        漏掉相机位置那一项是这一层最容易犯的错：结果会稳定地偏 −l 那么远，
        看起来「有输出」，实际上让车永远差一整条臂长。
        """
        l_mm, alpha01 = struct.unpack("<HH", msg.payload[:4])
        state_l = float(l_mm)
        # α 用补码传：0.1° 的有符号值（文档附录二第二节）
        state_a = math.radians((alpha01 - 65536 if alpha01 >= 32768 else alpha01) / 10.0)

        obs = self._perceive("ring")
        obs = obs() if callable(obs) else None
        if obs is None:
            self.counts[TASK_ALIGN] += 1
            self.log_events.append(("ALIGN_NO_RING", msg.seq, now))
            return self._reply_align(no_target_solution(), msg.seq, now)

        mm_per_px, grade = self._align_scale(obs.radius_px)
        if mm_per_px is None:
            # 外圈量不到、也没有外参：没有任何依据能把像素换成毫米。
            # 拿假设值硬算给出的位置大小全错，电控照着挪只会越挪越偏。
            self.counts[TASK_ALIGN] += 1
            self.log_events.append(("ALIGN_NO_SCALE", msg.seq, now))
            return self._reply_align(no_target_solution(), msg.seq, now)

        # 主圆相对光学中心的像素偏移 → 毫米。y 轴取反：图像 y 朝下，电机系 y 朝外。
        size = self._frame_size()
        cx, cy = (size[0] / 2.0, size[1] / 2.0) if size else (0.0, 0.0)
        off_px = (obs.center[0] - cx, -(obs.center[1] - cy))
        dx, dy = scale_offset(off_px, mm_per_px)

        # 相机随臂转过 α，所以相机系里的偏移也要转同样的角度才对得上电机系。
        ca, sa = math.cos(state_a), math.sin(state_a)
        target = (state_l * ca + dx * ca - dy * sa,
                  state_l * sa + dx * sa + dy * ca)

        sol = solve_align(target, state_l, grade=grade)
        self.counts[TASK_ALIGN] += 1
        self.log_events.append(
            ("ALIGN", round(sol.d_mm, 2), round(sol.alpha_deg, 3),
             sol.status, grade, round(mm_per_px, 4), msg.seq, now))
        return self._reply_align(sol, msg.seq, now)

    def _align_scale(self, radius_px):
        """尺度与质量档：优先用主圆外圈实测；退回外参标定值；都没有则返回 None。

        三种情况的区别：
        - 外圈量到了 → 实测尺度，质量档 ``GRADE_GOOD``；
        - 外圈量不到（被裁掉、半径不足 2px）但有外参 → 用标定尺度，降为 ``GRADE_MID``
          （尺度是估的，不假装精确）；
        - 两者都没有 → 返回 ``None``。此时**没有任何依据**把像素换成毫米，调用方
          必须回 ``ST_NOT_READY``，绝不能拿 1.0mm/px 这种假设值充数。
        """
        outer_mm = float((self.cfg.get("target") or {}).get(
            "ring_outer_mm", _DEFAULT_RING_OUTER_MM))
        try:
            return mm_per_px_from_ring(radius_px, outer_mm), GRADE_GOOD
        except (ValueError, TypeError):
            pass
        k = self._perceive("mm_per_px")
        if callable(k):
            k = k()
        if not isinstance(k, (int, float)) or k <= 0:
            return None, GRADE_BAD
        return float(k), GRADE_MID

    def _perceive(self, key):
        """取 perceive 里的一个入口。dict 与对象两种都支持。

        ``SimScene`` 走 ``__getitem__``；单元测试里常直接传裸 dict，而
        ``VisionPerceive`` 是对象。三条任务与两个辅助量都从这一处取，
        免得每种容器各写一遍兼容代码。
        """
        try:
            return self.perceive[key]
        except (KeyError, TypeError, IndexError):
            return getattr(self.perceive, key, None)

    def _frame_size(self):
        getter = self._perceive("frame_size")
        if not callable(getter):
            return None
        try:
            return getter()
        except (AttributeError, TypeError):
            return None

    def _reply_align(self, sol: AlignSolution, echo_seq, now):
        payload = struct.pack("<hh", _i16(sol.d_mm), _angle01(sol.alpha_deg))
        return self._send_result(TASK_ALIGN, sol.status, payload, echo_seq, now)

    # ---------- 终态 ----------

    def _s_SAFESTOP(self, now):
        pass


class Message:
    """一条已校验的请求：任务码 + seq + 载荷（不含任务码那 1 字节）。"""

    __slots__ = ("task", "seq", "payload")

    def __init__(self, task, seq, payload):
        self.task, self.seq, self.payload = task, seq, payload


class _Job:
    """一个还没到条件的请求。``last_poll`` 用来按 ``frame_interval_ms`` 取帧。"""

    __slots__ = ("msg", "started", "timeout_s", "last_poll")

    def __init__(self, msg, started, timeout_s):
        self.msg = msg
        self.started = started
        self.timeout_s = timeout_s
        self.last_poll = None


def _i16(v):
    """int16 饱和钳位（文档附录二第三节：超量程饱和，不回绕）。"""
    return max(-32768, min(32767, int(round(v))))


def _angle01(deg):
    """角度归一到 0.1° 的有符号区间（文档附录二第二节）。"""
    return int(((deg * 10.0 + 1800) % 3600) - 1800)
