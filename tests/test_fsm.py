"""状态机测试（V2.2.0 语义）。

V2.2.0 的视觉是**按需应答的测量仪器**：只有 SERVICE（随时待命）与 SAFESTOP
（双方实现明显对不上）两个状态。没有心跳、没有周期上报、没有 done。

这里测的是新契约里真正重要的东西：

- 只在收到请求时发帧，其余时间一片安静；
- 三条任务各自的返回体与状态码；
- 同一个 seq 的重复请求要原样重发上一次的结果，不重新测；
- 转盘要等到停稳才答，等不到就超时回「没准备好」；
- 连续不合法请求才进安全停止，偶发一次不算。
"""
import struct

import pytest
import yaml

from src.algorithms.arm import GRADE_GOOD
from src.core.fsm import MatchFSM
from src.core.protocol import (CMD_REQUEST, CMD_RESULT, ST_BAD_REQUEST,
                              ST_NOT_READY, ST_OK, ST_UNSOLVABLE, TASK_ALIGN,
                              TASK_SCAN, TASK_TURNTABLE, Frame, REQ_LEN)
from src.core.protocol import RETIRED_CMDS
from src.devices.sim_scene import SimScene
from tests.conftest import repo_path


def _cfg():
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class FakeLink:
    """可编程链路：记录视觉发出的帧，``poll`` 依次交付电控发来的请求。"""

    def __init__(self):
        self.sent = []
        self.rx = []

    def send(self, frame):
        self.sent.append(frame)

    def poll(self):
        return self.rx.pop(0) if self.rx else None

    def tick(self, now):
        pass

    def results(self):
        return [f for f in self.sent if f.cmd_id == CMD_RESULT]

    def last(self):
        return self.results()[-1]

    def push(self, frame):
        self.rx.append(frame)


def _fsm(link, scene=None):
    return MatchFSM(_cfg(), link, scene if scene is not None else SimScene())


def _req(task, seq=1, payload=b""):
    return Frame(seq, CMD_REQUEST, bytes((task,)) + bytes(payload))


def _parse(frame):
    """把结果帧拆成 (echo_seq, task, status, body)。"""
    assert frame.data[:1], "结果帧至少要有 echo_seq"
    return frame.data[0], frame.data[1], frame.data[2], frame.data[3:]


def _run(fsm, seconds=1.0, dt=0.02, start=0.0):
    now = start
    end = start + seconds
    while now < end:
        fsm.step(now)
        if fsm.state == "SAFESTOP":
            break
        now += dt
    return now


# ---------- 基本契约 ----------

def test_starts_in_service_and_stays_quiet():
    """开机即待命：没有请求就一帧都不发。

    这是 V2.2.0 最重要的一条——上一版在这里会每 50ms 发两帧。
    """
    link = FakeLink()
    fsm = _fsm(link)
    assert fsm.state == "SERVICE"
    _run(fsm, seconds=3.0)
    assert link.sent == [], "没人问就不该说话"


def test_step_returns_the_state_name():
    link = FakeLink()
    fsm = _fsm(link)
    assert fsm.step(0.0) == "SERVICE"


def test_retired_commands_are_silently_dropped():
    """旧版命令码一律静默丢弃：不报错、不改变状态、不回任何帧。"""
    link = FakeLink()
    fsm = _fsm(link)
    for cmd in sorted(RETIRED_CMDS):
        link.push(Frame(1, cmd, b"\x01\x02"))
    _run(fsm, seconds=0.5)
    assert link.sent == []
    assert fsm.state == "SERVICE"


# ---------- 任务 1：扫码 ----------

def test_scan_returns_the_plan_in_one_round_trip():
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1))
    link.push(_req(TASK_SCAN, seq=7))
    fsm.step(0.0)

    echo, task, status, body = _parse(link.last())
    assert (echo, task, status) == (7, TASK_SCAN, ST_OK)
    assert len(body) == 16, "任务计划固定 16 字节"
    assert body[:3] == bytes((4, 5, 2)), "前 3 字节是第一批颜色"


def test_scan_stays_quiet_while_it_has_nothing_to_say():
    """还没扫到码时先不说话，等到扫到或超时再回一帧。

    为什么不是「立刻回一帧没准备好」：那样电控会立刻换新 seq 再问一次，扫码窗口
    被无限重开，永远等不到「看一眼」的时间。一次请求只回一帧，才是真正的
    「一问一答」。
    """
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=999))
    link.push(_req(TASK_SCAN))
    fsm.step(0.0)
    assert link.sent == [], "码还没读到就不该发帧"
    assert fsm._job is not None, "作业要挂着，等超时或读到码"


def test_inconsistent_qr_is_rejected_not_guessed():
    """二维码内容不自洽时绝不回一份计划——那会让车开到错的靶位且不报错。"""
    link = FakeLink()
    scene = SimScene(qr_at_frame=1)
    scene.qr = lambda: "452+321+254+612"      # 第二批颜色 6 在第一批里找不到
    fsm = _fsm(link, scene)
    link.push(_req(TASK_SCAN))
    fsm.step(0.0)
    assert _parse(link.last())[2] == ST_NOT_READY
    assert any(e[0] == "PLAN_REJECTED" for e in fsm.log_events)


def test_scan_times_out_instead_of_silence():
    """一直扫不到也必须有尽头：超时回一帧，不让电控无限等。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=999))
    cfg_timeout = fsm.cfg["timing"]["scan_timeout_s"]
    link.push(_req(TASK_SCAN))
    _run(fsm, seconds=cfg_timeout + 0.5)
    assert len(link.results()) == 1, "超时之后只回一帧，之后继续沉默"
    assert _parse(link.last())[2] == ST_NOT_READY
    assert any(e[0] == "SCAN_TIMEOUT" for e in fsm.log_events)


def test_scan_timeout_is_read_as_seconds_not_milliseconds():
    """3s 的扫码超时不能被当成 3ms——那样一发请求就立刻失败，永远扫不到码。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=999))
    link.push(_req(TASK_SCAN))
    fsm.step(0.0)
    _run(fsm, seconds=2.5, start=0.02)
    assert link.results() == [], "2.5s 时还没到 3s 超时"
    assert not any(e[0] == "SCAN_TIMEOUT" for e in fsm.log_events)


# ---------- 任务 2：转盘取料点 ----------

def _turntable_scene(stop_after_s=1.0):
    """转盘先转一秒再停；停机时 0 号工位（颜色 4）正对取料点。"""
    return SimScene(station_colors=(4, 5, 2), omega01=600,
                    stop_after_s=stop_after_s, jitter_mm=0.0)


def test_turntable_answers_once_it_stops():
    """先转后停：停机之后才回颜色，而且颜色必须是停在取料点上的那个。"""
    link = FakeLink()
    scene = _turntable_scene(stop_after_s=0.6)
    fsm = _fsm(link, scene)
    fsm.cfg["timing"]["turntable_timeout_s"] = 3.0

    link.push(_req(TASK_TURNTABLE, seq=3))
    now = 0.0
    while now < 2.0 and not link.results():
        fsm.step(now)
        scene.advance(0.02)
        now += 0.02

    assert link.results(), "转盘停下后必须有应答"
    echo, task, status, body = _parse(link.last())
    assert (echo, task, status) == (3, TASK_TURNTABLE, ST_OK)
    assert body == bytes((4,)), "0 号工位（颜色 4）停在取料点上"


def test_turntable_does_not_answer_while_still_moving():
    """转着的时候不能答：答早了电控会照着错颜色去取。"""
    link = FakeLink()
    scene = SimScene(station_colors=(4, 5, 2), omega01=600,
                     stop_after_s=None, jitter_mm=0.0)      # 一直转
    fsm = _fsm(link, scene)
    fsm.cfg["timing"]["turntable_timeout_s"] = 0.8

    link.push(_req(TASK_TURNTABLE))
    now = 0.0
    while now < 0.7:
        fsm.step(now)
        scene.advance(0.02)
        now += 0.02
    assert link.results() == [], "还在转，不该有结果"


def test_turntable_times_out_when_never_stopping():
    """一直不停就超时回「没准备好」：电控据此决定重发还是放弃。"""
    link = FakeLink()
    scene = SimScene(station_colors=(4, 5, 2), omega01=600,
                     stop_after_s=None, jitter_mm=0.0)
    fsm = _fsm(link, scene)
    fsm.cfg["timing"]["turntable_timeout_s"] = 0.5

    link.push(_req(TASK_TURNTABLE))
    now = 0.0
    while now < 1.0 and not link.results():
        fsm.step(now)
        scene.advance(0.02)
        now += 0.02

    echo, task, status, body = _parse(link.last())
    assert (task, status) == (TASK_TURNTABLE, ST_NOT_READY)
    assert body == b"\x00", "失败帧也要按 RES_LEN 补齐，只填 0"
    assert len(link.last().data) == 4
    assert any(e[0] == "TURNTABLE_TIMEOUT" for e in fsm.log_events)


def test_turntable_reset_between_requests():
    """同一个判定器会被复用：新请求必须重新计时，不能沿用上一次的静止结论。"""
    link = FakeLink()
    scene = _turntable_scene(stop_after_s=0.3)
    fsm = _fsm(link, scene)
    fsm.cfg["timing"]["turntable_timeout_s"] = 3.0

    link.push(_req(TASK_TURNTABLE, seq=1))
    now = 0.0
    while now < 1.5 and not link.results():
        fsm.step(now)
        scene.advance(0.02)
        now += 0.02
    assert len(link.results()) == 1

    # 第二次请求：立刻问，此时转盘仍是静止的，但判定器必须从零开始累计
    link.push(_req(TASK_TURNTABLE, seq=2))
    fsm.step(now)
    _2 = link.results()
    assert len(_2) == 1, "第一拍不该立即作答"
    assert fsm._settle is not None and fsm._settle._anchor is not None


# ---------- 任务 3：对准主圆 ----------

def test_align_with_ring_at_the_optical_centre_needs_no_motion():
    """主圆正落在光轴上时 d 与 α 都应当是 0。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(ring_body=(0.0, 0.0), jitter_mm=0.0))
    link.push(_req(TASK_ALIGN, seq=9, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)

    echo, task, status, body = _parse(link.last())
    assert (echo, task, status) == (9, TASK_ALIGN, ST_OK)
    d_mm, alpha01 = struct.unpack("<hh", body)
    assert abs(d_mm) <= 1 and alpha01 == 0


def test_align_solves_the_offset_direction():
    """主圆偏向车外（mm +y）时 α 必须为正；偏向车头（+x）时 d 为正。"""
    link = FakeLink()
    scene = SimScene(ring_body=(120.0, 80.0), jitter_mm=0.0)
    fsm = _fsm(link, scene)
    link.push(_req(TASK_ALIGN, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)

    d_mm, alpha01 = struct.unpack("<hh", _parse(link.last())[3])
    assert alpha01 > 0, "y 为正（朝车外）时臂 yaw 应为正"
    assert d_mm > 0, "目标在车头前方时车要往前挪"


def test_align_end_to_end_matches_the_arm_geometry():
    """端到端：解出的 d 与 α 代回正解，必须落在主圆**在电机坐标系里的位置**上。

    ``ring_body`` 是相对相机光轴的偏移，而相机在臂末端。所以主圆的绝对位置是
    ``(l·cos α + dx, l·sin α + dy)``，不是 ``(dx, dy)``——漏掉相机自身那一项，
    结果会稳定地少一整条臂长。
    """
    import math

    link = FakeLink()
    # 尺度自洽：外径 120mm / 像素直径 120px = 1mm/px，画面也是 1mm/px 渲染的。
    scene = SimScene(ring_body=(200.0, 150.0), jitter_mm=0.0, mm_per_px=1.0,
                     ring_radius_px=60.0)
    fsm = _fsm(link, scene)
    fsm.cfg["target"]["ring_outer_mm"] = 120.0
    link.push(_req(TASK_ALIGN, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)

    d_mm, alpha01 = struct.unpack("<hh", _parse(link.last())[3])
    a = math.radians(alpha01 / 10.0)
    assert abs(d_mm + 600.0 * math.cos(a) - (600.0 + 200.0)) < 2.0
    assert abs(600.0 * math.sin(a) - 150.0) < 2.0


def test_align_uses_ring_outer_diameter_for_the_scale():
    """尺度必须来自主圆外圈实测，而不是外参里的标定值。

    外径 120mm、画面里半径 30px（直径 60px）→ 2mm/px。主圆偏离光轴 50mm，
    画面里是 50px；按实测 2mm/px 换出 100mm，按标定 1mm/px 只有 50mm。
    """
    link = FakeLink()
    scene = SimScene(ring_body=(50.0, 0.0), jitter_mm=0.0, mm_per_px=1.0,
                     ring_radius_px=30.0)
    fsm = _fsm(link, scene)
    fsm.cfg["target"]["ring_outer_mm"] = 120.0
    link.push(_req(TASK_ALIGN, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)

    d_mm, _alpha = struct.unpack("<hh", _parse(link.last())[3])
    # 主圆绝对位置 = 600 + 100 = 700，d = 700 − 600 = 100；
    # 若误用标定值 1mm/px，d 会只有 50。
    assert abs(d_mm - 100.0) < 2.0, f"d={d_mm}，尺度没用外圈实测"
    entry = [e for e in fsm.log_events if e[0] == "ALIGN"][0]
    assert entry[4] == GRADE_GOOD, "外圈实测成功时质量档应为 GOOD"


def test_align_falls_back_to_calibrated_scale():
    """外圈太小量不出尺度时退回标定值，并且把质量档降一级。"""
    link = FakeLink()
    scene = SimScene(ring_body=(50.0, 0.0), jitter_mm=0.0, mm_per_px=1.0,
                     ring_radius_px=1.0)     # 半径 1px，低于门限
    fsm = _fsm(link, scene)
    fsm.cfg["target"]["ring_outer_mm"] = 120.0
    link.push(_req(TASK_ALIGN, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)

    status = _parse(link.last())[2]
    assert status == ST_OK, "量不到外圈不等于测不到主圆"
    entry = [e for e in fsm.log_events if e[0] == "ALIGN"][0]
    assert entry[4] != GRADE_GOOD, "尺度是估的，质量档必须降级"


def test_align_without_a_ring_reports_not_ready():
    """画面里没有主圆时必须明确报「没准备好」，不能拿 (0,0) 当答案。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(ring_body=None))
    link.push(_req(TASK_ALIGN, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)

    echo, task, status, body = _parse(link.last())
    assert (task, status) == (TASK_ALIGN, ST_NOT_READY)
    assert body == struct.pack("<hh", 0, 0), "正文仍要有 4 字节，但值为 0"
    assert any(e[0] == "ALIGN_NO_RING" for e in fsm.log_events)


def test_align_without_scale_reports_not_ready():
    """外圈量不到、又没有外参时，不许拿 1.0mm/px 这种假设值硬算。

    尺度错了的结果形状与方向都对、只有大小错——电控照着挪会越挪越偏，
    全程还不报错。宁可回「没准备好」。
    """
    link = FakeLink()

    class TinyRing:
        center = (320.0, 240.0)
        radius_px = 1.0            # 低于 mm_per_px_from_ring 的 2px 门限
        ring_count = 3

    perceive = {
        "ring": lambda: TinyRing(),
        "qr": lambda: None,
        "turntable": lambda: ((), 1.0),
        "mm_per_px": None,          # 没有外参
        "frame_size": lambda: (640, 480),
    }
    fsm = MatchFSM(_cfg(), link, perceive)
    fsm.cfg["target"]["ring_outer_mm"] = 120.0
    link.push(_req(TASK_ALIGN, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)

    echo, task, status, body = _parse(link.last())
    assert (task, status) == (TASK_ALIGN, ST_NOT_READY)
    assert body == struct.pack("<hh", 0, 0)
    assert any(e[0] == "ALIGN_NO_SCALE" for e in fsm.log_events)
    assert not any(e[0] == "ALIGN" for e in fsm.log_events), "不该产出一个假的解"


def test_align_out_of_reach_reports_unsolvable():
    """横向超出臂长：明确报无解，让电控自己决定挪车。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(ring_body=(0.0, 900.0), jitter_mm=0.0))
    link.push(_req(TASK_ALIGN, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.0)
    assert _parse(link.last())[2] == ST_UNSOLVABLE


# ---------- 请求校验 ----------

def test_unknown_task_code_is_a_bad_request():
    link = FakeLink()
    fsm = _fsm(link)
    link.push(Frame(1, CMD_REQUEST, bytes((9,))))
    fsm.step(0.0)
    echo, task, status, _body = _parse(link.last())
    assert status == ST_BAD_REQUEST
    assert task == 9, "回显任务码，便于电控定位是谁发错了"


def test_wrong_payload_length_is_a_bad_request():
    link = FakeLink()
    fsm = _fsm(link)
    link.push(Frame(1, CMD_REQUEST, bytes((TASK_ALIGN, 1, 2))))   # 应为 5 字节
    fsm.step(0.0)
    assert _parse(link.last())[2] == ST_BAD_REQUEST


def test_empty_request_is_a_bad_request():
    link = FakeLink()
    fsm = _fsm(link)
    link.push(Frame(1, CMD_REQUEST, b""))
    fsm.step(0.0)
    assert _parse(link.last())[2] == ST_BAD_REQUEST


def test_a_single_bad_request_does_not_stop_the_vision():
    """偶发一次不合法（多半是断包残帧）不该让整台机器停下来。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1))
    link.push(Frame(1, CMD_REQUEST, bytes((9,))))
    fsm.step(0.0)
    assert fsm.state == "SERVICE"

    link.push(_req(TASK_SCAN, seq=2))
    fsm.step(0.1)
    assert _parse(link.last())[2] == ST_OK, "之后正常请求仍应被服务"


def test_repeated_bad_requests_lead_to_safestop():
    """连续不合法说明两边实现已经对不上，继续答只会产生垃圾数据。"""
    link = FakeLink()
    fsm = _fsm(link)
    limit = fsm.cfg["timing"]["max_bad_requests"]
    for k in range(limit):
        link.push(Frame(k + 1, CMD_REQUEST, bytes((9,))))
        fsm.step(k * 0.1)
    assert fsm.state == "SAFESTOP"
    assert any(e[0] == "PROTOCOL_MISMATCH" for e in fsm.log_events)


def test_safestop_sends_nothing_more():
    link = FakeLink()
    fsm = _fsm(link)
    fsm._goto("SAFESTOP", 0.0)
    n = len(link.sent)
    link.push(_req(TASK_SCAN))
    _run(fsm, seconds=1.0)
    assert len(link.sent) == n
    assert fsm.state == "SAFESTOP"


def test_a_good_request_resets_the_bad_counter():
    """「连续」才计数：中间成功过一次就该重新计。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1))
    limit = fsm.cfg["timing"]["max_bad_requests"]
    for k in range(limit - 1):
        link.push(Frame(k + 1, CMD_REQUEST, bytes((9,))))
        fsm.step(k * 0.01)
    link.push(_req(TASK_SCAN, seq=100))
    fsm.step(0.2)
    assert fsm.state == "SERVICE"
    assert _parse(link.last())[2] == ST_OK


# ---------- 重复请求与重发 ----------

def test_duplicate_request_resends_the_same_answer():
    """电控没收到结果而重发同一个请求：原样再发一遍，不重新测。

    内容必须逐字节一致（除了帧自己的 seq），电控按 echo_seq 去重。
    """
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1))
    link.push(_req(TASK_SCAN, seq=11))
    fsm.step(0.0)
    first = link.last()

    link.push(_req(TASK_SCAN, seq=11))
    fsm.step(0.2)
    assert len(link.results()) == 2
    again = link.last()
    assert again.data == first.data, "重发必须逐字节一致"
    assert again.seq != first.seq, "帧 seq 是视觉自己递增的，要换新的"
    assert fsm.resends == 1
    assert fsm.counts[TASK_SCAN] == 1, "重发不算一次新的测量"


def test_a_bad_request_does_not_clobber_the_cached_answer():
    """一条残帧不该让「原样重发上次结果」失效。

    电控没收到任务 1 的结果、重发原请求之前，中间混进一条坏帧是很常见的
    （断包、噪声）。如果坏帧把缓存答案覆盖掉，电控的重发会被当成全新请求，
    视觉要么重测一遍、要么白等一轮——都表现为「偶发超时」。
    """
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1))
    link.push(_req(TASK_SCAN, seq=11))
    fsm.step(0.0)
    good = link.last()

    # 中间来一条长度不对的坏帧
    link.push(Frame(77, CMD_REQUEST, bytes((TASK_ALIGN, 1, 2))))
    fsm.step(0.1)
    assert _parse(link.last())[2] == ST_BAD_REQUEST

    # 电控仍没收到任务 1 的结果，重发原请求：必须拿回逐字节相同的答案
    link.push(_req(TASK_SCAN, seq=11))
    fsm.step(0.2)
    assert link.last().data == good.data
    assert fsm.counts[TASK_SCAN] == 1, "重发不该触发第二次测量"


def test_same_seq_with_different_content_is_a_new_request():
    """seq 回绕到同一个值时，内容不同就必须当成新请求。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1))
    link.push(_req(TASK_SCAN, seq=5))
    fsm.step(0.0)

    link.push(_req(TASK_ALIGN, seq=5, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.1)
    assert _parse(link.last())[1] == TASK_ALIGN, "不同任务不能被当成重发"


def test_new_request_abandons_the_pending_one():
    """电控改问别的了：旧作业丢掉，不等它了。"""
    link = FakeLink()
    scene = SimScene(station_colors=(4, 5, 2), omega01=600,
                     stop_after_s=None, jitter_mm=0.0)
    fsm = _fsm(link, scene)
    link.push(_req(TASK_TURNTABLE, seq=1))
    fsm.step(0.0)
    assert fsm._job is not None
    old = fsm._job

    link.push(_req(TASK_SCAN, seq=2))
    fsm.step(0.1)
    assert fsm._job is not old, "旧作业必须被丢掉"
    assert fsm._job.msg.task == TASK_SCAN, "现在挂着的是新的扫码作业"
    assert any(e[0] == "PICK_COLOR" or e[0] == "TURNTABLE_TIMEOUT"
               for e in fsm.log_events) is False, "旧作业不该再产出结果"


def test_result_frame_is_addressed_to_the_echoed_request():
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1))
    link.push(_req(TASK_SCAN, seq=200))
    fsm.step(0.0)
    assert link.last().data[0] == 200


def test_every_result_length_matches_the_contract():
    """实际发出的结果帧长度必须等于 RES_LEN 里登记的值。"""
    from src.core.protocol import RES_LEN

    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1, ring_body=(0.0, 0.0)))
    link.push(_req(TASK_SCAN, seq=1))
    fsm.step(0.0)
    link.push(_req(TASK_ALIGN, seq=2, payload=struct.pack("<HH", 600, 0)))
    fsm.step(0.1)

    seen = {}
    for f in link.results():
        seen[f.data[1]] = len(f.data)
    assert seen[TASK_SCAN] == RES_LEN[TASK_SCAN]
    assert seen[TASK_ALIGN] == RES_LEN[TASK_ALIGN]


def test_requests_are_validated_against_req_len():
    """REQ_LEN 是双方唯一的长度契约，FSM 必须真的照它校验。

    载荷多一字节必须被拒；对本来就有载荷的任务，少一字节也要被拒。
    """
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1, ring_body=(0.0, 0.0)))
    for task, want in REQ_LEN.items():
        bad_lengths = {want}                       # 多一字节（want 含任务码）
        if want > 1:
            bad_lengths.add(want - 2)              # 少一字节
        for n in bad_lengths:
            link.push(Frame(1, CMD_REQUEST, bytes((task,)) + bytes(n)))
            fsm.step(0.0)
            assert _parse(link.last())[2] == ST_BAD_REQUEST, (
                f"任务 {task} 的载荷长度 {n + 1} 应被拒（应为 {want}）")


def test_correct_length_is_accepted():
    """长度正确的请求必须被服务，不能因为校验写严了而一律拒绝。"""
    link = FakeLink()
    fsm = _fsm(link, SimScene(qr_at_frame=1, ring_body=(0.0, 0.0)))
    for task, want in REQ_LEN.items():
        link.push(Frame(1, CMD_REQUEST, bytes((task,)) + bytes(want - 1)))
        fsm.step(0.0)
        assert _parse(link.last())[2] != ST_BAD_REQUEST, f"任务 {task} 被误拒"
