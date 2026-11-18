"""v2.0.0 主链路端到端：电控漂移 → 视觉看见靶 → 电控误差被清零。

这是新方案最核心的断言。若这条不成立，整个「视觉只测量、电控全规划」的分工
就没有价值。

测试分两类，刻意分开：

- **测量类**（``_measure_fsm``）：用一个「不说话」的 MockMcu（``route=[]``，
  不发去向提示），再由测试直接设 ``fsm._region``。这样可以单看视觉侧行为，
  不会被 MockMcu 自己发的区域提示扰动。
- **流程类**：用带完整路线的 MockMcu，验证扫到码 → 发计划 → 进 SERVICE 的全流程。
"""
import struct

import yaml

from src.algorithms.fieldmap import load_field_map
from src.algorithms.types import Pose
from src.core.fsm import MatchFSM
from src.core.protocol import (CMD_ARM_STATE, CMD_POSE, CMD_REGION,
                               CMD_TASK_PLAN, CMD_TURNTABLE,
                               REGION_MATERIAL, REGION_ROUGH, REGION_STAGING)
from src.devices.link import MockMcu
from src.devices.sim_scene import SimScene
from tests.conftest import repo_path


def _cfg():
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _field():
    return load_field_map(repo_path("config", "field.example.yaml"))


def _measure_fsm(region=None, jitter_mm=0.1, truth=None, omega01=600):
    """测视觉侧行为：电控不发任何帧，区域由测试指定。"""
    field = _field()
    mcu = MockMcu(route=[])          # 不说话
    scene = SimScene(field, jitter_mm=jitter_mm, omega01=omega01)
    if truth is not None:
        scene.pose_truth = truth
    fsm = MatchFSM(_cfg(), mcu, field, scene)
    fsm.state = "SERVICE"
    if region is not None:
        fsm._region = region
        fsm._region_at = 0.0
    return field, mcu, scene, fsm


def _poses(mcu):
    return [d for cid, d in mcu.received if cid == CMD_POSE]


def _decode_pose(data):
    x, y, h01 = struct.unpack("<hhh", data[:6])
    return Pose(x, y, h01 / 10.0)


def _ang_diff(a, b):
    """两个角度的最小差值（度）：协议里 270° 会以 −90° 发送，直接相减会得 360。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------- 流程类 ----------


def test_scanning_sends_task_plan_then_enters_service():
    field = _field()
    mcu = MockMcu()
    scene = SimScene(field)
    fsm = MatchFSM(_cfg(), mcu, field, scene)

    assert fsm.state == "SCANNING"
    for k in range(50):
        fsm.step(k * 0.02)
        if fsm.state != "SCANNING":
            break
    assert fsm.state == "SERVICE"
    assert CMD_TASK_PLAN in mcu.sent_cmd_ids()


def test_task_plan_bytes_match_doc_sample():
    """文档 §1.4 的黄金样例：452+321+254+312 → 16 字节。"""
    field = _field()
    mcu = MockMcu(route=[])
    fsm = MatchFSM(_cfg(), mcu, field, SimScene(field))
    for k in range(50):
        fsm.step(k * 0.02)
        if fsm.state == "SERVICE":
            break
    plans = [d for cid, d in mcu.received if cid == CMD_TASK_PLAN]
    assert plans
    assert plans[0] == bytes.fromhex("04 05 02 03 02 01 02 05 04 03 01 02 01 02 03 8d")


def test_periodic_pose_frames_are_sent_without_any_ack():
    """没有心跳：链路的活跃由周期测量帧自己证明。

    只要画面里有靶，位姿就会持续发出；一帧应答都没收到也不影响。主循环 20ms
    一拍、上报周期 50ms，实际按 60ms 节流（20 的倍数里第一个 ≥50 的），
    所以 2 秒约 34 帧。
    """
    field, mcu, scene, fsm = _measure_fsm(
        REGION_ROUGH, truth=Pose(1500.0, 1000.0, 270.0))
    for k in range(100):                 # 2 秒
        fsm.step(k * 0.02)
    poses = [c for c in mcu.sent_cmd_ids() if c == CMD_POSE]
    assert len(poses) >= 33, f"2 秒内应约 34 帧位姿，实际 {len(poses)}"


def test_bad_qr_does_not_send_plan():
    field = _field()
    mcu = MockMcu(route=[])
    scene = SimScene(field)
    scene.qr = lambda: "452+321+254+612"     # 颜色 6 不在第一批，配不上
    fsm = MatchFSM(_cfg(), mcu, field, scene)
    for k in range(50):
        fsm.step(k * 0.02)
        if fsm.state != "SCANNING":
            break
    assert fsm.state == "SCANNING", "不自洽的二维码不应进入 SERVICE"
    assert CMD_TASK_PLAN not in mcu.sent_cmd_ids()


def test_ack_timeout_goes_safestop():
    """任务计划得不到应答 → 重发 3 次后安全停止，且重发的是同一帧。"""
    field = _field()

    class Silent:
        def __init__(self):
            self.sent = []

        def send(self, f):
            self.sent.append(f)

        def poll(self):
            return None

        def tick(self, now):
            pass

    link = Silent()
    fsm = MatchFSM(_cfg(), link, field, SimScene(field))
    now = 0.0
    for _ in range(200):
        fsm.step(now)
        now += 0.02
        if fsm.state == "SAFESTOP":
            break
    assert fsm.state == "SAFESTOP"
    plans = [f for f in link.sent if f.cmd_id == CMD_TASK_PLAN]
    # 首发 1 帧 + 重发 3 帧 = 4 帧（文档表 1-14：应答重试上限 3 次）
    assert len(plans) == 4, f"首发 + 重发 3 次，实际发出 {len(plans)} 帧"
    assert len({f.seq for f in plans}) == 1, "重发的必须是原帧（seq 不变）"
    assert len({bytes(f.data) for f in plans}) == 1, "重发不得改动内容"


# ---------- 测量类 ----------


def test_pose_only_after_region_hint():
    """没收到 0x0005 就不知道该用哪组靶，不得发位姿。"""
    field, mcu, scene, fsm = _measure_fsm(region=None,
                                          truth=Pose(1500.0, 1000.0, 270.0))
    for k in range(20):
        fsm.step(k * 0.02)
    assert fsm._pose_sent == 0


def test_pose_sent_when_region_and_targets_visible():
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH,
                                          truth=Pose(1500.0, 1000.0, 270.0))
    fsm.step(0.0)
    assert fsm._pose_sent >= 1


def test_solved_pose_matches_truth():
    """视觉解出的位姿必须等于场景真值（误差在抖动量级内）。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH, jitter_mm=0.1, truth=truth)
    fsm.step(0.0)

    data = _poses(mcu)[0]
    got = _decode_pose(data)
    assert abs(got.x - truth.x) < 1.0, f"x={got.x} 期望≈{truth.x}"
    assert abs(got.y - truth.y) < 1.0, f"y={got.y} 期望≈{truth.y}"
    assert _ang_diff(got.heading, truth.heading) < 0.2, f"heading={got.heading}"


def test_solved_pose_can_be_off_nominal_entry():
    """车停偏 60mm、转头 8° 仍能解对——标称入场位姿只用于消歧，不是答案。"""
    truth = Pose(1560.0, 940.0, 262.0)
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH, jitter_mm=0.1, truth=truth)
    fsm.step(0.0)
    got = _decode_pose(_poses(mcu)[0])
    assert abs(got.x - truth.x) < 1.0 and abs(got.y - truth.y) < 1.0
    assert _ang_diff(got.heading, truth.heading) < 0.5


def test_digits_override_prior():
    """读到靶心数字时以数字为准，即使标称入场位姿是错的。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH, jitter_mm=0.1, truth=truth)
    # 把两个区域的 entry_pose 都故意写成反的，只有数字能救回来
    import dataclasses
    bad = dataclasses.replace
    fsm.field = bad(
        field,
        groups={k: bad(g, entry_pose=Pose(0.0, 0.0, 0.0)) for k, g in field.groups.items()},
    )
    scene.read_digits = True
    fsm.step(0.0)
    got = _decode_pose(_poses(mcu)[0])
    assert _ang_diff(got.heading, truth.heading) < 0.5, (
        f"读到数字时应解出 {truth.heading}，实际 {got.heading}")
    assert _poses(mcu)[0][7] & 0x0C == 0, "编号来源应为「读到数字」(0)"


def test_pose_quality_byte_is_usable():
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH,
                                          truth=Pose(1500.0, 1000.0, 270.0))
    fsm.step(0.0)
    data = _poses(mcu)[0]
    quality = data[7]
    assert quality & 0x80, "无噪声时应标记为可用"
    assert quality & 0x03 == 3, "三个靶都参与解算"
    assert data[8] == REGION_ROUGH, "第 8 字节回填当前区域码"


def test_correction_clears_drift_error():
    """核心性质：电控带着漂移到达靶区，收到视觉位姿后误差归零。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH, jitter_mm=0.1, truth=truth)

    # 电控自己走了一段，误差积累（位置偏 40mm 余、朝向偏 1.2°）
    mcu.true_x, mcu.true_y, mcu.true_h = truth.x, truth.y, truth.heading
    mcu.est_x, mcu.est_y, mcu.est_h = 1540.0, 1015.0, truth.heading + 1.2
    assert mcu.position_error_mm > 30.0

    fsm.step(0.0)
    mcu.apply_correction(_decode_pose(_poses(mcu)[0]))

    assert mcu.position_error_mm < 1.0, (
        f"收到位姿后误差应清零，实际 {mcu.position_error_mm:.2f}mm")
    assert _ang_diff(mcu.heading_error_deg, 0.0) < 0.2


def test_drift_accumulates_but_cleared_on_each_visit():
    """两趟之间的里程误差在下一次看见靶时被一次性清除（文档 §1.5 关键性质）。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH, jitter_mm=0.1, truth=truth)
    mcu.true_x, mcu.true_y, mcu.true_h = truth.x, truth.y, truth.heading
    mcu.est_x, mcu.est_y, mcu.est_h = 1500.0, 0.0, 270.0
    assert mcu.position_error_mm > 500.0

    fsm.step(0.0)
    mcu.apply_correction(_decode_pose(_poses(mcu)[0]))
    assert mcu.position_error_mm < 1.0


def test_no_pose_in_material_region():
    """原料区没有参考靶，不得发位姿。"""
    field, mcu, scene, fsm = _measure_fsm(REGION_MATERIAL, truth=Pose(600, 1400, 0))
    fsm.step(0.0)
    assert fsm._pose_sent == 0


def test_staging_region_uses_staging_targets():
    """同样的画面，区域提示不同 → 解出不同位姿（证明必须靠提示区分两组靶）。"""
    field, mcu, scene, fsm = _measure_fsm(REGION_STAGING, jitter_mm=0.1,
                                          truth=Pose(1000.0, 2200.0, 0.0))
    fsm.step(0.0)
    got = _decode_pose(_poses(mcu)[0])
    assert abs(got.y - 2200.0) < 1.0, f"暂存区解出 y={got.y}"


def test_partial_targets_do_not_send_pose():
    """只看两个靶时朝向不可观，必须一帧都不发。"""
    field, mcu, scene, fsm = _measure_fsm(REGION_ROUGH, truth=Pose(1000, 400, 0))
    real = scene.targets

    def only_two(region):
        obs = real(region)
        from src.algorithms.types import TargetSetObs
        return TargetSetObs(obs.points[:2], collinear=True) if obs else None

    scene.targets = only_two
    fsm.step(0.0)
    assert fsm._pose_sent == 0, "两靶时不得发位姿"


def test_turntable_frame_sent_in_material_region():
    field, mcu, scene, fsm = _measure_fsm(REGION_MATERIAL, truth=Pose(600, 1400, 0))
    fsm.step(0.0)
    assert fsm._turntable_sent >= 1
    data = [d for cid, d in mcu.received if cid == CMD_TURNTABLE][0]
    assert len(data) == 16
    assert data[10] == 3, "工位数量固定为 3"
    assert tuple(data[12:15]) == (4, 5, 2), "三工位颜色"


def test_turntable_angle_advances_with_time():
    """转盘匀速转动：角度必须随时间推进，否则电控算不出伸出时刻的位置。"""
    field = _field()
    scene = SimScene(field, omega01=600)     # 60°/s
    scene.pose_truth = Pose(600.0, 1400.0, 0.0)
    before = scene.turntable().angle01
    scene.advance(1.0)
    after = scene.turntable().angle01
    assert abs(((after - before) % 3600) - 600) <= 2


def test_turntable_angle_wraps_mod_3600():
    field = _field()
    scene = SimScene(field, omega01=600)
    scene.pose_truth = Pose(600.0, 1400.0, 0.0)
    scene._angle01 = 3500
    scene.advance(1.0)                        # 3500 + 600 = 4100 → 500
    assert scene.turntable().angle01 == 500


# ---------- 整场比赛的闭环：路线行驶 + 视觉清零 ----------


def test_full_route_keeps_drift_cleared_and_car_inside_field():
    """按场地图航点跑完整条路线：车始终在场内，电控误差被视觉反复清零。

    这是 sim 的「总验收」：单看某一帧的位姿对不等于整场都稳——只有电控真的按
    视觉位姿替换自身估计（``apply_pose=True``）时，里程误差才会被逐次清掉。
    """
    from src.devices.link import DEFAULT_ROUTE
    from src.main import _sim_waypoints

    field = _field()
    mcu = MockMcu(waypoints=_sim_waypoints(field, DEFAULT_ROUTE), apply_pose=True)
    scene = SimScene(field, jitter_mm=0.3)
    fsm = MatchFSM(_cfg(), mcu, field, scene)

    errors, track = [], []
    now, dt = 0.0, 0.02
    while now < 20.0:
        mcu.tick(now)
        scene.pose_truth = Pose(mcu.true_x, mcu.true_y, mcu.true_h)
        scene.advance(dt)
        fsm.step(now)
        track.append((mcu.true_x, mcu.true_y))
        errors.append(mcu.position_error_mm)
        now += dt

    xs = [p[0] for p in track]
    ys = [p[1] for p in track]
    assert 0.0 <= min(xs) and max(xs) <= 1500.0 + 1e-6, (min(xs), max(xs))
    assert 0.0 <= min(ys) and max(ys) <= 2000.0 + 1e-6, (min(ys), max(ys))
    assert fsm.state == "SERVICE"
    assert mcu.corrections > 20, f"整场应被清零很多次，实际 {mcu.corrections}"
    assert mcu.regions_sent[:3] == [4, 1, 2], mcu.regions_sent
    assert max(errors) < 100.0, f"误差应始终小于 100mm，实际最大 {max(errors):.1f}"
    assert mcu.position_error_mm < 5.0, (
        f"最后一次看见靶后误差应接近零，实际 {mcu.position_error_mm:.1f}mm")
