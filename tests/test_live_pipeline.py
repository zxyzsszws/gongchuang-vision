"""live 视觉链路端到端：合成相机帧 → 环检测 → 外参换算 → 校准闭环。

这是此前完全缺失的能力证明。旧实现里 ``detect_rings`` 输出像素、``calibrate``
拿像素比毫米，残差必然远超 15mm，live 模式永远走 CAL_SKIPPED——看起来「完赛」，
实际全靠裸 OPS 漂移。本文件用一台**假的相机**（按场地真值 + 位姿误差渲染三色环）
逐帧驱动完整管线，断言校准真的被算出来且方向正确。

渲染口径（与 SimScene 一致的物理模型）：
    图像点 = image_to_robot⁻¹( R(−heading_true)·(场地点 − pos_true) )
即「场地 → 车体（真位姿）→ 像素」。管线再用**带误差的位姿**做反向换算，
误差 ε 于是进入观测，正是 calibration.calibrate 期望的输入。
"""
import math

import numpy as np
import pytest

from src.algorithms.fieldmap import load_field_map
from src.algorithms.geometry import Similarity
from src.algorithms.ring_detect import detect_rings, detect_tri_ring
from src.algorithms.types import Pose
from src.devices.vision import VisionPerceive

FIELD = load_field_map("config/field.example.yaml")

# 相机外参：1mm/px、无旋转，图像中心 (320,240) 对应车体原点。
# robot = pixel − (320,240)，故 tx/ty 为负。三色环间距 150mm → 150px，落在 spacing 带内。
CALIB = Similarity(scale=1.0, rot_deg=0.0, tx=-320.0, ty=-240.0)
INV = CALIB.invert()


def _render_tri(pose_true, field_tri, calib=INV, radius_px=None):
    """按真位姿把三色环投影到图像，返回 (img, 理论像素环心列表)。

    只画环（黑描边圆），不画其它东西——检测层要能独立跑通。
    """
    W, H = 640, 480
    img = np.full((H, W, 3), 210, np.uint8)
    import cv2

    px_centers = []
    for ring in (field_tri.left, field_tri.middle, field_tri.right):
        # 场地 → 车体（真位姿的逆）
        a = math.radians(-pose_true.heading)
        dx = ring.center[0] - pose_true.x
        dy = ring.center[1] - pose_true.y
        rb = (math.cos(a) * dx - math.sin(a) * dy,
              math.sin(a) * dx + math.cos(a) * dy)
        px = calib.apply(rb)  # 车体 → 像素
        px_centers.append(px)
        r = radius_px if radius_px is not None else 20.0
        cv2.circle(img, (int(round(px[0])), int(round(px[1]))), int(r), (0, 0, 0), 6)
    return img, px_centers


class _ScriptedCamera:
    """按脚本逐帧返回图像（模拟连续帧流）。"""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self):
        if not self._frames:
            return False, None
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return True, frame


def _cfg(**ring_over):
    import yaml
    with open("config/default.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["ring"].update(ring_over)
    return cfg


def test_rings_are_found_in_projected_image():
    """渲染的三色环必须能被霍夫检出（否则后面所有换算都是空中楼阁）。"""
    pose_true = Pose(600.0, 300.0, 0.0)   # 站在三色环正前方
    img, px = _render_tri(pose_true, FIELD.tri)
    cfg = _cfg()
    rings = detect_rings(img, cfg["ring"]["geo"], cfg["ring"]["hough"],
                        cfg["ring"]["search_radius"])
    assert len(rings) >= 3
    # 投影点附近必须有检出
    for cx, cy in px:
        assert any(math.hypot(r.center[0] - cx, r.center[1] - cy) < 8 for r in rings)


def test_pixel_observation_converts_to_field_mm():
    """核心回归：tri() 必须返回**场地毫米**观测，而非像素。"""
    pose_true = Pose(600.0, 300.0, 0.0)
    img, _ = _render_tri(pose_true, FIELD.tri)
    cfg = _cfg()
    perce = VisionPerceive(_ScriptedCamera([img]), cfg, calib=_Calib(CALIB), field=FIELD)
    perce.pose = pose_true  # 无误差：观测应回到理论环心

    obs = perce.tri()
    assert obs is not None
    for got, want in ((obs.left, FIELD.tri.left), (obs.middle, FIELD.tri.middle),
                      (obs.right, FIELD.tri.right)):
        assert abs(got.center[0] - want.center[0]) < 12.0
        assert abs(got.center[1] - want.center[1]) < 12.0


class _Calib:
    """最简外参容器（VisionPerceive 只要求 .image_to_robot / .auto_scale_from_tri）。"""

    def __init__(self, tr, auto=False):
        self.image_to_robot = tr
        self.auto_scale_from_tri = auto

    def with_auto_scale(self, px_tri, field_tri):
        return self


def test_live_calibration_produces_correct_correction():
    """端到端：位姿带 30mm/−15mm 误差 → 校准修正量应约为 (−30, +15)。

    这是 live 模式此前的死区：像素当毫米时修正量会错到米级或被残差闸门丢弃。
    """
    from src.algorithms.calibration import calibrate

    truth = Pose(600.0, 300.0, 0.0)
    err = (30.0, -15.0, 0.0)                      # 上报 = 真值 + ε
    pose_reported = Pose(truth.x + err[0], truth.y + err[1], truth.heading)
    img, _ = _render_tri(truth, FIELD.tri)        # 图像按**真值**渲染

    perce = VisionPerceive(_ScriptedCamera([img]), _cfg(), calib=_Calib(CALIB), field=FIELD)
    perce.pose = pose_reported                    # 管线用**带误差**位姿换算
    obs = perce.tri()
    assert obs is not None

    fixed = calibrate(pose_reported, obs, FIELD, 0.7, 0.3, 15.0, 3.0)
    assert fixed is not None, "live 观测必须能通过残差闸门（旧实现此处恒失败）"
    dx, dy = fixed.x - pose_reported.x, fixed.y - pose_reported.y
    assert abs(dx + 30.0) < 8.0, f"dx={dx} 应约 −30（把上报位姿拉回真值）"
    assert abs(dy - 15.0) < 8.0, f"dy={dy} 应约 +15"


def test_live_calibration_rejects_without_extrinsics():
    """无外参：tri() 返回 None（宁可跳过校准，也不发方向错位的修正量）。"""
    truth = Pose(600.0, 300.0, 0.0)
    img, _ = _render_tri(truth, FIELD.tri)
    perce = VisionPerceive(_ScriptedCamera([img]), _cfg(), calib=None, field=FIELD)
    perce.pose = truth
    assert perce.tri() is None


def test_tri_requires_pose():
    """无位姿：无法进入场地坐标系 → None。"""
    truth = Pose(600.0, 300.0, 0.0)
    img, _ = _render_tri(truth, FIELD.tri)
    perce = VisionPerceive(_ScriptedCamera([img]), _cfg(), calib=_Calib(CALIB), field=FIELD)
    perce.pose = None
    assert perce.tri() is None


def test_digit_ring_and_tri_ring_do_not_cross_match():
    """数字环与三色环不得互相凑组：等半径判据 + kind 分流。"""
    from src.algorithms.types import RingObs

    tri_rings = [RingObs((100, 240), 20, kind="tri"),
                 RingObs((240, 240), 20, kind="tri"),
                 RingObs((380, 240), 20, kind="tri")]
    digit_rings = [RingObs((520, 100), 38, 1, kind="digit"),
                   RingObs((620, 200), 58, 2, kind="digit")]
    obs = detect_tri_ring(tri_rings + digit_rings, (120, 200), 5.0)
    assert obs is not None
    assert all(r.kind == "tri" for r in (obs.left, obs.middle, obs.right))


def test_digit_rings_are_labelled_by_radius_band():
    from src.algorithms.ring_detect import detect_digit_rings
    from src.algorithms.types import RingObs

    rings = [RingObs((0, 0), 80, 3, kind="digit"),
             RingObs((0, 0), 40, 1, kind="digit"),
             RingObs((0, 0), 60, 2, kind="digit"),
             RingObs((0, 0), 20, None, kind="tri")]
    assert [r.label for r in detect_digit_rings(rings)] == [1, 2, 3]


def test_mixed_radius_triple_is_rejected():
    """半径差异过大（远环被透视放大）不得凑成一联。"""
    from src.algorithms.types import RingObs

    rings = [RingObs((100, 240), 20, kind="tri"),
             RingObs((240, 240), 34, kind="tri"),
             RingObs((380, 240), 20, kind="tri")]
    assert detect_tri_ring(rings, (120, 200), 5.0) is None


def test_spacing_ratio_blocks_bad_triple():
    """两侧间距比超限：挡住把远处大环与近处小环凑成一联。"""
    from src.algorithms.types import RingObs

    rings = [RingObs((100, 240), 20, kind="tri"),
             RingObs((240, 240), 20, kind="tri"),
             RingObs((340, 240), 20, kind="tri")]  # 100 vs 140 → 比值 1.4 可通过
    assert detect_tri_ring(rings, (90, 200), 5.0, max_spacing_ratio=2.0) is not None
    assert detect_tri_ring(rings, (90, 200), 5.0, max_spacing_ratio=1.2) is None


def test_vision_perceive_exposes_fsm_keys():
    """perceive 必须支持 FSM 的 perceive["key"]() 调用约定。"""
    img, _ = _render_tri(Pose(600.0, 300.0, 0.0), FIELD.tri)
    perce = VisionPerceive(_ScriptedCamera([img]), _cfg(), calib=_Calib(CALIB), field=FIELD)
    perce.pose = Pose(600.0, 300.0, 0.0)
    assert callable(perce["tri"]) and callable(perce["rings"])
    assert isinstance(perce.rings(), list)


def test_multi_camera_routes_qr_to_second_camera():
    """qr_index 生效：扫码走第二台相机，环识别走主摄。"""
    from src.devices.camera import MultiCamera

    main = _ScriptedCamera([np.zeros((4, 4, 3), np.uint8)])
    qr = _ScriptedCamera([np.full((4, 4, 3), 7, np.uint8)])
    cam = MultiCamera(main, qr)
    ok, frame = cam.read()
    assert ok and frame.max() == 0          # 主摄
    ok, frame = cam.qr_read()
    assert ok and frame.max() == 7          # 扫码相机


def test_multi_camera_falls_back_to_main():
    from src.devices.camera import MultiCamera

    main = _ScriptedCamera([np.full((4, 4, 3), 3, np.uint8)])
    cam = MultiCamera(main, None)
    ok, frame = cam.qr_read()
    assert ok and frame.max() == 3
