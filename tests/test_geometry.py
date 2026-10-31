"""几何变换单测：相似/单应的拟合与逆变换、位姿合成、尺度自估。"""
import math

import numpy as np
import pytest

from src.algorithms.geometry import (CameraCalib, Homography, Similarity,
                                     compose_with_pose, estimate_scale_from_tri,
                                     fit_homography, fit_similarity,
                                     load_camera_calib)
from src.algorithms.types import Pose, RingObs, TriRingObs


def _tri(centers, radius=20.0):
    return TriRingObs(*[RingObs(c, radius) for c in centers])


# ---------- Similarity ----------

def test_similarity_apply_matches_expected_transform():
    # 2 倍放大 + 90° 旋转 + 平移 (10, 20)：点 (1,0) → 旋转后 (0,2) → 平移 (10, 22)
    s = Similarity(scale=2.0, rot_deg=90.0, tx=10.0, ty=20.0)
    x, y = s.apply((1.0, 0.0))
    assert abs(x - 10.0) < 1e-9 and abs(y - 22.0) < 1e-9


def test_similarity_invert_roundtrip():
    s = Similarity(scale=1.37, rot_deg=173.2, tx=-421.0, ty=88.5)
    inv = s.invert()
    for pt in [(0, 0), (640, 0), (320.5, 479.5), (-12.0, 7.25)]:
        back = inv.apply(s.apply(pt))
        assert abs(back[0] - pt[0]) < 1e-6 and abs(back[1] - pt[1]) < 1e-6


def test_fit_similarity_recovers_known_transform():
    truth = Similarity(scale=1.25, rot_deg=-168.0, tx=-300.0, ty=250.0)
    src = [(100, 120), (500, 130), (320, 400), (10, 470)]
    dst = [truth.apply(p) for p in src]
    fit = fit_similarity(src, dst)
    assert abs(fit.scale - truth.scale) < 1e-9
    assert abs(((fit.rot_deg - truth.rot_deg + 180) % 360) - 180) < 1e-6
    assert abs(fit.tx - truth.tx) < 1e-6 and abs(fit.ty - truth.ty) < 1e-6


def test_fit_similarity_rejects_degenerate_and_short_input():
    with pytest.raises(ValueError):
        fit_similarity([(0, 0)], [(1, 1)])
    with pytest.raises(ValueError):
        fit_similarity([(5, 5), (5, 5)], [(0, 0), (1, 1)])


# ---------- Homography ----------

def test_fit_homography_recovers_projective_mapping():
    h = np.array([[1.2, 0.15, -40.0],
                  [-0.08, 1.31, 25.0],
                  [0.0, 1e-4, 1.0]])
    truth = Homography(h)
    src = [(0, 0), (640, 0), (640, 480), (0, 480), (320, 240)]
    dst = [truth.apply(p) for p in src]
    fit = fit_homography(src, dst)
    for s, d in zip(src, dst):
        got = fit.apply(s)
        assert abs(got[0] - d[0]) < 1e-6 and abs(got[1] - d[1]) < 1e-6


def test_fit_homography_requires_four_points():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])


# ---------- 位姿合成：误差进入观测的唯一入口 ----------

def test_compose_with_pose_is_rigid_and_carries_error():
    # 位姿带误差 ε 时，合成结果 = 真值 + ε（校准算法的 obs = theory + ε 口径）
    truth = Pose(1000.0, 200.0, 0.0)
    err = Pose(30.0, -12.0, 0.0)
    drift = Pose(truth.x + err.x, truth.y + err.y, truth.heading + err.heading)
    pts = [(0.0, 100.0), (50.0, -20.0)]
    got = compose_with_pose(drift, pts)
    base = compose_with_pose(truth, pts)
    # 纯平移误差时，观测与理论之差必须**精确**等于 ε（这是校准能回收位姿的前提）
    for (bx, by), (gx, gy) in zip(base, got):
        assert abs((gx - bx) - err.x) < 1e-9
        assert abs((gy - by) - err.y) < 1e-9
    # 带航向误差时：差值 = ε_xy + [R(h+εh) − R(h)]·p（显式公式验证，模长不守恒）
    eh = 1.5
    drift_h = Pose(truth.x + err.x, truth.y + err.y, truth.heading + eh)
    got_h = compose_with_pose(drift_h, pts)
    for (px, py), (bx, by), (gx, gy) in zip(pts, base, got_h):
        c0, s0 = math.cos(0.0), math.sin(0.0)
        c1, s1 = math.cos(math.radians(eh)), math.sin(math.radians(eh))
        want_x = (c1 * px - s1 * py) - (c0 * px - s0 * py) + err.x
        want_y = (s1 * px + c1 * py) - (s0 * px + c0 * py) + err.y
        assert abs((gx - bx) - want_x) < 1e-9
        assert abs((gy - by) - want_y) < 1e-9
    # 车体原点处（p=0）差值恒等于 ε，与航向误差无关 —— 校准原点口径的锚点
    origin_gap = compose_with_pose(Pose(err.x, err.y, 77.0), [(0.0, 0.0)])[0]
    assert abs(origin_gap[0] - err.x) < 1e-9 and abs(origin_gap[1] - err.y) < 1e-9


def test_compose_with_pose_rotates_heading():
    # heading=90° 时车体 +x 应映射到场地 +y
    (x, y), = compose_with_pose(Pose(0, 0, 90.0), [(100.0, 0.0)])
    assert abs(x) < 1e-9 and abs(y - 100.0) < 1e-9


# ---------- 尺度自估 ----------

def test_estimate_scale_from_tri_uses_radii():
    # 已知环半径 20mm，图像里量到 10px → 2mm/px
    px = _tri([(0, 0), (100, 0), (200, 0)], radius=10.0)
    field = _tri([(0, 0), (300, 0), (600, 0)], radius=20.0)
    assert abs(estimate_scale_from_tri(px, field) - 2.0) < 1e-9


def test_estimate_scale_from_tri_is_rotation_invariant():
    base = estimate_scale_from_tri(_tri([(0, 0), (0, 100), (0, 200)], radius=10.0),
                                   _tri([(0, 0), (300, 0), (600, 0)], radius=20.0))
    assert abs(base - 2.0) < 1e-9


def test_estimate_scale_from_tri_rejects_degenerate():
    with pytest.raises(ValueError):
        estimate_scale_from_tri(_tri([(0, 0), (1, 0), (2, 0)], radius=0.0),
                                _tri([(0, 0), (300, 0), (600, 0)], radius=20.0))


def test_camera_calib_auto_scale_only_affects_similarity():
    px = _tri([(0, 0), (100, 0), (200, 0)], radius=10.0)
    field = _tri([(0, 0), (300, 0), (600, 0)], radius=20.0)
    calib = CameraCalib(Similarity(scale=1.0), auto_scale_from_tri=True)
    out = calib.with_auto_scale(px, field)
    assert abs(out.image_to_robot.scale - 2.0) < 1e-9
    # 关闭自估 → 原样返回
    off = CameraCalib(Similarity(scale=1.0), auto_scale_from_tri=False)
    assert off.with_auto_scale(px, field).image_to_robot.scale == 1.0
    # 单应透视为自带尺度，不参与自估
    hom = CameraCalib(Homography(np.eye(3)), auto_scale_from_tri=True)
    assert hom.with_auto_scale(px, field) is hom


# ---------- yaml 加载 ----------

def test_load_camera_calib_similarity(tmp_path):
    p = tmp_path / "calib.yaml"
    p.write_text("image_to_robot: {scale_mm_per_px: 1.5, rot_deg: 180, tx_mm: -480, ty_mm: -360}\n"
                 "auto_scale_from_tri: true\n", encoding="utf-8")
    calib = load_camera_calib(str(p))
    assert isinstance(calib.image_to_robot, Similarity)
    assert calib.image_to_robot.scale == 1.5 and calib.auto_scale_from_tri is True


def test_load_camera_calib_homography(tmp_path):
    p = tmp_path / "calib.yaml"
    p.write_text("homography:\n  - [1, 0, 5]\n  - [0, 1, 7]\n  - [0, 0, 1]\n", encoding="utf-8")
    calib = load_camera_calib(str(p))
    assert isinstance(calib.image_to_robot, Homography)
    x, y = calib.image_to_robot.apply((1, 2))
    assert (x, y) == (6.0, 9.0)


def test_load_camera_calib_rejects_missing_and_bad_shape(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_camera_calib(str(empty))
    bad = tmp_path / "bad.yaml"
    bad.write_text("homography:\n  - [1, 0]\n  - [0, 1]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_camera_calib(str(bad))


def test_example_calib_file_loads():
    """仓库自带的示例外参必须始终可解析（结构回归）。"""
    calib = load_camera_calib("config/camera_calib.example.yaml")
    assert isinstance(calib.image_to_robot, Similarity)
