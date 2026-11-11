"""几何层测试：相似/单应的拟合与逆变换、相机外参加载。

v2.0.0 删除了 ``compose_with_pose``（车体毫米→场地毫米）：位姿现在是视觉自己
解算的未知量，再让它反过来喂给感知层会形成循环依赖。位姿解算见
``tests/test_localize.py``。
"""
import math

import numpy as np
import pytest

from src.algorithms.geometry import (CameraCalib, Homography, Similarity,
                                     fit_homography, fit_similarity,
                                     load_camera_calib)


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


# ---------- yaml 加载 ----------

def test_load_camera_calib_similarity(tmp_path):
    p = tmp_path / "calib.yaml"
    p.write_text("image_to_robot: {scale_mm_per_px: 1.5, rot_deg: 180, tx_mm: -480, ty_mm: -360}\n",
                 encoding="utf-8")
    calib = load_camera_calib(str(p))
    assert isinstance(calib.image_to_robot, Similarity)
    assert calib.image_to_robot.scale == 1.5


def test_load_camera_calib_homography(tmp_path):
    p = tmp_path / "calib.yaml"
    p.write_text("homography:\n  - [1, 0, 5]\n  - [0, 1, 7]\n  - [0, 0, 1]\n",
                 encoding="utf-8")
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


def test_example_calib_is_identity_ish_and_monotonic():
    """示例外参的尺度必须为正且合理，否则符号搞反会让整幅图像左右翻转。"""
    calib = load_camera_calib("config/camera_calib.example.yaml")
    s = calib.image_to_robot
    assert 0.1 < s.scale < 20.0
    # 相邻像素映射到相邻毫米，且距离随像素距离等比放大
    a, b = s.apply((0, 0)), s.apply((100, 0))
    assert abs(math.hypot(b[0] - a[0], b[1] - a[1]) - 100.0 * s.scale) < 1e-6


def test_camera_calib_has_no_auto_scale():
    """v1.0.0 的 auto_scale_from_tri 已删除：尺度由标定固定，不再用三色环自估。"""
    calib = CameraCalib()
    assert not hasattr(calib, "auto_scale_from_tri")
    assert not hasattr(calib, "with_auto_scale")
