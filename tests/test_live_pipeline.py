"""live 感知链路：相机帧 → 检测 → 三件测量结果。

V2.2.0 的感知层只做三件事：读码、量主圆、量转盘色块。它**不再解车身位姿**，
所以这里也不再需要场地图与位姿投影——直接渲染一张图，看能不能把该量的量出来。

这一层最容易被「看起来能跑」掩盖的缺陷是坐标系与尺度：
- 主圆的像素偏移若忘了换毫米，d 会整体按 mm/px 缩小；
- 转盘色块若没绕取料点缩放，颜色判定的距离门限全部失效；
- 没有外参时若照常输出，电控会拿到方向错位的测量结果。

所以每条都用一个「偏移量已知」的合成图去反查，而不是只看函数不抛异常。
"""
import math

import cv2
import numpy as np

from src.algorithms.geometry import CameraCalib, Similarity
from src.devices.vision import VisionPerceive
from tests.conftest import repo_path

SCALE = 2.0      # 2mm/px：640×480 的画面覆盖 1280×960mm，够放主圆与转盘
CALIB = Similarity(scale=SCALE, rot_deg=0.0, tx=-320.0 * SCALE,
                   ty=-240.0 * SCALE)
FRAME_W, FRAME_H = 640, 480
CENTRE = (FRAME_W / 2.0, FRAME_H / 2.0)


def _cfg():
    import yaml
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["target"] = dict(cfg["target"], search_radius=[15, 90], min_rings=3)
    return cfg


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


def _blank():
    return np.full((FRAME_H, FRAME_W, 3), 240, np.uint8)


def _draw_ring(img, center, radii=(30, 45, 60), thickness=2):
    for r in radii:
        cv2.circle(img, (int(round(center[0])), int(round(center[1]))),
                   int(r), (0, 0, 0), thickness)
    return img


def _perceive(img, calib=CALIB):
    return VisionPerceive(_ScriptedCamera([img]), _cfg(),
                          calib=CameraCalib(calib) if calib else None)


# ---------- 主圆 ----------

def test_ring_is_found_at_the_centre():
    """画面中心的同心圆必须被当成主圆，外圈半径要报出来（尺度靠它）。"""
    img = _draw_ring(_blank(), CENTRE)
    obs = _perceive(img).ring()
    assert obs is not None
    assert abs(obs.center[0] - CENTRE[0]) < 3
    assert abs(obs.center[1] - CENTRE[1]) < 3
    assert abs(obs.radius_px - 60) <= 4
    assert obs.ring_count >= 3


def test_ring_picks_the_one_nearest_the_optical_axis():
    """画面上另有一个更大的靶时，仍要挑中心那个：电控已经把它伸到光轴附近了。"""
    img = _blank()
    _draw_ring(img, (120, 120), radii=(40, 60, 80))
    _draw_ring(img, CENTRE)
    obs = _perceive(img).ring()
    assert obs is not None
    assert abs(obs.center[0] - CENTRE[0]) < 6


def test_empty_frame_has_no_ring():
    assert _perceive(_blank()).ring() is None


def test_lone_circle_is_not_a_ring():
    """单圈无法与反光伪影区分，不能当成主圆——那会把 d 与 α 算到一张纸的边上。"""
    img = _blank()
    cv2.circle(img, (320, 240), 50, (0, 0, 0), 2)
    assert _perceive(img).ring() is None


def test_ring_offset_direction_is_preserved():
    """主圆偏在画面右侧时，检出到的圆心也必须偏右（符号不能反）。"""
    img = _draw_ring(_blank(), (CENTRE[0] + 80, CENTRE[1] - 40))
    obs = _perceive(img).ring()
    assert obs is not None
    assert obs.center[0] > CENTRE[0] + 60
    assert obs.center[1] < CENTRE[1] - 20


# ---------- 尺度 ----------

def test_mm_per_px_comes_from_the_calibration():
    per = _perceive(_blank())
    assert per.mm_per_px == SCALE


def test_mm_per_px_is_none_without_calibration():
    """没有外参时**不许**编一个尺度出来。

    转盘上没有已知尺寸物，尺度只能来自外参。凭空取 1.0mm/px 会让毫米坐标大小
    全错，而形状与方向都正常——电控照着挪会越挪越偏，且不报错。
    """
    per = _perceive(_blank(), calib=None)
    assert per.mm_per_px is None


def test_frame_size_is_reported():
    """像素偏移要相对画面中心算，所以这一层必须能报出画面尺寸。"""
    assert _perceive(_blank()).frame_size() == (FRAME_W, FRAME_H)


# ---------- 转盘色块 ----------

def _red(x, y, r=25):
    return int(x), int(y), r


def test_turntable_blobs_are_returned_in_millimetres():
    """色块要换到毫米：距画面中心 100px、2mm/px → 200mm。"""
    img = _blank()
    cv2.circle(img, (320 + 100, 240), 25, (0, 0, 255), -1)   # 纯红，色号 1
    per = _perceive(img)
    blobs, k = per.turntable()
    assert blobs, "红色块必须被检出来"
    assert k == SCALE
    reds = [b for b in blobs if b.color == 1]
    assert reds, f"色号 1 缺失，实际 {[b.color for b in blobs]}"
    assert abs(reds[0].center[0] - 200.0) < 8.0, "像素偏移没换毫米"


def test_turntable_radius_comes_from_the_area():
    """半径由面积换算：画一个半径 25px 的圆，2mm/px → 约 50mm。"""
    img = _blank()
    cv2.circle(img, (320, 240), 25, (0, 0, 255), -1)
    blobs, _k = _perceive(img).turntable()
    reds = [b for b in blobs if b.color == 1]
    assert reds
    assert abs(reds[0].radius - 50.0) < 6.0


def test_turntable_without_colour_returns_empty_not_none():
    """画面里没有物料是正常状态，应返回空的色块列表而不是 None。

    返回 None 表示「相机没帧」，两者混在一起会让 FSM 分不清该等还是该报错。
    """
    blobs, _k = _perceive(_blank()).turntable()
    assert blobs == ()


def test_turntable_returns_none_without_a_frame():
    per = VisionPerceive(_ScriptedCamera([]), _cfg(), calib=CameraCalib(CALIB))
    blobs, k = per.turntable()
    assert blobs is None and k is None


def test_no_extrinsics_only_blocks_the_turntable_task():
    """没有外参时：转盘那条路没尺度可用，必须不给结果；对准仍能靠主圆外圈自量。

    这条差别很容易搞混，所以分开断言：任务 3 的尺度来自主圆外圈实测
    （``target.ring_outer_mm``），本来就是设计上不依赖外参的。
    """
    img = _blank()
    _draw_ring(img, CENTRE)
    cv2.circle(img, (int(CENTRE[0]) + 120, int(CENTRE[1])), 25, (0, 0, 255), -1)
    per = VisionPerceive(_ScriptedCamera([img]), _cfg(), calib=None)

    assert per.has_extrinsics() is False

    blobs, k = per.turntable()
    assert blobs is None and k is None, "转盘没尺度可用，不该给毫米坐标"

    obs = per.ring()
    assert obs is not None, "对准靠主圆外圈自量尺度，不需要外参"
    assert obs.radius_px > 0


# ---------- 装配 ----------

def test_vision_perceive_exposes_the_three_keys():
    per = _perceive(_blank())
    for key in ("qr", "turntable", "ring"):
        assert callable(per[key]), f"perceive 缺少可调用键 {key}"


def test_old_pose_keys_are_gone():
    """位姿与区域这两条通道已下架，不能还挂在感知层上。"""
    per = _perceive(_blank())
    for dead in ("targets", "pose", "region", "arm_target"):
        assert not hasattr(per, dead), f"感知层仍暴露 {dead}"


def test_multi_camera_routes_qr_to_second_camera():
    """qr_index 生效：扫码走第二台相机，主圆识别走主摄。"""
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
