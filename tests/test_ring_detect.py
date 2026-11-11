"""同心圆参考靶检测：圆拟合、聚类、排序、抗噪（像素域，纯函数）。"""
import math
import random

import cv2
import numpy as np
import pytest

from src.algorithms.ring_detect import (TargetObs, cluster_concentric,
                                        detect_circles, detect_targets,
                                        order_targets)


def _target(cx, cy, radii=(20, 30, 40, 50, 60), jitter=0.4, seed=0):
    """一个靶的多圈同心圆，圆心带抖动（模拟圆拟合误差）。"""
    rng = random.Random(seed)
    return [(cx + rng.gauss(0, jitter), cy + rng.gauss(0, jitter), float(r))
            for r in radii]


def _ring_image(centers, radii=(30, 40, 50), size=(480, 640), thickness=2):
    img = np.full((size[0], size[1], 3), 255, np.uint8)
    for cx in centers:
        for r in radii:
            cv2.circle(img, (cx, 240), r, (0, 0, 0), thickness)
    return img


# ---------- 聚类 ----------

def test_clusters_concentric_circles_into_one_target():
    out = cluster_concentric(_target(100, 200, seed=1))
    assert len(out) == 1
    t = out[0]
    assert abs(t.center[0] - 100) < 1.0 and abs(t.center[1] - 200) < 1.0
    assert t.ring_count == 5
    assert t.radius_px == 60.0


def test_separate_targets_stay_separate():
    circles = (_target(100, 200, seed=1)
               + _target(400, 210, seed=2)
               + _target(700, 190, seed=3))
    out = cluster_concentric(circles)
    assert len(out) == 3
    assert all(t.ring_count == 5 for t in out)
    assert sorted(round(t.center[0]) for t in out) == [100, 400, 700]


def test_duplicate_detections_do_not_inflate_ring_count():
    """同一圈的重复检出不能把圈数抬高，否则伪检会被当成「圈数够多」的真靶。"""
    circles = _target(100, 200, radii=(30, 40, 50), seed=1)
    # 每个圆再来两次略有偏移的重复检出
    for x, y, r in list(circles):
        circles.append((x + 1.5, y - 1.0, r))
        circles.append((x - 1.2, y + 0.8, r))
    out = cluster_concentric(circles)
    assert len(out) == 1
    assert out[0].ring_count == 3, f"仍应是 3 圈，实际 {out[0].ring_count}"


def test_single_isolated_circle_is_dropped():
    """只有一个圈不能与反光伪影区分，宁可丢掉也不误判成一个靶。"""
    circles = _target(100, 200, seed=1) + [(250.0, 400.0, 25.0)]
    out = cluster_concentric(circles)
    assert len(out) == 1


def test_min_rings_is_configurable():
    circles = _target(100, 200, seed=1)
    assert len(cluster_concentric(circles, min_rings=5)) == 1
    assert cluster_concentric(circles, min_rings=6) == []


def test_empty_input():
    assert cluster_concentric([]) == []


def test_center_accuracy_better_than_single_ring():
    """多圈拟合应显著优于单圈：这是「用同心圆而不是单圆」的定量依据。"""
    rng = random.Random(99)
    multi, single = [], []
    for k in range(200):
        circles = _target(300, 300, jitter=0.5, seed=k)
        t = cluster_concentric(circles)[0]
        multi.append(math.hypot(t.center[0] - 300, t.center[1] - 300))
        outer = [c for c in circles if c[2] == 60.0][0]
        single.append(math.hypot(outer[0] - 300, outer[1] - 300))
    m = sum(multi) / len(multi)
    s = sum(single) / len(single)
    assert m < s, f"多圈均值 {m:.3f}px 应优于单圈 {s:.3f}px"
    assert m < 0.5, f"0.5px 抖动时多圈圆心误差应明显小于 0.5px，实际 {m:.3f}px"


# ---------- 排序 ----------

def test_order_targets_left_to_right_on_horizontal_line():
    pts = [TargetObs((700.0, 190.0), 60.0, 5),
           TargetObs((100.0, 200.0), 60.0, 5),
           TargetObs((400.0, 210.0), 60.0, 5)]
    ordered, collinear = order_targets(pts)
    assert [round(t.center[0]) for t in ordered] == [100, 400, 700]
    assert collinear


def test_order_targets_on_vertical_line():
    """车正对靶排时三点在画面里是竖着排的，「左右」无意义，必须沿连线排。

    排序方向做了规范化（+x 优先、同 x 取 +y），因此竖排时会自上而下或自下而上，
    具体朝哪边不重要——哪一端是 1 号本来就未知（180° 歧义）。重要的是**沿连线
    单调**：中间点必须落在两端之间。
    """
    pts = [TargetObs((302.0, 400.0), 60.0, 5),
           TargetObs((300.0, 100.0), 60.0, 5),
           TargetObs((298.0, 700.0), 60.0, 5)]
    ordered, collinear = order_targets(pts)
    assert collinear
    ys = [t.center[1] for t in ordered]
    assert min(ys) == 100 and max(ys) == 700
    assert sorted(ys)[1] == ys[1], f"中间点必须在两端之间，实际 {ys}"
    # 相邻间距相等（三点等距排列）
    assert abs(abs(ys[1] - ys[0]) - abs(ys[2] - ys[1])) < 1e-6


def test_order_targets_detects_non_collinear():
    pts = [TargetObs((100.0, 100.0), 60.0, 5),
           TargetObs((300.0, 400.0), 60.0, 5),
           TargetObs((500.0, 100.0), 60.0, 5)]
    ordered, collinear = order_targets(pts)
    assert not collinear, "明显不共线必须报出来（门控交给调用方，但信息不能藏）"
    assert len(ordered) == 3


def test_order_targets_with_fewer_than_three():
    pts = [TargetObs((100.0, 100.0), 60.0, 5)]
    ordered, collinear = order_targets(pts)
    assert ordered == pts and not collinear


def test_order_targets_is_rotation_invariant():
    """三点连线方向任意旋转，排序结果仍是「沿连线一端到另一端」。"""
    for ang in (0, 30, 90, 135, 200, 315):
        a = math.radians(ang)
        base = [(-100.0, 0.0), (0.0, 0.0), (100.0, 0.0)]
        pts = [TargetObs((300 + x * math.cos(a) - y * math.sin(a),
                          300 + x * math.sin(a) + y * math.cos(a)), 60.0, 5)
               for x, y in base]
        ordered, collinear = order_targets(pts)
        assert collinear
        c0, c1, c2 = [t.center for t in ordered]
        d1 = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
        d2 = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
        assert abs(d1 - 100) < 1e-6 and abs(d2 - 100) < 1e-6


# ---------- 轮廓圆检出 ----------

def test_detect_circles_returns_empty_without_search_radius():
    """没有搜索半径时不做圆检出（避免在全图上以默认半径乱找）。"""
    img = np.full((480, 640, 3), 200, np.uint8)
    assert detect_circles(img, {}) == []


def test_detect_circles_finds_concentric_rings_exactly():
    """同心圆共用圆心，必须每一圈都独立检出（这正是霍夫做不到的）。"""
    img = _ring_image([320], radii=(30, 40, 50))
    circles = detect_circles(img, {"search_radius": [20, 70]})
    assert len(circles) == 3, f"应检出 3 圈，实际 {len(circles)}"
    rs = sorted(round(r) for _x, _y, r in circles)
    assert abs(rs[0] - 30) <= 3 and abs(rs[1] - 40) <= 3 and abs(rs[2] - 50) <= 3
    for x, y, _r in circles:
        assert abs(x - 320) <= 3 and abs(y - 240) <= 3


def test_detect_circles_rejects_non_circular_contours():
    """矩形不是圆，必须被圆度门限挡住。"""
    img = np.full((480, 640, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), 2)
    assert detect_circles(img, {"search_radius": [20, 200]}) == []


def test_detect_circles_respects_radius_range():
    img = _ring_image([320], radii=(10, 40, 90))
    circles = detect_circles(img, {"search_radius": [30, 60]})
    assert len(circles) == 1
    assert abs(circles[0][2] - 40) <= 4


# ---------- detect_targets 组合 ----------

def test_detect_targets_on_synthetic_three_targets():
    img = _ring_image([150, 320, 490])
    out = detect_targets(img, {"search_radius": [20, 70], "min_rings": 3})
    assert len(out) == 3, f"应检出三个靶，实际 {len(out)}"
    xs = sorted(round(t.center[0]) for t in out)
    assert all(abs(x - want) < 3 for x, want in zip(xs, (150, 320, 490)))
    assert all(t.ring_count == 3 for t in out)


def test_detect_targets_drops_lone_circle_target():
    img = _ring_image([150, 320, 490])
    cv2.circle(img, (320, 400), 40, (0, 0, 0), 2)     # 孤立单圈
    out = detect_targets(img, {"search_radius": [20, 70], "min_rings": 3})
    assert len(out) == 3, "孤立单圈不得被当成第四个靶"


def test_detect_targets_result_feeds_order_and_solve():
    """端到端：合成图像 → 检出 → 排序，三点必须共线。"""
    img = _ring_image([150, 320, 490])
    out = detect_targets(img, {"search_radius": [20, 70], "min_rings": 3})
    ordered, collinear = order_targets(out)
    assert collinear and len(ordered) == 3
    assert [round(t.center[0]) for t in ordered] == [150, 320, 490]
