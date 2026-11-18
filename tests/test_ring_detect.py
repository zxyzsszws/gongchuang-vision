"""同心圆主圆检测：圆拟合、聚类、挑选、抗噪（像素域，纯函数）。

V2.2.0 只挑「离画面中心最近的那个主圆」，不再给三个共线靶排序——排序是位姿
解算的一部分，已随位姿上报一起下架。
"""
import math
import random

import cv2
import numpy as np

from src.algorithms.ring_detect import (TargetObs, cluster_concentric,
                                        detect_circles, detect_targets,
                                        pick_main)


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


# ---------- 挑主圆 ----------

def test_pick_main_prefers_the_one_nearest_the_centre():
    """主圆就是画面中心附近那个；边上的靶即使圈数更多也不该被选走。

    电控已经把臂伸到放置点上方，主圆必然靠近光轴，所以「离中心最近」比「圈数
    最多」更可靠：反光切断某一圈时圈数会变，但位置不会突然跳到画面上。
    """
    near = TargetObs((330.0, 245.0), 60.0, 3)
    far_more_rings = TargetObs((600.0, 100.0), 80.0, 6)
    assert pick_main([far_more_rings, near], (320.0, 240.0)) is near


def test_pick_main_breaks_ties_by_ring_count():
    """距离相同时取圈数多的：圈数多说明圆心更可靠。"""
    a = TargetObs((330.0, 240.0), 60.0, 2)
    b = TargetObs((310.0, 240.0), 60.0, 5)
    assert pick_main([a, b], (320.0, 240.0)) is b


def test_pick_main_returns_none_when_empty():
    assert pick_main([], (320.0, 240.0)) is None


def test_pick_main_skips_clusters_below_min_rings():
    """圈数不足的簇不算靶——它的圆心没有多圈约束，位置不可信。"""
    weak = TargetObs((321.0, 240.0), 30.0, 1)
    assert pick_main([weak], (320.0, 240.0)) is None
    assert pick_main([weak], (320.0, 240.0), min_rings=1) is weak


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
    """矩形不是圆，必须被充满度门限挡住。"""
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


def test_detect_targets_result_feeds_pick_main():
    """端到端：合成图像 → 检出 → 挑中心那个，中心靶必须被选中。"""
    img = _ring_image([150, 320, 490])
    out = detect_targets(img, {"search_radius": [20, 70], "min_rings": 3})
    main = pick_main(out, (320.0, 240.0))
    assert main is not None
    assert abs(main.center[0] - 320) < 3
    assert abs(main.radius_px - 50) < 4, "外圈半径必须报出来，尺度靠它反算"
