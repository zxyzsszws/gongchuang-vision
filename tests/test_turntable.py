"""转盘取料点：两种停稳判定策略（毫米域，纯函数）。

两种策略回答的是同一个问题——「现在停在取料点上的是哪个颜色」——但思路不同，
用配置项 ``turntable.stop_strategy`` 切换。这里两者都测，因为现场可能只适用其中一种。
"""
import math

import pytest

from src.algorithms.turntable import (DEFAULT_CONTAINMENT_MARGIN, SettleByContainment,
                                      SettleByMotion, blobs_in_mm, make_strategy)
from src.algorithms.types import Blob, BlobMm, TurntableFrame


def _frame(t, centers, colors=(1,), radius=25.0):
    """造一帧：``centers`` 是各色块圆心（mm），``colors`` 是各自的颜色。"""
    blobs = tuple(BlobMm(color=colors[i % len(colors)], center=tuple(c), radius=radius)
                  for i, c in enumerate(centers))
    return TurntableFrame(t=t, blobs=blobs)


# ---------- SettleByMotion：连续几帧不动 ----------

def test_motion_needs_to_hold_still_long_enough():
    """刚进画面就算停稳是错的：必须连续不动达到 hold_s 才认。"""
    s = SettleByMotion((0.0, 0.0), tol_mm=8.0, hold_s=0.25, pick_tol_mm=25.0)
    assert s.feed(_frame(0.00, [(0.0, 0.0)])) is None    # 第一帧只记下起点
    assert s.feed(_frame(0.10, [(0.5, 0.0)])) is None    # 才 100ms
    assert s.feed(_frame(0.30, [(1.0, 0.0)])) == 1       # 满 250ms，答颜色 1


def test_motion_reports_the_colour_that_is_still():
    s = SettleByMotion((0.0, 0.0), tol_mm=8.0, hold_s=0.25)
    s.feed(_frame(0.0, [(0.0, 0.0)], colors=(4,)))
    s.feed(_frame(0.1, [(0.0, 0.0)], colors=(4,)))
    assert s.feed(_frame(0.3, [(0.0, 0.0)], colors=(4,))) == 4


def test_motion_restarts_the_clock_when_it_moves():
    """中途动了就要重新计时，否则「慢慢爬」会被误判成停稳。"""
    # pick_tol 放宽到 100mm：这里要测的是「位移让计时重来」，
    # 不能让「离取料点太远」那条规则先把帧丢掉（40mm 已超过缺省的 25mm）。
    s = SettleByMotion((0.0, 0.0), tol_mm=8.0, hold_s=0.25, pick_tol_mm=100.0)
    s.feed(_frame(0.0, [(0.0, 0.0)]))
    s.feed(_frame(0.15, [(40.0, 0.0)]))     # 走了 40mm，明显在转
    assert s.feed(_frame(0.20, [(40.0, 0.0)])) is None
    assert s.feed(_frame(0.45, [(40.0, 0.0)])) == 1


def test_motion_ignores_blobs_far_from_the_pick_point():
    """停在别处的物料不算到位：它还没转到取料点。"""
    s = SettleByMotion((0.0, 0.0), tol_mm=8.0, hold_s=0.1, pick_tol_mm=25.0)
    s.feed(_frame(0.0, [(150.0, 0.0)]))
    assert s.feed(_frame(0.5, [(150.0, 0.0)])) is None


def test_motion_never_answers_without_blobs():
    """画面里没有色块时不得返回颜色，更不得返回 0 当「空位」。"""
    s = SettleByMotion((0.0, 0.0))
    assert s.feed(_frame(0.0, [])) is None
    assert s.feed(_frame(1.0, [])) is None


def test_motion_reset_forgets_history():
    """同一个 FSM 会复用同一个判定器；新请求必须从零开始计时。"""
    s = SettleByMotion((0.0, 0.0), tol_mm=8.0, hold_s=0.25)
    s.feed(_frame(0.0, [(0.0, 0.0)]))
    s.feed(_frame(0.3, [(0.0, 0.0)]))
    s.reset()
    assert s.feed(_frame(1.0, [(0.0, 0.0)])) is None, "reset 后必须重新累计"


def test_motion_picks_the_nearest_blob():
    """画面里有两块物料时，判定要看离取料点最近的那块。"""
    s = SettleByMotion((0.0, 0.0), tol_mm=8.0, hold_s=0.1, pick_tol_mm=25.0)
    s.feed(_frame(0.0, [(120.0, 0.0), (0.0, 0.0)], colors=(2, 5)))
    assert s.feed(_frame(0.2, [(120.0, 0.0), (0.0, 0.0)], colors=(2, 5))) == 5


# ---------- SettleByContainment：完全落进边界圆 ----------

def test_containment_answers_when_blob_is_fully_inside():
    s = SettleByContainment((0.0, 0.0), material_radius_mm=25.0)
    assert s.boundary_radius_mm == pytest.approx(25.0 * DEFAULT_CONTAINMENT_MARGIN)
    assert s.feed(_frame(0.0, [(0.0, 0.0)])) == 1


def test_containment_accepts_small_offset_within_the_margin():
    """边界圆比物料大一点，所以允许一点停位误差。"""
    s = SettleByContainment((0.0, 0.0), material_radius_mm=25.0, margin=1.15)
    limit = s.boundary_radius_mm
    assert s.feed(_frame(0.0, [(limit - 25.0 - 0.1, 0.0)])) == 1


def test_containment_rejects_partially_outside():
    """只有一部分压进边界圆不算到位——那正是转盘还在转的时刻。"""
    s = SettleByContainment((0.0, 0.0), material_radius_mm=25.0, margin=1.15)
    limit = s.boundary_radius_mm
    assert s.feed(_frame(0.0, [(limit - 25.0 + 1.0, 0.0)])) is None


def test_containment_rejects_blob_at_another_station():
    """相邻工位离得很远，绝不可能落进边界圆。"""
    s = SettleByContainment((0.0, 0.0), material_radius_mm=25.0)
    assert s.feed(_frame(0.0, [(150.0, 0.0)])) is None


def test_containment_skips_unknown_colour():
    """认不出颜色的色块不能拿来回答，否则会把「不知道」变成「猜一个」。"""
    s = SettleByContainment((0.0, 0.0), material_radius_mm=25.0)
    assert s.feed(_frame(0.0, [(0.0, 0.0)], colors=(0,))) is None


def test_containment_is_stateless():
    """它只看位置对不对，与历史帧无关，所以单帧就能作答。"""
    s = SettleByContainment((0.0, 0.0), material_radius_mm=25.0)
    assert s.feed(_frame(12.3, [(0.0, 0.0)])) == 1


# ---------- make_strategy ----------

def test_make_strategy_defaults_to_motion():
    s = make_strategy({}, (0.0, 0.0), 25.0)
    assert isinstance(s, SettleByMotion)
    assert s.tol_mm == pytest.approx(8.0)
    assert s.hold_s == pytest.approx(0.25)


def test_make_strategy_reads_motion_parameters():
    s = make_strategy({"stop_strategy": "motion", "tol_mm": 3.0,
                       "hold_s": 0.4, "pick_tol_mm": 12.0}, (5.0, -5.0), 25.0)
    assert isinstance(s, SettleByMotion)
    assert (s.tol_mm, s.hold_s, s.pick_tol_mm) == (3.0, 0.4, 12.0)
    assert s.pick_point == (5.0, -5.0)


def test_make_strategy_builds_containment():
    s = make_strategy({"stop_strategy": "containment",
                       "containment_margin": 1.2}, (0.0, 0.0), 30.0)
    assert isinstance(s, SettleByContainment)
    assert s.boundary_radius_mm == pytest.approx(36.0)


def test_make_strategy_rejects_unknown_name():
    """写错策略名必须早报错，而不是静默退回某个缺省值。"""
    with pytest.raises(ValueError, match="stop_strategy"):
        make_strategy({"stop_strategy": "magic"}, (0.0, 0.0), 25.0)


# ---------- blobs_in_mm ----------

def test_blobs_in_mm_scales_and_shifts():
    """像素域色块 → 毫米域：绕取料点缩放，半径用面积换算。"""
    out = blobs_in_mm([Blob(center=(100.0, 50.0), area=math.pi * 100.0, color=3)],
                      mm_per_px=2.0, origin_px=(20.0, 10.0))
    assert len(out) == 1
    b = out[0]
    assert b.color == 3
    assert b.center == pytest.approx((160.0, 80.0))
    assert b.radius == pytest.approx(20.0), "面积 πr² 换回半径再乘尺度"


def test_blobs_in_mm_keeps_distance_order():
    """换算是等距的，所以「谁离取料点近」不会因为换算而改变。"""
    px = [Blob(center=(30.0, 0.0), area=100.0, color=1),
          Blob(center=(90.0, 0.0), area=100.0, color=2)]
    a, b = blobs_in_mm(px, 1.5, origin_px=(0.0, 0.0))
    assert a.distance_to((0.0, 0.0)) < b.distance_to((0.0, 0.0))
