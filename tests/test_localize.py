"""三点解位姿与 180° 镜像消歧（v2.0.0 的核心算法）。

这套数学是整个方案的地基：它不成立，视觉给电控的位姿就是错的。
"""
import math

import pytest

from src.algorithms.localize import (DegenerateInput, PoseSolution,
                                     resolve_order, solve_pose)
from src.algorithms.types import Pose

# 场地粗加工区的三个靶：共线、间距 300mm
THEORY = [(1200.0, 600.0), (1500.0, 600.0), (1800.0, 600.0)]


def body_of(pose, field_pts):
    """场地点 → 车身坐标（用位姿的逆变换）。"""
    a = math.radians(pose.heading)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for fx, fy in field_pts:
        dx, dy = fx - pose.x, fy - pose.y
        out.append((ca * dx + sa * dy, -sa * dx + ca * dy))
    return out


def test_solve_pose_recovers_truth_exactly():
    truth = Pose(1500.0, 1000.0, 270.0)
    obs = body_of(truth, THEORY)
    sol = solve_pose(obs, THEORY)
    assert abs(sol.pose.x - truth.x) < 1e-6
    assert abs(sol.pose.y - truth.y) < 1e-6
    # 270° 会以 −90° 表示（归一到 ±180），比角度时必须按差值比
    assert abs((sol.pose.heading - truth.heading + 180) % 360 - 180) < 1e-6
    assert abs(sol.scale - 1.0) < 1e-9
    assert sol.residual_mm < 1e-9


@pytest.mark.parametrize("heading", [0.0, 37.5, 90.0, 180.0, -135.0, 270.0])
def test_solve_pose_all_headings(heading):
    truth = Pose(1400.0, 900.0, heading)
    sol = solve_pose(body_of(truth, THEORY), THEORY)
    diff = (sol.pose.heading - truth.heading + 180) % 360 - 180
    assert abs(diff) < 1e-6


def test_scale_estimated_away_from_one():
    """观测整体放大 2% 时尺度自估应得 1/1.02，残差仍为 0。"""
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = [(x * 1.02, y * 1.02) for x, y in body_of(truth, THEORY)]
    sol = solve_pose(obs, THEORY)
    assert abs(sol.scale - 1 / 1.02) < 1e-9
    assert sol.residual_mm < 1e-9


def test_fixed_scale_keeps_scale_and_shows_residual():
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = [(x * 1.02, y * 1.02) for x, y in body_of(truth, THEORY)]
    sol = solve_pose(obs, THEORY, fixed_scale=1.0)
    assert sol.scale == 1.0
    assert sol.residual_mm > 1.0, "尺度不匹配时必须体现在残差上"


def test_degenerate_input_raises():
    with pytest.raises(DegenerateInput):
        solve_pose([(0, 0), (0, 0), (0, 0)], THEORY)
    with pytest.raises(DegenerateInput):
        solve_pose(THEORY, [(5.0, 5.0)] * 3)      # 理论点重合同样不可解


def test_requires_equal_point_counts():
    with pytest.raises(ValueError):
        solve_pose([(0, 0), (1, 0)], THEORY)


# ---------- 180° 镜像歧义 ----------

def test_reversed_order_gives_180_degree_flipped_solution():
    """整体反向对应会得到朝向差 180° 的解，且残差同样为 0 —— 这正是歧义的来源。"""
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = body_of(truth, THEORY)
    fwd = solve_pose(obs, THEORY)
    rev = solve_pose(list(reversed(obs)), THEORY)
    assert rev.residual_mm < 1e-9, "反解残差也是 0，所以残差无法分辨"
    diff = (rev.pose.heading - fwd.pose.heading + 180) % 360 - 180
    assert abs(abs(diff) - 180.0) < 1e-6


def test_resolve_order_picks_by_prior():
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = list(reversed(body_of(truth, THEORY)))
    cand = resolve_order(obs, THEORY, prior=truth)
    diff = (cand.solution.pose.heading - truth.heading + 180) % 360 - 180
    assert abs(diff) < 1e-6, "有正确先验时必须选对"
    assert abs(cand.solution.pose.x - truth.x) < 1e-6


def test_resolve_order_picks_digits_over_prior():
    """读到靶心数字时以数字为准，即使先验是错的。"""
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = list(reversed(body_of(truth, THEORY)))
    wrong_prior = Pose(1500.0, 1000.0, 210.0)
    by_prior = resolve_order(obs, THEORY, prior=wrong_prior)
    assert abs((by_prior.solution.pose.heading - wrong_prior.heading + 180) % 360
               - 180) < 1e-6, "先验错时会被带偏（这是先验的作用）"

    # obs 顺序是 3,2,1，所以数字标签依次是 (3,2,1)
    by_digits = resolve_order(obs, THEORY, prior=wrong_prior, digits=(3, 2, 1))
    diff = (by_digits.solution.pose.heading - truth.heading + 180) % 360 - 180
    assert abs(diff) < 1e-6, "数字是直接证据，必须压过先验"


def test_resolve_order_without_prior_takes_forward():
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = body_of(truth, THEORY)
    cand = resolve_order(obs, THEORY)
    assert cand.order == (0, 1, 2)
    assert abs(cand.solution.pose.heading - truth.heading) < 1e-6


def test_digits_must_be_a_permutation():
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = body_of(truth, THEORY)
    with pytest.raises(ValueError, match="排列"):
        resolve_order(obs, THEORY, digits=(1, 1, 2))


def test_digits_length_must_match():
    truth = Pose(1500.0, 1000.0, 30.0)
    obs = body_of(truth, THEORY)
    with pytest.raises(ValueError, match="长度"):
        resolve_order(obs, THEORY, digits=(1, 2))


# ---------- 精度敏感性 ----------

def test_heading_error_scales_with_jitter_and_spacing():
    """抖动固定时，三靶间距越大朝向越准；这是「靶间距要摆大」的定量依据。"""
    import random
    rng = random.Random(7)
    truth = Pose(1500.0, 1000.0, 45.0)

    def spread(gap):
        theory = [(1200.0, 600.0), (1200.0 + gap, 600.0), (1200.0 + 2 * gap, 600.0)]
        worst = 0.0
        for _ in range(200):
            obs = [(x + rng.gauss(0, 0.5), y + rng.gauss(0, 0.5))
                   for x, y in body_of(truth, theory)]
            sol = solve_pose(obs, theory)
            worst = max(worst, abs(sol.pose.heading - truth.heading))
        return worst

    small, large = spread(150.0), spread(600.0)
    assert large < small, f"间距 600mm 的朝向误差应小于 150mm，实际 {large} vs {small}"
    assert large < 0.5, f"预期 0.5px 抖动、600mm 间距时朝向误差很小，实际 {large}°"


def test_position_accuracy_under_jitter():
    """0.5px 抖动、三靶间距 300mm 时位置误差应在毫米量级以内。"""
    import random
    rng = random.Random(11)
    truth = Pose(1500.0, 1000.0, 20.0)
    errs = []
    for _ in range(200):
        obs = [(x + rng.gauss(0, 0.5), y + rng.gauss(0, 0.5))
               for x, y in body_of(truth, THEORY)]
        sol = solve_pose(obs, THEORY)
        errs.append(math.hypot(sol.pose.x - truth.x, sol.pose.y - truth.y))
    mean = sum(errs) / len(errs)
    assert mean < 2.0, f"平均位置误差 {mean:.3f}mm 偏大"
