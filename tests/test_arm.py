"""主圆对准解算：主圆在电机坐标系下的位置 → 车沿 x 的位移 d + 臂 yaw α。

这一层的正确性必须用**正解反查**来验证，而不是只比对算出来的数：
把解出的 d、α 代回末端位置公式，看是否落在目标点上。公式 `(l·cos α, l·sin α)`
是物理事实，反查能抓住「符号反了」「用了另一支解」这类靠肉眼看不出的错误。
"""
import math

import pytest

from src.algorithms.arm import (GRADE_BAD, GRADE_GOOD, GRADE_MID, ArmState,
                                mm_per_px_from_ring, no_target_solution,
                                scale_offset, solve_align)
from src.core.protocol import ST_NOT_READY, ST_OK, ST_UNSOLVABLE


def end_effector(l, d, alpha_deg):
    """把解代回正解：车挪 d、臂转 α 之后，末端在**原**电机坐标系下的位置。"""
    a = math.radians(alpha_deg)
    return (d + l * math.cos(a), l * math.sin(a))


@pytest.mark.parametrize("x,y,l", [
    (700.0, 0.0, 600.0),         # 正前方，臂不用偏
    (700.0, 300.0, 600.0),       # 偏一侧
    (700.0, -300.0, 600.0),      # 另一侧
    (350.0, 550.0, 600.0),       # 很近、很偏
    (1200.0, 120.0, 600.0),      # 很远
    (700.0, 600.0, 600.0),       # 刚好在极限上
    (700.0, -600.0, 600.0),
])
def test_solution_reaches_the_target(x, y, l):
    """正解反查：代回去必须落在目标上，误差在浮点噪声量级。"""
    sol = solve_align((x, y), l)
    assert sol.status == ST_OK
    ex, ey = end_effector(l, sol.d_mm, sol.alpha_deg)
    assert abs(ex - x) < 1e-6, f"x 反查得 {ex}，目标 {x}"
    assert abs(ey - y) < 1e-6, f"y 反查得 {ey}，目标 {y}"


def test_alpha_is_asin_y_over_l():
    l, y = 600.0, 300.0
    sol = solve_align((700.0, y), l)
    assert abs(sol.alpha_deg - math.degrees(math.asin(y / l))) < 1e-9


def test_d_is_x_minus_horizontal_reach():
    """d 是「车要往前开多少」，等于 x 减去臂当前能挡住的前伸量。"""
    l, x, y = 600.0, 700.0, 300.0
    sol = solve_align((x, y), l)
    assert abs(sol.d_mm - (x - l * math.cos(math.radians(sol.alpha_deg)))) < 1e-9


def test_centre_target_needs_no_turning():
    sol = solve_align((500.0, 0.0), 600.0)
    assert abs(sol.alpha_deg) < 1e-12
    assert abs(sol.d_mm - (500.0 - 600.0)) < 1e-9, "正前方时 d = x − l"


def test_side_sign_follows_y():
    """α 的符号必须与 y 同号，否则臂会朝反方向摆。"""
    assert solve_align((700.0, 200.0), 600.0).alpha_deg > 0
    assert solve_align((700.0, -200.0), 600.0).alpha_deg < 0


def test_unreachable_reports_status_not_clamped():
    """|y| > l 时明确报「够不着」，不得钳到边界假装能到。

    钳位会给出一个末端其实差几十毫米的「可用」结果，这是最危险的一种失败：
    电控会照它伸手，抓空却不报错。
    """
    l = 600.0
    sol = solve_align((700.0, 700.0), l)
    assert sol.status == ST_UNSOLVABLE
    assert not sol.ok
    assert sol.d_mm == 0.0 and sol.alpha_deg == 0.0, "无解时不要给半截答案"
    # 目标点仍然报回去，便于现场排查「差多少」
    assert sol.target == (700.0, 700.0)


def test_unreachable_just_past_the_limit():
    l = 600.0
    assert solve_align((700.0, 600.0), l).status == ST_OK
    assert solve_align((700.0, 600.5), l).status == ST_UNSOLVABLE


def test_zero_length_is_unsolvable_not_a_crash():
    """l 为 0 时臂长没有任何意义：报无解，不让异常冒到主循环里。"""
    sol = solve_align((100.0, 0.0), 0.0)
    assert sol.status == ST_UNSOLVABLE


def test_no_target_solution_is_explicit():
    """没测到主圆时必须报「没准备好」，不能编一个 (0,0) 当答案。

    用「没准备好」而不是「无解」：没测到往往是相机还没伸到位，重发一次可能就好了。
    """
    sol = no_target_solution()
    assert sol.status == ST_NOT_READY
    assert not sol.ok
    assert sol.grade == GRADE_BAD


def test_grade_is_carried_through():
    """质量档只在成功时有意义，并且要能传下去（现场排查靠它）。"""
    assert solve_align((700.0, 0.0), 600.0, grade=GRADE_GOOD).grade == GRADE_GOOD
    assert solve_align((700.0, 0.0), 600.0, grade=GRADE_MID).grade == GRADE_MID


def test_arm_state_carries_units():
    """臂状态只带 l 与 α：尺度由视觉自己从主圆外圈量，不需要电控报相机高度。"""
    st = ArmState(l_mm=600.0, alpha_deg=12.3)
    assert st.l_mm == 600.0
    assert st.alpha_deg == 12.3
    assert not hasattr(st, "h_mm")


# ---------- 尺度：从主圆外圈直径反算 ----------

def test_scale_from_ring_uses_outer_diameter():
    """外圈直径 120mm、画面里测到半径 60px → 1.0 mm/px。"""
    assert mm_per_px_from_ring(60.0, 120.0) == pytest.approx(1.0)


def test_scale_from_ring_inverse():
    """反查：同一个圆在画面里越小，说明拍得越高、每像素对应的毫米越多。"""
    near = mm_per_px_from_ring(100.0, 120.0)
    far = mm_per_px_from_ring(50.0, 120.0)
    assert far == pytest.approx(near * 2.0)


def test_scale_from_ring_rejects_tiny_radius():
    """半径只有一两像素时相对误差会被放大，宁可报错也不要给虚高精度。"""
    with pytest.raises(ValueError):
        mm_per_px_from_ring(1.0, 120.0)


def test_scale_from_ring_rejects_non_positive_diameter():
    with pytest.raises(ValueError):
        mm_per_px_from_ring(50.0, 0.0)


def test_scale_offset_maps_pixel_offset_to_mm():
    """像素偏移 (40, 30)、1.5 mm/px → (60, 45)mm。"""
    assert scale_offset((40.0, 30.0), 1.5) == pytest.approx((60.0, 45.0))


def test_scale_offset_is_zero_at_pivot():
    """偏移为零时必须是零，不允许留系统性残差。"""
    assert scale_offset((0.0, 0.0), 1.234) == pytest.approx((0.0, 0.0))
