"""机械臂目标解算：三维/平面目标点 → 车沿 x 位移 d + 臂 yaw α。

这一层的正确性必须用**正解反查**来验证，而不是只比对算出来的数：
把解出的 d、α 代回末端位置公式，看是否落在目标点上。公式 `(l·cos α, l·sin α)`
是物理事实，反查能抓住「符号反了」「用了另一支解」这类靠肉眼看不出的错误。
"""
import math

import pytest

from src.algorithms.arm import (FLAG_SWEEP, FLAG_USABLE, GRADE_BEST,
                                GRADE_BAD, OUTCOME_MASK, OUTCOME_NO_TARGET,
                                OUTCOME_OK, OUTCOME_UNREACHABLE, ArmState,
                                encode_status, no_target_solution, solve_arm)


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
    sol = solve_arm((x, y), l)
    assert sol.status & OUTCOME_MASK == OUTCOME_OK
    ex, ey = end_effector(l, sol.d_mm, sol.alpha_deg)
    assert abs(ex - x) < 1e-6, f"x 反查得 {ex}，目标 {x}"
    assert abs(ey - y) < 1e-6, f"y 反查得 {ey}，目标 {y}"


def test_alpha_is_asin_y_over_l():
    l, y = 600.0, 300.0
    sol = solve_arm((700.0, y), l)
    assert abs(sol.alpha_deg - math.degrees(math.asin(y / l))) < 1e-9


def test_d_is_x_minus_horizontal_reach():
    """d 是「车要往前开多少」，等于 x 减去臂当前能挡住的前伸量。"""
    l, x, y = 600.0, 700.0, 300.0
    sol = solve_arm((x, y), l)
    assert abs(sol.d_mm - (x - l * math.cos(math.radians(sol.alpha_deg)))) < 1e-9


def test_centre_target_needs_no_turning():
    sol = solve_arm((500.0, 0.0), 600.0)
    assert abs(sol.alpha_deg) < 1e-12
    assert abs(sol.d_mm - (500.0 - 600.0)) < 1e-9, "正前方时 d = x − l"


def test_side_sign_follows_y():
    """α 的符号必须与 y 同号，否则臂会朝反方向摆。"""
    assert solve_arm((700.0, 200.0), 600.0).alpha_deg > 0
    assert solve_arm((700.0, -200.0), 600.0).alpha_deg < 0


def test_unreachable_reports_outcome_not_clamped():
    """|y| > l 时明确报「够不着」，不得钳到边界假装能到。

    钳位会给出一个末端其实差几十毫米的「可用」结果，这是最危险的一种失败：
    电控会照它伸手，抓空却不报错。
    """
    l = 600.0
    sol = solve_arm((700.0, 700.0), l)
    assert sol.status & OUTCOME_MASK == OUTCOME_UNREACHABLE
    assert not sol.usable, "够不着时不得置可用位"
    assert sol.d_mm == 0.0 and sol.alpha_deg == 0.0, "无解时不要给半截答案"
    # 目标点仍然报回去，便于电控判断「差多少」
    assert sol.target == (700.0, 700.0)


def test_unreachable_just_past_the_limit():
    l = 600.0
    assert solve_arm((700.0, 600.0), l).status & OUTCOME_MASK == OUTCOME_OK
    assert solve_arm((700.0, 600.5), l).status & OUTCOME_MASK == OUTCOME_UNREACHABLE


def test_no_target_solution_is_explicit():
    """没测到目标时必须报「没测到」，不能编一个 (0,0) 当答案。"""
    sol = no_target_solution(echo_seq=9)
    assert sol.status & OUTCOME_MASK == OUTCOME_NO_TARGET
    assert not sol.usable
    assert sol.echo_seq == 9


def test_echo_seq_is_carried_through():
    """echo_seq 必须原样带回，电控靠它配对自己的请求。"""
    assert solve_arm((700.0, 100.0), 600.0, echo_seq=42).echo_seq == 42


def test_zero_length_is_a_programming_error():
    with pytest.raises(ValueError):
        solve_arm((100.0, 0.0), 0.0)


def test_status_byte_layout():
    """status：bit0~1 结果、bit4~5 等级、bit7 可用。"""
    ok = encode_status(OUTCOME_OK, GRADE_BEST)
    assert ok & OUTCOME_MASK == OUTCOME_OK
    assert (ok >> 4) & 0x03 == GRADE_BEST
    assert ok & FLAG_USABLE

    bad = encode_status(OUTCOME_UNREACHABLE, GRADE_BAD)
    assert bad & OUTCOME_MASK == OUTCOME_UNREACHABLE
    assert not bad & FLAG_USABLE


def test_sweep_flag_when_alpha_passes_90_degrees():
    """|α| > 90° 时置告警位——那意味着臂会从车身里穿过去。

    ``|y| ≤ l`` 时 asin 的主值不会超过 90°，所以正常解永远不置这一位；
    保留它是为了将来若改用「另一支解」时能被发现。
    """
    sol = solve_arm((700.0, 600.0), 600.0)
    assert abs(sol.alpha_deg) <= 90.0
    assert not sol.status & FLAG_SWEEP


def test_arm_state_carries_units():
    st = ArmState(l_mm=600.0, h_mm=300.0, alpha_deg=12.3, seq=7)
    assert st.l_mm == 600.0 and st.h_mm == 300.0
    assert st.alpha_deg == 12.3 and st.seq == 7
