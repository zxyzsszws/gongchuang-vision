import math

from .types import Pose


def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0


def calibrate(pose, obs, field, w_mid, w_pair, max_residual_mm, max_residual_deg):
    """三色环视觉校准（spec 2.3，观测与理论均在场地坐标系）。

    - Z 轴：meas_angle = atan2(right - left)，dhead = norm180(meas_angle - base)，
      heading' = pose.heading - dhead。
    - 位置：e_mid = 理论中环 - 实测中环；e_pair = 理论左右中点 - 实测左右中点；
      delta = w_mid*e_mid + w_pair*e_pair；pos' = pose.xy + delta。
    - residual = |e_mid - e_pair| 表示两路修正的不一致程度；角度或残差任一超阈值
      判为误匹配，返回 None。

    符号约定：航位推算的 pose 带着相对真值的误差 ε 时，投影到场地的观测为
    theory + ε，故 theory - obs 的修正量恰把 pose 回收到真值。下游 FSM 依据
    dx = fixed - last 下发修正。
    """
    t = field.tri
    lx, ly = obs.left.center
    mx, my = obs.middle.center
    rx, ry = obs.right.center

    meas_angle = math.degrees(math.atan2(ry - ly, rx - lx))
    dhead = _norm180(meas_angle - t.base_angle_deg)
    if abs(dhead) > max_residual_deg:
        return None

    e_mid = (t.middle.center[0] - mx, t.middle.center[1] - my)
    pair_meas = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    pair_theory = ((t.left.center[0] + t.right.center[0]) / 2.0,
                   (t.left.center[1] + t.right.center[1]) / 2.0)
    e_pair = (pair_theory[0] - pair_meas[0], pair_theory[1] - pair_meas[1])

    residual = math.hypot(e_mid[0] - e_pair[0], e_mid[1] - e_pair[1])
    if residual > max_residual_mm:
        return None

    dx = w_mid * e_mid[0] + w_pair * e_pair[0]
    dy = w_mid * e_mid[1] + w_pair * e_pair[1]
    return Pose(pose.x + dx, pose.y + dy, _norm180(pose.heading - dhead))
