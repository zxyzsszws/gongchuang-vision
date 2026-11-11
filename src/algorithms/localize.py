"""由三个参考靶反解车体位姿（纯函数层，无硬件依赖）。

## 为什么必须三个靶

参考靶由多圈同心圆组成，关于自身中心**旋转对称**。只看见一个靶时，只能得到
2 个测量量（靶心在车身坐标系下的 x、y），而未知量是 3 个（x、y、朝向），
**缺少的正是朝向**：沿以靶心为圆心、以距离为半径的圆周行走一整圈，观测完全相同。

三个靶心连成一条直线，该直线在场地坐标系下的方向已知，量出它在图像中的方向
即可解出朝向；三个点还提供冗余，可用于残差门控与误检剔除。

## 求解量

车身坐标 → 场地坐标是一个相似变换（4 自由度）：

    field = scale · R(heading) · body + (x, y)

- ``heading`` 即车头朝向，``(x, y)`` 即车身参考点在场地中的位置。
- ``scale`` 应为 1.0（车身坐标已是毫米）；偏离 1 说明像素→毫米的尺度标定有误，
  可作为诊断量。是否启用尺度自估由调用方决定（文档 0x0003 质量字节 bit4）。

## 180° 镜像歧义

三个点共线，把对应关系整个反向会得到「旋转 180°、残差同样为 0」的另一个解，
**残差无法分辨**。必须靠外部先验消歧：

1. 靶心数字识别结果（最可靠，但识别率受限于靶心数字尺寸）；
2. 与航位推算的预测位姿比对（几乎免费，见 ``resolve_order``）。

``resolve_order`` 优先采用显式编号，其次取与先验位姿更接近的那个解。
"""
import math
from dataclasses import dataclass

from .geometry import fit_similarity
from .types import Pose


@dataclass(frozen=True)
class PoseSolution:
    """一次位姿解算的结果。"""

    pose: Pose                 # 车身参考点在场地中的位姿
    scale: float               # 相似变换尺度，理想值 1.0
    residual_mm: float         # 三个点的最大重投影误差（mm）
    order: tuple               # 观测点 → 场地靶编号的对应关系


class DegenerateInput(ValueError):
    """观测点重合/共点，无法定出尺度或朝向。"""


def _reprojection_error(obs, theory, scale, heading, tx, ty):
    """逐点重投影误差的最大值（mm）。"""
    a = math.radians(heading)
    ca, sa = math.cos(a), math.sin(a)
    worst = 0.0
    for (bx, by), (fx, fy) in zip(obs, theory):
        px = scale * (ca * bx - sa * by) + tx
        py = scale * (sa * bx + ca * by) + ty
        worst = max(worst, math.hypot(px - fx, py - fy))
    return worst


def _spread(points):
    """点的最大两两距离，用于判定退化（全部重合成一个点）。"""
    best = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            best = max(best, math.hypot(points[i][0] - points[j][0],
                                        points[i][1] - points[j][1]))
    return best


def solve_pose(obs_pts, theory_pts, fixed_scale=None):
    """由成对的（车身坐标点, 场地坐标点）解出车身在场地中的位姿。

    obs_pts:    观测到的靶心，**车身坐标系**，单位 mm，顺序与 theory_pts 对应。
    theory_pts: 同一批靶心在场地坐标系下的已知坐标，单位 mm。
    fixed_scale: 给定则按该尺度求解（3 个点解 3 自由度）；None 则连同尺度一起
                 最小二乘自估。

    返回 ``PoseSolution``；点全部重合时抛 ``DegenerateInput``。
    """
    obs = [(float(x), float(y)) for x, y in obs_pts]
    theory = [(float(x), float(y)) for x, y in theory_pts]
    if len(obs) != len(theory) or len(obs) < 2:
        raise ValueError("obs_pts 与 theory_pts 必须等长且至少 2 个点")
    if _spread(obs) < 1e-6 or _spread(theory) < 1e-6:
        raise DegenerateInput("观测点或理论点全部重合，无法求解位姿")

    if fixed_scale is None:
        sim = fit_similarity(obs, theory)          # 最小二乘相似变换
        scale, heading, tx, ty = sim.scale, sim.rot_deg, sim.tx, sim.ty
        if scale <= 1e-9:
            raise DegenerateInput("解出的尺度为零")
    else:
        # 尺度固定时，把观测点先缩放到目标尺度再求纯刚体变换
        scale = float(fixed_scale)
        if scale <= 1e-9:
            raise DegenerateInput("fixed_scale 必须为正")
        scaled = [(scale * x, scale * y) for x, y in obs]
        sim = fit_similarity(scaled, theory)
        heading, tx, ty = sim.rot_deg, sim.tx, sim.ty

    return PoseSolution(
        pose=Pose(tx, ty, _norm180(heading)),
        scale=scale,
        residual_mm=_reprojection_error(obs, theory, scale, heading, tx, ty),
        order=tuple(range(len(obs))),
    )


def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class OrderCandidate:
    """一种「观测点 → 场地靶」的对应假设及其解算结果。"""

    order: tuple
    solution: PoseSolution


def _pose_distance(a, b):
    """两个位姿的差异，位置按 mm、朝向按折算的等效位移综合（松耦合打分）。"""
    dpos = math.hypot(a.x - b.x, a.y - b.y)
    dhead = abs(_norm180(a.heading - b.heading))
    # 朝向 1° 约等价于 10mm 的位移误差，仅用于在两种假设间排序
    return dpos + 10.0 * dhead


def resolve_order(obs_pts, theory_pts, prior=None, digits=None, fixed_scale=None):
    """在 180° 镜像歧义下选出正确的对应关系。

    obs_pts:   观测到的靶心（车身坐标，mm），已按「沿三个点连线从一端到另一端」
               排序，即顺序本身不携带哪端是 1 号的信息。
    theory_pts: 场地图中该组三个靶的坐标，按编号 1→2→3 排列。
    prior:     航位推算给出的预测位姿 ``Pose``；两种假设中取与之更接近者。
    digits:    靶心数字识别结果，``(数字1, 数字2)`` 形式或与 obs_pts 等长的编号元组；
               给了就直接按它对应，不再依赖 prior。
    fixed_scale: 透传给 ``solve_pose``。

    返回 ``OrderCandidate``；两种假设都解不出时抛 ``DegenerateInput``。
    """
    n = len(obs_pts)
    if n != len(theory_pts):
        raise ValueError("观测点与理论点数量必须一致")

    # 数字是最直接的证据：digits[k] 是 obs_pts[k] 的靶编号，直接据此定序
    if digits is not None:
        digits = tuple(int(d) for d in digits)
        if len(digits) != n:
            raise ValueError("digits 长度必须与观测点数量一致")
        try:
            order = tuple(digits.index(label + 1) for label in range(n))
        except ValueError:
            raise ValueError(f"digits 必须是 1..{n} 的一个排列，收到 {digits}")
        ordered = [obs_pts[i] for i in order]
        return OrderCandidate(order, solve_pose(ordered, theory_pts, fixed_scale))

    fwd = tuple(range(n))
    rev = tuple(reversed(range(n)))
    candidates = []
    for order in (fwd, rev):
        ordered = [obs_pts[i] for i in order]
        candidates.append(OrderCandidate(order,
                                         solve_pose(ordered, theory_pts, fixed_scale)))

    if prior is None:
        return candidates[0]
    return min(candidates, key=lambda c: _pose_distance(c.solution.pose, prior))
