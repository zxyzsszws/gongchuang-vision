"""参考靶检测：把「多圈同心圆」拟合出一个公共圆心（纯函数层，像素域）。

## 为什么用同心圆而不是单个圆

参考靶由若干圈同心圆组成（类似射箭靶）。单个圆拟合只能得到一个圆心与该圆
半径；而多圈同心圆共同约束同一个圆心——**圈数越多、半径跨度越大，圆心越准**。
这正是位置精度能到亚像素级的物理来源。

## 为什么不用霍夫圆

霍夫的 ``minDist`` 会抑制同心的多个圆：同心圆圆心完全相同，属于霍夫「重复
检出」要压制的情形。实测在合成靶图上，霍夫把 3 个靶检成 55 个圆心互有偏移的
圆，聚类后又被劈成十几簇。轮廓法没有这个矛盾：每一圈是一条独立闭合曲线，
用最小外接圆拟合即得精确圆心与半径。

因此检测流程是：

1. 自适应阈值 → 找所有闭合轮廓；
2. 用「圆度 = 4πA/P²」和半径范围筛掉非圆轮廓（矩形、弧段、噪声）；
3. 把「圆心彼此足够接近」的圆聚成一簇，一簇即一个靶；簇心即靶心，
   簇内**不同半径的圈数**即「可见圈数」。

## 为什么不用「按半径分档映射编号」

旧实现靠半径带区分数字环 1/2/3。新方案里三个靶结构完全相同、编号只由排列
顺序与靶心数字决定，半径不再承载编号信息，故不按半径分档。
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

# 圆心聚类容差（像素）：多圈同心圆的圆心误差远小于此值，不同靶的间距远大于此值。
_DEFAULT_CLUSTER_TOL_PX = 12.0
# 圆度下限：正圆为 1.0；2:1 椭圆约 0.84。
_DEFAULT_MIN_CIRCULARITY = 0.70
# 充满度下限：轮廓面积 / 最小外接圆面积。圆与椭圆都接近 1.0，正方形仅 2/π≈0.64。
# 这是区分「圆」与「方形/多边形」最有效的判据——圆度对方形给 0.785，门限很难设。
_DEFAULT_MIN_FILL = 0.85
# 半径相对差小于此值视为「同一圈」的重复检出，不重复计圈。
_DEFAULT_SAME_RADIUS_RATIO = 0.12
# 去重容差：圆心距与半径都小于此值时视为同一次检出（环带的内外轮廓会被各检出一次）。
_DEFAULT_DEDUP_PX = 3.0


@dataclass(frozen=True)
class TargetObs:
    """一个参考靶的像素域观测。"""

    center: tuple        # 靶心像素坐标
    radius_px: float     # 最外圈半径
    ring_count: int      # 该靶可见的不同半径圈数

    @property
    def spread_px(self) -> float:
        """最外圈半径与可见圈数之比，可作尺度诊断。"""
        return self.radius_px / self.ring_count if self.ring_count else 0.0


def detect_circles(img_bgr, cfg=None, search_radius=None):
    """检出画面中所有圆，返回 ``[(x, y, r), ...]``（像素），按半径从大到小。

    ## 两道几何判据

    轮廓法会给出**每一圈的两条轮廓**（环带的内边界与外边界），所以同一个圆会
    出现两次，半径相差约等于线宽。不做处理会让「圈数」翻倍、把两圈的伪影也
    凑成「四圈真靶」。处理方式是最后按「圆心距 + 半径差」去重。

    区分圆与方/多边形靠**充满度**（轮廓面积 / 最小外接圆面积）而不是圆度：
    正方形的圆度是 4πA/P² = 4π·s²/(4s)² = π/4 ≈ 0.785，与透视压扁的椭圆
    （2:1 时为 0.84）几乎重叠，门限没法设；而充满度对正方形只有 2/π ≈ 0.64，
    对圆与椭圆都接近 1.0，判据干净。两条判据同时用：圆度挡细长弧段，充满度挡
    方形与多边形。

    cfg 可含 ``min_circularity``、``min_fill``、``threshold_block``、
    ``threshold_c``，以及 ``search_radius``（优先用参数传入的值）。
    """
    cfg = cfg or {}
    search = search_radius or cfg.get("search_radius")
    if not search:
        return []
    r_lo, r_hi = float(search[0]), float(search[1])
    min_circ = float(cfg.get("min_circularity", _DEFAULT_MIN_CIRCULARITY))
    min_fill = float(cfg.get("min_fill", _DEFAULT_MIN_FILL))
    dedup_px = float(cfg.get("dedup_px", _DEFAULT_DEDUP_PX))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 自适应阈值：靶的圆环是暗线，光照不均时全局阈值会漏掉一侧
    block = int(cfg.get("threshold_block", 25)) | 1     # 必须为奇数
    c_val = float(cfg.get("threshold_c", 10))
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, block, c_val)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    found = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        circularity = 4.0 * math.pi * area / (peri * peri)
        if circularity < min_circ:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r <= 0 or not r_lo <= r <= r_hi:
            continue
        fill = area / (math.pi * r * r)
        if fill < min_fill:
            continue
        found.append((float(x), float(y), float(r)))

    # 去重：环带内外两条轮廓是同一个圆。按半径降序处理，保留先出现的（外边界，
    # 与真实半径更接近），半径或圆心接近的一律视为重复。
    found.sort(key=lambda c: -c[2])
    kept = []
    for x, y, r in found:
        if any(math.hypot(x - kx, y - ky) < dedup_px and abs(r - kr) < dedup_px
               for kx, ky, kr in kept):
            continue
        kept.append((x, y, r))
    return kept


def cluster_concentric(circles, tol_px=_DEFAULT_CLUSTER_TOL_PX, min_rings=2,
                       same_radius_ratio=_DEFAULT_SAME_RADIUS_RATIO):
    """把圆心相近的圆聚成一簇，每簇即一个同心圆靶。

    以**最外圈为种子**（外圈圆心估计最可靠），把圆心距种子 ``tol_px`` 内的圆
    归入该簇；簇内**不同半径**的个数即可见圈数（重复检出只增加计数、不增加
    圈数）；圈数不足 ``min_rings`` 的簇丢弃。

    靶心取半径加权均值：外圈像素多、圆心更准。
    """
    if not circles:
        return []

    pts = [(float(x), float(y), float(r)) for x, y, r in circles]
    assigned = [False] * len(pts)
    out = []

    for seed in sorted(range(len(pts)), key=lambda i: -pts[i][2]):
        if assigned[seed]:
            continue
        sx, sy, _sr = pts[seed]
        members = [j for j, (x, y, _r) in enumerate(pts)
                   if math.hypot(x - sx, y - sy) <= tol_px]

        radii = []
        for j in members:
            rj = pts[j][2]
            if all(abs(rj - have) / max(rj, have, 1e-9) > same_radius_ratio
                   for have in radii):
                radii.append(rj)
        if len(radii) < min_rings:
            continue

        wsum = sum(pts[j][2] for j in members)
        mx = sum(pts[j][0] * pts[j][2] for j in members) / wsum
        my = sum(pts[j][1] * pts[j][2] for j in members) / wsum
        r_max = max(pts[j][2] for j in members)
        for j in members:
            assigned[j] = True
        out.append(TargetObs((mx, my), r_max, len(radii)))

    out.sort(key=lambda t: -t.radius_px)
    return out


def detect_targets(img_bgr, cfg=None, search_radius=None):
    """一步完成「检出所有圆 + 聚成同心圆靶」。

    cfg 可含 ``hough``（保留键，当前实现用轮廓法，不再读取）、
    ``cluster_tol_px``、``min_rings``、``search_radius``、``min_circularity``。
    ``search_radius`` 参数优先于 cfg 内的同名项。
    """
    cfg = cfg or {}
    circles = detect_circles(img_bgr, cfg, search_radius)
    return cluster_concentric(
        circles,
        tol_px=float(cfg.get("cluster_tol_px", _DEFAULT_CLUSTER_TOL_PX)),
        min_rings=int(cfg.get("min_rings", 2)),
        same_radius_ratio=float(cfg.get("same_radius_ratio",
                                        _DEFAULT_SAME_RADIUS_RATIO)),
    )


def order_targets(obs, collinear_tol_deg=8.0):
    """把检测到的靶排成「沿三点连线从一端到另一端」的顺序。

    返回 ``(ordered, ok)``：``ordered`` 按连线方向的投影坐标升序；``ok`` 表示三点
    是否确实近共线。三点不共线时**仍然返回排序结果**，由调用方决定是否采用——
    检测层不隐藏信息，门控交给解算后的残差。

    为什么不用「左到右」：车正对靶排时三点在图像里是竖着排的，「左右」无意义；
    用连线方向的投影排序对任意朝向都成立。排序方向做了规范化（+x 优先、同 x
    取 +y），因此结果是确定的；但哪一端是 1 号仍然未知——那正是 180° 歧义，
    必须由靶心数字或先验消解，不在本函数职责内。
    """
    pts = list(obs)
    if len(pts) < 3:
        return pts, False

    best, axis = -1.0, (1.0, 0.0)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[j].center[0] - pts[i].center[0]
            dy = pts[j].center[1] - pts[i].center[1]
            d = math.hypot(dx, dy)
            if d > best:
                best, axis = d, (dx / d, dy / d) if d > 0 else (1.0, 0.0)

    if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
        axis = (-axis[0], -axis[1])

    ordered = sorted(pts, key=lambda t: t.center[0] * axis[0] + t.center[1] * axis[1])

    # 共线判定：中间点到首尾连线的垂距，换算成角度
    a, b, m = ordered[0].center, ordered[-1].center, ordered[1].center
    ab = math.hypot(b[0] - a[0], b[1] - a[1])
    if ab < 1e-9:
        return ordered, False
    cross = abs((b[0] - a[0]) * (m[1] - a[1]) - (b[1] - a[1]) * (m[0] - a[0]))
    perp = cross / ab
    ang = math.degrees(math.atan2(perp, max(1e-9, ab / 2.0)))
    return ordered, ang <= collinear_tol_deg
