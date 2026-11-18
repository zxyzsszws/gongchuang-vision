"""主圆检测：把「多圈同心圆」拟合出一个公共圆心（纯函数层，像素域）。

## 为什么用同心圆而不是单个圆

主圆由若干圈同心圆组成（类似射箭靶）。单个圆拟合只能得到一个圆心与该圆半径；
而多圈同心圆共同约束同一个圆心——**圈数越多、半径跨度越大，圆心越准**。这正是
对准精度能到亚像素级的物理来源。

## 为什么不用霍夫圆

霍夫的 ``minDist`` 会抑制同心的多个圆：同心圆圆心完全相同，属于霍夫「重复检出」
要压制的情形。实测在合成靶图上，霍夫把 3 个靶检成 55 个圆心互有偏移的圆，聚类
后又被劈成十几簇。轮廓法没有这个矛盾：每一圈是一条独立闭合曲线，用最小外接圆
拟合即得精确圆心与半径。

因此检测流程是：

1. 自适应阈值 → 找所有闭合轮廓；
2. 用「圆度 = 4πA/P²」和半径范围筛掉非圆轮廓（矩形、弧段、噪声）；
3. 把「圆心彼此足够接近」的圆聚成一簇，一簇即一个靶；簇心即靶心，
   簇内**不同半径的圈数**即「可见圈数」。

## 为什么不再给三个靶排序

V2.1.0 靠三个共线靶解车身位姿，所以要把它们排成一条线的顺序。V2.2.0 只做
「对准画面里最近的那个主圆」，没有多靶排序这回事了——排序代码随位姿解算一起
下架，避免留下一段没人调用、却还在暗示「视觉会解位姿」的逻辑。
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
    """一个同心圆靶的像素域观测。"""

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

    cfg 可含 ``cluster_tol_px``、``min_rings``、``search_radius``、
    ``min_circularity``。``search_radius`` 参数优先于 cfg 内的同名项。
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


def pick_main(obs, center_px, min_rings=2):
    """从若干同心圆靶里挑出「主圆」：离画面中心最近的那个；没有就返回 None。

    为什么按「离中心最近」挑而不是按圈数最多：电控已经先把臂伸到放置点上方，
    主圆就在画面中心附近。按位置挑能挡住画面边上其他靶（甚至转盘上的圆）的干扰；
    按圈数挑则在某圈恰好被反光切断时会把旁边的靶认成主圆，那比不认更糟。

    距离相同时取圈数多的那个（圈数多说明圆心更可靠）。圈数低于 ``min_rings``
    的簇根本不算靶，``detect_targets`` 已经滤掉。
    """
    cand = [t for t in obs if t.ring_count >= min_rings]
    if not cand:
        return None
    cx, cy = float(center_px[0]), float(center_px[1])
    return min(cand, key=lambda t: (math.hypot(t.center[0] - cx,
                                               t.center[1] - cy),
                                    -t.ring_count))
