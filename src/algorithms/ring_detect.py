"""霍夫圆环检测：按半径打「环种类 + 编号」标签，并做三色环分组。

两类环此前共用一套半径分档，导致三色环与数字环互相串味（三色环被当成数字环、
或被整段丢弃）。现在按物理不变量区分，而非靠配置硬凑：

- **数字环 1/2/3** 的半径**各不相同**（`ring.geo` 半径带 → 环编号），
  这也是「按半径分档映射编号」的物理依据；命中打 ``kind="digit"`` + label。
- **三色环三联** 三个环**同尺寸**，靠「等半径 + 近共线 + 间距达标」分组，
  与数字环天然可分；不在任何数字环带内的圆记为 ``kind="tri"`` 候选。

分组约束（宁缺勿错）：同种类、半径彼此接近（``radius_tol``）、相邻间距落在
``spacing_*`` 带内、两侧间距比不超过 ``max_spacing_ratio``、整体近共线。
"""
import math

import cv2
import numpy as np

from .types import RingObs, TriRingObs

# HoughCircles 默认识别阈值：现场调优可写入 config `ring.hough: {param1, param2}`。
_DEFAULT_HOUGH = {"param1": 120, "param2": 30}


def _bands(geo_cfg):
    """半径带配置解析：[{r_min, r_max, label}|{r_min, r_max}] 或元组列表。"""
    out = []
    for item in geo_cfg or []:
        if isinstance(item, dict):
            out.append((float(item["r_min"]), float(item["r_max"]), item.get("label")))
        else:
            lo, hi = float(item[0]), float(item[1])
            out.append((lo, hi, item[2] if len(item) > 2 else None))
    return out


def detect_rings(img_bgr, geo_cfg, hough_cfg=None, search_radius=None):
    """霍夫圆环检测：返回 RingObs 列表（带 kind/label）。

    geo_cfg: [{r_min, r_max, label}, ...] 数字环半径带 → label(1-3)，
             命中打 kind="digit"；不在任何带内的圆打 kind="tri"（三色环候选）。
    search_radius: (min_r, max_r) 霍夫搜索范围（像素）。缺省按 geo 带 ×0.7/1.3
             推导；三色环半径（像素）不在数字环带附近时**必须**显式给足范围，
             否则三联根本进不了候选集。
    hough_cfg: 可选 {param1, param2}，缺省用 _DEFAULT_HOUGH。
    """
    hough_cfg = hough_cfg or _DEFAULT_HOUGH
    param1 = float(hough_cfg.get("param1", _DEFAULT_HOUGH["param1"]))
    param2 = float(hough_cfg.get("param2", _DEFAULT_HOUGH["param2"]))
    bands = _bands(geo_cfg)
    if search_radius:
        r_lo, r_hi = int(search_radius[0]), int(search_radius[1])
    elif bands:
        r_lo = int(min(b[0] for b in bands) * 0.7)
        r_hi = int(max(b[1] for b in bands) * 1.3)
    else:
        return []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    grad = cv2.morphologyEx(clahe, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    blur = cv2.GaussianBlur(grad, (9, 9), 2)

    raw = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
                           param1=param1, param2=param2,
                           minRadius=max(1, r_lo), maxRadius=max(2, r_hi))

    rings = []
    for det in (raw[0] if raw is not None else []):
        x, y, r = float(det[0]), float(det[1]), float(det[2])
        label = next((lb for lo, hi, lb in bands if lo <= r <= hi and lb is not None), None)
        if label is not None:
            rings.append(RingObs((x, y), r, label, kind="digit"))
        else:
            rings.append(RingObs((x, y), r, None, kind="tri"))
    return rings


def _spacing_ok(d1, d2, spacing, max_ratio):
    if not all(spacing[0] <= d <= spacing[1] for d in (d1, d2)):
        return False
    if max_ratio and max_ratio > 0:
        hi, lo = max(d1, d2), min(d1, d2)
        if lo <= 0 or hi / lo > max_ratio:
            return False
    return True


def _radius_ok(a, b, c, tol):
    """三色环等尺寸：三环半径相对差不超过 tol（数字环 1/2/3 半径各异，天然排除）。"""
    if not tol or tol <= 0:
        return True
    rs = (a.radius, b.radius, c.radius)
    if min(rs) <= 0:
        return False
    return (max(rs) - min(rs)) / max(rs) <= tol


def detect_tri_ring(rings, spacing_mm, collinear_tol_deg, max_spacing_ratio=2.0,
                    radius_tol=0.25, prefer_kind="tri"):
    """三色环分组：找「等半径 + 近水平共线 + 相邻间距达标」的三元组。

    按中心 x 升序取左/中/右；两侧间距都须在 spacing_mm 带内、比值不超过
    max_spacing_ratio，三环半径相对差不超过 radius_tol，且整体与水平夹角
    （折叠到 [0,90]）不超过 collinear_tol_deg。找不到返回 None（宁缺勿错）。

    prefer_kind 指定的种类有候选时只在该种类内分组（三色环 vs 数字环不串味）；
    无该种类候选时退回全部圆（兼容未配置 tri 半径带的老场景/历史回放）。
    """
    cand = [r for r in rings if r.kind == prefer_kind]
    if not cand:
        cand = list(rings)
    for a in cand:
        for b in cand:
            for c in cand:
                if not (a.center[0] < b.center[0] < c.center[0]):
                    continue
                d1 = math.hypot(b.center[0] - a.center[0], b.center[1] - a.center[1])
                d2 = math.hypot(c.center[0] - b.center[0], c.center[1] - b.center[1])
                if not _spacing_ok(d1, d2, spacing_mm, max_spacing_ratio):
                    continue
                if not _radius_ok(a, b, c, radius_tol):
                    continue
                ang = abs(math.degrees(math.atan2(c.center[1] - a.center[1],
                                                  c.center[0] - a.center[0])))
                if min(ang, 180 - ang) > collinear_tol_deg:
                    continue
                return TriRingObs(left=a, middle=b, right=c)
    return None


def detect_digit_rings(rings):
    """挑出数字环（kind="digit" 且带 label），按 label 升序返回。"""
    return sorted((r for r in rings if r.kind == "digit" and r.label is not None),
                  key=lambda r: r.label)
