import math

import cv2
import numpy as np

from .types import RingObs, TriRingObs

# HoughCircles 默认识别阈值：现场调优可写入 config `ring.hough: {param1, param2}`。
_DEFAULT_HOUGH = {"param1": 120, "param2": 30}


def detect_rings(img_bgr, geo_cfg, hough_cfg=None):
    """霍夫圆环检测：按 geo 半径带（label1-3）给每个圆打标签，返回 RingObs 列表。

    geo_cfg: [{r_min, r_max, label}, ...] 半径带列表。
    hough_cfg: 可选 {param1, param2}，缺省用 _DEFAULT_HOUGH；供现场通过 config
              `ring.hough` 调整检出灵敏度。
    """
    hough_cfg = hough_cfg or _DEFAULT_HOUGH
    param1 = float(hough_cfg.get("param1", _DEFAULT_HOUGH["param1"]))
    param2 = float(hough_cfg.get("param2", _DEFAULT_HOUGH["param2"]))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    grad = cv2.morphologyEx(clahe, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    blur = cv2.GaussianBlur(grad, (9, 9), 2)

    raw = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
                           param1=param1, param2=param2,
                           minRadius=int(min(g["r_min"] for g in geo_cfg) * 0.7),
                           maxRadius=int(max(g["r_max"] for g in geo_cfg) * 1.3))

    rings = []
    for det in (raw[0] if raw is not None else []):
        x, y, r = det[0], det[1], det[2]
        label = next((g["label"] for g in geo_cfg if g["r_min"] <= r <= g["r_max"]), None)
        if label is not None:
            rings.append(RingObs((float(x), float(y)), float(r), label))
    return rings


def detect_tri_ring(rings, spacing_mm, collinear_tol_deg):
    """三色环分组：在带标签的圆环里找「近水平共线 + 相邻间距落在 [min,max]」的三元组。

    按中心 x 升序取左/中/右；两侧间距都须在 spacing_mm 带内，且整体与水平夹角
    （折叠到 [0,90]）不超过 collinear_tol_deg。找不到返回 None。
    """
    same = [r for r in rings if r.label is not None]
    for a in same:
        for b in same:
            for c in same:
                if not (a.center[0] < b.center[0] < c.center[0]):
                    continue
                d1 = math.hypot(b.center[0] - a.center[0], b.center[1] - a.center[1])
                d2 = math.hypot(c.center[0] - b.center[0], c.center[1] - b.center[1])
                if not all(spacing_mm[0] <= d <= spacing_mm[1] for d in (d1, d2)):
                    continue
                ang = abs(math.degrees(math.atan2(c.center[1] - a.center[1],
                                                  c.center[0] - a.center[0])))
                if min(ang, 180 - ang) > collinear_tol_deg:
                    continue
                return TriRingObs(left=a, middle=b, right=c)
    return None
