import cv2, numpy as np, yaml
from src.algorithms.ring_detect import detect_rings, detect_tri_ring

def _geo():
    return yaml.safe_load(open("config/default.yaml", encoding="utf-8"))["ring"]["geo"]

def _ring_img(centers_radii, size=(480, 640)):
    img = np.full((*size, 3), 200, np.uint8)
    for (cx, cy, r) in centers_radii:
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 6)
    return img

def test_detect_two_rings():
    img = _ring_img([(200, 240, 38), (420, 240, 58)])
    rings = detect_rings(img, _geo())
    assert len(rings) == 2
    labels = sorted(r.label for r in rings)
    assert labels == [1, 2]

def test_tri_grouping():
    img = _ring_img([(180, 240, 38), (320, 240, 38), (460, 240, 38)])
    rings = detect_rings(img, _geo())
    tri = detect_tri_ring(rings, spacing_mm=(120, 200), collinear_tol_deg=5)
    assert tri is not None
    assert tri.left.center[0] < tri.middle.center[0] < tri.right.center[0]

def test_no_tri_when_scattered():
    img = _ring_img([(100, 100, 38), (500, 150, 38), (300, 400, 38)])
    rings = detect_rings(img, _geo())
    assert detect_tri_ring(rings, spacing_mm=(120, 200), collinear_tol_deg=5) is None
