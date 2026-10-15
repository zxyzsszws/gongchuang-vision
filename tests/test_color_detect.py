import cv2, numpy as np, yaml
from src.algorithms.color_detect import detect_colors

def _cfg():
    return yaml.safe_load(open("config/default.yaml", encoding="utf-8"))["color"]

def test_two_blobs():
    img = np.full((480, 640, 3), 128, np.uint8)
    cv2.rectangle(img, (50, 50), (130, 130), (0, 0, 255), -1)      # 红=BGR(0,0,255)
    cv2.rectangle(img, (300, 200), (400, 300), (255, 0, 0), -1)    # 蓝
    blobs = detect_colors(img, _cfg()["hsv_table"], _cfg()["min_area"])
    found = {b.color for b in blobs}
    assert 1 in found and 3 in found
    red = next(b for b in blobs if b.color == 1)
    assert abs(red.center[0] - 90) < 10 and abs(red.center[1] - 90) < 10

def test_small_noise_ignored():
    img = np.full((480, 480, 3), 128, np.uint8)
    cv2.rectangle(img, (10, 10), (20, 20), (0, 0, 255), -1)  # 面积 100 < min_area
    assert detect_colors(img, _cfg()["hsv_table"], _cfg()["min_area"]) == []
