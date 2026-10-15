import cv2
import numpy as np

from .types import Blob


def detect_colors(img_bgr, hsv_table, min_area):
    """按 HSV 颜色表检测色块，返回面积 >= min_area 的 Blob 列表。

    hsv_table: {color_id: ([h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi])}
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blobs = []
    for color_id, (lo, hi) in hsv_table.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, *_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                m = cv2.moments(c)
                blobs.append(Blob(center=(m["m10"] / m["m00"], m["m01"] / m["m00"]),
                                  area=area, color=int(color_id)))
    return blobs
