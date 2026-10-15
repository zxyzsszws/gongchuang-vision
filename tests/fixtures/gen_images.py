"""生成相机源测试图：3 张 480x640x3 带环图，存到 tests/fixtures/imgs/。

可重复执行（幂等）：每次运行覆盖同名文件。画图逻辑复用 Task 8 的 _ring_img。
"""
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs")


def _ring_img(centers_radii, size=(480, 640)):
    img = np.full((*size, 3), 200, np.uint8)
    for (cx, cy, r) in centers_radii:
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 6)
    return img


# 三帧：环组从左到右平移，模拟目标移动
_FRAMES = [
    [(180, 240, 38), (320, 240, 38), (460, 240, 38)],
    [(200, 240, 38), (340, 240, 38), (480, 240, 38)],
    [(220, 240, 38), (360, 240, 38), (500, 240, 38)],
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, centers in enumerate(_FRAMES):
        path = os.path.join(OUT_DIR, f"frame_{i:03d}.png")
        cv2.imwrite(path, _ring_img(centers))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
