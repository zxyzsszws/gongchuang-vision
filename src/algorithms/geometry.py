"""平面几何：相似变换、单应、以及相机外参的加载（纯函数层）。

## v2.0.0 里这层的职责变小了

v1.0.0 需要「像素 → 车体毫米 → 场地毫米」两层换算，因为视觉要拿观测去反推
位姿修正量，而修正量依赖观测的绝对坐标系。v2.0.0 里位姿本身就是视觉要解算
的未知量，不可能反过来依赖它，所以换算**只做一层**：像素 → 车身毫米。

因此 ``compose_with_pose``（车体毫米 → 场地毫米）已被删除，它只会诱使后来者
又把位姿循环地喂回感知层。位姿解算在 ``algorithms/localize.py``，那里的输入
是车身毫米坐标，输出是场地坐标下的位姿。

## 两种外参形式

- ``Similarity``：尺度 + 旋转 + 平移。相机光轴垂直地面、镜头无明显透视时够用，
  两点实测即可标定。
- ``Homography``：3×3 单应。相机有倾角或视场边缘畸变明显时更准，需四点以上。

两者都提供 ``apply(pt)``，可互换。
"""
import math
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass(frozen=True)
class Similarity:
    """相似变换（尺度 + 旋转 + 平移）：dst = scale·R(rot)·src + (tx, ty)。

    像素→车身毫米的缺省外参形式：scale 由「已知尺寸物」实测，rot_deg 为相机
    相对车体的安装角（图像 y 朝下，故通常需要 180° 或 ±90° 的翻转量级）。
    """
    scale: float = 1.0
    rot_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, pt):
        x, y = float(pt[0]), float(pt[1])
        c = self.scale * math.cos(math.radians(self.rot_deg))
        s = self.scale * math.sin(math.radians(self.rot_deg))
        return (c * x - s * y + self.tx, s * x + c * y + self.ty)

    def invert(self) -> "Similarity":
        """逆变换：车身毫米 → 像素（标定时由「已知 mm 点」反查图像位置用）。"""
        if self.scale == 0:
            raise ZeroDivisionError("similarity scale must be non-zero")
        rot = -self.rot_deg
        inv = 1.0 / self.scale
        c = inv * math.cos(math.radians(rot))
        s = inv * math.sin(math.radians(rot))
        # dst = A·src + t 的逆：src = A⁻¹·(dst − t)
        return Similarity(inv, rot, -(c * self.tx - s * self.ty),
                          -(s * self.tx + c * self.ty))


@dataclass(frozen=True, eq=False)
class Homography:
    """3×3 单应矩阵（透视变换）；eq=False 避免 ndarray 参与相等比较。"""
    matrix: np.ndarray

    def apply(self, pt):
        h = self.matrix
        x, y = float(pt[0]), float(pt[1])
        w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        if abs(w) < 1e-12:
            raise ZeroDivisionError("homography maps point to infinity")
        return ((h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w,
                (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w)


def fit_similarity(src, dst) -> Similarity:
    """最小二乘拟合相似变换（≥2 对点，≥3 更稳）：复数闭式解。

    c = Σ conj(z)·w / Σ |z|²（z/w 为去均值后的源/目标点），scale=|c|、rot=arg(c)、
    t = mean(dst) − c·mean(src)。
    """
    src_arr, dst_arr = np.asarray(src, float), np.asarray(dst, float)
    if src_arr.shape != dst_arr.shape or src_arr.ndim != 2 or src_arr.shape[0] < 2:
        raise ValueError("fit_similarity needs >=2 paired points of equal shape")
    ms, md = src_arr.mean(axis=0), dst_arr.mean(axis=0)
    zc = (src_arr[:, 0] - ms[0]) + 1j * (src_arr[:, 1] - ms[1])
    wc = (dst_arr[:, 0] - md[0]) + 1j * (dst_arr[:, 1] - md[1])
    denom = float((zc.real ** 2 + zc.imag ** 2).sum())
    if denom <= 0.0:
        raise ValueError("fit_similarity: degenerate source points (all identical)")
    c = complex((np.conjugate(zc) * wc).sum() / denom)
    t = (md[0] + 1j * md[1]) - c * (ms[0] + 1j * ms[1])
    return Similarity(scale=abs(c), rot_deg=math.degrees(math.atan2(c.imag, c.real)),
                      tx=t.real, ty=t.imag)


def fit_homography(src, dst) -> Homography:
    """标准 DLT（≥4 对点）：四点标定产物可直接落成 Homography。"""
    src_arr, dst_arr = np.asarray(src, float), np.asarray(dst, float)
    if src_arr.shape != dst_arr.shape or src_arr.ndim != 2 or src_arr.shape[0] < 4:
        raise ValueError("fit_homography needs >=4 paired points of equal shape")
    rows = []
    for (x, y), (u, v) in zip(src_arr, dst_arr):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, float))
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return Homography(h)


@dataclass(frozen=True)
class CameraCalib:
    """像素 → 车身毫米的外参。"""

    image_to_robot: object = Similarity()


def load_camera_calib(path: str) -> CameraCalib:
    """读相机外参 yaml。

    两种写法（二选一）：
        image_to_robot: {scale_mm_per_px: 1.2, rot_deg: 180.0, tx_mm: -320, ty_mm: -240}
        homography: [[...], [...], [...]]      # 四点标定，3×3

    车身坐标系原点必须是**机械臂底座中心在地面的投影**，不是相机光轴
    （文档附录二第四节）；两者差几十毫米且不会报错。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "homography" in raw:
        h = np.asarray(raw["homography"], float)
        if h.shape != (3, 3):
            raise ValueError(f"{path}: homography must be 3x3, got {h.shape}")
        return CameraCalib(Homography(h))
    if "image_to_robot" not in raw:
        raise ValueError(f"{path}: need either 'image_to_robot' or 'homography'")
    t = raw["image_to_robot"]
    return CameraCalib(Similarity(float(t.get("scale_mm_per_px", 1.0)),
                                  float(t.get("rot_deg", 0.0)),
                                  float(t.get("tx_mm", 0.0)),
                                  float(t.get("ty_mm", 0.0))))
