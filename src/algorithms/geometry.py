"""平面几何：像素 → 车体毫米 → 场地毫米（纯函数层，无硬件依赖）。

为什么必须两层换算：
- 相机固定在车上，像素先换到**车体坐标系**（mm）。只依赖安装外参，与车走到哪无关。
- 再叠加位姿换到**场地坐标系**：dst = R(heading)·p + (x, y)。航位推算误差 ε 由此
  进入观测，恰好满足 ``calibration.calibrate`` 的「obs = theory + ε」口径。

**外参不可省**（这条曾被我误判，记录在此防复发）：``calibrate`` 的修正量
``delta = w·(theory − obs)`` 依赖 obs 的**绝对坐标系**，不是平移不变的；只有
``residual`` 与 ``dhead`` 是相对量。把像素直接当毫米（或只做尺度缩放）会得到
方向/原点都错位的观测，算出的 delta 可以达到几米，而残差闸门与角度闸门未必能拦。
故：
- 有安装外参 → 走 ``image_to_robot`` 精确换算（推荐，赛前用标定工具一次标定）。
- 无外参 → live 模式**拒绝启动**（见 ``src/main.py``），绝不外发未校数据。

单应（Homography）与相似（Similarity）共用 ``apply`` 接口，可互换。
"""
import math
from dataclasses import dataclass, replace

import numpy as np
import yaml


@dataclass(frozen=True)
class Similarity:
    """相似变换（尺度 + 旋转 + 平移）：dst = scale·R(rot)·src + (tx, ty)。

    像素→车体毫米的缺省外参形式：scale 由「已知尺寸物」或环半径给，rot_deg 为
    相机相对车体的安装角（图像 y 朝下，故通常需要 180° 或 ±90° 的翻转量级）。
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
        """逆变换：车体毫米 → 像素（标定时由「已知 mm 点」反查图像位置用）。"""
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
    """标准 DLT（≥4 对点）：棋盘格标定产物可直接落成 Homography。"""
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


def compose_with_pose(pose, points):
    """车体毫米 → 场地毫米：dst = R(pose.heading)·p + (pose.x, pose.y)。

    这是航位推算误差进入观测的唯一入口：pose 带 ε 时 dst ≈ theory + ε。
    """
    a = math.radians(pose.heading)
    ca, sa = math.cos(a), math.sin(a)
    return [(ca * float(x) - sa * float(y) + pose.x,
             sa * float(x) + ca * float(y) + pose.y) for x, y in points]


def estimate_scale_from_tri(px_tri, field_tri) -> float:
    """用三色环**环半径**估计 mm/px（对旋转/平移不敏感，不需要已知位姿）。

    比「环间距」更稳：半径是单个环的局部量，分组间距受透视/形变影响更大。
    取三环半径均值与理论半径均值之比；退化时抛 ValueError。
    """
    px_r = (px_tri.left.radius + px_tri.middle.radius + px_tri.right.radius) / 3.0
    f = field_tri
    mm_r = (f.left.radius + f.middle.radius + f.right.radius) / 3.0
    if px_r <= 1e-9 or mm_r <= 1e-9:
        raise ValueError("estimate_scale_from_tri: degenerate radius")
    return mm_r / px_r


@dataclass(frozen=True)
class CameraCalib:
    """像素 → 车体毫米的外参（+ 尺度自估开关）。"""
    image_to_robot: object = Similarity()
    auto_scale_from_tri: bool = False

    def with_auto_scale(self, px_tri, field_tri) -> "CameraCalib":
        """按实测环半径重估尺度（仅相似变换有意义；单应自带透视尺度）。"""
        if not self.auto_scale_from_tri or not isinstance(self.image_to_robot, Similarity):
            return self
        scale = estimate_scale_from_tri(px_tri, field_tri)
        return CameraCalib(replace(self.image_to_robot, scale=scale), True)


def load_camera_calib(path: str) -> CameraCalib:
    """读相机外参 yaml。

    两种写法（二选一）：
        image_to_robot: {scale_mm_per_px: 1.2, rot_deg: 180.0, tx_mm: -320, ty_mm: -240}
        homography: [[...], [...], [...]]      # 棋盘格/四点标定，3×3
    可选 ``auto_scale_from_tri: true``：忽略 scale_mm_per_px，用三色环半径自估
    （距离变化自适应，赛前只需量安装角与偏移）。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    auto = bool(raw.get("auto_scale_from_tri", False))
    if "homography" in raw:
        h = np.asarray(raw["homography"], float)
        if h.shape != (3, 3):
            raise ValueError(f"{path}: homography must be 3x3, got {h.shape}")
        return CameraCalib(Homography(h), auto)
    if "image_to_robot" not in raw:
        raise ValueError(f"{path}: need either 'image_to_robot' or 'homography'")
    t = raw["image_to_robot"]
    return CameraCalib(Similarity(float(t.get("scale_mm_per_px", 1.0)),
                                  float(t.get("rot_deg", 0.0)),
                                  float(t.get("tx_mm", 0.0)),
                                  float(t.get("ty_mm", 0.0))), auto)
