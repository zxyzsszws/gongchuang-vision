"""转盘取料点：判定「物料已经停稳」并认出它的颜色（毫米域，纯函数层）。

## 这件事为什么不能靠「转盘在转/不转」

电控知道转盘什么时候停，但视觉不知道，也不该被告知——停机时刻是电控的时序，
视觉只需要看画面。所以判定完全靠画面：**同一块物料连续若干帧都不动，就算停稳**。

## 两种判定策略（接口一致，可配置切换）

用户提的「边界圆」与「连续几帧不动」是两种思路，现场可能各有优劣，所以两种都
实现，用配置项 ``turntable.stop_strategy`` 选择：

- ``motion``：最近取料点的那块物料，连续 ``hold_s`` 秒内位移都小于 ``tol_mm``，
  并且它离取料点不超过 ``pick_tol_mm``，就算停稳。这是通用做法，对停机位置没有
  假设。缺点是转盘爬行（极慢地转）时可能被误判成停稳——所以 ``tol_mm`` 要比
  爬行速度乘上 ``hold_s`` 大得多。
- ``containment``：画面中心（或标定给出的取料点）画一个比物料略大的边界圆，
  某块物料**完全落进边界圆内部**就算到位。这是用户画的那张图的思路：它直接检查
  「位置对不对」，不关心速度，因此对爬行不敏感；代价是要求转盘停位精度高、
  边界圆半径标得准。

两者都**只回答颜色**，不回答角度、不回答什么时候停——电控要的是「这次能取什么
颜色」，其余都由它自己规划。

## 为什么认不出就返回 None

``None`` 表示「此刻还判断不出」，不是「没有物料」。FSM 会继续等，等到自己的
超时上限再回一帧「没准备好」。绝不能返回一个猜的颜色：取错颜色整个码垛都错，
而且不会报任何错。
"""
import math

from .types import BlobMm, TurntableFrame

# 停稳判定的缺省值。物料直径约 50mm，工位半径 150mm、6s 一转时，取料点附近的
# 线速度约 150mm/s，即 20ms 一帧走 3mm 左右。所以判据不能看「相邻两帧差多少」
# ——3mm 的位移永远小于任何能容忍抖动的门限，转着的转盘会被判成停稳。
# 这里改成看「相对计时起点的累计位移」：转着的转盘 250ms 就漂出 38mm，一定超限；
# 真停住时只剩标定与检测的抖动（亚毫米级）。
DEFAULT_TOL_MM = 8.0
DEFAULT_HOLD_S = 0.25
DEFAULT_PICK_TOL_MM = 25.0
# 边界圆比物料大多少：留 15% 余量，太小会因抖动一直判不过，太大则相邻工位也满足。
DEFAULT_CONTAINMENT_MARGIN = 1.15


def _nearest(blobs, point):
    """离 ``point`` 最近的色块；没有色块时返回 None。"""
    best, best_d = None, None
    for b in blobs:
        d = b.distance_to(point)
        if best_d is None or d < best_d:
            best, best_d = b, d
    return best


class SettleByMotion:
    """连续若干帧不动就算停稳（``stop_strategy: motion``）。

    判据是**相对计时起点的累计位移**，不是相邻两帧的位移。原因见 ``DEFAULT_TOL_MM``
    的注释：慢速转动时单帧位移很小，只有累计量才能把「转」与「停」分开。
    """

    name = "motion"

    def __init__(self, pick_point, tol_mm=DEFAULT_TOL_MM,
                 hold_s=DEFAULT_HOLD_S, pick_tol_mm=DEFAULT_PICK_TOL_MM):
        self.pick_point = (float(pick_point[0]), float(pick_point[1]))
        self.tol_mm = float(tol_mm)
        self.hold_s = float(hold_s)
        self.pick_tol_mm = float(pick_tol_mm)
        self._anchor = None        # 计时起点的位置（累计位移的基准）
        self._still_since = None   # 从哪一刻开始不动的

    def reset(self):
        self._anchor = None
        self._still_since = None

    def feed(self, frame: TurntableFrame):
        """喂一帧；判定停稳时返回颜色（1~6），否则返回 None。"""
        near = _nearest(frame.blobs, self.pick_point)
        if near is None:
            self.reset()           # 画面里没东西，之前的「静止」不再成立
            return None

        if (near.distance_to(self.pick_point) > self.pick_tol_mm
                or near.color <= 0):
            self.reset()           # 还没转到取料点，或者颜色认不出
            return None

        if self._anchor is None:
            self._anchor = near.center
            self._still_since = frame.t
        elif near.distance_to(self._anchor) > self.tol_mm:
            # 相对起点已经漂出容差：它还在动，计时重来
            self._anchor = near.center
            self._still_since = frame.t

        if frame.t - self._still_since >= self.hold_s - 1e-9:
            return near.color
        return None


class SettleByContainment:
    """物料完全落进取料点的边界圆内部就算到位（``stop_strategy: containment``）。

    边界圆半径取 ``material_radius_mm * margin``：比物料略大，于是「完全进入」
    等价于「圆心与取料点的偏差不超过 (margin-1)·r」，这个量级正好在停位精度之内，
    同时相邻工位（相隔 120°、距离约 2.6 倍工位半径）绝不可能落进来。
    """

    name = "containment"

    def __init__(self, pick_point, material_radius_mm,
                 margin=DEFAULT_CONTAINMENT_MARGIN):
        self.pick_point = (float(pick_point[0]), float(pick_point[1]))
        self.material_radius_mm = float(material_radius_mm)
        self.margin = float(margin)

    def reset(self):
        pass                        # 无状态：每帧独立判断

    @property
    def boundary_radius_mm(self):
        return self.material_radius_mm * self.margin

    def feed(self, frame: TurntableFrame):
        """完全落进边界圆内部就返回颜色，否则 None。"""
        limit = self.boundary_radius_mm
        for b in frame.blobs:
            if b.color <= 0:
                continue
            if b.distance_to(self.pick_point) + b.radius <= limit:
                return b.color
        return None


def make_strategy(cfg, pick_point, material_radius_mm):
    """按配置造一个判定器；配置不合法时抛 ValueError（早暴露好过静默取错）。"""
    cfg = cfg or {}
    name = cfg.get("stop_strategy", "motion")
    if name == "motion":
        return SettleByMotion(
            pick_point,
            tol_mm=float(cfg.get("tol_mm", DEFAULT_TOL_MM)),
            hold_s=float(cfg.get("hold_s", DEFAULT_HOLD_S)),
            pick_tol_mm=float(cfg.get("pick_tol_mm", DEFAULT_PICK_TOL_MM)),
        )
    if name == "containment":
        return SettleByContainment(
            pick_point, material_radius_mm,
            margin=float(cfg.get("containment_margin",
                                 DEFAULT_CONTAINMENT_MARGIN)),
        )
    raise ValueError(
        f"turntable.stop_strategy 只能是 motion 或 containment，收到 {name!r}")


def blobs_in_mm(blobs_px, mm_per_px, origin_px=(0.0, 0.0)):
    """把像素色块换成毫米色块（绕 ``origin_px`` 缩放）。

    ``origin_px`` 是取料点在画面里的位置：转盘取料时相机固定悬停，取料点就是
    画面中心附近的一个固定点，由标定给出。等距缩放即可，不需要旋转——颜色判定
    只用到距离，方向由 ``pick_point`` 自己表达。
    """
    k = float(mm_per_px)
    ox, oy = float(origin_px[0]), float(origin_px[1])
    out = []
    for b in blobs_px:
        area = max(0.0, float(b.area))
        r_mm = math.sqrt(area / math.pi) * k
        out.append(BlobMm(color=int(b.color),
                          center=((b.center[0] - ox) * k,
                                  (b.center[1] - oy) * k),
                          radius=r_mm))
    return tuple(out)
