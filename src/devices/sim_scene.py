"""sim 场景：纯合成的 perceive（不依赖相机/cv2 采集），供端到端联测与 CI。

它按「东西相对相机光轴在哪」直接合成三件测量，不依赖电控的真值位姿——V2.2.0 的
视觉不解位姿，也就不需要知道车在哪。

## 三件测量各自合成什么

- ``qr``：第 ``qr_at_frame`` 次调用之后返回固定码；
- ``turntable``：转盘按 ``omega01`` 匀速转动，工位颜色固定；``stop_after_s``
  之后停住，模拟赛场上「转一会儿就停一段时间」；
- ``ring``：一个主圆，**像素域**观测（圆心像素 + 外圈像素半径 + 圈数）。

## 转盘为什么会正好停在取料点

赛场上的转盘是间歇机构：它转到位才停，停下时总有一个工位正对取料点。合成场景
照这个事实建模——起点角度由 ``stop_angle01 - omega01 * stop_after_s`` 反推，于是
停机那一刻角度恰好是 ``stop_angle01``（缺省 0），0 号工位正好落在取料点上。

这样「停稳判定」测的是真实情形：先转一阵，停住，然后被认出来；而不是一开始就
静止，让判定器白捡一个结果。

## 坐标系约定

主圆的圆心用**毫米**给出（``ring_body``，相对相机光轴），再按 ``mm_per_px``
折算成像素。这样「电控执行完 d 之后主圆应该落到画面中心」可以在测试里直接验证：
把 ``ring_body`` 设成 (0, 0)，FSM 解出的 d 与 α 就都应当是 0。

转盘色块同理，用毫米给出（相对光轴），因为颜色判定只用到距离。
"""
import math
import random

from ..algorithms.ring_detect import TargetObs
from ..algorithms.types import Blob, TurntableFrame
from ..algorithms.turntable import blobs_in_mm

QR_TEXT = "452+321+254+312"

# 物料直径（mm）。赛题物料是直径约 50mm 的圆柱，取 50 作为缺省。
MATERIAL_DIAMETER_MM = 50.0

# 合成画面的尺寸（与 config/default.yaml 的 camera 段一致）
FRAME_W, FRAME_H = 640, 480


class SimScene:
    """合成感知：``qr`` / ``turntable`` / ``ring`` 三个可调用键。"""

    KEYS = ("qr", "turntable", "ring")

    def __init__(self, qr_at_frame=3, jitter_mm=0.3, seed=42,
                 station_colors=(4, 5, 2), omega01=600, angle01=0,
                 station_radius_mm=150.0, stop_after_s=None, stop_angle01=0,
                 ring_body=(120.0, 80.0), ring_radius_px=60.0,
                 ring_rings=4, mm_per_px=1.0):
        self._qr_at = qr_at_frame
        self._sigma = jitter_mm
        self._rng = random.Random(seed)
        self._qr_calls = 0
        self.station_colors = list(station_colors)
        self.omega01 = float(omega01)          # 0.1°/s
        self.station_radius_mm = float(station_radius_mm)
        self.stop_after_s = stop_after_s       # None = 一直转
        # 转盘圆心放在光轴一侧，于是「角度为 0 的工位」正好落在光轴（取料点）上
        self.center = (-self.station_radius_mm, 0.0)
        if stop_after_s is None:
            self._angle01 = float(angle01)
        else:
            # 反推起点，使停机时角度恰好是 stop_angle01
            self._angle01 = float(stop_angle01 - self.omega01 * stop_after_s)
        # 主圆圆心（相对相机光轴，mm）；None = 画面里没有主圆
        self.ring_body = ring_body
        self.ring_radius_px = float(ring_radius_px)
        self.ring_rings = int(ring_rings)
        self.mm_per_px = float(mm_per_px)
        self.now = 0.0
        self.frame_size = lambda: (FRAME_W, FRAME_H)

    # ---------- 内部 ----------

    def _jitter(self):
        return self._rng.gauss(0.0, self._sigma)

    # ---------- perceive 三键 ----------

    def qr(self):
        self._qr_calls += 1
        return QR_TEXT if self._qr_calls >= self._qr_at else None

    def ring(self):
        """主圆的像素域观测；``ring_body`` 为 None 表示画面里没有主圆。

        图像 y 朝下，所以毫米域的 +y（朝车外）对应像素的 −y。
        """
        if self.ring_body is None:
            return None
        bx, by = self.ring_body
        k = self.mm_per_px
        cx = FRAME_W / 2.0 + (bx + self._jitter()) / k
        cy = FRAME_H / 2.0 - (by + self._jitter()) / k
        return TargetObs((cx, cy), self.ring_radius_px, self.ring_rings)

    def turntable(self):
        """转盘那一帧的色块（毫米，原点在相机光轴落点）与尺度。

        转盘停住后位置不再变化，「连续几帧不动」这条策略自然能判出停稳。
        """
        out = []
        r = self.station_radius_mm
        for k, color in enumerate(self.station_colors):
            ang01 = (self._angle01 + 1200 * k) % 3600
            a = math.radians(ang01 / 10.0)
            out.append(Blob(center=(self.center[0] + r * math.cos(a)
                                    + self._jitter(),
                                    self.center[1] + r * math.sin(a)
                                    + self._jitter()),
                            area=math.pi * (MATERIAL_DIAMETER_MM / 2.0) ** 2,
                            color=color))
        # 合成场景里色块本来就在毫米域，换算比例取 1
        return tuple(blobs_in_mm(out, 1.0, origin_px=(0.0, 0.0))), 1.0

    # ---------- 场景推进 ----------

    def advance(self, dt):
        """推进虚拟时钟与转盘角度；过了 ``stop_after_s`` 就停住。

        跨过停机时刻的那一步只按**剩下的那段时间**转：``min(dt, 剩余)``。
        写成 ``stop_after_s - before`` 会把整段停机前的角度一次性补上，转盘等于
        在第一步就跳到终点——那样「先转后停」根本没被测到。
        """
        before = self.now
        self.now += float(dt)
        if self.stop_after_s is None:
            eff = float(dt)
        elif before >= self.stop_after_s:
            eff = 0.0
        else:
            eff = max(0.0, min(float(dt), self.stop_after_s - before))
        self._angle01 = (self._angle01 + self.omega01 * eff) % 3600

    def frame(self):
        """把当前转盘状态打成一帧 ``TurntableFrame``，供停稳判定使用。"""
        blobs, _k = self.turntable()
        return TurntableFrame(t=self.now, blobs=blobs)

    def __getitem__(self, key):
        return getattr(self, key)
