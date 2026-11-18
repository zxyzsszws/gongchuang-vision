"""sim 场景：纯合成的 perceive（不依赖相机/cv2 采集），供端到端联测与 CI。

它按「电控真实位姿」把场地图里的参考靶与转盘投影到**车身坐标系**，再叠加
毫米级抖动——这正是真实视觉要做的事：视觉量到的是「靶在我车身坐标系下哪」，
与电控估计得准不准无关（电控不知道自己在哪，视觉自己解）。

关键设计：``pose_truth`` 是**真值**，不是电控的估计。所以 SimScene 与 MockMcu
的漂移模型解耦——电控漂得再厉害，视觉交给它的是真值，于是「电控收到位姿后
整体替换」这一步会把误差清零。这就是要验证的核心性质。

**观测顺序不携带编号。** 真实检测只能把三个靶排成「沿连线从一端到另一端」，
哪端是 1 号看不出来（这正是 180° 歧义）。这里复用
``ring_detect.order_targets`` 做同样的排序，因此 sim 与实拍走的是同一条代码路径；
若 sim 直接按编号排好，等于把答案喂给解算器，测不出歧义消解是否真的有效。

``read_digits=True`` 时模拟靶心数字识别成功（给每个观测附上编号），用于覆盖
「读到数字」这条更可靠的分支。
"""
import math
import random

from ..algorithms.ring_detect import order_targets
from ..algorithms.types import (Pose, StationObs, TargetObs, TargetSetObs,
                                TurntableObs)

QR_TEXT = "452+321+254+312"


class SimScene:
    """合成感知：``qr`` / ``targets`` / ``turntable`` 三个可调用键。"""

    KEYS = ("qr", "targets", "turntable", "arm_target")

    def __init__(self, field, qr_at_frame=3, jitter_mm=0.3, seed=42,
                 station_colors=(4, 5, 2), omega01=600, angle01=600,
                 truth_pose=None, auto_advance=True, read_digits=False,
                 arm_target_body=None, car_width_mm=620.0):
        self.field = field
        self._qr_at = qr_at_frame
        self._sigma = jitter_mm
        self._rng = random.Random(seed)
        self._qr_calls = 0
        self.station_colors = list(station_colors)
        self.omega01 = omega01          # 0.1°/s
        self._angle01 = angle01
        self.pose_truth = truth_pose or Pose(0.0, 0.0, 0.0)
        self.auto_advance = auto_advance
        self.read_digits = read_digits
        self.visible = True             # 测试可关掉，模拟离开靶区
        # 相机要看的那个目标点在车身坐标系下的位置（None = 画面里没有目标）
        self.arm_target_body = arm_target_body
        self.arm_width_mm = car_width_mm

    # ---------- 坐标系变换 ----------

    def _to_body(self, field_pt):
        """场地坐标 → 车身坐标（用真值位姿的逆变换）。"""
        a = math.radians(self.pose_truth.heading)
        ca, sa = math.cos(a), math.sin(a)
        dx = field_pt[0] - self.pose_truth.x
        dy = field_pt[1] - self.pose_truth.y
        return (ca * dx + sa * dy, -sa * dx + ca * dy)

    def _jitter(self):
        return self._rng.gauss(0.0, self._sigma)

    # ---------- perceive 三键 ----------

    def qr(self):
        self._qr_calls += 1
        return QR_TEXT if self._qr_calls >= self._qr_at else None

    def targets(self, region=None):
        """指定区域三个参考靶的观测；该区域无靶或不可见时返回 None。

        ``region`` 由 FSM 按去向提示 0x0005 传入——两组靶结构完全相同，
        没有这个提示就无法判断画面里的三个靶属于哪一组。
        """
        group = self.field.group_for_region(region) if region is not None else None
        if group is None or not self.visible:
            return None

        pts = []
        for t in group.targets:
            bx, by = self._to_body(t.center)
            pts.append(TargetObs((bx + self._jitter(), by + self._jitter()),
                                 ring_count=t.rings,
                                 label=t.label if self.read_digits else None))
        ordered, collinear = order_targets(pts)
        return TargetSetObs(tuple(ordered), collinear=collinear)

    def turntable(self):
        """转盘在车身坐标系下的观测；不可见或无转盘时返回 None。"""
        tt = self.field.turntable
        if tt is None or not self.visible:
            return None
        cx, cy = self._to_body(tt.center)
        stations = []
        for k in range(tt.station_count):
            ang01 = (self._angle01 + 1200 * k) % 3600
            a = math.radians(ang01 / 10.0)
            sx = cx + tt.station_radius * math.cos(a)
            sy = cy + tt.station_radius * math.sin(a)
            stations.append(StationObs(color=self.station_colors[k % 3],
                                       center=(sx + self._jitter(),
                                               sy + self._jitter()),
                                       angle01=ang01))
        return TurntableObs(center=(cx + self._jitter(), cy + self._jitter()),
                            radius=tt.station_radius,
                            angle01=self._angle01,
                            omega01=self.omega01,
                            stations=tuple(stations),
                            status=0x8F)

    # ---------- 场景推进 ----------

    def advance(self, dt):
        """按虚拟时钟推进转盘角度（视觉侧也要按 dt 走，才能跟上真实转动）。"""
        if self.auto_advance:
            self._angle01 = int((self._angle01 + self.omega01 * dt) % 3600)

    def arm_target(self, target_id=0):
        """相机在臂末端看下去，测到的目标点在**电机坐标系**下的位置。

        机位随臂动：臂 yaw 与伸出长度一变，同一个场地目标在电机坐标系里的坐标就
        变了。这里按真值反算——把目标点从场地系换到电机系（``_to_body`` 给的是
        车身系，电机系只是它沿 y 平移 width/2 再按整车平移约定取反）。

        测试用固定目标：``self.arm_target_body``（车身系）。没设就返回 None，
        表示画面里没看到目标——这时视觉必须报「没测到」，不能编一个坐标出来。
        """
        if self.arm_target_body is None:
            return None
        bx, by = self.arm_target_body
        # 车身系 → 电机系：原点在车身右边缘中点，车身在 y<0 一侧，
        # 电机系的 y 朝车外，故取负并减去车宽一半的偏移。
        return (bx, by + self.arm_width_mm / 2.0)

    def __getitem__(self, key):
        return getattr(self, key)
