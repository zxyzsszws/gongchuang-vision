"""live/replay 视觉感知：相机帧 → 检测 → 车速体毫米，供 FSM 使用。

## 坐标链路（v2.0.0）

    pixel ──[相机外参 image_to_robot]──> 车身毫米

**只做一次换算，停在车身坐标系。** v1.0.0 里还要再叠加「当前位姿」换到场地
坐标系，那是因为当时视觉要用观测反推位姿修正量；v2.0.0 里位姿正是视觉自己要
解算的东西，不可能反过来依赖它。所以这里换到车身毫米就交出去，由
``algorithms/localize.py`` 用三个靶解出车在场地中的位姿。

这个区别很关键：它意味着**相机外参只需标定一次**，与车走到哪无关；也意味着
外参里的平移量必须按「机械臂底座中心在地面的投影」为原点标定，而不是相机
光轴（文档附录二第四节）。

## 无外参时的行为

``targets()`` 返回 None，FSM 因此不发位姿（宁可让电控继续用里程推算，也不发
方向错位的测量结果）。``qr()`` 不受影响，可以正常扫码。
"""
import itertools
import math

from ..algorithms.ring_detect import detect_targets, order_targets
from ..algorithms.types import TargetObs as DomainTargetObs
from ..algorithms.types import TargetSetObs

# 选靶时最多参与组合的候选数：按「离预测最近」截断，避免画面里圆一多组合数爆炸。
_MAX_CANDIDATES = 8
# 挑出的靶离预测位置的最大允许偏差（mm）的缺省值；配置里可用
# ``target.max_match_mm`` 覆盖。超过就认为画面里没有当前区域的靶。
# 参考量级：两组靶相距约 1800mm，而电控停位偏差通常在几十毫米量级（里程漂移
# 8mm/m，走十米才 80mm），500mm 足够宽松，又能挡住「拿错组」。
_DEFAULT_MAX_MATCH_MM = 500.0


class VisionPerceive:
    """把相机帧包装成 FSM 需要的 ``qr / targets / turntable`` 三个可调用键。"""

    KEYS = ("qr", "targets", "turntable")

    def __init__(self, cam, cfg, calib=None, field=None, color_cfg=None):
        self._cam = cam
        self._cfg = cfg
        self._calib = calib
        self._field = field
        self._color_cfg = color_cfg or (cfg.get("color") or {})
        self._frame = None
        self._targets_cache = None
        target_cfg = cfg.get("target") or {}
        self._target_cfg = target_cfg

    # ---------- 帧与检出 ----------

    def _read(self):
        """取最新帧；新帧才让检测缓存失效（targets/turntable 共享一次检出）。"""
        ok, img = self._cam.read()
        if ok and img is not None:
            self._frame, self._targets_cache = img, None
        return self._frame

    def _detect(self):
        img = self._read()
        if img is None:
            return []
        if self._targets_cache is None:
            self._targets_cache = detect_targets(img, self._target_cfg)
        return self._targets_cache

    # ---------- FSM 接口 ----------

    def qr(self):
        from ..algorithms.qr_decode import decode_qr
        # 扫码走独立相机（若装配了 MultiCamera），否则回落主摄
        reader = getattr(self._cam, "qr_read", None)
        ok, img = reader() if reader is not None else self._cam.read()
        if not ok or img is None:
            return None
        model_dir = (self._cfg.get("camera") or {}).get("qr_model_dir")
        return decode_qr(img, model_dir)

    def has_extrinsics(self) -> bool:
        return self._calib is not None

    def targets(self, region=None):
        """当前区域三个参考靶在**车身坐标系**（mm）下的观测。

        三靶不全、不共线、或没有相机外参时返回 None —— 这几种情况下位姿解不
        出来或不唯一，FSM 会因此不发 0x0003。

        ``region`` 用来**选哪三个**：画面里可能同时混着另一组靶（车正好停两组
        之间）以及转盘上的圆。给了区域就用该区域的标称入场位姿预测三个靶在车身
        坐标系里该在什么位置，按距离挑最贴合的；预测只决定「挑哪三个」，位姿仍然
        完全由解算得出，不依赖这个先验。
        """
        if self._calib is None:
            return None
        px_obs = self._detect()
        if len(px_obs) < 3:
            return None

        body = []
        for t in px_obs:
            bx, by = self._calib.image_to_robot.apply(t.center)
            body.append((bx, by, t.ring_count))

        picked = self._pick_three(body, region)
        if picked is None:
            return None
        pts = [DomainTargetObs((bx, by), ring_count=n) for bx, by, n in picked]
        ordered, collinear = order_targets(pts)
        return TargetSetObs(tuple(ordered), collinear=collinear)

    # ---------- 选靶 ----------

    def _pick_three(self, body, region):
        """从若干候选圆心里挑出属于当前区域的三个靶；挑不出返回 ``None``。

        优先按区域预测位置挑（能排除另一组靶与转盘上的圆）；没有区域或没有场地图
        时退回「圈数多、离车近」的置信度排序。

        挑法是「在所有三元组里，取与三个预测位置整体最贴合的那一组」，贴合程度用
        双向最近距离之和衡量：预测点到三元组的距离、三元组到预测点的距离，两边都
        要小。只按「预测点各取最近候选」挑会在一侧预测偏得较多时把另一组的靶拉
        进来；只按「全局最近的三个候选」挑又可能三个全落在同一个预测点附近（靶旁
        多出一圈伪影时）。双向和同时要求「每个靶都有着落」和「每个预测点都被占
        住」，两种退化都能挡住。

        最后还要过一道**贴合门限**：挑出的三个靶只要有一个离它的预测位置超过
        ``target.max_match_mm``，就说明画面里根本没有当前区域的靶（车开过头、或
        提示给错），返回 None 让 FSM 不发位姿。这一步不能省——三个共线点用相似
        变换可以精确映射到任意另外三个共线点，**残差恒为 0，拦不住拿错组**，只能
        在选靶阶段用位置先验挡。

        预测只决定「挑哪三个」，位姿仍然完全由解算得出，不依赖这个先验。
        """
        predicted = self._predicted_body_points(region)
        if predicted is None:
            return sorted(body, key=lambda t: (-t[2], -(t[0] ** 2 + t[1] ** 2)))[:3]

        def near_predicted(item):
            return min(math.hypot(item[0] - px, item[1] - py) for px, py in predicted)

        # 先按「离预测最近」截到至多 8 个候选，三元组最多 56 种，20Hz 下毫无压力；
        # 不截的话画面里圆一多组合数会爆。
        cands = sorted(body, key=near_predicted)[:_MAX_CANDIDATES]

        best, best_cost, best_far = None, None, 0.0
        for combo in itertools.combinations(cands, 3):
            to_pred = [near_predicted(c) for c in combo]
            to_combo = [min(math.hypot(c[0] - px, c[1] - py) for c in combo)
                        for px, py in predicted]
            cost = sum(to_pred) + sum(to_combo)
            if best_cost is None or cost < best_cost:
                best, best_cost, best_far = combo, cost, max(to_pred)

        if best is None or best_far > self._max_match_mm():
            return None
        return list(best)

    def _max_match_mm(self):
        """选靶时允许的最大预测偏差；配置里可覆盖（现场可放宽到 800mm）。"""
        return float(self._target_cfg.get("max_match_mm",
                                          _DEFAULT_MAX_MATCH_MM))

    def _predicted_body_points(self, region):
        """用该区域的标称入场位姿，预测三个靶在车身坐标系下的位置（mm）。"""
        if self._field is None or region is None:
            return None
        group = self._field.group_for_region(region)
        if group is None:
            return None
        pose = group.entry_pose
        a = math.radians(pose.heading)
        ca, sa = math.cos(a), math.sin(a)
        out = []
        for t in group.targets:
            dx = t.center[0] - pose.x
            dy = t.center[1] - pose.y
            out.append((ca * dx + sa * dy, -sa * dx + ca * dy))
        return out

    def turntable(self, region=None):
        """转盘在车身坐标系下的观测。

        真正的转盘识别（圆心、半径、三工位角度与颜色）还需要按颜色分割与
        拟合圆，当前版本尚未实现，返回 None 让 FSM 停发 0x0004。
        这里显式返回 None 而不是编造数据：发一份假的转盘模型会让电控按错的
        伸出时刻去取料，比不取更糟。
        """
        return None

    def __getitem__(self, key):
        return getattr(self, key)
