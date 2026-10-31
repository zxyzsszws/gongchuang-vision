"""live/replay 视觉感知：相机帧 → 检测 → 坐标换算 → FSM 的 perceive 键。

链路（关键：让校准算法拿到**场地毫米**观测，而不是像素）：

    pixel ──[image_to_robot]──> 车体毫米 ──[当前位姿]──> 场地毫米

- ``image_to_robot``：相机安装外参，来自 ``camera_calib_file``。尺度可用
  ``auto_scale_from_tri`` 由三色环半径自估（距离自适应），安装角/偏移量尺可得。
- 位姿由 main 循环每拍注入 ``perceive.pose``，观测因此带「上报−真值」误差 ε，
  与 ``calibration.calibrate`` 的符号约定一致。
- **无外参时 tri() 返回 None**（宁可跳过校准，也不发方向错位的修正量）。

三色环在**像素域**分组（间距门限是像素），换算后交给校准；数字环换算后供
放置位置校核（Phase-4 精细偏移的输入）。
"""
from ..algorithms.geometry import compose_with_pose
from ..algorithms.ring_detect import detect_rings, detect_tri_ring
from ..algorithms.types import RingObs, TriRingObs


class VisionPerceive:
    """把相机帧包装成 FSM 需要的 ``qr / tri / rings`` 三个可调用键。"""

    def __init__(self, cam, cfg, calib=None, field=None):
        self._cam = cam
        self._cfg = cfg
        self._calib = calib
        self._field = field
        self.pose = None            # main 每拍注入（滤波后 Pose）
        self._frame = None
        self._rings = None
        ring_cfg = cfg.get("ring") or {}
        self._geo = ring_cfg.get("geo")
        self._hough = ring_cfg.get("hough")
        sr = ring_cfg.get("search_radius")
        self._search = (float(sr[0]), float(sr[1])) if sr else None
        # 三色环分组门限**以像素计**（现场对着实拍画面标一次）；缺省沿用旧键名，
        # 兼容历史配置与回放脚本。
        self._spacing = (float(ring_cfg.get("spacing_px_min",
                                            ring_cfg.get("spacing_min_mm", 120))),
                         float(ring_cfg.get("spacing_px_max",
                                            ring_cfg.get("spacing_max_mm", 200))))
        self._collinear = float(ring_cfg.get("collinear_tol_deg", 5))
        self._ratio = float(ring_cfg.get("max_spacing_ratio", 2.0))
        self._radius_tol = float(ring_cfg.get("radius_tol", 0.25))

    # ---------- 帧与检出 ----------

    def _read(self):
        """取最新帧；新帧才让圆环缓存失效（tri/rings 共享一次霍夫检出）。"""
        ok, img = self._cam.read()
        if ok and img is not None:
            self._frame, self._rings = img, None
        return self._frame

    def _detect(self):
        img = self._read()
        if img is None:
            return []
        if self._rings is None:
            self._rings = detect_rings(img, self._geo, self._hough, self._search)
        return self._rings

    # ---------- 坐标换算 ----------

    def _to_field(self, px_pt, calib):
        rb = calib.image_to_robot.apply(px_pt)
        return compose_with_pose(self.pose, [rb])[0]

    def _calib_for(self, px_tri):
        """本次换算用的外参；开启自估尺度时按本帧环半径更新。"""
        if self._calib is None:
            return None
        if self._field is None or not self._calib.auto_scale_from_tri:
            return self._calib
        try:
            return self._calib.with_auto_scale(px_tri, self._field.tri)
        except ValueError:
            return self._calib

    def has_extrinsics(self) -> bool:
        return self._calib is not None

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

    def tri(self):
        rings = self._detect()
        px_obs = detect_tri_ring(rings, self._spacing, self._collinear,
                                 self._ratio, self._radius_tol)
        if px_obs is None or self.pose is None:
            return None
        calib = self._calib_for(px_obs)
        if calib is None:
            return None  # 无外参：不发错修正量，FSM 会走 CAL_SKIPPED 降级
        return TriRingObs(
            RingObs(self._to_field(px_obs.left.center, calib), px_obs.left.radius, kind="tri"),
            RingObs(self._to_field(px_obs.middle.center, calib), px_obs.middle.radius, kind="tri"),
            RingObs(self._to_field(px_obs.right.center, calib), px_obs.right.radius, kind="tri"),
        )

    def rings(self):
        """全部环换算到场地毫米；无位姿/无外参时退回像素（仅调试与日志）。

        注意自动尺度（auto_scale_from_tri）：数字环半径与三色环不同，不能借用三色环
        自估的尺度，故此处只用配置里的静态 scale；精细放置属 Phase-4，另行处理。
        """
        rings = self._detect()
        if self.pose is None or self._calib is None:
            return list(rings)
        return [RingObs(self._to_field(r.center, self._calib), r.radius, r.label, kind=r.kind)
                for r in rings]

    def __getitem__(self, key):
        return getattr(self, key)
