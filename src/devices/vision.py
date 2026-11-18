"""live/replay 视觉感知：相机帧 → 检测 → 三件测量结果（V2.2.0）。

## 只做三件事

V2.2.0 里视觉不再主动上报任何东西，也不再解车身位姿。电控问什么就答什么：

| 任务码 | 要什么 | 这一层提供什么 |
|---|---|---|
| 1 | 扫码 | ``qr()`` |
| 2 | 转盘取料点的颜色 | ``turntable()``（一帧画面里的色块 + 停稳判定） |
| 3 | 对准主圆 | ``ring()``（主圆像素观测，含外圈半径） |

## 坐标系：只用到外参里的一个数

    pixel ──[只有 scale_mm_per_px]──> 毫米

V2.2.0 的像素换毫米**只需要尺度**，不需要平移与旋转：

- 图像原点 = **画面中心**（即光轴落点），像素偏移直接减 `(w/2, h/2)`；
- 光轴在地面上的绝对位置由电控报来的 `l` 与 `α` 给出（任务码 3），不必从外参读；
- 相机随臂转过的角度也是 `α`，同样不必从外参读。

所以外参文件里 `tx_mm` / `ty_mm` / `rot_deg` 在当前实现中**不参与计算**，只有
`scale_mm_per_px` 起作用（而且它只是兜底：对准时优先用主圆外圈实测尺度）。
标定时不必纠结把原点对准转轴——那个偏移在当前协议下没有字段承载。

## 无外参时的行为

| 任务 | 无外参时 | 为什么 |
|---|---|---|
| 1 扫码 | 正常 | 扫码与尺度无关 |
| 2 转盘颜色 | 回 `ST_NOT_READY` | 转盘上没有已知尺寸物，尺度只能来自外参 |
| 3 对准主圆 | **仍可正常** | 尺度从主圆外圈实测反算（`target.ring_outer_mm`）；只有外圈量不到时才需要外参兜底，那时回 `ST_NOT_READY` |

所以外参不是「不填就不许跑」，而是「转盘那条路和兜底尺度要用它」。没有外参时
宁可回「没准备好」，也不拿一个凭空假设的 mm/px 去算——那种结果大小全错、形状
与方向却正常，电控照着挪只会越挪越偏，还不报错。
"""
from ..algorithms.color_detect import detect_colors
from ..algorithms.ring_detect import detect_targets, pick_main
from ..algorithms.turntable import blobs_in_mm


class VisionPerceive:
    """把相机帧包装成 FSM 需要的 ``qr / turntable / ring`` 三个可调用键。"""

    KEYS = ("qr", "turntable", "ring")

    def __init__(self, cam, cfg, calib=None, color_cfg=None):
        self._cam = cam
        self._cfg = cfg
        self._calib = calib
        self._color_cfg = color_cfg or (cfg.get("color") or {})
        self._frame = None
        self._targets_cache = None
        self._blobs_cache = None
        target_cfg = cfg.get("target") or {}
        self._target_cfg = target_cfg

    # ---------- 帧与检出 ----------

    def _read(self):
        """取最新帧；新帧才让检测缓存失效（ring/turntable 共享一次检出）。"""
        ok, img = self._cam.read()
        if ok and img is not None:
            self._frame = img
            self._targets_cache = None
            self._blobs_cache = None
        return self._frame

    def _detect(self):
        img = self._read()
        if img is None:
            return []
        if self._targets_cache is None:
            self._targets_cache = detect_targets(img, self._target_cfg)
        return self._targets_cache

    def _detect_blobs(self):
        img = self._read()
        if img is None:
            return []
        if self._blobs_cache is None:
            self._blobs_cache = detect_colors(
                img,
                self._color_cfg.get("hsv_table") or {},
                float(self._color_cfg.get("min_area", 200.0)))
        return self._blobs_cache

    @property
    def mm_per_px(self):
        """像素 → 毫米的**标定**尺度；没有外参时返回 ``None``。

        对准主圆时这个值会被「外圈实测」覆盖（见 ``ring``）；转盘取料时画面里
        没有已知尺寸物，只能用标定值。

        没有外参时返回 ``None`` 而不是 1.0：凭空假设的尺度会让毫米坐标大小全错，
        而形状与方向都正常——电控照着挪会越挪越偏，且不报错。调用方据此回
        「没准备好」，比发一个看着正常的错数安全得多。
        """
        if self._calib is None:
            return None
        scale = getattr(self._calib.image_to_robot, "scale", None)
        if scale is None or float(scale) <= 0:
            return None
        return float(scale)

    def frame_size(self):
        """画面尺寸 ``(w, h)``；还没读到帧时返回 None。"""
        img = self._read()
        if img is None:
            return None
        return (int(img.shape[1]), int(img.shape[0]))

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

    def ring(self):
        """画面里**离中心最近**的那个同心圆（主圆），像素域观测；没有返回 None。

        返回的是 ``ring_detect.TargetObs``，调用方要用 ``radius_px`` 反算尺度、
        用 ``center`` 算像素偏移。挑法见 ``pick_main``。

        这里**不检查外参**：对准的尺度是从主圆外圈实测反算的（见 FSM 的
        ``_align_scale``），外参只是外圈量不到时的兜底。这条路上外参缺失不影响
        正常工作。
        """
        obs = self._detect()
        if not obs:
            return None
        size = self.frame_size()
        if size is None:
            return None
        center_px = (size[0] / 2.0, size[1] / 2.0)
        return pick_main(obs, center_px,
                         min_rings=int(self._target_cfg.get("min_rings", 2)))

    def turntable(self):
        """转盘那一帧的色块，已换到以画面中心为原点的毫米坐标。

        返回 ``(blobs_mm, mm_per_px)``。转盘上没有已知尺寸的参考物，尺度**只能**
        来自相机外参，所以没有外参时返回 ``(None, None)``：此时给出的毫米坐标
        大小全错，而停稳判定与「离取料点多远」都依赖它。相机没帧时同样返回
        ``(None, None)``。

        停稳判定与颜色选择在 ``algorithms/turntable.py``，这一层只提供数据——
        判定策略要能按配置切换，把它写进设备层会让两种策略共享一堆隐藏状态。
        """
        k = self.mm_per_px
        if k is None:
            return None, None
        img = self._read()
        if img is None:
            return None, None
        blobs = self._detect_blobs()
        size = self.frame_size()
        center_px = (size[0] / 2.0, size[1] / 2.0)
        return blobs_in_mm(blobs, k, origin_px=center_px), k

    def __getitem__(self, key):
        return getattr(self, key)
