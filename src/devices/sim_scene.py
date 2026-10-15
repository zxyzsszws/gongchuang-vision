"""sim 场景：纯合成 perceive 三键（不依赖相机/cv2 采集），供端到端联测与 CI。

- qr()：第 3 帧起返回竞赛二维码文本（Task 12 简报固定值）。
- tri()：每帧返回「与场地图理论位对齐 + ±3mm 高斯抖动」的 TriRingObs。
  抖动的误检风险由 FSM 自身防抖（stable_frames 连续稳定帧）与 CAL 态门控
  消解：只有残差/角度双过闸的帧才产生 CAL_APPLIED。
- rings()：sim 无环摆放任务，返回 []。

SimScene 支持 s["qr"] 下标访问（MatchFSM 以 perceive[key]() 约定调用）。
"""
import random

from ..algorithms.types import RingObs, TriRingObs

QR_TEXT = "452+321+254+312"


class SimScene:
    def __init__(self, field, qr_at_frame=3, jitter_mm=3.0, seed=42):
        self.field = field
        self._qr_at = qr_at_frame
        self._sigma = jitter_mm
        self._rng = random.Random(seed)  # 固定种子：e2e 行为可复现
        self._qr_calls = 0

    def qr(self):
        self._qr_calls += 1
        return QR_TEXT if self._qr_calls >= self._qr_at else None

    def tri(self):
        t = self.field.tri

        def jittered(r):
            center = (r.center[0] + self._rng.gauss(0.0, self._sigma),
                      r.center[1] + self._rng.gauss(0.0, self._sigma))
            return RingObs(center, r.radius)

        return TriRingObs(jittered(t.left), jittered(t.middle), jittered(t.right))

    def rings(self):
        return []

    def __getitem__(self, key):
        return getattr(self, key)
