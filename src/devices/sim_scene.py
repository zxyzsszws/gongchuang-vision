"""sim 场景：纯合成 perceive 三键（不依赖相机/cv2 采集），供端到端联测与 CI。

- qr()：第 3 帧起返回竞赛二维码文本（Task 12 简报固定值）。
- tri()：field-as-seen —— 把场地图理论环心经**当前位姿误差**（主控上报−标称真值，
  经 link.get_pose_report() 只读访问器取得）做刚体变换（绕中环旋转 err_head +
  平移 err_x/err_y），再叠加 ±3mm 高斯抖动。观测与机器人漂移真正耦合，
  校准闭环在 sim 端到端被激励（final-review F5）。
  抖动的误检风险由 FSM 自身防抖（stable_frames 连续稳定帧 + K 帧中位数）与
  CAL 态门控消解：只有残差/角度双过闸的帧才产生 CAL_APPLIED。
- link=None 时退回「与理论位对齐 + 抖动」的旧行为（无主控的纯场景测试用）。
- rings()：sim 无环摆放任务，返回 []。

SimScene 支持 s["qr"] 下标访问（MatchFSM 以 perceive[key]() 约定调用）。
"""
import math
import random

from ..algorithms.types import RingObs, TriRingObs

QR_TEXT = "452+321+254+312"


class SimScene:
    def __init__(self, field, qr_at_frame=3, jitter_mm=3.0, seed=42, link=None):
        self.field = field
        self._qr_at = qr_at_frame
        self._sigma = jitter_mm
        self._rng = random.Random(seed)  # 固定种子：e2e 行为可复现
        self._qr_calls = 0
        self._link = link  # MockMcu 引用（只读其位姿快照，绝不写）

    def qr(self):
        self._qr_calls += 1
        return QR_TEXT if self._qr_calls >= self._qr_at else None

    def tri(self):
        t = self.field.tri
        ex = ey = eh = 0.0
        if self._link is not None:
            rep = self._link.get_pose_report()
            ex, ey, eh = rep["err_x_mm"], rep["err_y_mm"], rep["err_head_deg"]
        # 刚体变换绕中环心（与 calibration 符号约定同一口径：obs = theory + ε）
        mx0, my0 = t.middle.center
        a = math.radians(eh)
        ca, sa = math.cos(a), math.sin(a)

        def seen(r):
            px, py = r.center[0] - mx0, r.center[1] - my0
            x = px * ca - py * sa + mx0 + ex
            y = px * sa + py * ca + my0 + ey
            return RingObs((x + self._rng.gauss(0.0, self._sigma),
                            y + self._rng.gauss(0.0, self._sigma)), r.radius)

        return TriRingObs(seen(t.left), seen(t.middle), seen(t.right))

    def rings(self):
        return []

    def __getitem__(self, key):
        return getattr(self, key)
