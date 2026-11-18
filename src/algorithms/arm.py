"""机械臂目标解算：目标点在**电机坐标系**下的位置 → 车沿自身 x 的位移量 d + 臂 yaw α。

## 电机坐标系

原点在**车身右侧边的中点**，也就是臂的转轴，单位 mm：

- +x 沿车头方向
- +y 指向车身外侧（俯视图里画面右）
- 车身本体落在 y < 0 一侧，x ∈ [−车长/2, +车长/2]

臂的水平伸出长度 l 比赛中固定死，所以臂末端距转轴恒为 l。臂 yaw α 从 +x 量起，
向 +y 为正。末端位置就是 `(l·cos α, l·sin α)`。

要让末端落在目标 (x, y) 上：

```
α = asin(y / l)          横向位置只能靠臂转出来
d = x − l·cos α          前后位置靠车沿自身 x 挪
```

## 为什么要明确报「无解」而不是钳位

`|y| > l` 时 `asin` 的宗量超出 ±1，臂无论怎么转都够不到那个横向距离，只能挪车。
这时返回的 status 带 ``OUTCOME_UNREACHABLE``，d 与 α 填 0 而不是钳到边界值：
钳位会给出一个「看起来能到、实际差一截」的答案，比明确报无解难查得多。

另一支解 `α = 180° − asin(y/l)` 数学上也满足方程，但它 |α| > 90°，臂会从车身里
穿过去，所以不用。当解出的 |α| 接近 90° 时置告警位，提醒电控这一帧动作余量很小。

## d 是相对当前位置的增量

正的 d 表示沿车头方向前进，负的表示后退。视觉每次都从零算，不做跨帧累积；
电控执行完一次动作后，下一轮重新给 l 与 α 即可。

## 尺度从哪来（不需要电控上报相机高度）

`l` 与 `α` 只能定出相机相对转轴在哪、朝哪，还差一项像素到毫米的换算比例。
这一项视觉自己量：参考靶的外圈直径是固定值，量出它在画面里的像素直径，
`mm/px = 外圈直径 ÷ 像素直径`（``mm_per_px_from_ring``）。台面上没有靶时
（例如对着转盘取料）沿用标定时的固定 ``scale_mm_per_px``。
"""
import math
from dataclasses import dataclass

# status 字节（协议文档「表 1-8」）
OUTCOME_OK = 0            # 解算成功
OUTCOME_UNREACHABLE = 1   # 横向够不着（|y| > l）
OUTCOME_NO_TARGET = 2     # 画面里没测到目标
OUTCOME_MASK = 0x03

FLAG_SWEEP = 0x04         # |α| > 90°，臂会扫过车身
GRADE_SHIFT = 4           # bit4~5：质量等级 0~3
FLAG_USABLE = 0x80        # bit7：本帧可用

# 质量等级
GRADE_BAD, GRADE_MID, GRADE_GOOD, GRADE_BEST = 0, 1, 2, 3


@dataclass(frozen=True)
class ArmState:
    """电控发来的臂状态（协议 0x0006）。

    ``l`` 是臂的水平伸出长度、``alpha_deg`` 是臂当前 yaw。相机装在臂末端且正对
    地面，这两项给出转轴到相机的平移与旋转；像素到毫米的尺度由视觉用参考靶的外圈
    直径自行量出（见 ``mm_per_px_from_ring``），不需要电控上报相机高度。
    """

    l_mm: float
    alpha_deg: float
    seq: int = 0


@dataclass(frozen=True)
class ArmSolution:
    """一帧解算结果（协议 0x0008）。"""

    d_mm: float
    alpha_deg: float
    status: int
    target: tuple = (0.0, 0.0)   # 测得的目标点在电机坐标系下的 (x, y)，mm
    echo_seq: int = 0            # 对应 0x0006 的 seq

    @property
    def usable(self) -> bool:
        return bool(self.status & FLAG_USABLE)


def encode_status(outcome, grade=GRADE_GOOD, sweeps=False):
    """拼 status 字节。只有解算成功才置「可用」位。"""
    status = (outcome & OUTCOME_MASK) | ((grade & 0x03) << GRADE_SHIFT)
    if sweeps:
        status |= FLAG_SWEEP
    if outcome == OUTCOME_OK:
        status |= FLAG_USABLE
    return status


def solve_arm(target, l_mm, echo_seq=0, grade=GRADE_GOOD):
    """由目标点解出 d 与 α。

    target：目标在电机坐标系下的 ``(x, y)``，mm
    l_mm：  臂的水平伸出长度，mm（电控在 0x0006 里给出）

    返回 ``ArmSolution``。横向够不着时返回 outcome 为
    ``OUTCOME_UNREACHABLE`` 的解，而不是抛异常——「够不着」是正常的业务结果。
    """
    l = float(l_mm)
    if l <= 0.0:
        raise ValueError(f"水平伸出长度必须为正，收到 {l}")

    x, y = float(target[0]), float(target[1])
    if abs(y) > l:
        return ArmSolution(0.0, 0.0,
                           encode_status(OUTCOME_UNREACHABLE, GRADE_BAD),
                           (x, y), echo_seq)

    alpha = math.asin(max(-1.0, min(1.0, y / l)))
    alpha_deg = math.degrees(alpha)
    d = x - l * math.cos(alpha)
    return ArmSolution(d, alpha_deg,
                       encode_status(OUTCOME_OK, grade,
                                     sweeps=abs(alpha_deg) > 90.0),
                       (x, y), echo_seq)


def no_target_solution(echo_seq=0):
    """没测到目标时的解算结果：明确报「没测到」，不要发一个假的坐标。"""
    return ArmSolution(0.0, 0.0, encode_status(OUTCOME_NO_TARGET, GRADE_BAD),
                       (0.0, 0.0), echo_seq)


def mm_per_px_from_ring(ring_radius_px, ring_outer_mm):
    """由参考靶外圈的像素半径反算尺度 mm/px。

    参考靶的外圈直径是固定值（赛前量一次写进配置），所以「它在画面里有多大」
    就直接给出像素到毫米的换算比例：``mm/px = 外圈直径 ÷ 像素直径``。

    比让电控上报相机高度可靠：高度算尺度的前提是光轴严格垂直地面、镜头无畸变、
    高度值准确，三项任一不成立都会系统性偏；而用靶自己量，尺度与被测物在同一张
    画面里，倾斜与畸变的影响大部分相消。

    ``ring_radius_px`` 是检出到的**最外圈**半径（像素）。半径太小（远小于几像素）
    时相对误差会被放大，这种情况抛 ValueError 而不是返回一个虚高的精度。
    """
    r_px = float(ring_radius_px)
    if r_px < 2.0:
        raise ValueError(f"外圈像素半径 {r_px} 太小，尺度不可信")
    outer_mm = float(ring_outer_mm)
    if outer_mm <= 0.0:
        raise ValueError(f"外圈实际直径必须为正，收到 {outer_mm}")
    return outer_mm / (2.0 * r_px)


def scale_target(target_px, center_px, mm_per_px):
    """把像素域的目标点绕画面中心缩放到毫米域，得到电机坐标系下的 (x, y)。

    ``target_px`` / ``center_px`` 都是 ``(x, y)`` 像素坐标；后者是**转轴在画面里的
    位置**（即相机光轴落点，由标定给出）。结果仍以画面坐标系为准，调用方再按安装
    角旋转到电机坐标系——这一层只负责长度换算，不猜朝向。
    """
    k = float(mm_per_px)
    tx, ty = float(target_px[0]), float(target_px[1])
    cx, cy = float(center_px[0]), float(center_px[1])
    return ((tx - cx) * k, (ty - cy) * k)
