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
电控执行完一次动作后，下一轮重新给 l、h、α 即可。
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
    """电控发来的臂状态（协议 0x0006），也是相机的安装状态。

    ``l`` 是臂的水平伸出长度、``h`` 是相机离地高度、``alpha_deg`` 是臂当前 yaw。
    相机装在臂末端且正对地面，所以这三项决定了「像素里的点对应电机坐标系哪里」：
    l 与 α 给出转轴到相机的平移与旋转，h 给出尺度。
    """

    l_mm: float
    h_mm: float
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
