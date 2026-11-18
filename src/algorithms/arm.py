"""主圆对准：目标点在**电机坐标系**下的位置 → 车沿自身 x 的位移量 d + 臂 yaw α。

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

## 「主圆」是什么

V2.2.0 里视觉不再解车身位姿。电控把臂伸到放置点附近之后，画面里会有一个主同心圆
（比周围其他圆都醒目、且离画面中心最近），视觉量它的圆心在像素里的偏移，换算成
毫米，解出「车该挪多少、臂该转多少」，于是相机就能正对主圆。

因此这里**不需要知道这个圆在场地上的编号**：视觉只回答「怎么把相机移到它正上方」，
至于它是 1 号还是 3 号靶、放什么颜色的物料，全部由电控按任务计划决定。

## 为什么要明确报「无解」而不是钳位

`|y| > l` 时 `asin` 的宗量超出 ±1，臂无论怎么转都够不到那个横向距离，只能挪车。
这时返回的 status 为 ``ST_UNSOLVABLE``，d 与 α 填 0 而不是钳到边界值：钳位会给出
一个「看起来能到、实际差一截」的答案，比明确报无解难查得多。

另一支解 `α = 180° − asin(y/l)` 数学上也满足方程，但它 |α| > 90°，臂会从车身里
穿过去，所以不用。

## d 是相对当前位置的增量

正的 d 表示沿车头方向前进，负的表示后退。视觉每次都从零算，不做跨帧累积；
电控执行完一次动作后，下一轮重新给 l 与 α 即可。

## 尺度从哪来（不需要电控上报相机高度）

`l` 与 `α` 只能定出相机相对转轴在哪、朝哪，还差一项像素到毫米的换算比例。
这一项视觉自己量：主圆的外圈直径是固定值，量出它在画面里的像素直径，
`mm/px = 外圈直径 ÷ 像素直径`（``mm_per_px_from_ring``）。画面里量不到主圆外圈
（例如它被裁掉一半）时退回标定时的固定 ``scale_mm_per_px``，此时状态字节仍报成功，
但质量档降一级——尺度是估的，不假装精确。
"""
import math
from dataclasses import dataclass

from ..core.protocol import ST_NOT_READY, ST_OK, ST_UNSOLVABLE

# 质量档 0~3，放在 status 之外的字段里（V2.2.0 的 status 只表示结果类别）。
GRADE_BAD, GRADE_MID, GRADE_GOOD, GRADE_BEST = 0, 1, 2, 3

# 外圈像素半径的下限：低于 2px 时相对误差被放大到不可接受，宁可退回标定值。
MIN_RING_RADIUS_PX = 2.0


@dataclass(frozen=True)
class ArmState:
    """电控报来的臂状态（请求载荷的 l 与 α）。

    ``l`` 是臂的水平伸出长度、``alpha_deg`` 是臂当前 yaw。相机装在臂末端且正对
    地面，这两项给出转轴到相机的平移与旋转；像素到毫米的尺度由视觉用主圆的外圈
    直径自行量出（见 ``mm_per_px_from_ring``），不需要电控上报相机高度。
    """

    l_mm: float
    alpha_deg: float


@dataclass(frozen=True)
class AlignSolution:
    """一次对准的解算结果。

    ``status`` 取协议的状态码（``ST_OK`` / ``ST_UNSOLVABLE`` / ``ST_NOT_READY``）；
    ``grade`` 是 0~3 的质量档，只在 ``status`` 为成功时才有意义。
    """

    d_mm: float = 0.0
    alpha_deg: float = 0.0
    status: int = ST_OK
    grade: int = GRADE_GOOD
    # 测得的主圆圆心在电机坐标系下的位置（mm），仅供日志与现场排查。
    target: tuple = (0.0, 0.0)

    @property
    def ok(self) -> bool:
        return self.status == ST_OK


def no_target_solution():
    """画面里没量到主圆：明确报「没准备好」，不要发一个假的坐标。

    用 ``ST_NOT_READY`` 而不是 ``ST_UNSOLVABLE``：没测到往往是相机还没伸到位、
    或者刚好被机械臂自己挡住，重发一次可能就好了，电控值得重试。
    """
    return AlignSolution(status=ST_NOT_READY, grade=GRADE_BAD)


def solve_align(target, l_mm, grade=GRADE_GOOD):
    """由主圆圆心解出 d 与 α。横向够不着时返回 ``ST_UNSOLVABLE``，不抛异常。

    「够不着」是正常的业务结果：目标偏在车身侧面之外，臂转不过去，得挪车。
    """
    l = float(l_mm)
    if l <= 0.0:
        return AlignSolution(status=ST_UNSOLVABLE, grade=GRADE_BAD,
                             target=(float(target[0]), float(target[1])))

    x, y = float(target[0]), float(target[1])
    if abs(y) > l:
        return AlignSolution(status=ST_UNSOLVABLE, grade=GRADE_BAD,
                             target=(x, y))

    alpha = math.asin(max(-1.0, min(1.0, y / l)))
    d = x - l * math.cos(alpha)
    return AlignSolution(d_mm=d, alpha_deg=math.degrees(alpha),
                         status=ST_OK, grade=grade, target=(x, y))


def mm_per_px_from_ring(ring_radius_px, ring_outer_mm):
    """由主圆外圈的像素半径反算尺度 mm/px。

    外圈直径是固定值（赛前量一次写进配置），所以「它在画面里有多大」就直接给出
    像素到毫米的换算比例：``mm/px = 外圈直径 ÷ 像素直径``。

    比让电控上报相机高度可靠：高度算尺度的前提是光轴严格垂直地面、镜头无畸变、
    高度值准确，三项任一不成立都会系统性偏；而用圆自己量，尺度与被测物在同一张
    画面里，倾斜与畸变的影响大部分相消。

    ``ring_radius_px`` 是检出到的**最外圈**半径（像素）。半径太小时相对误差会被
    放大，这种情况抛 ValueError 而不是返回一个虚高的精度。
    """
    r_px = float(ring_radius_px)
    if r_px < MIN_RING_RADIUS_PX:
        raise ValueError(f"外圈像素半径 {r_px} 太小，尺度不可信")
    outer_mm = float(ring_outer_mm)
    if outer_mm <= 0.0:
        raise ValueError(f"外圈实际直径必须为正，收到 {outer_mm}")
    return outer_mm / (2.0 * r_px)


def scale_offset(offset_px, mm_per_px):
    """把像素偏移换成毫米偏移（只缩放，不旋转）。

    ``offset_px`` 是「主圆圆心 − 相机光轴落点」的像素差。按安装角旋转到电机坐标
    系这一步由调用方做——这一层只负责长度换算，不猜朝向。
    """
    k = float(mm_per_px)
    return (float(offset_px[0]) * k, float(offset_px[1]) * k)
