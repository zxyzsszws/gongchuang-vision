"""视觉侧的领域数据类型（V2.2.0）。

V2.2.0 里视觉只回答三件事：任务计划、取料点的颜色、相机要挪多远才能对准主圆。
因此这里的类型只承载这三件事的输入输出，不再有「三个共线靶 → 位姿」那一套。
"""
from dataclasses import dataclass


class QrFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Pose:
    """一个位姿（mm / 度）。sim 里表示电控的真值与估计，不参与协议。"""

    x: float
    y: float
    heading: float  # deg


@dataclass(frozen=True)
class Blob:
    """一个颜色块（像素域），color_detect 的输出。"""

    center: tuple
    area: float
    color: int


@dataclass(frozen=True)
class BlobMm:
    """一个颜色块换到毫米之后的样子（毫米域）。

    ``radius`` 是**等效圆半径**：用面积换算，`r = sqrt(A/π)`。圆形物料这样换算
    与直接用最小外接圆一致；物料被画面边缘切掉一块时会偏小，但那属于测量缺陷，
    宁可让判定不通过（不报颜色），也不要去补一个猜出来的半径。
    """

    color: int
    center: tuple          # (x, y)，毫米，坐标系由调用方约定
    radius: float

    def distance_to(self, point):
        dx = self.center[0] - point[0]
        dy = self.center[1] - point[1]
        return (dx * dx + dy * dy) ** 0.5


@dataclass(frozen=True)
class TurntableFrame:
    """转盘的一帧观测，供停稳判定使用（毫米域）。

    ``t`` 是该帧的时刻（秒），停稳判定要靠它算「已经多久没动」。同一帧可以来自
    相机，也可以来自合成场景；判定策略只认这个结构，不关心数据来源。
    """

    t: float
    blobs: tuple           # tuple[BlobMm, ...]

    def __len__(self):
        return len(self.blobs)


@dataclass(frozen=True)
class TaskPlanRaw:
    """二维码四组数字解析出的原始值（未做码垛配对）。"""

    batch1_colors: tuple
    batch1_positions: tuple
    batch2_colors: tuple
    batch2_positions: tuple
