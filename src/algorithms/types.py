from dataclasses import dataclass

class QrFormatError(ValueError):
    pass

@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float  # deg

@dataclass(frozen=True)
class RingObs:
    center: tuple
    radius: float
    label: int | None = None
    # 环的种类："target"=参考靶（定位用）、"digit"=旧数字环、None=未标注。
    kind: str | None = None

@dataclass(frozen=True)
class TargetObs:
    """一个参考靶在**车身坐标系**（mm）下的观测。"""

    center: tuple
    ring_count: int = 0
    label: int | None = None   # 靶心数字识别结果；未识别为 None

@dataclass(frozen=True)
class TargetSetObs:
    """当前区域三个参考靶的观测，按「沿三点连线从一端到另一端」排序。"""

    points: tuple          # tuple[TargetObs, ...]，长度 3
    collinear: bool = True

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, i):
        return self.points[i]

@dataclass(frozen=True)
class StationObs:
    """转盘上一个工位的观测（颜色 + 在车身坐标系下的位置）。"""

    color: int
    center: tuple
    angle01: int   # 相对车头方向的角度，0.1°

@dataclass(frozen=True)
class TurntableObs:
    """转盘在**车身坐标系**下的观测（文档「表 1-9」的输入）。"""

    center: tuple
    radius: float
    angle01: int
    omega01: float
    stations: tuple            # tuple[StationObs, ...]，按工位序号排列
    status: int = 0

@dataclass(frozen=True)
class TaskPlanRaw:
    """二维码四组数字解析出的原始值（未做码垛配对）。"""

    batch1_colors: tuple
    batch1_positions: tuple
    batch2_colors: tuple
    batch2_positions: tuple

@dataclass(frozen=True)
class Blob:
    center: tuple
    area: float
    color: int
