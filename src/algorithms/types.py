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
    # 环的种类："tri"=三色环（校准用）、"digit"=数字环（放置位置用）、None=未标注。
    # 由 detect_rings 按配置的半径带打标；标定/校准只认 "tri"，避免两类环串味。
    kind: str | None = None

@dataclass(frozen=True)
class TriRingObs:
    left: RingObs
    middle: RingObs
    right: RingObs

@dataclass(frozen=True)
class TaskPlan:
    batch1_colors: tuple
    batch1_positions: tuple
    batch2_colors: tuple
    batch2_positions: tuple

@dataclass(frozen=True)
class Blob:
    center: tuple
    area: float
    color: int
