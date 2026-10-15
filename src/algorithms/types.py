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
