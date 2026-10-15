from dataclasses import dataclass, field
import yaml
from .types import RingObs

@dataclass(frozen=True)
class TriRingField:
    left: RingObs
    middle: RingObs
    right: RingObs
    base_angle_deg: float

@dataclass(frozen=True)
class FieldMap:
    tri: TriRingField
    rings: list = field(default_factory=list)

def load_field_map(path: str) -> FieldMap:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    t = raw["tri"]
    tri = TriRingField(
        RingObs(tuple(t["left"]["center"]), t["left"]["radius"]),
        RingObs(tuple(t["middle"]["center"]), t["middle"]["radius"]),
        RingObs(tuple(t["right"]["center"]), t["right"]["radius"]),
        float(t["base_angle_deg"]),
    )
    rings = [RingObs(tuple(r["center"]), r["radius"], r.get("label")) for r in raw.get("rings", [])]
    return FieldMap(tri=tri, rings=rings)
