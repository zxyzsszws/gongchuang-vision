import pytest
from src.algorithms.types import Pose, RingObs, TriRingObs, QrFormatError
from src.algorithms.fieldmap import load_field_map

def test_pose_frozen():
    p = Pose(1.0, 2.0, 30.0)
    with pytest.raises(Exception):
        p.x = 5.0

def test_load_field_example():
    fm = load_field_map("config/field.example.yaml")
    assert fm.tri.middle.center == (600, 300)
    assert abs(fm.tri.base_angle_deg) < 1e-9
