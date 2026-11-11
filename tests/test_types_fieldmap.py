"""数据类型与场地图加载测试。"""
import pytest

import yaml

from src.algorithms.fieldmap import (REGION_TO_GROUP, FieldMap, TargetField,
                                     load_field_map)
from src.algorithms.types import (Pose, RingObs, StationObs, TargetObs,
                                  TargetSetObs, TurntableObs)


def test_pose_frozen():
    p = Pose(1.0, 2.0, 30.0)
    with pytest.raises(Exception):
        p.x = 5.0


def test_load_field_example_has_two_groups():
    fm = load_field_map("config/field.example.yaml")
    assert set(fm.groups) == {"rough", "staging"}
    for name in ("rough", "staging"):
        assert len(fm.targets(name)) == 3


def test_targets_are_sorted_by_label():
    fm = load_field_map("config/field.example.yaml")
    labels = [t.label for t in fm.targets("rough")]
    assert labels == [1, 2, 3]


def test_group_for_region_mapping():
    fm = load_field_map("config/field.example.yaml")
    assert fm.group_for_region(2) is fm.groups["rough"]
    assert fm.group_for_region(3) is fm.groups["staging"]
    # 原料区与起点没有参考靶
    assert fm.group_for_region(1) is None
    assert fm.group_for_region(4) is None
    assert fm.group_for_region(None) is None


def test_region_to_group_constants():
    assert REGION_TO_GROUP == {2: "rough", 3: "staging"}


def test_entry_pose_is_loaded():
    fm = load_field_map("config/field.example.yaml")
    ep = fm.entry_pose_for_region(2)
    assert ep is not None
    assert abs(ep.x - 1500.0) < 1e-9 and abs(ep.y - 1000.0) < 1e-9
    assert fm.entry_pose_for_region(1) is None


def test_region_poses_cover_every_route_region():
    """sim 要用 region_poses 生成行驶路线：三个作业区都必须有标称停位。"""
    fm = load_field_map("config/field.example.yaml")
    for region, want in ((1, (600.0, 1600.0)),
                         (2, (1500.0, 1000.0)),
                         (3, (1500.0, 2000.0))):
        pose = fm.pose_for_region(region)
        assert pose is not None, f"区域 {region} 缺少 region_poses"
        assert (pose.x, pose.y) == want


def test_region_poses_start_falls_back_to_start():
    """起点没有单独的 region_poses：必须退回 start。"""
    fm = load_field_map("config/field.example.yaml")
    assert fm.pose_for_region(4) == fm.start


def test_unknown_region_pose_name_rejected(tmp_path):
    p = tmp_path / "f.yaml"
    p.write_text(yaml.safe_dump({
        "reference_groups": {"g": {"targets": [
            {"label": 1, "center": [0, 0], "radius": 10},
            {"label": 2, "center": [100, 0], "radius": 10},
            {"label": 3, "center": [200, 0], "radius": 10},
        ]}},
        "region_poses": {"no_such_region": {"x": 0, "y": 0}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="region_poses"):
        load_field_map(str(p))


def test_target_lookup_and_missing():
    fm = load_field_map("config/field.example.yaml")
    assert fm.target("rough", 2).center == (1500.0, 600.0)
    with pytest.raises(KeyError):
        fm.target("rough", 9)


def test_turntable_and_palletizing_loaded():
    fm = load_field_map("config/field.example.yaml")
    assert fm.turntable is not None
    assert fm.turntable.station_count == 3
    assert fm.turntable.station_radius == 150.0
    assert fm.layer_height_mm == 40.0


def test_missing_reference_groups_rejected(tmp_path):
    p = tmp_path / "f.yaml"
    p.write_text("start: {pose: {x: 0, y: 0}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference_groups"):
        load_field_map(str(p))


def test_duplicate_labels_rejected(tmp_path):
    p = tmp_path / "f.yaml"
    p.write_text(yaml.safe_dump({"reference_groups": {"g": {"targets": [
        {"label": 1, "center": [0, 0], "radius": 10},
        {"label": 1, "center": [100, 0], "radius": 10},
        {"label": 2, "center": [200, 0], "radius": 10},
    ]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_field_map(str(p))


def test_non_collinear_targets_rejected(tmp_path):
    """三个靶不共线时朝向解不出来，必须在加载配置时就拦下，而不是场上解出错的值。"""
    p = tmp_path / "f.yaml"
    p.write_text(yaml.safe_dump({"reference_groups": {"g": {"targets": [
        {"label": 1, "center": [0, 0], "radius": 10},
        {"label": 2, "center": [100, 300], "radius": 10},   # 明显偏出
        {"label": 3, "center": [200, 0], "radius": 10},
    ]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="不在一条直线"):
        load_field_map(str(p))


def test_target_set_obs_is_iterable():
    """FSM 会把观测当作序列用（len/迭代/下标），三者必须都支持。"""
    pts = (TargetObs((0.0, 0.0)), TargetObs((100.0, 0.0)), TargetObs((200.0, 0.0)))
    obs = TargetSetObs(pts)
    assert len(obs) == 3
    assert [p.center[0] for p in obs] == [0.0, 100.0, 200.0]
    assert obs[1].center[0] == 100.0


def test_turntable_obs_fields():
    st = StationObs(color=4, center=(10.0, 20.0), angle01=600)
    tt = TurntableObs(center=(0.0, 0.0), radius=150.0, angle01=600,
                      omega01=600, stations=(st,), status=0x8F)
    assert tt.stations[0].color == 4
    assert tt.status & 0x80


def test_target_field_obs_helper():
    t = TargetField(label=2, center=(1500.0, 600.0), radius=60.0, rings=5)
    obs = t.obs
    assert isinstance(obs, RingObs)
    assert obs.label == 2 and obs.kind == "target"
