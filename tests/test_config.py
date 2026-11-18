"""配置层测试：V2.2.0 的 default.yaml 必须自洽，且不再残留旧版的死键。"""
import pytest

from src.config import cfg_get, load_config
from src.core.protocol import TASK_ALIGN, TASK_SCAN, TASK_TURNTABLE


def test_load_default_yaml():
    cfg = load_config("config/default.yaml")
    assert cfg_get(cfg, "serial.baudrate") == 115200
    assert cfg_get(cfg, "mode") in ("sim", "live", "replay")


def test_timing_keys_match_doc_table():
    """文档「表 1-12 时序参数」的每一项都必须在配置里有对应项。"""
    cfg = load_config("config/default.yaml")
    assert cfg_get(cfg, "timing.frame_interval_ms") == 20
    assert cfg_get(cfg, "timing.scan_timeout_s") == 3
    assert cfg_get(cfg, "timing.turntable_timeout_s") == 3
    assert cfg_get(cfg, "timing.max_bad_requests") == 5


def test_obsolete_timing_keys_are_gone():
    """V2.2.0 没有周期上报、没有 done，这些键留着只会诱导实现者加回来。"""
    cfg = load_config("config/default.yaml")
    for dead in ("ack_timeout_ms", "exec_timeout_ms", "ack_retry",
                 "pose_interval_ms", "turntable_interval_ms",
                 "heartbeat_ms", "link_timeout_s"):
        assert dead not in cfg["timing"], f"配置里仍残留 {dead!r}"


def test_retired_v1_keys_are_gone():
    """v1.0.0 / v2.0.0 的校准、导航、场地图配置在此版本已无对应代码。"""
    cfg = load_config("config/default.yaml")
    for dead in ("calib", "nav_timeout_s", "cal_timeout_s", "ring", "pose_hz",
                 "field_file"):
        assert dead not in cfg, f"配置里仍残留旧键 {dead!r}"


def test_target_detection_keys_present():
    cfg = load_config("config/default.yaml")
    for key in ("search_radius", "cluster_tol_px", "min_rings",
                "min_circularity", "min_fill", "same_radius_ratio",
                "dedup_px", "ring_outer_mm", "threshold_block", "threshold_c"):
        assert key in cfg["target"], f"target 配置缺 {key}"
    lo, hi = cfg["target"]["search_radius"]
    assert 0 < lo < hi
    assert cfg["target"]["ring_outer_mm"] > 0, "尺度全靠这个直径，必须为正"


def test_max_match_mm_is_gone():
    """选靶不再按「离预测位置多远」筛（没有场地图也没有预测），这个键该消失。"""
    cfg = load_config("config/default.yaml")
    assert "max_match_mm" not in cfg["target"]


def test_fill_and_dedup_thresholds_are_sane():
    """充满度门限用来挡方形（正方形只有 0.64），设太低就挡不住了。"""
    cfg = load_config("config/default.yaml")
    assert 0.6 < cfg["target"]["min_fill"] <= 1.0
    assert 0.0 < cfg["target"]["dedup_px"] < cfg["target"]["cluster_tol_px"]


def test_sim_keys_present():
    cfg = load_config("config/default.yaml")
    for key in ("jitter_mm", "turntable_omega01", "turntable_stop_after_s"):
        assert key in cfg["sim"], f"sim 配置缺 {key}"
    for dead in ("drift_mm_per_m", "drift_deg_per_m", "leg_mm", "speed_mm_s"):
        assert dead not in cfg["sim"], f"sim 里仍残留已删除的 {dead!r}"


def test_turntable_section_is_complete():
    """停稳判定的两套策略都要能配；缺参数会让现场无法切换。"""
    cfg = load_config("config/default.yaml")
    tt = cfg["turntable"]
    assert tt["stop_strategy"] in ("motion", "containment")
    for key in ("pick_point_mm", "material_radius_mm", "tol_mm", "hold_s",
                "pick_tol_mm", "containment_margin"):
        assert key in tt, f"turntable 配置缺 {key}"
    assert len(tt["pick_point_mm"]) == 2
    assert tt["material_radius_mm"] > 0
    assert tt["hold_s"] > 0 and tt["tol_mm"] > 0
    assert tt["containment_margin"] > 1.0, "边界圆必须比物料大，否则永远判不过"


def test_fsm_accepts_the_shipped_config():
    """配置必须能真的构造出一个可用的 FSM，而不只是键齐全。

    这里用最小假对象代替链路与感知：只要配置里的取值合法，构造与首次推进就不该抛错。
    """
    from src.core.fsm import MatchFSM

    cfg = load_config("config/default.yaml")

    class _Link:
        def send(self, frame):
            pass

        def poll(self):
            return None

        def tick(self, now):
            pass

    fsm = MatchFSM(cfg, _Link(), {})
    assert fsm.state == "SERVICE"
    assert set(fsm.counts) == {TASK_SCAN, TASK_TURNTABLE, TASK_ALIGN}


def test_cfg_get_missing_raises():
    cfg = load_config("config/default.yaml")
    with pytest.raises(KeyError):
        cfg_get(cfg, "no.such.key")
