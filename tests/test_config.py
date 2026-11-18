"""配置层测试：v2.0.0 的 default.yaml 必须自洽，且不再残留 v1.0.0 的死键。"""
import pytest

from src.config import cfg_get, load_config


def test_load_default_yaml():
    cfg = load_config("config/default.yaml")
    assert cfg_get(cfg, "serial.baudrate") == 115200
    assert cfg_get(cfg, "mode") in ("sim", "live", "replay")


def test_timing_keys_match_doc_table():
    """表 1-15 的每个时序参数都必须在配置文件里有对应项，且取值一致。"""
    cfg = load_config("config/default.yaml")
    assert cfg_get(cfg, "timing.ack_timeout_ms") == 100
    assert cfg_get(cfg, "timing.exec_timeout_ms") == 2000
    assert cfg_get(cfg, "timing.ack_retry") == 3
    assert cfg_get(cfg, "timing.pose_interval_ms") == 50
    assert cfg_get(cfg, "timing.turntable_interval_ms") == 50


def test_heartbeat_keys_are_gone():
    """V2.1.0 取消心跳：配置里不得残留 heartbeat_ms，也不该有断链超时。

    断链是靠不出来的（电控事件驱动，停着作业时可以长时间不发帧），所以这条
    配置项本身就不该存在——留着会诱导实现者去写一个必然误报的判断。
    """
    cfg = load_config("config/default.yaml")
    for dead in ("heartbeat_ms", "link_timeout_s"):
        assert dead not in cfg["timing"], f"配置里仍残留 {dead!r}"


def test_retired_v1_keys_are_gone():
    """v1.0.0 的校准/导航/滤波配置在 v2.0.0 已无对应代码，留着只会误导。"""
    cfg = load_config("config/default.yaml")
    for dead in ("calib", "nav_timeout_s", "cal_timeout_s", "ring", "pose_hz"):
        assert dead not in cfg, f"配置里仍残留 v1.0.0 的键 {dead!r}"


def test_target_detection_keys_present():
    cfg = load_config("config/default.yaml")
    for key in ("search_radius", "cluster_tol_px", "min_rings",
                "min_circularity", "min_fill", "same_radius_ratio",
                "dedup_px", "max_match_mm", "threshold_block", "threshold_c"):
        assert key in cfg["target"], f"target 配置缺 {key}"
    lo, hi = cfg["target"]["search_radius"]
    assert 0 < lo < hi


def test_fill_and_dedup_thresholds_are_sane():
    """充满度门限用来挡方形（正方形只有 0.64），设太低就挡不住了。"""
    cfg = load_config("config/default.yaml")
    assert 0.6 < cfg["target"]["min_fill"] <= 1.0
    assert 0.0 < cfg["target"]["dedup_px"] < cfg["target"]["cluster_tol_px"]


def test_sim_keys_present():
    cfg = load_config("config/default.yaml")
    for key in ("jitter_mm", "leg_mm", "speed_mm_s", "turntable_omega01"):
        assert key in cfg["sim"], f"sim 配置缺 {key}"


def test_cfg_get_missing_raises():
    cfg = load_config("config/default.yaml")
    with pytest.raises(KeyError):
        cfg_get(cfg, "no.such.key")
