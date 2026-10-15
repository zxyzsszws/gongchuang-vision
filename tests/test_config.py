from pathlib import Path
from src.config import load_config, cfg_get

def test_load_default_yaml():
    cfg = load_config("config/default.yaml")
    assert cfg_get(cfg, "serial.baudrate") == 115200
    assert cfg_get(cfg, "calib.stable_frames") == 5

def test_cfg_get_missing_raises():
    cfg = load_config("config/default.yaml")
    import pytest
    with pytest.raises(KeyError):
        cfg_get(cfg, "no.such.key")
