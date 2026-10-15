import json, logging
from src.logsetup import setup_logging

def test_jsonl_written(tmp_path):
    path = setup_logging(str(tmp_path))
    logging.getLogger("fsm").info("evt", extra={"fsm_state": "NAV1", "t": 1.23})
    for h in logging.getLogger().handlers:
        h.flush()
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert lines[-1]["fsm_state"] == "NAV1"


def test_replay_prints_t_state_event(tmp_path, capsys):
    from tools.replay_log import main

    path = setup_logging(str(tmp_path))
    logging.getLogger("fsm").info("evt", extra={"fsm_state": "NAV1", "t": 1.23})
    for h in logging.getLogger().handlers:
        h.flush()

    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "1.23 NAV1 evt" in out


def test_replay_missing_file(tmp_path, capsys):
    from tools.replay_log import main

    assert main([str(tmp_path / "nope.jsonl")]) != 0
    assert "nope.jsonl" in capsys.readouterr().err
