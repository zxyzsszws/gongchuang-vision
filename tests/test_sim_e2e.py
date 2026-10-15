from src.main import run


def test_sim_full_match():
    fsm = run("config/default.yaml", "sim", max_seconds=120.0)
    assert fsm.state == "DONE"
    assert any(e[0] == "CAL_APPLIED" for e in fsm.log_events)  # sim 中校准至少成功 1 次
