"""sim 端到端：入口装配全图跑通 + 校准闭环被真正激励（final-review F5）。

test_sim_full_match 的旧断言（DONE + 任意 CAL_APPLIED）在无环耦合的 SimScene
下也能通过（评审 Important-5），故收紧为：
- 恰好 2 次 CAL_APPLIED（CAL1/CAL2 各一次，无 CAL_SKIPPED 降级）；
- 每次应用时 |dx|+|dy| ≥ 20mm 且 ≥ 当时标称位姿误差的 80%
  （修正量必须来自漂移误差本身，而非 ±3mm 抖动）；
- 每次修正后主控「上报−标称」误差收缩过半（闭环收敛的直接证据）；
- 收尾 out_of_range_clamps == 0（评审遗留 Task-10 deferred 断言）；
- 虚拟时钟逐步驱动，无墙钟竞态，固定种子，可复现。
"""
import yaml

from src.core.fsm import MatchFSM
from src.core.kalman import PoseFilter
from src.algorithms.fieldmap import load_field_map
from src.devices.link import MockMcu
from src.devices.sim_scene import SimScene
from src.main import run
from tests.conftest import repo_path

# 每次校准应用时的修正量下限（mm）：|dx|+|dy| 必须达到此漂移级量才算激励了闭环
APPLIED_FLOOR_MM = 20.0


def _cfg():
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _err_l1(rep):
    """主控模型「上报 − 标称真值」的 L1 距离（mm），修正生效前=标称漂移+已叠加修正。"""
    return abs(rep["err_x_mm"]) + abs(rep["err_y_mm"])


def _run_match(drift_mm_per_m, drift_deg_per_m=0.1, pose_hz=10.0,
               steps=2000, dt=0.02):
    """虚拟时钟确定性驱动 sim 全栈；返回 (fsm, link, applied)。

    applied: 每次出现 CAL_APPLIED 事件的那一拍，记录 (事件, 应用前误差, 应用后误差)。
    MockMcu.send 同步叠加 bias，故「应用后」在同一拍即可读取。
    """
    cfg = _cfg()
    field = load_field_map(repo_path("config", "field.example.yaml"))
    link = MockMcu(drift_mm_per_m=drift_mm_per_m,
                   drift_deg_per_m=drift_deg_per_m, pose_hz=pose_hz)
    sim = SimScene(field, link=link)
    fsm = MatchFSM(cfg, link, PoseFilter(), field, sim)
    applied, seen = [], 0
    for k in range(steps):
        now = (k + 1) * dt
        pre_err = _err_l1(link.get_pose_report())
        fsm.step(now)
        for e in fsm.log_events[seen:]:
            if e[0] == "CAL_APPLIED":
                post_err = _err_l1(link.get_pose_report())
                applied.append((e, pre_err, post_err))
        seen = len(fsm.log_events)
        if fsm.state in ("DONE", "SAFESTOP"):
            break
    return fsm, link, applied


def test_sim_entrypoint_reaches_done_and_uses_config_drift():
    # 入口 run()（虚拟时钟，确定性）以 config/sim 的漂移参数装配 MockMcu。
    fsm = run(repo_path("config", "default.yaml"), "sim", max_seconds=120.0)
    assert fsm.state == "DONE"
    assert fsm.link.drift_mm_per_m == _cfg()["sim"]["drift_mm_per_m"]
    cals = [e for e in fsm.log_events if e[0] == "CAL_APPLIED"]
    assert len(cals) == 2
    assert not any(e[0] in ("ACK_TIMEOUT", "ACTION_TIMEOUT", "LINK_TIMEOUT",
                            "CAL_SKIPPED", "SCAN_TIMEOUT") for e in fsm.log_events)
    assert fsm.link.out_of_range_clamps == 0


def test_sim_calibration_loop_is_closed():
    # F5 回归：观测必须由「当前位姿误差」合成（上报−标称），校准闭环真正被激励。
    fsm, link, applied = _run_match(drift_mm_per_m=20.0)
    assert fsm.state == "DONE"
    assert len(applied) == 2  # 恰好 CAL1/CAL2 各一次
    for event, pre_err, post_err in applied:
        dx, dy = event[1], event[2]
        assert abs(dx) + abs(dy) >= APPLIED_FLOOR_MM  # 漂移级修正，非抖动量级
        assert abs(dx) + abs(dy) >= 0.8 * pre_err     # 修正量≈当时的标称误差
        assert abs(post_err) < abs(pre_err) / 2.0     # 应用后误差收缩过半（收敛闭环）
    assert link.out_of_range_clamps == 0
    assert not any(e[0] in ("ACK_TIMEOUT", "ACTION_TIMEOUT", "LINK_TIMEOUT",
                            "CAL_SKIPPED") for e in fsm.log_events)
