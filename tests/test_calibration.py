import math
import random

from src.algorithms.calibration import calibrate
from src.algorithms.fieldmap import load_field_map
from src.algorithms.types import Pose, RingObs, TriRingObs
from tests.conftest import repo_path

FIELD = load_field_map(repo_path("config", "field.example.yaml"))


def _obs_with_error(ex, ey, ehead):
    """合成一个三色环观测。

    符号约定（Ruling 1，以实现为准）：航位推算得到的 pose 带着相对真值的误差 ε，
    投影到场地就得到 obs = theory + ε，而 calibrate 的修正量 (theory - obs) 恰好
    把 pose 拉回真值。因此这里把观测构造成「理论环经 -(ex, ey) 平移、绕中环旋转
    -ehead」，传入 ex = truth.x - pose.x 时，calibrate 就能把 pose 回收到 truth。
    """
    c = FIELD.tri
    mx0, my0 = c.middle.center

    def rot(p):
        a = math.radians(-ehead)
        px, py = p[0] - mx0, p[1] - my0
        return (px * math.cos(a) - py * math.sin(a) + mx0 - ex,
                px * math.sin(a) + py * math.cos(a) + my0 - ey)

    return TriRingObs(RingObs(rot(c.left.center), 20),
                      RingObs(rot(c.middle.center), 20),
                      RingObs(rot(c.right.center), 20))


def test_exact_correction_recovers_truth():
    obs = _obs_with_error(12.0, -7.0, 2.5)
    wrong = Pose(0, 0, 0)
    fixed = calibrate(wrong, obs, FIELD, 0.7, 0.3, 15, 3)
    assert fixed is not None
    assert abs(fixed.x - 12.0) < 0.5 and abs(fixed.y + 7.0) < 0.5
    assert abs(fixed.heading - 2.5) < 0.1


def test_residual_gate_rejects_mismatch():
    l, r = FIELD.tri.left.center, FIELD.tri.right.center
    bad = TriRingObs(RingObs(l, 20), RingObs((900, 900), 20), RingObs(r, 20))
    assert calibrate(Pose(0, 0, 0), bad, FIELD, 0.7, 0.3, 15, 3) is None


def test_two_calibrations_bound_drift():
    """量化验收：漂移模型下，校准瞬间的位置误差收缩到未校准误差的 1/5 以下。

    意图（Ruling 2 重写）：truth 按确定步长推进；推算 pose = truth + 逐段累积的漂移；
    观测由该漂移经 _obs_with_error 生成并叠加小幅测量噪声。在第 2、4 段各校准一次，
    断言校准瞬间 |fixed - truth| < |pose - truth| / 5。若门限或符号/权重计算被破坏，
    fixed 无法回收到 truth，断言即失败——不放宽断言凑过。
    """
    random.seed(0)
    pose = Pose(0, 0, 0)
    truth = Pose(0, 0, 0)
    calibrations = []

    for leg in range(1, 5):
        truth = Pose(truth.x + 1000.0, truth.y + 20.0, truth.heading + 0.8)
        drift = 60.0 * leg  # 推算漂移逐段累积
        pose = Pose(truth.x + drift, truth.y + drift * 0.5, truth.heading + drift * 0.005)
        if leg not in (2, 4):
            continue
        # 观测 = 理论环 + pose 误差（即 -(truth-pose) 平移），叠加每环 ±2mm 噪声
        obs = _obs_with_error(truth.x - pose.x, truth.y - pose.y, truth.heading - pose.heading)

        def jitter(ring):
            x, y = ring.center
            return RingObs((x + random.uniform(-2, 2), y + random.uniform(-2, 2)), ring.radius)

        obs = TriRingObs(jitter(obs.left), jitter(obs.middle), jitter(obs.right))

        fixed = calibrate(pose, obs, FIELD, 0.7, 0.3, 15, 3)
        assert fixed is not None, f"leg {leg}: gate must not reject a small-noise calibration"
        pre = math.hypot(pose.x - truth.x, pose.y - truth.y)
        post = math.hypot(fixed.x - truth.x, fixed.y - truth.y)
        calibrations.append((leg, pre, post))

    assert len(calibrations) == 2
    for leg, pre, post in calibrations:
        assert pre > 0.0, "drift model must produce a real pre-calibration error"
        assert post < pre / 5.0, f"leg {leg}: post {post:.3f} not < pre/5 ({pre / 5.0:.3f})"
