import random
from src.core.kalman import PoseFilter
from src.algorithms.types import Pose


def test_noise_reduced():
    random.seed(42)
    pf = PoseFilter()
    truth_x = 100.0
    errs = []
    for _ in range(30):
        meas = Pose(truth_x + random.gauss(0, 5), 0, 0)
        est = pf.update(meas)
        errs.append(abs(est.x - truth_x))
    assert errs[-1] < 2.0 and errs[-1] < max(errs[:5])


def test_constant_velocity_tracks():
    pf = PoseFilter()
    for i in range(60):
        est = pf.update(Pose(10.0 * i, 0, 0))
    assert abs(est.x - 590.0) < 15.0
