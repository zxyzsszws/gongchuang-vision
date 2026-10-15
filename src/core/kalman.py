"""Pose filtering: per-axis position+velocity Kalman filter and heading EMA."""

import time

from src.algorithms.types import Pose

# Covariance entry floors guarding the tight-loop case (dt at the 1e-3 floor).
_MIN_P = 1e-9


class _PosVelKalman:
    """2-state (position, velocity) Kalman filter for one scalar axis.

    Model: x = [pos, vel], F = [[1, dt], [0, 1]], H = [1, 0] (position measured).
    Process noise Q = [[0, 0], [0, q]] (velocity random walk), measurement
    variance R = r. The first measurement seeds the position; the second seeds
    the velocity by finite difference, which makes constant-velocity segments
    lock on immediately instead of waiting for the Riccati gain to teach it.
    """

    def __init__(self, q: float, r: float):
        self.q, self.r = q, r
        self.x = None  # state: [pos, vel]
        self.p = [[1e4, 0.0], [0.0, 1e4]]  # covariance
        self.z0 = None  # first measurement, for velocity seeding

    def update(self, z: float, dt: float) -> float:
        if self.x is None:
            self.x = [z, 0.0]
            self.z0 = z
            return self.x[0]
        if self.z0 is not None:
            self.x[1] = (z - self.z0) / dt
            self.z0 = None
        x0, x1 = self.x
        p00, p01 = self.p[0]
        p10, p11 = self.p[1]
        # Predict: x <- F x; P <- F P F^T + Q.
        pred0 = x0 + dt * x1
        pp00 = p00 + dt * (p10 + p01) + dt * dt * p11
        pp01 = p01 + dt * p11
        pp10 = p10 + dt * p11
        pp11 = p11 + self.q
        # Scalar position measurement: S = H P- H^T + R, K = P- H^T / S.
        s = pp00 + self.r
        k0 = pp00 / s
        k1 = pp10 / s
        innovation = z - pred0
        self.x = [pred0 + k0 * innovation, x1 + k1 * innovation]
        # Covariance update P <- (I - K H) P-, kept symmetric; optimal-gain
        # form is equivalent to the Joseph form for this scalar measurement.
        np00 = max(pp00 * (1.0 - k0), _MIN_P)
        np01 = pp01 * (1.0 - k0)
        np11 = max(pp11 - k1 * pp01, _MIN_P)
        self.p = [[np00, np01], [np01, np11]]
        return self.x[0]


class PoseFilter:
    """Smooth raw Pose estimates: Kalman on x/y, exponential smoothing on heading."""

    def __init__(self, q: float = 1.0, r: float = 25.0, heading_alpha: float = 0.3):
        self.q, self.r, self.alpha = q, r, heading_alpha
        self.kx = _PosVelKalman(q, r)
        self.ky = _PosVelKalman(q, r)
        self.h: float | None = None
        self.last_t: float | None = None

    def reset(self, pose: Pose) -> None:
        """Re-initialize the filter state at the given pose (e.g. robot reset)."""
        self.kx = _PosVelKalman(self.q, self.r)
        self.ky = _PosVelKalman(self.q, self.r)
        self.h = pose.heading
        self.last_t = None

    def update(self, pose: Pose) -> Pose:
        """Filter one raw pose observation and return the smoothed estimate."""
        now = time.monotonic()
        dt = 0.033 if self.last_t is None else max(1e-3, now - self.last_t)
        self.last_t = now
        x = self.kx.update(pose.x, dt)
        y = self.ky.update(pose.y, dt)
        if self.h is None:
            self.h = pose.heading
        else:
            diff = (pose.heading - self.h + 180.0) % 360.0 - 180.0
            self.h = (self.h + self.alpha * diff + 180.0) % 360.0 - 180.0
        return Pose(x, y, self.h)
