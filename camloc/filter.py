"""Particle filter: VO motion model + track-border map correction.

State per particle: (x, y, yaw) in the map frame, plus a shared running speed.

predict()  applies the VO body-frame increment (dx, dy, dyaw) to every particle
           with Gaussian process noise, so the cloud spreads along the directions
           VO is least certain about (scale -> longitudinal, yaw drift).

update()   scores each particle against the surveyed borders. The car always
           drives *between* the left and right borders at a roughly constant
           half-width, so a hypothesised pose whose implied lateral offset to the
           nearest border disagrees with the expected track half-width is
           down-weighted. This removes lateral drift (the metric's dominant term)
           without ever reading the ground truth.

The filter stays causal: it is only ever advanced with increments and
observations derived from images at or before the current timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry
from .track import TrackMap


@dataclass
class FilterConfig:
    n_particles: int = 600
    # process noise (per step), std devs
    sigma_lon: float = 0.15      # m, longitudinal (scale uncertainty)
    sigma_lat: float = 0.03      # m, lateral
    sigma_yaw: float = 0.004     # rad
    # measurement model
    expected_half_width: float = 6.0   # m, car-centre to nearest border (prior)
    half_width_sigma: float = 2.5      # m, tolerance of the above
    resample_ratio: float = 0.5        # resample when ESS/N drops below this


class ParticleFilter:
    def __init__(self, init_pose, track: TrackMap, config: FilterConfig | None = None):
        self.cfg = config or FilterConfig()
        self.track = track
        n = self.cfg.n_particles
        x, y, yaw = init_pose
        self.particles = np.tile([x, y, yaw], (n, 1)).astype(float)
        # tiny initial jitter so the cloud is not degenerate
        self.particles[:, 0] += np.random.normal(0, 0.1, n)
        self.particles[:, 1] += np.random.normal(0, 0.1, n)
        self.particles[:, 2] += np.random.normal(0, 0.01, n)
        self.weights = np.full(n, 1.0 / n)
        self.speed = 0.0

    # ------------------------------------------------------------------ #
    def predict(self, increment, dt):
        """Apply the body-frame VO increment with process noise."""
        dx, dy, dyaw = increment
        n = len(self.particles)
        nlon = np.random.normal(0, self.cfg.sigma_lon, n)
        nlat = np.random.normal(0, self.cfg.sigma_lat, n)
        nyaw = np.random.normal(0, self.cfg.sigma_yaw, n)
        yaw = self.particles[:, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        # increment expressed in each particle's body frame, then noise
        bx = dx + nlon
        by = dy + nlat
        self.particles[:, 0] += c * bx - s * by
        self.particles[:, 1] += s * bx + c * by
        self.particles[:, 2] = np.arctan2(
            np.sin(yaw + dyaw + nyaw), np.cos(yaw + dyaw + nyaw)
        )
        if dt > 1e-6:
            self.speed = float(np.hypot(dx, dy) / dt)

    # ------------------------------------------------------------------ #
    def update(self):
        """Re-weight particles using the border map prior."""
        if self.track.segment_count == 0:
            return
        d = self.track.nearest_distance(self.particles[:, :2])  # (N,)
        # likelihood: the car centre should sit ~half-width from the nearest edge
        err = d - self.cfg.expected_half_width
        logw = -(err ** 2) / (2.0 * self.cfg.half_width_sigma ** 2)
        logw -= logw.max()
        w = self.weights * np.exp(logw)
        total = w.sum()
        if total <= 0 or not np.isfinite(total):
            self.weights[:] = 1.0 / len(self.weights)
            return
        self.weights = w / total
        if self._ess() < self.cfg.resample_ratio * len(self.weights):
            self._resample()

    def _ess(self):
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        n = len(self.weights)
        # systematic resampling
        positions = (np.arange(n) + np.random.random()) / n
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        idx = np.searchsorted(cumsum, positions)
        self.particles = self.particles[idx]
        self.weights = np.full(n, 1.0 / n)

    # ------------------------------------------------------------------ #
    def estimate(self):
        """Weighted mean pose (x, y, yaw), yaw averaged on the circle."""
        w = self.weights
        x = float(np.sum(w * self.particles[:, 0]))
        y = float(np.sum(w * self.particles[:, 1]))
        yaw = float(
            np.arctan2(
                np.sum(w * np.sin(self.particles[:, 2])),
                np.sum(w * np.cos(self.particles[:, 2])),
            )
        )
        return x, y, yaw
