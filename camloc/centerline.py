"""Track centerline model + arc-length progress.

Rationale
---------
Full-pose monocular VO diverged (547 m) because the only absolute correction —
distance to the nearest border — carries *no* along-track information, so any VO
scale bias integrates without bound. But the car physically drives *between* the
surveyed borders, so the border midline (the centerline) is an excellent position
prior: it pins the lateral error (the metric's dominant term) by construction.

This module builds a dense, arc-length-parameterised centerline from the
`track_left` / `track_right` polylines and lets the pipeline:
  - find the progress `s` (metres along the lap) nearest a given xy,
  - map a progress `s` back to a map pose (x, y, yaw=tangent),
  - advance progress by a travelled distance (laps wrap around automatically).

We only need a scalar *speed* to drive the progress forward — direction comes
from the track geometry — which is far more robust than estimating full 6-DOF
camera motion from night-time race footage.
"""
from __future__ import annotations

import numpy as np


def _resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline to roughly uniform `step`-metre spacing."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return pts
    n = max(int(total / step) + 1, 2)
    s_new = np.linspace(0.0, total, n)
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x, y])


def _nearest_index(poly: np.ndarray, xy) -> int:
    d = np.linalg.norm(poly - np.asarray(xy, float), axis=1)
    return int(np.argmin(d))


class Centerline:
    """Arc-length-parameterised track midline built from left/right borders."""

    def __init__(self, left: np.ndarray, right: np.ndarray, step: float = 1.0):
        self.step = step
        self.points = self._build(left, right, step)
        self.closed = np.linalg.norm(self.points[0] - self.points[-1]) < 5.0
        self._arc = self._arclength(self.points)
        self.length = float(self._arc[-1])
        self._tangents = self._compute_tangents(self.points, self.closed)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build(left, right, step):
        """Midline: resample both borders to equal counts, average pointwise.

        The two polylines are walked in the same arc-length direction, so the
        average follows the drivable corridor. We first resample both to the
        same number of samples by normalised arc length.
        """
        left = _resample_polyline(left, step)
        right = _resample_polyline(right, step)
        n = max(len(left), len(right))
        tl = np.linspace(0, 1, len(left))
        tr = np.linspace(0, 1, len(right))
        tt = np.linspace(0, 1, n)
        lx = np.interp(tt, tl, left[:, 0])
        ly = np.interp(tt, tl, left[:, 1])
        rx = np.interp(tt, tr, right[:, 0])
        ry = np.interp(tt, tr, right[:, 1])
        mid = np.column_stack([(lx + rx) / 2.0, (ly + ry) / 2.0])
        return _resample_polyline(mid, step)

    @staticmethod
    def _arclength(poly):
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    @staticmethod
    def _compute_tangents(poly, closed):
        if closed:
            nxt = np.roll(poly, -1, axis=0)
            prv = np.roll(poly, 1, axis=0)
        else:
            nxt = np.vstack([poly[1:], poly[-1]])
            prv = np.vstack([poly[0], poly[:-1]])
        d = nxt - prv
        return np.arctan2(d[:, 1], d[:, 0])

    # ------------------------------------------------------------------ #
    def progress_of(self, xy) -> float:
        """Arc length (m) of the centerline point nearest `xy`."""
        return float(self._arc[_nearest_index(self.points, xy)])

    def tangent_sign_toward(self, xy, heading: float) -> float:
        """+1 if increasing progress agrees with `heading`, else -1.

        Used once at init to decide the direction of travel along the lap.
        """
        i = _nearest_index(self.points, xy)
        t = self._tangents[i]
        return 1.0 if np.cos(heading - t) >= 0 else -1.0

    def pose_at(self, s: float, direction: float = 1.0):
        """Map pose (x, y, yaw) at arc length `s` (metres).

        `direction` (+1/-1) flips the tangent so yaw points the way the car goes.
        Progress wraps around the lap for a closed track.
        """
        if self.closed and self.length > 0:
            s = s % self.length
        s = float(np.clip(s, 0.0, self.length))
        x = float(np.interp(s, self._arc, self.points[:, 0]))
        y = float(np.interp(s, self._arc, self.points[:, 1]))
        i = _nearest_index(self.points, (x, y))
        yaw = self._tangents[i]
        if direction < 0:
            yaw = float(np.arctan2(np.sin(yaw + np.pi), np.cos(yaw + np.pi)))
        return x, y, yaw
