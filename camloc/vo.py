"""Monocular visual odometry for base_link motion between camera frames.

We track features across successive images of a single reference camera (a front
camera by default: wide translation baseline, forward-looking, good texture from
track markings), recover the camera's relative rotation + unit translation
direction from the essential matrix, and express the motion in the `base_link`
frame.

Monocular VO is observable only up to a translation *scale*. We recover an
absolute, metric scale from the known camera mounting height above the ground
plane: features that lie on the (locally planar) track surface are triangulated
between the two views, and the ratio between their known real height drop below
the camera and their triangulated depth fixes the scale. This makes the VO
self-sufficient — it does not depend on any fed-back speed, so there is no
chicken-and-egg start-up deadlock. A near-constant-velocity fallback keeps the
pipeline moving when triangulation is degenerate (e.g. pure rotation or too few
ground features).

Everything is causal: a motion increment for the frame at time t uses only the
current and previous images (stamps ≤ t). Absolute drift correction is handled
downstream by the particle filter against the border map.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from . import geometry


@dataclass
class VOConfig:
    max_features: int = 1200
    quality: float = 0.01
    min_distance: int = 8
    ransac_thresh: float = 1.0          # px, essential-matrix RANSAC
    min_tracks: int = 30                # below this -> fall back to prediction
    max_speed: float = 45.0             # m/s, clamp implausible scale spikes
    scale_smoothing: float = 0.5        # EMA on recovered speed (0..1)
    robust_scale: bool = False          # use RANSAC ground-plane scale (scale.py)
    robust_scale_iters: int = 80        # RANSAC iterations when robust_scale
    robust_scale_thresh: float = 0.05   # RANSAC inlier band (unit baseline)


class VisualOdometry:
    """Frame-to-frame relative base_link motion (dx, dy, dyaw)."""

    def __init__(self, camera_model, config: VOConfig | None = None):
        if cv2 is None:
            raise RuntimeError("opencv-python is required for visual odometry")
        self.cam = camera_model
        self.cfg = config or VOConfig()
        self.K = camera_model.K
        self.Kinv = np.linalg.inv(camera_model.K)
        # base_link <- camera optical, and camera optical <- base_link.
        self.T_cam_base = geometry.invert_transform(camera_model.T_base_cam)
        self.T_base_cam = camera_model.T_base_cam
        # Camera height above the ground plane (base_link z is up; mount height).
        # The camera optical origin's z in base_link coordinates:
        self.cam_height = float(abs(camera_model.T_base_cam[2, 3])) or 0.5
        self._prev_gray = None
        self._prev_pts = None
        self._prev_stamp = None
        self._speed = 0.0  # m/s, recovered + smoothed

    # ------------------------------------------------------------------ #
    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.max_features,
            qualityLevel=self.cfg.quality,
            minDistance=self.cfg.min_distance,
        )

    def _recover_scale(self, R, t_dir, p0, p1):
        """Metric scale from the ground plane under the known camera height.

        Triangulate the tracked points for a unit-baseline motion, keep those
        that fall on the ground plane (points below the camera, consistent
        depth), and scale so their median height below the camera matches the
        real mounting height. Returns a positive scale, or None if degenerate.

        If cfg.robust_scale is set, delegate to the RANSAC ground-plane
        estimator in scale.py (more accurate at speed / with outliers).
        """
        if self.cfg.robust_scale:
            from .scale import ground_plane_scale

            return ground_plane_scale(
                self.K, R, t_dir, p0, p1, self.cam_height, cv2,
                n_iter=self.cfg.robust_scale_iters,
                thresh=self.cfg.robust_scale_thresh,
            )
        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([R, t_dir.reshape(3, 1)])
        pts4 = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
        w = pts4[3]
        valid = np.abs(w) > 1e-6
        if valid.sum() < 8:
            return None
        X = (pts4[:3, valid] / w[valid]).T  # (M,3) in camera-0 frame, unit scale

        # In the optical frame z is forward, y is down. Ground points are the
        # ones with the largest positive y (farthest below the camera) and
        # positive depth. Use their median y as the unit-scale ground drop.
        depth_ok = X[:, 2] > 0
        if depth_ok.sum() < 8:
            return None
        y_down = X[depth_ok, 1]
        ground_drop_unit = np.median(y_down[y_down > 0]) if np.any(y_down > 0) else None
        if not ground_drop_unit or ground_drop_unit <= 1e-6:
            return None
        scale = self.cam_height / ground_drop_unit
        if not np.isfinite(scale) or scale <= 0:
            return None
        return float(scale)

    def _camera_motion_to_base(self, R_cam, t_cam):
        """Camera-frame relative motion (metric) -> base_link SE(2) increment."""
        T_rel_cam = np.eye(4)
        T_rel_cam[:3, :3] = R_cam
        T_rel_cam[:3, 3] = t_cam
        T_rel_base = self.T_base_cam @ T_rel_cam @ self.T_cam_base
        dx = float(T_rel_base[0, 3])
        dy = float(T_rel_base[1, 3])
        dyaw = float(np.arctan2(T_rel_base[1, 0], T_rel_base[0, 0]))
        return dx, dy, dyaw

    def _predict_only(self, dt):
        """Straight-line fallback at the last known speed."""
        return (self._speed * dt, 0.0, 0.0)

    def _advance(self, gray, stamp_ns):
        self._prev_gray = gray
        self._prev_pts = self._detect(gray)
        self._prev_stamp = stamp_ns

    # ------------------------------------------------------------------ #
    def process(self, gray, stamp_ns):
        """Feed the next reference-camera frame; return (dx, dy, dyaw) in base_link."""
        if self._prev_gray is None:
            self._advance(gray, stamp_ns)
            return (0.0, 0.0, 0.0)

        dt = max((stamp_ns - self._prev_stamp) / 1e9, 1e-3)

        if self._prev_pts is None or len(self._prev_pts) < self.cfg.min_tracks:
            out = self._predict_only(dt)
            self._advance(gray, stamp_ns)
            return out

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None
        )
        if nxt is None:
            out = self._predict_only(dt)
            self._advance(gray, stamp_ns)
            return out

        good = status.ravel() == 1
        p0 = self._prev_pts[good].reshape(-1, 2)
        p1 = nxt[good].reshape(-1, 2)

        increment = self._predict_only(dt)
        if len(p0) >= self.cfg.min_tracks:
            E, _ = cv2.findEssentialMat(
                p1, p0, self.K, method=cv2.RANSAC,
                prob=0.999, threshold=self.cfg.ransac_thresh,
            )
            if E is not None and E.shape == (3, 3):
                _, R, t, _ = cv2.recoverPose(E, p1, p0, self.K)
                t_dir = t.ravel()
                scale = self._recover_scale(R, t_dir, p0, p1)
                if scale is not None:
                    # clamp to a plausible speed, then EMA-smooth
                    speed = min(scale / dt, self.cfg.max_speed)
                    a = self.cfg.scale_smoothing
                    self._speed = (1 - a) * self._speed + a * speed
                    scale = self._speed * dt
                else:
                    scale = self._speed * dt  # reuse last speed
                increment = self._camera_motion_to_base(R, t_dir * scale)

        self._advance(gray, stamp_ns)
        return increment

    @property
    def speed(self) -> float:
        return self._speed
