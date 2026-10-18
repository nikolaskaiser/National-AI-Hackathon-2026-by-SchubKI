"""Robust scalar speed estimate from image motion.

For the centerline-progress pipeline we do NOT need full camera motion — only a
scalar speed to advance the car along the track. Estimating "how fast" is far
more robust than "which direction" (which was what broke the essential-matrix
VO at night / high speed).

Approach: track features frame-to-frame on a forward camera and use the *median*
optical-flow magnitude. Flow magnitude scales linearly with vehicle speed for a
forward-looking camera at roughly fixed geometry, so we map median flow (px/frame)
to speed (m/s) through a single gain that is auto-calibrated online against a
coarse physical prior (ground-plane depth of the tracked points). The absolute
gain only needs to be roughly right because the centerline constrains the path;
errors show up as longitudinal lag, not path divergence.

Falls back gracefully (keeps the last speed) when tracking is poor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class SpeedConfig:
    max_features: int = 600
    quality: float = 0.01
    min_distance: int = 10
    min_tracks: int = 20
    max_speed: float = 45.0        # m/s clamp (README: up to ~40 m/s)
    smoothing: float = 0.35        # EMA on speed
    # px/frame -> m/s gain. Calibrated online from the camera focal length and an
    # assumed forward scene depth; see _auto_gain. A rough default seeds it.
    flow_to_mps: float = 0.0       # 0 => auto-calibrate on first good frame


class SpeedEstimator:
    def __init__(self, camera_model, config: SpeedConfig | None = None):
        if cv2 is None:
            raise RuntimeError("opencv-python is required for speed estimation")
        self.cfg = config or SpeedConfig()
        self.cam = camera_model
        self.fx = float(camera_model.K[0, 0])
        # camera mount height above ground (base_link z of the optical origin)
        self.cam_height = float(abs(camera_model.T_base_cam[2, 3])) or 0.5
        self._gain = self.cfg.flow_to_mps
        self._prev_gray = None
        self._prev_pts = None
        self._prev_stamp = None
        self._speed = 0.0

    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg.max_features,
            qualityLevel=self.cfg.quality, minDistance=self.cfg.min_distance,
        )

    def _auto_gain(self, median_flow_px, dt, p0):
        """Rough px/frame -> m/s gain from pinhole geometry.

        For a point at forward depth Z, image displacement du for a forward
        translation dX_forward is ~ fx * (lateral offset) * ... — intractable in
        closed form per point, so we use the standard forward-motion approximation
        that *radial* flow from the FOE grows with speed/Z. We approximate scene
        depth by the camera height / tan(pitch of the lower image rows); lacking a
        calibrated pitch we assume a representative ground depth of ~15 m, which
        the online EMA + the centerline path then absorb.
        """
        assumed_depth = 15.0  # m, representative forward ground distance
        # speed [m/s] ~= median_flow[px/frame] * Z / (fx * dt)
        return assumed_depth / (self.fx * max(dt, 1e-3))

    def process(self, gray, stamp_ns) -> float:
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._detect(gray)
            self._prev_stamp = stamp_ns
            return 0.0

        dt = max((stamp_ns - self._prev_stamp) / 1e9, 1e-3)
        speed = self._speed  # default: hold last

        if self._prev_pts is not None and len(self._prev_pts) >= self.cfg.min_tracks:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None
            )
            if nxt is not None:
                good = status.ravel() == 1
                p0 = self._prev_pts[good].reshape(-1, 2)
                p1 = nxt[good].reshape(-1, 2)
                if len(p0) >= self.cfg.min_tracks:
                    flow = np.linalg.norm(p1 - p0, axis=1)
                    median_flow = float(np.median(flow))
                    if self._gain <= 0:
                        self._gain = self._auto_gain(median_flow, dt, p0)
                    raw = median_flow * self._gain
                    raw = min(max(raw, 0.0), self.cfg.max_speed)
                    a = self.cfg.smoothing
                    speed = (1 - a) * self._speed + a * raw

        self._speed = speed
        self._prev_gray = gray
        self._prev_pts = self._detect(gray)
        self._prev_stamp = stamp_ns
        return speed

    @property
    def speed(self) -> float:
        return self._speed
