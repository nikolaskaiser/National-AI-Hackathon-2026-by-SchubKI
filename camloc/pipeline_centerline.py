"""Centerline-progress localization pipeline (robust default).

Position comes from the track centerline (the midline of the surveyed borders),
which the car physically follows — so lateral error, the metric's dominant term,
is small by construction. Along-track progress is advanced by a scalar speed
estimate from image motion; yaw is the centerline tangent in the direction of
travel. Only the first GT pose is used (for the initial progress), as allowed.

This replaces the full-pose monocular VO (which diverged to 547 m because the
border prior gives no along-track information). It is causal: progress at time t
uses only images with stamp <= t.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .bag import Dataset
from .centerline import Centerline
from .speed import SpeedConfig, SpeedEstimator


@dataclass
class CenterlineConfig:
    reference_camera: str = "fl"
    centerline_step: float = 1.0        # m, centerline resample spacing
    speed: SpeedConfig = None
    left_name: str = "track_left"
    right_name: str = "track_right"

    def __post_init__(self):
        if self.speed is None:
            self.speed = SpeedConfig()


class CenterlinePipeline:
    def __init__(self, bag_path: str, config: CenterlineConfig | None = None):
        if cv2 is None:
            raise RuntimeError("opencv-python is required to run the pipeline")
        self.cfg = config or CenterlineConfig()
        self.dataset = Dataset(bag_path).load_static()
        self._validate()
        left, right = self._pick_borders()
        self.centerline = Centerline(left, right, self.cfg.centerline_step)
        cam = self.dataset.cameras[self.cfg.reference_camera]
        self.speed_est = SpeedEstimator(cam, self.cfg.speed)

        x0, y0, yaw0 = self.dataset.init_pose
        self.s = self.centerline.progress_of((x0, y0))
        self.direction = self.centerline.tangent_sign_toward((x0, y0), yaw0)
        self._last_stamp = None

    def _validate(self):
        ref = self.cfg.reference_camera
        if ref not in self.dataset.cameras or not self.dataset.cameras[ref].ready:
            raise RuntimeError(f"reference camera '{ref}' missing intrinsics/extrinsics")
        if self.dataset.init_pose is None:
            raise RuntimeError("no initial pose found in /tf (map->base_link)")
        if not self.dataset.track_bounds:
            raise RuntimeError("no track_bounds found in the bag")

    def _pick_borders(self):
        b = self.dataset.track_bounds
        left = b.get(self.cfg.left_name)
        right = b.get(self.cfg.right_name)
        if left is None or right is None:
            # fall back to the two longest polylines (the main track edges)
            ordered = sorted(b.items(), key=lambda kv: -len(kv[1]))
            names = [k for k, _ in ordered if "pit" not in k.lower()][:2]
            if len(names) < 2:
                names = [k for k, _ in ordered][:2]
            left, right = b[names[0]], b[names[1]]
        return np.asarray(left, float), np.asarray(right, float)

    # ------------------------------------------------------------------ #
    def run(self, out_csv: str, progress_every: int = 500) -> int:
        n = 0
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ns", "x", "y", "yaw"])
            for frame in self.dataset.iter_camera_frames(
                cameras=[self.cfg.reference_camera]
            ):
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                speed = self.speed_est.process(gray, frame.stamp_ns)
                if self._last_stamp is not None:
                    dt = max((frame.stamp_ns - self._last_stamp) / 1e9, 0.0)
                    self.s += self.direction * speed * dt
                self._last_stamp = frame.stamp_ns
                x, y, yaw = self.centerline.pose_at(self.s, self.direction)
                writer.writerow([frame.stamp_ns, f"{x:.4f}", f"{y:.4f}", f"{yaw:.6f}"])
                n += 1
                if progress_every and n % progress_every == 0:
                    print(f"  {n} poses  (s={self.s:.1f} m, v={speed:.1f} m/s)", flush=True)
        print(f"wrote {n} poses -> {out_csv}")
        return n
