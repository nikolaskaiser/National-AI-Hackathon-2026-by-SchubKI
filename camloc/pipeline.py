"""End-to-end causal localization pipeline.

Streams the bag in log order, runs visual odometry on a reference camera to get
frame-to-frame base_link motion, fuses it with the track-border map prior through
the particle filter, and emits one pose per reference-camera frame (10 Hz) into
submission.csv.

Causality: we iterate messages in timestamp order and, for each reference-camera
image at time t, we predict + update + emit a pose using only data with stamp <= t.
The single exception the task permits is the first GT pose, used purely to
initialise the filter.
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
from .filter import FilterConfig, ParticleFilter
from .track import TrackMap
from .vo import VisualOdometry, VOConfig


@dataclass
class PipelineConfig:
    reference_camera: str = "fl"     # front camera used for VO
    vo: VOConfig = None
    filt: FilterConfig = None

    def __post_init__(self):
        if self.vo is None:
            self.vo = VOConfig()
        if self.filt is None:
            self.filt = FilterConfig()


class Pipeline:
    def __init__(self, bag_path: str, config: PipelineConfig | None = None):
        if cv2 is None:
            raise RuntimeError("opencv-python is required to run the pipeline")
        self.cfg = config or PipelineConfig()
        self.dataset = Dataset(bag_path).load_static()
        self._validate()
        self.track = TrackMap(self.dataset.track_bounds)
        cam_model = self.dataset.cameras[self.cfg.reference_camera]
        self.vo = VisualOdometry(cam_model, self.cfg.vo)
        self.filter = ParticleFilter(
            self.dataset.init_pose, self.track, self.cfg.filt
        )
        self._last_stamp = self.dataset.init_stamp_ns

    def _validate(self):
        ref = self.cfg.reference_camera
        if ref not in self.dataset.cameras or not self.dataset.cameras[ref].ready:
            raise RuntimeError(
                f"reference camera '{ref}' missing intrinsics/extrinsics"
            )
        if self.dataset.init_pose is None:
            raise RuntimeError("no initial pose found in /tf (map->base_link)")
        if not self.dataset.track_bounds:
            raise RuntimeError("no track_bounds found in the bag")

    # ------------------------------------------------------------------ #
    def run(self, out_csv: str, progress_every: int = 500) -> int:
        """Process the bag and write submission.csv. Returns #poses written."""
        n = 0
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ns", "x", "y", "yaw"])
            for frame in self.dataset.iter_camera_frames(
                cameras=[self.cfg.reference_camera]
            ):
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                increment = self.vo.process(gray, frame.stamp_ns)
                dt = max((frame.stamp_ns - (self._last_stamp or frame.stamp_ns)) / 1e9, 0.0)
                self.filter.predict(increment, dt)
                self.filter.update()
                x, y, yaw = self.filter.estimate()
                writer.writerow([frame.stamp_ns, f"{x:.4f}", f"{y:.4f}", f"{yaw:.6f}"])
                self._last_stamp = frame.stamp_ns
                n += 1
                if progress_every and n % progress_every == 0:
                    print(f"  {n} poses written ...", flush=True)
        print(f"wrote {n} poses -> {out_csv}")
        return n
