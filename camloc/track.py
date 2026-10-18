"""The track-border map prior.

The task provides the surveyed borders (`track_left`, `track_right`,
`pitlane_left`, `pitlane_right`) as LINE_STRIPs in the `map` frame. They are the
only absolute reference the method is allowed to use (besides the first pose), so
they are what pins down the long-term, drift-free position — especially the
lateral position, which the metric weights most.

`TrackMap` turns those polylines into:
  - a dense set of border line segments, and
  - a fast nearest-point / signed-distance query for an arbitrary map point,
used by the particle filter to score how well a hypothesised pose places the
camera-observed borders onto the real ones.
"""
from __future__ import annotations

import numpy as np


class TrackMap:
    def __init__(self, bounds: dict):
        """bounds: name -> (N, 2) array of map-frame xy polyline vertices."""
        self.bounds = {k: np.asarray(v, float) for k, v in bounds.items() if len(v) >= 2}
        self._seg_a, self._seg_b = self._build_segments()

    def _build_segments(self):
        a_list, b_list = [], []
        for pts in self.bounds.values():
            a_list.append(pts[:-1])
            b_list.append(pts[1:])
        if not a_list:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return np.concatenate(a_list), np.concatenate(b_list)

    @property
    def segment_count(self) -> int:
        return len(self._seg_a)

    def nearest_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from each map point (M,2) to the nearest border segment.

        Vectorised point-to-segment distance against all segments.
        Returns an (M,) array of distances in metres.
        """
        points = np.atleast_2d(np.asarray(points, float))
        if self.segment_count == 0:
            return np.full(len(points), np.inf)

        a = self._seg_a[None, :, :]        # (1, S, 2)
        b = self._seg_b[None, :, :]        # (1, S, 2)
        p = points[:, None, :]             # (M, 1, 2)
        ab = b - a                         # (1, S, 2)
        ap = p - a                         # (M, S, 2)
        denom = np.sum(ab * ab, axis=2)    # (1, S)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        t = np.clip(np.sum(ap * ab, axis=2) / denom, 0.0, 1.0)  # (M, S)
        proj = a + t[:, :, None] * ab      # (M, S, 2)
        d = np.linalg.norm(p - proj, axis=2)  # (M, S)
        return d.min(axis=1)

    def bounding_box(self):
        if self.segment_count == 0:
            return None
        allpts = np.concatenate([self._seg_a, self._seg_b])
        return allpts.min(axis=0), allpts.max(axis=0)
