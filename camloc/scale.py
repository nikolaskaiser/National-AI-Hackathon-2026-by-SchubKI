"""Robust ground-plane metric-scale recovery for monocular VO (drop-in upgrade).

Why this exists
---------------
The baseline scale estimator in vo.py (`VisualOdometry._recover_scale`) takes the
*median* depth of forward-and-below points and matches it to the known camera
mounting height. That is fragile on a race track: the triangulated cloud mixes
true ground points with barriers, kerbs, distant structure and triangulation
noise, and at ~40 m/s a few outliers drag the median — corrupting the scale and,
through it, the longitudinal position (the error the metric splits out as
`longitudinal RMSE`).

This module recovers scale the way the monocular-VO literature recommends:
fit the local ground *plane* to the triangulated points with RANSAC, keep only
the inliers, and solve for the scale that puts the camera at its known height
above that plane. This is the standard ground-plane scale cue
(e.g. ground-plane absolute-scale estimation, arXiv:1903.00912; plane-geometry
scale recovery, arXiv:2101.05995; "known height of the camera from the ground
plane" for scale-drift correction, Chandraker et al., TPAMI 2015). Sources are
rephrased for compliance with licensing restrictions.

Usage (swap into vo.py):
    from .scale import ground_plane_scale
    scale = ground_plane_scale(self.K, R, t_dir, p0, p1, self.cam_height, cv2)
    # falls back to None exactly like the baseline, so the caller is unchanged.
"""
from __future__ import annotations

import numpy as np


def _triangulate(K, R, t_dir, p0, p1, cv2):
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t_dir.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    w = pts4[3]
    ok = np.abs(w) > 1e-6
    if ok.sum() < 8:
        return None
    X = (pts4[:3, ok] / w[ok]).T  # (M, 3), unit-baseline camera-0 frame
    # keep points in front of both cameras
    X = X[X[:, 2] > 0]
    return X if len(X) >= 8 else None


def _ransac_plane(points, n_iter=80, thresh=0.05, rng=None):
    """Fit a plane n·x + d = 0 to (M,3) points; return (normal, d, inlier_mask).

    thresh is the inlier band in *unit-baseline* coordinates; it is a fraction of
    the scene, not metres, so it stays scale-free until we apply the camera height.
    """
    rng = rng or np.random.default_rng(0)
    m = len(points)
    best_inliers = None
    best_count = 0
    for _ in range(n_iter):
        idx = rng.choice(m, 3, replace=False)
        a, b, c = points[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ a
        dist = np.abs(points @ n + d)
        inliers = dist < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_count < max(8, int(0.2 * m)):
        return None
    # refit on all inliers (least squares plane through centroid)
    pin = points[best_inliers]
    centroid = pin.mean(axis=0)
    _, _, vh = np.linalg.svd(pin - centroid)
    n = vh[-1]
    d = -n @ centroid
    return n, float(d), best_inliers


def ground_plane_scale(K, R, t_dir, p0, p1, cam_height, cv2,
                       n_iter=80, thresh=0.05):
    """Robust metric scale from the ground plane under a known camera height.

    Returns a positive float scale (metres per unit baseline), or None if the
    geometry is degenerate — identical contract to the baseline estimator, so it
    is a safe drop-in.
    """
    X = _triangulate(K, R, t_dir, p0, p1, cv2)
    if X is None:
        return None

    # bias the sample toward the lower half of the image cloud (optical y is
    # down, so ground points have larger y) to help RANSAC lock onto the road.
    y = X[:, 1]
    lower = X[y > np.median(y)]
    candidates = lower if len(lower) >= 8 else X

    fit = _ransac_plane(candidates, n_iter=n_iter, thresh=thresh)
    if fit is None:
        return None
    n, d, inliers = fit

    # Distance from the camera origin (0,0,0) to the plane = |d| (n is unit).
    # That distance, in unit-baseline units, should equal cam_height in metres.
    ground_dist_unit = abs(d)
    if ground_dist_unit <= 1e-6:
        return None
    scale = cam_height / ground_dist_unit
    if not np.isfinite(scale) or scale <= 0:
        return None
    return float(scale)
