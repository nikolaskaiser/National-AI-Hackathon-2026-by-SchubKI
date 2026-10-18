"""Pose / transform helpers, numpy-only (no scipy dependency).

Conventions
-----------
- Quaternions are stored (x, y, z, w) to match ROS geometry_msgs/Quaternion.
- SE(3) transforms are 4x4 homogeneous matrices.
- `map` frame and `base_link` follow REP-103: x forward, y left, z up.
- Camera optical frames follow the ROS optical convention: z forward, x right,
  y down. The extrinsics in /tf_static map base_link -> each camera frame.
- A 2-D pose is (x, y, yaw) in the map frame.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Quaternions
# --------------------------------------------------------------------------- #
def quat_to_matrix(q) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def quat_to_yaw(q) -> float:
    """Yaw (rotation about map z) from a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_to_quat(yaw: float):
    """Yaw about z -> quaternion (x, y, z, w)."""
    return (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))


# --------------------------------------------------------------------------- #
# SE(3)
# --------------------------------------------------------------------------- #
def make_transform(translation, quat) -> np.ndarray:
    """(t, q) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat)
    T[:3, 3] = translation
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# --------------------------------------------------------------------------- #
# SE(2)
# --------------------------------------------------------------------------- #
def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def pose2d_to_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """(x, y, yaw) -> 3x3 SE(2) matrix."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def matrix_to_pose2d(T: np.ndarray):
    """3x3 SE(2) matrix -> (x, y, yaw)."""
    return float(T[0, 2]), float(T[1, 2]), float(np.arctan2(T[1, 0], T[0, 0]))


def compose2d(pose, delta):
    """Apply a body-frame increment `delta` to a map-frame pose.

    pose, delta are (x, y, yaw). Returns the new map-frame pose.
    """
    return matrix_to_pose2d(
        pose2d_to_matrix(*pose) @ pose2d_to_matrix(*delta)
    )


def relative2d(pose_a, pose_b):
    """Body-frame motion from pose_a to pose_b (both map-frame (x, y, yaw))."""
    from_a = np.linalg.inv(pose2d_to_matrix(*pose_a))
    return matrix_to_pose2d(from_a @ pose2d_to_matrix(*pose_b))
