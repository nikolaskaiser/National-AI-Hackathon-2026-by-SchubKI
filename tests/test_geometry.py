"""Numpy-free checks of the geometry + track math.

These run without numpy / opencv / mcap, so they work even in a minimal
environment. They mirror the formulas in camloc/geometry.py and camloc/track.py
and assert the invariants the pipeline relies on.

    python -m pytest tests/            # if pytest is available
    python tests/test_geometry.py      # or run directly
"""
import math


def quat_to_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def mat3_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def pose2d_to_matrix(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, -s, x], [s, c, y], [0, 0, 1]]


def matrix_to_pose2d(T):
    return (T[0][2], T[1][2], math.atan2(T[1][0], T[0][0]))


def se2_inv(T):
    c, s = T[0][0], T[1][0]
    x, y = T[0][2], T[1][2]
    r00, r01, r10, r11 = c, s, -s, c
    return [[r00, r01, -(r00 * x + r01 * y)],
            [r10, r11, -(r10 * x + r11 * y)],
            [0, 0, 1]]


def compose2d(pose, delta):
    return matrix_to_pose2d(mat3_mul(pose2d_to_matrix(*pose), pose2d_to_matrix(*delta)))


def relative2d(a, b):
    return matrix_to_pose2d(mat3_mul(se2_inv(pose2d_to_matrix(*a)), pose2d_to_matrix(*b)))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def point_seg_dist(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    denom = abx * abx + aby * aby
    t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


def nearest_distance(p, polylines):
    best = float("inf")
    for pts in polylines:
        for i in range(len(pts) - 1):
            best = min(best, point_seg_dist(p, pts[i], pts[i + 1]))
    return best


def test_quat_yaw_roundtrip():
    for yaw in [-3.0, -1.2, 0.0, 0.5, 1.5, 3.1]:
        assert abs(wrap(quat_to_yaw(yaw_to_quat(yaw)) - yaw)) < 1e-9


def test_compose_relative_inverse():
    a, delta = (10.0, -5.0, 0.7), (2.0, 0.3, 0.15)
    rec = relative2d(a, compose2d(a, delta))
    assert all(abs(wrap(g - w)) < 1e-9 for g, w in zip(rec, delta))


def test_compose_body_frame_convention():
    pose = (0.0, 0.0, math.pi / 2)
    nx, ny, _ = compose2d(pose, (1.0, 0.0, 0.0))   # forward -> +y in map
    assert abs(nx) < 1e-9 and abs(ny - 1.0) < 1e-9
    nx, ny, _ = compose2d(pose, (0.0, 1.0, 0.0))   # left -> -x in map
    assert abs(nx + 1.0) < 1e-9 and abs(ny) < 1e-9


def test_track_distance():
    poly = [[(0, 0), (10, 0)]]
    assert abs(nearest_distance((5, 3), poly) - 3.0) < 1e-9
    assert abs(nearest_distance((13, 4), poly) - 5.0) < 1e-9
    poly2 = [[(0, 0), (10, 0)], [(0, 12), (10, 12)]]
    assert abs(nearest_distance((5, 5), poly2) - 5.0) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nALL TESTS PASSED")
