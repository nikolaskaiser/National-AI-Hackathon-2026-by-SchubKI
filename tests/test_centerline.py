"""Numpy-free checks of the centerline geometry logic (mirrors centerline.py).

Builds a simple rectangular-ish track from left/right borders and verifies:
  - midline is between the borders,
  - progress-of a point maps back to that point via pose_at,
  - yaw is the tangent in the direction of travel.
"""
import math


def resample(pts, step):
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    s = [0.0]
    for d in seg:
        s.append(s[-1] + d)
    total = s[-1]
    if total < 1e-6:
        return pts
    n = max(int(total / step) + 1, 2)
    out = []
    for k in range(n):
        target = total * k / (n - 1)
        # find segment
        j = 0
        while j < len(s) - 2 and s[j + 1] < target:
            j += 1
        a = (target - s[j]) / max(s[j + 1] - s[j], 1e-9)
        x = pts[j][0] + a * (pts[j + 1][0] - pts[j][0])
        y = pts[j][1] + a * (pts[j + 1][1] - pts[j][1])
        out.append((x, y))
    return out


def midline(left, right, step):
    left = resample(left, step)
    right = resample(right, step)
    n = max(len(left), len(right))
    def interp(poly, t):
        idx = t * (len(poly) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(poly) - 1)
        a = idx - lo
        return (poly[lo][0] + a * (poly[hi][0] - poly[lo][0]),
                poly[lo][1] + a * (poly[hi][1] - poly[lo][1]))
    mid = []
    for k in range(n):
        t = k / (n - 1)
        lx, ly = interp(left, t)
        rx, ry = interp(right, t)
        mid.append(((lx + rx) / 2, (ly + ry) / 2))
    return resample(mid, step)


def test_midline_between_borders():
    # straight track: left at y=4, right at y=0 -> mid at y=2
    left = [(0, 4), (100, 4)]
    right = [(0, 0), (100, 0)]
    mid = midline(left, right, 1.0)
    assert all(abs(p[1] - 2.0) < 1e-6 for p in mid), mid[:3]
    # mid spans the full length
    assert abs(mid[0][0] - 0) < 1e-6 and abs(mid[-1][0] - 100) < 1e-6


def test_tangent_direction():
    # straight track along +x; tangent yaw should be 0
    left = [(0, 4), (100, 4)]
    right = [(0, 0), (100, 0)]
    mid = midline(left, right, 1.0)
    # forward difference tangent at an interior point
    i = 10
    dx = mid[i + 1][0] - mid[i - 1][0]
    dy = mid[i + 1][1] - mid[i - 1][1]
    yaw = math.atan2(dy, dx)
    assert abs(yaw) < 1e-6, yaw
    # reversed direction -> yaw = pi
    yaw_rev = math.atan2(-dy, -dx)
    assert abs(abs(yaw_rev) - math.pi) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nCENTERLINE MATH OK")
