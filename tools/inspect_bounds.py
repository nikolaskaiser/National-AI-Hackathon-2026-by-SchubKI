#!/usr/bin/env python3
"""Quick diagnostic: inspect the track_bounds polylines in the bag.

    python tools/inspect_bounds.py <bag.mcap>

Prints each border's name, point count, endpoints and total arc length, plus the
GT start position, so we can confirm how to build the centerline + lap length.
"""
import sys

import numpy as np

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def arclen(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def main(bag):
    bounds = {}
    init = None
    with open(bag, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, msg in reader.iter_decoded_messages(
            topics=["/constructor0/track_bounds", "/tf"]
        ):
            if ch.topic == "/constructor0/track_bounds":
                for m in msg.markers:
                    pts = getattr(m, "points", None)
                    if pts:
                        name = m.ns or f"marker_{m.id}"
                        if name not in bounds:
                            bounds[name] = np.array([[p.x, p.y] for p in pts], float)
            elif ch.topic == "/tf" and init is None:
                for t in msg.transforms:
                    if t.child_frame_id == "constructor0/base_link":
                        p = t.transform.translation
                        init = (p.x, p.y)
            if bounds and init and len(bounds) >= 2:
                break

    print(f"GT start xy: {init}\n")
    for name, pts in bounds.items():
        closed = np.linalg.norm(pts[0] - pts[-1]) < 1.0
        print(f"{name:16s} {len(pts):5d} pts  len={arclen(pts):8.1f} m  "
              f"start={pts[0].round(1)} end={pts[-1].round(1)}  "
              f"{'CLOSED' if closed else 'open'}")


if __name__ == "__main__":
    main(sys.argv[1])
