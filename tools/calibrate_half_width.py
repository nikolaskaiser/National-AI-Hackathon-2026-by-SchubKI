#!/usr/bin/env python3
"""Calibrate the border-prior half-width used by the particle filter.

The filter in camloc assumes the car centre sits a roughly constant distance from
the nearest surveyed track border. The default (6.0 m) is a guess. This script
measures the *real* distribution from the ground truth + the track_bounds in the
bag, so you can pass the calibrated value:

    python tools/calibrate_half_width.py <bag.mcap> gt.csv

It prints the median and spread of the car-centre -> nearest-border distance,
and suggests --half-width / --half-width-sigma values for camloc.run. This uses
the GT only for *offline calibration* of a prior, not inside the causal method.
"""
import sys

import numpy as np

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, ".")
from camloc.track import TrackMap  # noqa: E402


def read_track_bounds(bag):
    bounds = {}
    with open(bag, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(
            topics=["/constructor0/track_bounds"]
        ):
            for marker in msg.markers:
                pts = getattr(marker, "points", None)
                if pts:
                    name = marker.ns or f"marker_{marker.id}"
                    bounds[name] = np.array([[p.x, p.y] for p in pts], float)
            if bounds:
                break
    return bounds


def main(bag, gt_csv):
    bounds = read_track_bounds(bag)
    if not bounds:
        print("no track_bounds found in bag")
        return
    track = TrackMap(bounds)
    print(f"borders: {list(bounds)}  ({track.segment_count} segments)")

    d = np.genfromtxt(gt_csv, delimiter=",", names=True)
    xy = np.column_stack([d["x"], d["y"]])
    # subsample for speed if huge
    if len(xy) > 20000:
        xy = xy[:: len(xy) // 20000]
    dist = track.nearest_distance(xy)

    print(f"\ncar-centre -> nearest border distance over {len(xy)} GT poses:")
    print(f"  median   {np.median(dist):.3f} m")
    print(f"  mean     {np.mean(dist):.3f} m")
    print(f"  std      {np.std(dist):.3f} m")
    print(f"  p05/p95  {np.percentile(dist, 5):.3f} / {np.percentile(dist, 95):.3f} m")
    print("\nsuggested flags:")
    print(f"  --half-width {np.median(dist):.2f} "
          f"--half-width-sigma {max(np.std(dist), 0.5):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
