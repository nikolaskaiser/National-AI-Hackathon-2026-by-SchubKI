"""Command-line entry point.

    python -m camloc.run <bag.mcap> submission.csv [--camera fl] [--particles 600]

Produces a submission.csv (timestamp_ns,x,y,yaw) that can be scored with the
provided tools/evaluate.py against the ground truth from tools/extract_gt.py.
"""
from __future__ import annotations

import argparse
import sys

from .filter import FilterConfig
from .pipeline import Pipeline, PipelineConfig
from .vo import VOConfig


def build_config(args) -> PipelineConfig:
    return PipelineConfig(
        reference_camera=args.camera,
        vo=VOConfig(max_features=args.features, robust_scale=args.robust_scale),
        filt=FilterConfig(
            n_particles=args.particles,
            expected_half_width=args.half_width,
            half_width_sigma=args.half_width_sigma,
        ),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="camera-only race-car localization")
    p.add_argument("bag", help="path to the .mcap dataset")
    p.add_argument("out", help="output submission.csv path")
    p.add_argument("--camera", default="fl", help="reference camera for VO (default fl)")
    p.add_argument("--particles", type=int, default=600)
    p.add_argument("--features", type=int, default=1200)
    p.add_argument("--robust-scale", action="store_true",
                   help="use RANSAC ground-plane scale recovery (scale.py)")
    p.add_argument("--half-width", type=float, default=6.0,
                   help="expected car-centre to nearest border distance [m]")
    p.add_argument("--half-width-sigma", type=float, default=2.5)
    args = p.parse_args(argv)

    cfg = build_config(args)
    pipeline = Pipeline(args.bag, cfg)
    print(f"init pose: {pipeline.dataset.init_pose}")
    print(f"track segments: {pipeline.track.segment_count}")
    pipeline.run(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
