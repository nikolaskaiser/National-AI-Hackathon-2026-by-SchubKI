"""Command-line entry point.

    python -m camloc.run <bag.mcap> submission.csv [--method centerline|vo] ...

Produces a submission.csv (timestamp_ns,x,y,yaw) that can be scored with the
provided tools/evaluate.py against the ground truth from tools/extract_gt.py.

Methods
-------
  centerline  (default)  follow the track midline, advance by a scalar speed
                         estimate. Robust: lateral error small by construction.
  vo                     full-pose monocular visual odometry + particle filter
                         against the borders (research baseline; drifts badly on
                         this night/high-speed data — kept for reference).
"""
from __future__ import annotations

import argparse
import sys


def run_vo(args) -> int:
    from .filter import FilterConfig
    from .pipeline import Pipeline, PipelineConfig
    from .vo import VOConfig

    cfg = PipelineConfig(
        reference_camera=args.camera,
        vo=VOConfig(max_features=args.features, robust_scale=args.robust_scale),
        filt=FilterConfig(
            n_particles=args.particles,
            expected_half_width=args.half_width,
            half_width_sigma=args.half_width_sigma,
        ),
    )
    pipeline = Pipeline(args.bag, cfg)
    print(f"init pose: {pipeline.dataset.init_pose}")
    print(f"track segments: {pipeline.track.segment_count}")
    pipeline.run(args.out)
    return 0


def run_centerline(args) -> int:
    from .pipeline_centerline import CenterlineConfig, CenterlinePipeline
    from .speed import SpeedConfig

    cfg = CenterlineConfig(
        reference_camera=args.camera,
        centerline_step=args.centerline_step,
        speed=SpeedConfig(max_features=args.features),
    )
    pipeline = CenterlinePipeline(args.bag, cfg)
    print(f"init pose: {pipeline.dataset.init_pose}")
    print(f"centerline length: {pipeline.centerline.length:.1f} m "
          f"({'closed loop' if pipeline.centerline.closed else 'open'})")
    print(f"init progress s0: {pipeline.s:.1f} m, direction: {pipeline.direction:+.0f}")
    pipeline.run(args.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="camera-only race-car localization")
    p.add_argument("bag", help="path to the .mcap dataset")
    p.add_argument("out", help="output submission.csv path")
    p.add_argument("--method", choices=["centerline", "vo"], default="centerline")
    p.add_argument("--camera", default="fl", help="reference camera (default fl)")
    # centerline method
    p.add_argument("--centerline-step", type=float, default=1.0,
                   help="centerline resample spacing [m]")
    # vo method
    p.add_argument("--particles", type=int, default=600)
    p.add_argument("--robust-scale", action="store_true",
                   help="use RANSAC ground-plane scale recovery (scale.py)")
    p.add_argument("--half-width", type=float, default=6.0,
                   help="expected car-centre to nearest border distance [m]")
    p.add_argument("--half-width-sigma", type=float, default=2.5)
    # shared
    p.add_argument("--features", type=int, default=1200)
    args = p.parse_args(argv)

    if args.method == "vo":
        return run_vo(args)
    return run_centerline(args)


if __name__ == "__main__":
    sys.exit(main())
