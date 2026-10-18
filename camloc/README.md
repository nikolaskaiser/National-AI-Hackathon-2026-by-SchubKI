# camloc — camera-only race-car localization

Estimates the pose (`x`, `y`, `yaw`) of `constructor0/base_link` in the `map`
frame at camera rate (10 Hz) from the 7 on-board cameras only, and writes a
`submission.csv` that can be scored with `tools/evaluate.py`.

## What it uses (all allowed by the task)

- the camera images (`compressed_image`),
- intrinsics (`camera_info`) and extrinsics (`/tf_static`),
- the surveyed track borders (`/constructor0/track_bounds`) as a map prior,
- the **first** ground-truth pose, only to initialise the filter.

It never reads `/tf` or `/state_estimation` afterwards, and every emitted pose
is computed from images with stamp ≤ its own timestamp — the method is causal.

## How it works

```
images ──► visual odometry (vo.py) ──► (dx, dy, dyaw) body-frame motion
                                              │
track_bounds ─► TrackMap (track.py) ─► particle filter (filter.py) ─► pose
                                              │
first GT pose ────────────────────────────────┘ (init only)
```

1. **Visual odometry** tracks features across successive frames of a front
   camera, recovers relative camera motion via the essential matrix, and maps it
   into `base_link` through the static extrinsic. The monocular scale is recovered
   metrically from the known camera mounting height over the (locally planar)
   track surface, so the VO is self-sufficient (no start-up speed deadlock); a
   constant-velocity fallback covers degenerate frames.
2. **Particle filter** propagates the VO increment with process noise and
   re-weights particles by how well their position agrees with the surveyed
   borders (the car drives between left/right borders at a roughly constant
   half-width). This cancels the long-term VO drift — especially lateral error,
   which the metric weights most.

The design is modular on purpose: each stage can be swapped (e.g. a learned
place-recognition front end, or line-level border matching) without touching the
others.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m camloc.run c0_2026-10-13_22-53-42_camloc_0.mcap submission.csv
```

Useful flags:

```
--camera fl            reference camera for VO (fl/fr are the narrow front cams)
--particles 600        particle count
--half-width 6.0       expected car-centre → nearest-border distance [m]
--half-width-sigma 2.5 tolerance on the above [m]
```

## Score it

```bash
python tools/extract_gt.py c0_2026-10-13_22-53-42_camloc_0.mcap gt.csv
python tools/evaluate.py gt.csv submission.csv
```

## Tuning notes

- `--half-width` is a track-geometry prior. If the start/finish straight is much
  wider or narrower than the surveyed median, lower `half_width_sigma` once the
  value is calibrated from a short GT segment to tighten the lateral correction.
- Front cameras (`fl`, `fr`) give the cleanest forward VO; `--camera fr` is a
  good cross-check. The rear camera `r` has the most reliable yaw calibration and
  is a candidate reference for a yaw-focused variant.
- Raising `--particles` improves robustness in ambiguous stretches (long
  straights where the border constraint is weak) at a linear cost.
