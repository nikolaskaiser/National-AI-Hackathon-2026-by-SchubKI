"""camloc — camera-only localization of the race car in the map frame.

A causal pipeline that estimates the pose (x, y, yaw) of `constructor0/base_link`
in the `map` frame at camera rate (10 Hz), using only the 7 on-board cameras,
their intrinsics / extrinsics, and the surveyed track-border map prior.

Pipeline stages (see the module docstrings for details):

    bag.py        read the ROS 2 / mcap dataset (images, camera_info, tf_static,
                  track_bounds, and the single allowed init pose)
    geometry.py   small SE(2)/SE(3) + quaternion helpers (no scipy dependency)
    track.py      the track-border map prior and point->border distance queries
    vo.py         visual odometry: relative base_link motion between frames
    filter.py     particle filter fusing VO motion with the border map prior
    pipeline.py   wires the stages together and writes submission.csv

The entry point is `python -m camloc.run <bag.mcap> submission.csv`.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
