"""Reading the ROS 2 / mcap dataset.

This wraps `mcap` + `mcap_ros2` so the rest of the pipeline never touches the
bag format directly. Everything here is *causal-friendly*: messages come back in
log order, so a streaming consumer only ever sees data up to the current time.

Only these inputs are read (all allowed by the task):
    - camera compressed_image         (the 7 cameras, jpeg, 10 Hz)
    - camera camera_info              (K, D per camera, 1 Hz, latched)
    - /tf_static                      (base_link -> camera extrinsics, latched)
    - /constructor0/track_bounds      (surveyed borders in map, latched)

The ground-truth topics (/tf, /constructor0/state_estimation) are read ONLY to
recover the very first pose of the run, which the task explicitly allows as the
initial pose. They are never used afterwards.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a runtime dependency
    cv2 = None

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from . import geometry

CAMERAS = ["fl", "fr", "cl", "cr", "r", "rl", "rr"]
NS = 10 ** 9


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * NS + int(header.stamp.nanosec)


@dataclass
class CameraModel:
    """Intrinsics + the static base_link -> optical-frame extrinsic."""

    name: str
    K: np.ndarray | None = None          # 3x3 intrinsic matrix
    D: np.ndarray | None = None          # distortion (plumb_bob)
    width: int = 0
    height: int = 0
    T_base_cam: np.ndarray | None = None  # 4x4, base_link -> camera optical

    @property
    def ready(self) -> bool:
        return self.K is not None and self.T_base_cam is not None


@dataclass
class CameraFrame:
    """A single decoded camera image with its exposure timestamp."""

    camera: str
    stamp_ns: int
    image: np.ndarray  # BGR uint8


@dataclass
class Dataset:
    """Static context + an iterator over camera frames in log order."""

    bag_path: str
    cameras: dict = field(default_factory=dict)
    track_bounds: dict = field(default_factory=dict)  # name -> (N,2) map xy
    init_pose: tuple | None = None  # (x, y, yaw) first GT pose, allowed as init
    init_stamp_ns: int | None = None

    # --------------------------------------------------------------------- #
    def load_static(self) -> "Dataset":
        """Read the latched topics + first GT pose; cheap, scans headers only."""
        topics = (
            [f"/constructor0/sensor/camera_{c}/camera_info" for c in CAMERAS]
            + ["/tf_static", "/constructor0/track_bounds", "/tf"]
        )
        self.cameras = {c: CameraModel(c) for c in CAMERAS}
        got_init = False
        with open(self.bag_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, msg in reader.iter_decoded_messages(
                topics=topics
            ):
                topic = channel.topic
                if topic.endswith("/camera_info"):
                    self._read_camera_info(topic, msg)
                elif topic == "/tf_static":
                    self._read_tf_static(msg)
                elif topic == "/constructor0/track_bounds":
                    self._read_track_bounds(msg)
                elif topic == "/tf" and not got_init:
                    got_init = self._read_init_pose(msg)
                if (
                    got_init
                    and self.track_bounds
                    and all(m.ready for m in self.cameras.values())
                ):
                    break
        return self

    def _read_camera_info(self, topic: str, msg) -> None:
        cam = topic.split("/camera_")[1].split("/")[0]
        m = self.cameras[cam]
        if m.K is None:
            m.K = np.asarray(msg.k, float).reshape(3, 3)
            m.D = np.asarray(msg.d, float)
            m.width, m.height = int(msg.width), int(msg.height)

    def _read_tf_static(self, msg) -> None:
        for t in msg.transforms:
            child = t.child_frame_id.replace("constructor0/", "")
            if child.startswith("camera_"):
                cam = child.split("camera_")[1]
                if cam in self.cameras and self.cameras[cam].T_base_cam is None:
                    q = t.transform.rotation
                    p = t.transform.translation
                    self.cameras[cam].T_base_cam = geometry.make_transform(
                        (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
                    )

    def _read_track_bounds(self, msg) -> None:
        for marker in msg.markers:
            pts = getattr(marker, "points", None)
            if not pts:
                continue
            name = marker.ns or f"marker_{marker.id}"
            self.track_bounds[name] = np.array([[p.x, p.y] for p in pts], float)

    def _read_init_pose(self, msg) -> bool:
        for t in msg.transforms:
            if (
                t.header.frame_id == "map"
                and t.child_frame_id == "constructor0/base_link"
            ):
                q = t.transform.rotation
                p = t.transform.translation
                self.init_pose = (p.x, p.y, geometry.quat_to_yaw((q.x, q.y, q.z, q.w)))
                self.init_stamp_ns = _stamp_ns(t.header)
                return True
        return False

    # --------------------------------------------------------------------- #
    def iter_camera_frames(self, cameras=None):
        """Yield CameraFrame objects in log (time) order — causal by construction."""
        if cv2 is None:
            raise RuntimeError("opencv-python is required to decode images")
        cams = cameras or CAMERAS
        topics = [f"/constructor0/sensor/camera_{c}/compressed_image" for c in cams]
        topic_to_cam = {
            f"/constructor0/sensor/camera_{c}/compressed_image": c for c in cams
        }
        with open(self.bag_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, msg in reader.iter_decoded_messages(
                topics=topics
            ):
                buf = np.frombuffer(bytes(msg.data), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                yield CameraFrame(
                    camera=topic_to_cam[channel.topic],
                    stamp_ns=_stamp_ns(msg.header),
                    image=img,
                )
