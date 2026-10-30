# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Narrow RGB-D / learned-label interchange for the isolated Hydra probe.

This is an input contract, not proof of a label array's provenance. The capture
driver must run the detector on RGB; this module intentionally has no dataset,
simulator, question, or annotation dependency.
"""

from collections.abc import Mapping

import numpy as np

PACKET_FIELDS = frozenset({"rgb", "depth", "camera_K", "camera_pose", "timestamp_ns", "learned_labels"})


def validate_packet(packet: Mapping[str, np.ndarray], *, num_labels: int) -> None:
    """Reject extra fields and malformed sensor/learned-perception outputs."""
    if set(packet) != PACKET_FIELDS:
        raise ValueError(f"Packet fields must be exactly {sorted(PACKET_FIELDS)}")
    rgb, depth, labels = packet["rgb"], packet["depth"], packet["learned_labels"]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("RGB must be nonempty HxWx3 uint8")
    if depth.dtype != np.float32 or depth.shape != rgb.shape[:2]:
        raise ValueError("Depth must be HxW float32 in meters")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("Depth must be finite and nonnegative (zero means unavailable)")
    if labels.dtype != np.int32 or labels.shape != depth.shape:
        raise ValueError("Learned labels must be HxW int32")
    if num_labels <= 0 or (labels < 0).any() or (labels >= num_labels).any():
        raise ValueError("Learned labels exceed the declared vocabulary")
    intrinsics = packet["camera_K"]
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
    ):
        raise ValueError("Invalid camera intrinsics")
    pose = packet["camera_pose"]
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Invalid camera pose")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0):
        raise ValueError("Camera rotation must be right handed and orthonormal")
    stamp = packet["timestamp_ns"]
    if stamp.shape != () or not np.issubdtype(stamp.dtype, np.integer) or int(stamp) <= 0:
        raise ValueError("Timestamp must be a positive integer scalar")


def learned_label_map(semantic: np.ndarray, instances: np.ndarray, label_ids: list[int]) -> np.ndarray:
    """Translate detector class indices; undetected pixels remain unknown (0).

    YOLOE's semantic background and its first class both use zero. Only its
    instance support distinguishes the two; never turn the background into an
    object or assume that undetected pixels are floor.
    """
    if semantic.shape != instances.shape or semantic.ndim != 2:
        raise ValueError("Detector outputs must be matching HxW arrays")
    if not label_ids or min(label_ids) <= 0 or len(set(label_ids)) != len(label_ids):
        raise ValueError("Detector label IDs must be unique positive integers")
    present = instances >= 0
    indices = semantic[present]
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Detector class indices must be finite integers")
    if (indices < 0).any() or (indices >= len(label_ids)).any():
        raise ValueError("Detector class index outside vocabulary")
    labels = np.zeros(semantic.shape, dtype=np.int32)
    labels[present] = np.asarray(label_ids, dtype=np.int32)[indices.astype(np.int64)]
    return labels


def command_camera_pose(*, yaw_deg: float, tilt_deg: float, sensor_height: float) -> np.ndarray:
    """OpenCV camera -> local Z-up world, from mount calibration and yaw command.

    Initial base is the origin facing +X. This in-place-turn probe does not use
    simulator global pose or map geometry as a pose estimator.
    """
    if not np.isfinite([yaw_deg, tilt_deg, sensor_height]).all() or sensor_height <= 0:
        raise ValueError("Invalid command pose / camera calibration")
    yaw, pitch = np.deg2rad([yaw_deg, -tilt_deg])
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    world_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    mount_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    optical_to_base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    pose = np.eye(4)
    pose[:3, :3] = world_yaw @ mount_pitch @ optical_to_base
    pose[2, 3] = sensor_height
    return pose
