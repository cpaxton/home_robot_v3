# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared DynaMem controller constants and small helpers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

# Env truthy check (same tokens as ``EMET_DYNAMEM_PERFECT_DEPTH``).
_TRUEISH = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUEISH


# Manipulation hyperparameters
INIT_LIFT_POS = 0.45
INIT_WRIST_PITCH = -1.57
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.57
INIT_HEAD_TILT = -0.65

# After look_front / move_to_nav_posture, wait briefly so the head reaches goal and depth/RGB
# stabilize before base motion (Stretch ZMQ + mapping).
DYNAMEM_HEAD_SETTLE_S = 0.25
# Head sweep: command non-blocking; exit on near-goal or settled motion.
# Soft-wait is for client settle (not because real Stretch is slow — Dynamixel head ~3 rad/s).
# Sim MJCF used to use head kp=10 (crawl); assets now use higher kp. Keep a short max wait anyway.
DYNAMEM_HEAD_SWEEP_MAX_WAIT_S = 0.75
DYNAMEM_HEAD_SWEEP_MIN_MOVE_S = 0.08
DYNAMEM_HEAD_SWEEP_STOPPED_HOLD_S = 0.05
DYNAMEM_HEAD_SWEEP_SPEED_TOL = 0.20
DYNAMEM_HEAD_SWEEP_POS_DELTA_TOL = 0.04
DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD = 0.35
DYNAMEM_HEAD_SWEEP_FRAME_SETTLE_S = 0.08

# A* is executed in 8-waypoint chunks so look-around can grow the map along a
# long path. Leftover must be resumed (empty-text explore used to drop it and
# pick a new frontier). One investigate/explore call hops until arrival.
DYNAMEM_NAV_CHUNK_WPS = 8
DYNAMEM_NAV_MAX_HOPS = 8
# Keep default-table rby1 scans on the workspace instead of 45° floor turns.
DEFAULT_TABLE_MAPPING_YAW_HALF_RAD = float(np.deg2rad(25.0))


def default_table_mapping_relative_yaws(n_extra: int) -> list[float]:
    """Relative yaw steps that stay on the default-table workspace (±25°).

    ``n_extra`` is the number of turns *after* the initial heading has already
    been captured. For the OVMM 4-view scan that is 3: +25°, −50°, +25° (back
    to the table-facing heading).
    """
    n = max(0, int(n_extra))
    if n == 0:
        return []
    half = DEFAULT_TABLE_MAPPING_YAW_HALF_RAD
    if n == 1:
        return [half]
    if n == 2:
        return [half, -2.0 * half]
    seq = [half, -2.0 * half, half]
    while len(seq) < n:
        seq.append(half)
    return seq[:n]


def _finite_xyz_traj_target(traj_target_point: Any) -> bool:
    """True if traj tail looks like a 3D world point (not a waypoint or NaN sentinel)."""
    if isinstance(traj_target_point, torch.Tensor):
        t = traj_target_point.detach().cpu().reshape(-1)
        return t.numel() >= 3 and bool(torch.isfinite(t[:3]).all())
    if isinstance(traj_target_point, np.ndarray):
        a = np.asarray(traj_target_point, dtype=np.float64).reshape(-1)
        return a.size >= 3 and bool(np.all(np.isfinite(a[:3])))
    if isinstance(traj_target_point, (list, tuple)) and len(traj_target_point) >= 3:
        a = np.asarray(traj_target_point[:3], dtype=np.float64)
        return bool(np.all(np.isfinite(a)))
    return False


# Batched OWL text queries for describe_head_camera_scene_text (single forward pass).
_DESCRIBE_SCENE_OWL_QUERIES: tuple[str, ...] = (
    "table",
    "chair",
    "person",
    "cup",
    "bottle",
    "laptop",
    "computer monitor",
    "television",
    "cabinet",
    "shelf",
    "door",
    "window",
    "couch",
    "bed",
    "counter",
    "box",
    "bowl",
    "plate",
    "plant",
    "book",
    "keyboard",
    "microwave",
    "refrigerator",
    "sink",
    "robot arm",
)

# Household-ish labels for user-facing YoloE describe_scene (not full ScanNet-200).
# Mapping still uses ScanNet-200 at a low confidence; describe uses this vocab + a higher bar.
_DESCRIBE_SCENE_YOLOE_LABELS: tuple[str, ...] = _DESCRIBE_SCENE_OWL_QUERIES + (
    "desk",
    "sofa",
    "lamp",
    "pillow",
    "curtain",
    "picture",
    "mirror",
    "trash can",
    "bag",
    "phone",
    "remote",
    "wall",
    "floor",
    "ceiling",
    "stairs",
    "rug",
    "blanket",
    "towel",
    "toilet",
    "bathtub",
    "oven",
    "dishwasher",
    "washer",
    "dryer",
    "fan",
    "clock",
    "vase",
    "apple",
    "banana",
    "orange",
    "mouse",
    "tv stand",
    "nightstand",
    "dresser",
    "wardrobe",
    "stool",
    "bench",
    "fireplace",
    "radiator",
)

# User-facing describe_scene: mapping keeps detection.confidence_threshold low for proposals;
# chat-only filtering uses describe_confidence_threshold (safe to raise; does not affect mapping).
_DEFAULT_DESCRIBE_CONFIDENCE = 0.30
_DEFAULT_DESCRIBE_MAX_LABELS = 12
_DESCRIBE_SCENE_STRUCTURE_LABELS = frozenset({"floor", "wall", "ceiling"})
