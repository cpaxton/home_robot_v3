# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Owned sensor pixels and calibration behind one coordinator decision image."""

from copy import deepcopy
from hashlib import sha256
from typing import Any
from uuid import uuid4

import numpy as np

from emet.core.interfaces import Observations


def camera_snapshot(observation: Observations) -> tuple[Observations, dict[str, Any]]:
    """Copy head sensor inputs; exclude task labels, wrist and third-person views.

    Stereo head inputs are retained for configured depth providers. Only the
    current decision is held by CHAT; voxel memory owns frames admitted later.
    A copied receive timestamp is not a new sensor acquisition timestamp.
    Missing head RGB is rejected explicitly; it cannot identify a decision view.
    """
    if observation.rgb is None:
        raise ValueError("camera_snapshot requires head RGB")
    names = (
        "gps",
        "compass",
        "rgb",
        "depth",
        "camera_K",
        "camera_pose",
        "head_rgb_right",
        "head_camera_K_right",
        "head_camera_pose_right",
        "emet_session",
        "image_timing",
        "received_monotonic",
    )
    values = {name: deepcopy(getattr(observation, name, None)) for name in names}
    sequence = getattr(observation, "seq_id", getattr(observation, "_seq_id", -1))
    values["seq_id"] = sequence if isinstance(sequence, int) else -1
    values["is_simulation"] = bool(getattr(observation, "is_simulation", False))
    snapshot = Observations(**values)
    identity = {
        "decision_view_id": f"decision-view:{uuid4().hex}",
        "rgb_sha256": sha256(np.ascontiguousarray(snapshot.rgb).tobytes()).hexdigest(),
        "rgb_shape": list(np.shape(snapshot.rgb)),
        "rgb_dtype": str(np.asarray(snapshot.rgb).dtype),
        "sensor_depth_sha256": (
            sha256(np.ascontiguousarray(snapshot.depth).tobytes()).hexdigest() if snapshot.depth is not None else None
        ),
        "frame_sequence": snapshot.seq_id if snapshot.seq_id >= 0 else None,
    }
    return snapshot, identity
