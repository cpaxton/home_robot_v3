# Copyright (c) Allen Institute for AI (MolmoSpaces). Apache-2.0.
# Vendored from molmo_spaces/utils/pose.py (subset).

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def pos_quat_to_pose_mat(
    pos: np.ndarray | list, quat: np.ndarray | list | None = None
) -> np.ndarray:
    if quat is None:
        assert len(pos) == 7
        quat = pos[3:7]
        pos = pos[0:3]
    assert len(pos) == 3
    assert len(quat) == 4
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    pose_matrix[:3, 3] = pos
    return pose_matrix
