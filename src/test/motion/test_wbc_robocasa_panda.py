# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""RoboCasa Panda adapter frame tests (no scene construction)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("robosuite")

from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend


def test_base_pose_uses_the_reset_anchored_yaw():
    backend = object.__new__(RoboCasaPandaBackend)
    backend._origin_xy = np.array([1.0, 2.0], dtype=np.float64)
    backend._origin_yaw = np.pi / 2
    backend._world_base_pose = lambda: (1.0, 3.0, np.pi)

    pose = backend._base_pose()
    # World offset (0, 1) rotates to (1, 0) in the reset-anchored frame; the
    # heading is the same -pi/2 offset as the EE local pose.
    np.testing.assert_allclose(pose[:2], [1.0, 0.0], atol=1e-9)
    assert pose[2] == pytest.approx(np.pi / 2)
