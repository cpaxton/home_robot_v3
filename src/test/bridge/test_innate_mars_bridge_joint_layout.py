# Copyright (c) Hello Robot, Inc.
# All rights reserved.

import numpy as np
from innate_mars_bridge.joint_layout import (
    INNATE_MARS_JOINT_NAMES,
    pack_innate_mars_joint_positions,
    pack_innate_mars_joint_velocities,
)


def test_pack_joint_positions_base_plus_arm():
    arm = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    base = np.array([1.0, 2.0, 0.25])
    q = pack_innate_mars_joint_positions(arm, base_xyt=base)
    assert q.shape == (len(INNATE_MARS_JOINT_NAMES),)
    np.testing.assert_allclose(q[:3], base)
    np.testing.assert_allclose(q[3:9], arm)
    assert q[9] == arm[5]


def test_pack_joint_velocities_gripper_mimic():
    arm_dq = np.ones(6)
    dq = pack_innate_mars_joint_velocities(arm_dq)
    assert dq[9] == 1.0
