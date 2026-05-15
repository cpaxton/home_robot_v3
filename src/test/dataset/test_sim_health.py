# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from emet.dataset.sim_health import RobotSimPhysicsExplodedError, check_robot_sim_stable


class _FakeRobotExplode:
    def get_joint_state(self, timeout: float = 0.5):
        q = np.zeros(4)
        dq = np.ones(4) * 999.0
        return q, dq, np.zeros(4)

    def get_observation(self):
        return None


class _FakeRobotOk:
    def get_joint_state(self, timeout: float = 0.5):
        q = np.zeros(8)
        dq = np.zeros(8)
        return q, dq, np.zeros(8)

    def get_observation(self):
        return None


def test_check_raises_on_absurd_velocities() -> None:
    with pytest.raises(RobotSimPhysicsExplodedError, match="unstable"):
        check_robot_sim_stable(_FakeRobotExplode(), stage="unit")


def test_check_ok_on_zeros() -> None:
    check_robot_sim_stable(_FakeRobotOk(), stage="unit")
