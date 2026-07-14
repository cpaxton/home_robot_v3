"""Stretch nav / look_front head pose should be room-scale, not floor-staring."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from emet.controller.base_controller import BaseController
from emet.motion import constants as motion_constants


def test_look_front_is_mild_downward_not_floor():
    pan, tilt = motion_constants.look_front
    assert pan == pytest.approx(0.0)
    assert tilt == pytest.approx(math.radians(-30))
    # Historical nav posture used ~-65°; keep look_front well above that.
    assert tilt > math.radians(-50)


def test_stretch_navigation_q_uses_look_front_head():
    q = motion_constants.STRETCH_NAVIGATION_Q
    assert q[9] == pytest.approx(motion_constants.look_front[0])
    assert q[10] == pytest.approx(motion_constants.look_front[1])


def test_base_controller_start_calls_look_front():
    class _Agent(BaseController):
        def get_voxel_map(self):
            return None

    robot = MagicMock()
    robot.start.return_value = True
    agent = _Agent.__new__(_Agent)
    agent.robot = robot
    BaseController.start(agent, can_move=True, verbose=False)
    robot.move_to_nav_posture.assert_called()
    robot.look_front.assert_called_with(blocking=True)
