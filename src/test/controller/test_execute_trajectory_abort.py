# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Both clients dispatch each waypoint once and require its command outcome."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("kind", ["stretch", "generic"])
@pytest.mark.parametrize(
    "outcomes,expected,calls",
    [([True, True, True], True, 3), ([True, False], False, 2), ([True, True, False], False, 3)],
)
def test_serial_waypoints_stop_on_failure(kind, outcomes, expected, calls):
    if kind == "stretch":
        from emet.controller.zmq_client import StretchZmqClient as Client
    else:
        from emet.controller.generic_zmq_client import GenericZmqClient as Client
    robot = SimpleNamespace(move_base_to=Mock(side_effect=outcomes), wait_for_waypoint=Mock())
    trajectory = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert (
        Client.execute_trajectory(
            robot, trajectory, per_waypoint_timeout=2, final_timeout=5, blocking=False, world_frame=True
        )
        is expected
    )
    assert robot.move_base_to.call_count == calls
    for index, call in enumerate(robot.move_base_to.call_args_list):
        assert call.args == (trajectory[index],)
        assert call.kwargs["blocking"] is True
        assert call.kwargs["world_frame"] is True
        assert call.kwargs["timeout"] == (5 if index == 2 else 2)
    robot.wait_for_waypoint.assert_not_called()
