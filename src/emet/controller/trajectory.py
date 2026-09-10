# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""One serial waypoint contract for ZMQ-backed robot clients."""

import logging

logger = logging.getLogger(__name__)


def execute_waypoints(robot, trajectory, *, relative, world_frame, per_waypoint_timeout, final_timeout):
    """Wait for each command's terminal success before dispatching the next.

    Trajectories are synchronous even when a client's legacy ``blocking``
    argument is false: the command server permits only one active base goal.
    Arrival tolerances are owned by that server, not a second client pose loop.
    """
    waypoints = list(trajectory)
    for index, waypoint in enumerate(waypoints):
        if not robot.move_base_to(
            waypoint,
            relative=relative,
            world_frame=world_frame,
            blocking=True,
            timeout=final_timeout if index == len(waypoints) - 1 else per_waypoint_timeout,
        ):
            logger.warning(
                "Waypoint %d/%d failed: goal=%s terminal_receipt=%s",
                index + 1,
                len(waypoints),
                waypoint,
                getattr(robot, "_command_receipt", None),
            )
            return False
    return True
