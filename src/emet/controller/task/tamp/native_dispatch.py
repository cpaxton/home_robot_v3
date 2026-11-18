# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Possible-dispatch accounting around a finalized native executor's transports."""

from contextlib import contextmanager


class CommandBoundary:
    def __init__(self, robot, entering):
        self._robot, self._entering = robot, entering

    def __getattr__(self, name):
        target = getattr(self._robot, name)
        if name not in {"open_gripper", "close_gripper", "gripper_to", "arm_to", "move_base_to"}:
            return target

        def send(*args, **kwargs):
            self._entering()
            return target(*args, **kwargs)

        return send


def robot_identity(robot):
    """Keep adapter-cache identity stable while a transport proxy is installed."""
    while isinstance(robot, CommandBoundary):
        robot = robot._robot
    return robot


@contextmanager
def dispatch_boundary(executor, entering, *, owner=None):
    """Validate before marking a possible send; restore callbacks on every exit.

    Mark before transport: a send can succeed even if its acknowledgement fails.
    Planning-only failures before this boundary can still assert no command.
    """
    robot, command = executor.robot, executor.command_joints
    gripper = executor.command_gripper_joints
    owner_robot = None if owner is None else owner.robot

    def arm(q):
        entering()
        return command(q)

    def grip(q):
        entering()
        return gripper(q)

    executor.robot, executor.command_joints = CommandBoundary(robot, entering), arm
    if owner is not None:
        # Native close/hold callbacks and base commands resolve the adapter's
        # robot at send time rather than going through executor.robot.
        owner.robot = CommandBoundary(owner_robot, entering)
    if gripper is not None:
        executor.command_gripper_joints = grip
    try:
        yield
    finally:
        executor.robot, executor.command_joints = robot, command
        executor.command_gripper_joints = gripper
        if owner is not None:
            owner.robot = owner_robot
