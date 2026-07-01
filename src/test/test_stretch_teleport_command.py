# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.simulation.stretch_mujoco.datamodels.status_command import (
    CommandBaseVelocity,
    CommandTeleportBase,
    StatusCommand,
)


def test_zero_base_velocity_does_not_clear_pending_teleport():
    cmd = StatusCommand.default()
    cmd.set_teleport_base(CommandTeleportBase(1.0, 2.0, 0.5, True))
    assert cmd.teleport_base.trigger
    cmd.set_base_velocity(CommandBaseVelocity(0.0, 0.0, True))
    assert cmd.teleport_base.trigger


def test_nonzero_base_velocity_clears_pending_teleport():
    cmd = StatusCommand.default()
    cmd.set_teleport_base(CommandTeleportBase(1.0, 2.0, 0.5, True))
    cmd.set_base_velocity(CommandBaseVelocity(0.1, 0.0, True))
    assert not cmd.teleport_base.trigger
