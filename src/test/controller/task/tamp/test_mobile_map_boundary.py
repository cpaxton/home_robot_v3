# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Retired map evidence cannot cross the mobile harness command boundary."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.motion.local_scene_store import LocalMapUnavailable


@pytest.fixture
def mobile_robot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "scripts"))
    from eval_mobile_mapped_contact import MobileRobot

    return MobileRobot


@pytest.mark.parametrize("reason", ["expired", "retired"])
def test_base_dispatch_rechecks_lease_and_leaves_cancellation_available(reason, mobile_robot):
    robot = mobile_robot.__new__(mobile_robot)
    robot.commands = []
    robot.nav_goal = None
    robot.data = SimpleNamespace(time=0.0, ctrl=np.ones(2))
    robot.model = SimpleNamespace(actuator=lambda name: SimpleNamespace(id=int(name.startswith("right"))))
    robot.motion_gate = Mock(side_effect=LocalMapUnavailable(reason))
    with pytest.raises(LocalMapUnavailable):
        robot.move_base_to([0.1, 0.0, 0.0])
    assert robot.nav_goal is None
    assert robot.commands == []
    assert robot.cancel_navigation() is True
    np.testing.assert_array_equal(robot.data.ctrl, [0, 0])
