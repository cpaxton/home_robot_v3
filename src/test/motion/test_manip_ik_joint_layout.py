# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from unittest.mock import Mock

import numpy as np
import pytest

from emet.motion.kinematics import HelloStretchKinematics


def model():
    robot = object.__new__(HelloStretchKinematics)
    robot.dof = 11
    robot._manip_dof = 9
    robot.manip_ik_solver = Mock()
    robot.manip_ik_solver.compute_ik.side_effect = lambda p, r, q, **kw: (
        np.zeros(9) if q is None else q.copy(),
        True,
        {},
    )
    return robot


@pytest.mark.parametrize("entry", ["direct", "grasp_frame"])
def test_full_ik_seed_is_converted_once_and_passive_joints_survive(entry):
    robot = model()
    seed = np.linspace(0, 1, 11)
    before = seed.copy()
    pose = (np.array([0, -0.4, 0.7]), np.array([0, 0, 0, 1]))
    if entry == "direct":
        result, success, _ = robot.manip_ik(pose, q0=seed)
    else:
        result, _, _, success, _ = robot.manip_ik_for_grasp_frame(*pose, q0=seed)
    assert success
    np.testing.assert_array_equal(robot.manip_ik_solver.compute_ik.call_args.args[2], robot._to_manip_format(seed))
    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(seed, before)
    assert result.shape == (11,)


@pytest.mark.parametrize("seed", [None, np.arange(9) * 0.01])
def test_controlled_or_default_seed_still_returns_full_joint_layout(seed):
    robot = model()
    result, _, _, success, _ = robot.manip_ik_for_grasp_frame([0, -0.4, 0.7], [0, 0, 0, 1], q0=seed)
    assert success and result.shape == (11,)
    expected = np.zeros(9) if seed is None else seed.copy()
    expected[2:6] = expected[2:6].mean()  # Full layout stores total telescoping extension.
    np.testing.assert_array_equal(robot._to_manip_format(result), expected)


def test_unknown_seed_layout_is_rejected_before_solver():
    robot = model()
    with pytest.raises(ValueError, match="joint layout"):
        robot.manip_ik(([0, -0.4, 0.7], [0, 0, 0, 1]), q0=np.zeros(6))
    robot.manip_ik_solver.compute_ik.assert_not_called()
