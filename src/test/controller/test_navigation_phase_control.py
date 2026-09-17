# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import numpy as np
import pytest

from emet.motion.control.goto_controller import GotoVelocityController


def controller(tolerance):
    control = GotoVelocityController()
    control.update_goal(np.zeros(3))
    control.control.set_linear_error_tolerance(tolerance)
    control.control.set_angular_error_tolerance(0.03)
    return control


@pytest.mark.parametrize("tolerance", [0.02, 0.07])
def test_final_turn_has_a_position_buffer(tolerance):
    control = controller(tolerance).control
    # The old controller immediately turned clockwise at the outer boundary,
    # then reversed to approach heading when XY drifted just outside it.
    for fraction in [1.01, 0.99, 1.01, 0.6]:
        v, w, done = control(np.array([fraction * tolerance, 0, -1.5]))
        assert v > 0 and w == 0 and not done
    for fraction in [0.49, 0.51, 0.9, 0.99]:
        v, w, done = control(np.array([fraction * tolerance, 0, -1.5]))
        assert v == 0 and w < 0 and not done


def test_drift_outside_acceptance_reacquires_position():
    control = controller(0.02).control
    control(np.array([0.009, 0, -1.5]))
    v, w, done = control(np.array([0.021, 0, -1.5]))
    assert v > 0 and w == 0 and not done


def test_arrival_tolerance_is_not_relaxed_or_unnecessarily_tightened():
    control = controller(0.02).control
    assert control(np.array([0.019, 0, 0.029])) == (0, 0, True)
    assert not control(np.array([0.021, 0, 0]))[2]
    assert not control(np.array([0, 0, 0.031]))[2]


def test_new_goal_does_not_inherit_final_turn_phase():
    control = controller(0.02)
    control.control(np.array([0.009, 0, -1.5]))
    assert control.control._at_goal_xy
    control.update_goal(np.array([1, 0, 0]))
    assert not control.control._at_goal_xy
