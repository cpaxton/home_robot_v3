# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import numpy as np
import pytest

from emet.motion.control.goto_controller import GotoVelocityController


def controller():
    control = GotoVelocityController()
    control.update_goal(np.zeros(3))
    control.control.set_linear_error_tolerance(0.005)
    control.control.set_angular_error_tolerance(0.03)
    control.translation_only = True
    return control


@pytest.mark.parametrize("longitudinal", [-0.04, 0.04])
def test_translation_joint_does_not_face_a_small_lateral_residual(longitudinal):
    control = controller()
    control.update_pose_feedback(np.array([-longitudinal, 0.009, 0]))
    v, w = control.compute_control()
    assert v * longitudinal > 0
    assert abs(v) <= 0.08
    assert w == 0
    assert not control.is_done()


def test_translation_preserves_reference_heading_and_bounds_turn_rate():
    control = controller()
    control.update_pose_feedback(np.array([-0.04, 0, 0.2]))
    v, w = control.compute_control()
    assert v == 0  # restore heading before translating
    assert -0.15 <= w < 0


def test_translation_completes_without_turning_to_correct_uncommanded_lateral_drift():
    control = controller()
    control.update_pose_feedback(np.array([-0.004, 0.009, 0.02]))
    assert control.compute_control() == (0, 0)
    assert control.is_done()


def test_new_navigation_goal_restores_general_pose_control():
    control = controller()
    control.update_goal(np.array([0, 0.5, 0]))
    assert not control.translation_only
    assert control.compute_control()[1] > 0
