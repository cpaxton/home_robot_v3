"""Lift diagnostics retain the existing acceptance contract and measured failure."""
from types import SimpleNamespace

import numpy as np

from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor


def executor(after):
    result = object.__new__(KinematicPickPlaceExecutor)
    result.robot = SimpleNamespace(_last_step=12, _emet_session_cache_step=10)
    result.grasp_lift_verify_tol_m = .03
    result.lift_m = .12
    result._body_pos = lambda body: None if after is None else np.asarray(after)
    return result


def test_failed_lift_records_actual_pose_and_observation_step():
    ex = executor([0., 0., .01])
    assert not ex._verify_grasp_lift('apple', np.array([0., 0., .13]), pre_pos=np.array([0., 0., .01]))
    evidence = ex.last_grasp_verification
    assert evidence['observed_xyz'] == [0., 0., .01]
    assert evidence['lift_dz_m'] == 0
    assert evidence['command_step'] == 12 and evidence['session_step'] == 10


def test_original_height_or_target_acceptance_is_preserved():
    ex = executor([0., 0., .13])
    assert ex._verify_grasp_lift('apple', np.array([0., 0., .13]), pre_pos=None)
    ex = executor([.2, 0., .1])
    assert ex._verify_grasp_lift('apple', np.array([0., 0., .13]), pre_pos=np.array([0., 0., .01]))
    assert ex.last_grasp_verification['target_error_m'] > ex.grasp_lift_verify_tol_m


def test_missing_pose_is_a_recorded_failure():
    ex = executor(None)
    assert not ex._verify_grasp_lift('apple', np.array([0., 0., .13]), pre_pos=None)
    assert ex.last_grasp_verification['observed_xyz'] is None
