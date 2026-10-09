"""Lift diagnostics reject displaced lifts and require measured arm arrival."""
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


def test_height_gain_cannot_accept_a_displaced_object():
    ex = executor([0., 0., .13])
    assert ex._verify_grasp_lift('apple', np.array([0., 0., .13]), pre_pos=None)
    ex = executor([.2, 0., .1])
    assert not ex._verify_grasp_lift('apple', np.array([0., 0., .13]), pre_pos=np.array([0., 0., .01]))
    assert ex.last_grasp_verification['target_error_m'] > ex.grasp_lift_verify_tol_m


def test_missing_pose_is_a_recorded_failure():
    ex = executor(None)
    assert not ex._verify_grasp_lift('apple', np.array([0., 0., .13]), pre_pos=None)
    assert ex.last_grasp_verification['observed_xyz'] is None



def test_planned_pose_is_not_a_substitute_for_measured_joint_state():
    ex = executor(None)
    ex.ik_tol_m = .035
    ex._data = SimpleNamespace(body=lambda name: SimpleNamespace(xpos=np.array([0., 0., .13])))
    ex.ee_body = 'tool'
    ex._sync_qpos_from_robot = lambda: False
    assert not ex._wait_measured_ee(np.array([0., 0., .13]), timeout_s=0)[0]
    assert ex.last_ee_verification['observed_xyz'] is None
    ex._sync_qpos_from_robot = lambda: True
    assert ex._wait_measured_ee(np.array([0., 0., .13]), timeout_s=0)[0]
    assert not ex._wait_measured_ee(np.array([1., 0., .13]), timeout_s=0)[0]


def test_tracking_evidence_distinguishes_unapplied_command_from_joint_error():
    import mujoco
    ex = executor(None)
    ex._model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="tool">
      <joint name="arm" type="slide"/><geom type="sphere" size=".1"/>
      </body></worldbody><actuator><position name="arm" joint="arm"/></actuator></mujoco>''')
    ex._data = mujoco.MjData(ex._model)
    ex._data.qpos[0] = .2
    ex.joint_names = ('arm',)
    ex._last_cmd_q = np.array([.8])
    ex._actuator_names = lambda: ['arm']
    ex.robot._state = {'actuator_targets': [.7], 'step': 42}
    evidence = ex._joint_tracking_evidence()
    assert evidence['planned_q'] == [.8]
    assert evidence['server_targets'] == [.7]
    assert evidence['observed_q'] == [.2]
    ex._last_motion_failure = 'tracking_failed'
    assert ex._stage_failure('pregrasp') == 'pregrasp_tracking_failed'
