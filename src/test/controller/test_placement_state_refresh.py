"""Measured placement state recovery must not bypass freshness or completeness."""
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation import kinematic_pick_place as module
from emet.controller.task.tamp.api import failure_code


@pytest.fixture
def rig(monkeypatch):
    now = [10.0]
    on_sleep = [lambda: None]

    def sleep(dt):
        now[0] += dt
        on_sleep[0]()

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="tool">
        <joint name="arm" type="slide"/><geom size=".05" mass="1"/>
        </body></worldbody></mujoco>''')
    ex = object.__new__(module.KinematicPickPlaceExecutor)
    ex._model, ex._data = model, mujoco.MjData(model)
    ex.joint_names = ["arm"]
    ex._actuator_names = lambda: ["arm"]
    ex._actuator_to_joint_name = lambda name: name
    ex._sync_base_freejoint = Mock()
    ex.robot = SimpleNamespace(_state_received_monotonic=10., get_joint_state=Mock(return_value=(np.array([.4]), None, None)))
    return ex, now, on_sleep


def test_transient_stale_stream_recovers_without_command_substitution(rig):
    ex, now, on_sleep = rig
    ex.robot._state_received_monotonic = 7.
    on_sleep[0] = lambda: setattr(ex.robot, "_state_received_monotonic", now[0])
    ex._last_cmd_q = np.array([.9])
    assert ex._sync_qpos_from_robot()
    assert ex._data.qpos[0] == .4
    assert ex.last_state_sync["code"] == "ok"
    assert ex.last_state_sync["wait_s"] > 0
    assert ex.last_state_sync["state_age_s"] <= 2.


@pytest.mark.parametrize("problem,code", [
    ("stale", "stale_observation"), ("missing", "missing_joint_state"),
    ("partial", "missing_joint_state"), ("nan", "nonfinite_joint_state"),
    ("infinite_timestamp", "stale_observation"), ("future_timestamp", "stale_observation"),
])
def test_invalid_stream_stops_without_mutating_model(rig, problem, code):
    ex, now, _ = rig
    if problem == "stale":
        ex.robot._state_received_monotonic = 7.
    elif problem == "infinite_timestamp":
        ex.robot._state_received_monotonic = float("inf")
    elif problem == "future_timestamp":
        ex.robot._state_received_monotonic = 50.
    elif problem == "missing":
        ex.robot.get_joint_state.return_value = (None, None, None)
    elif problem == "partial":
        ex.joint_names.append("unobserved")
    else:
        ex.robot.get_joint_state.return_value = (np.array([float("nan")]), None, None)
    assert not ex._sync_qpos_from_robot()
    assert ex._last_motion_failure == code
    assert now[0] <= 12.001
    np.testing.assert_array_equal(ex._data.qpos, [0.])
    ex._sync_base_freejoint.assert_not_called()


def test_age_is_checked_after_blocking_joint_read(rig):
    ex, now, _ = rig

    def read(**_):
        now[0] += 3.
        return np.array([.4]), None, None

    ex.robot.get_joint_state = read
    assert not ex._sync_qpos_from_robot()
    assert ex._last_motion_failure == "stale_observation"
    assert ex._data.qpos[0] == 0.


@pytest.mark.parametrize("code", [*module.PLACEMENT_STATE_FAILURES, "placement_path_invalidated"])
def test_api_keeps_placement_cause_through_nested_stage_failure(code):
    assert failure_code(f"place_failed:preplace_{code}") == code
