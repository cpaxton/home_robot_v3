"""A matched analytic baseline must never masquerade as learned inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.rl.manipulation.tamp_policy import TaskPoseTampReference


def test_reference_delegates_admission_and_settings_but_never_calls_learned_actor(monkeypatch):
    from emet.rl.manipulation import command_runtime, tamp_policy

    seen = []

    class BoundSettings:
        def __init__(self, directory, env, *, evidence):
            seen.append((directory, env, evidence))
            self.admission = {"weights_sha256": "fixed", "simulation_only": True}

        def __call__(self, obs):
            pytest.fail("reference called learned actor")

        def verify_unchanged(self):
            seen.append("verify")

    env = SimpleNamespace(control=object())
    monkeypatch.setattr(tamp_policy, "CheckpointTampPolicy", BoundSettings)
    monkeypatch.setattr(command_runtime, "task_pose_teacher_action", lambda c: np.ones(8) if c is env.control else None)
    reference = TaskPoseTampReference("actor", env, evidence="evidence")
    np.testing.assert_array_equal(reference({}), np.ones(8))
    reference.verify_unchanged()
    assert seen == [("actor", env, "evidence"), "verify"]
    assert reference.calls == 1 and reference.is_reference_controller
    assert reference.admission["matched_checkpoint_settings"]["weights_sha256"] == "fixed"
    assert reference.admission["reference_controller"] and not reference.admission["safety_promoted"]
    robot = object.__new__(MarsSimRobot)
    robot.policy, robot.policy_calls = reference, 1
    assert robot.controller_name == "task_pose_reference"
    assert robot.inference_counts == {"policy_calls": 0, "reference_controller_calls": 1}


def test_reference_admission_failure_prevents_teacher_construction(monkeypatch):
    from emet.rl.manipulation import tamp_policy

    def reject(*args, **kwargs):
        raise ValueError("unbound settings")

    monkeypatch.setattr(tamp_policy, "CheckpointTampPolicy", reject)
    with pytest.raises(ValueError, match="unbound settings"):
        TaskPoseTampReference("missing", object(), evidence="missing")


@pytest.mark.parametrize("enabled,mode", [(True, "joint"), (1, "cartesian"), (None, "cartesian")])
def test_reference_mode_must_be_explicit_before_loading_any_checkpoint(enabled, mode):
    with pytest.raises(ValueError, match="explicit Cartesian"):
        MarsSimRobot.from_checkpoint(None, "missing", evidence="missing", task_pose_reference=enabled, arm_execution=mode)
