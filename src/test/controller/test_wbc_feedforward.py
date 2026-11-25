# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Failure-driven embodiment calibration: loader, fit, and executor feedforward."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.manipulation.failure_dataset import load_manipulation_attempts
from emet.controller.manipulation.wbc_feedforward import (
    FeedforwardCalibration,
    FeedforwardModel,
    apply_bias,
    fit_feedforward,
)


def _record(target, observed, *, robot_id="rby1", mode="tracking_failed", joints=("torso_joint1", "torso_joint2")):
    return {
        "context": {"robot_id": robot_id, "failure_mode": mode, "op": "grasp"},
        "attempt": {
            "joint_names": list(joints),
            "planned_waypoints": [[0.0] * len(joints), list(target)],
            "trace": [{"qpos": list(observed)} for _ in range(4)],
            "failure_mode": mode,
            "error_m": 0.5,
        },
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "manip_failures.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_fit_recovers_steady_state_bias(tmp_path):
    path = _write(tmp_path, [_record([1.0, 0.5], [0.6, 0.5]), _record([1.2, 0.5], [0.8, 0.5])])
    attempts = load_manipulation_attempts(path)
    model = fit_feedforward(attempts, embodiment="rby1", min_samples=2, cluster_distance=0.3)

    assert model.samples == 2
    assert model.joint_bias["torso_joint1"] == pytest.approx(0.4)
    assert model.joint_bias["torso_joint2"] == pytest.approx(0.0)


def test_fit_fails_closed_without_enough_samples(tmp_path):
    path = _write(tmp_path, [_record([1.0, 0.5], [0.6, 0.5])])
    with pytest.raises(ValueError, match="no posture had enough"):
        fit_feedforward(load_manipulation_attempts(path), min_samples=2)


def test_fit_excludes_implausible_outlier(tmp_path):
    path = _write(
        tmp_path,
        [_record([1.0, 0.5], [0.0, 0.5]), _record([1.2, 0.5], [0.2, 0.5])],
    )
    model = fit_feedforward(
        load_manipulation_attempts(path), min_samples=2, max_abs_bias=0.5, cluster_distance=0.3
    )
    assert "torso_joint1" not in model.joint_bias
    assert model.joint_bias["torso_joint2"] == pytest.approx(0.0)


def test_model_round_trips_and_biases_by_joint_name(tmp_path):
    model = FeedforwardModel(
        embodiment="rby1",
        calibrations=(
            FeedforwardCalibration(reference_q={"torso_joint1": 0.0}, joint_bias={"torso_joint1": 0.3}),
        ),
        samples=2,
    )
    path = model.save(tmp_path / "ff.json")
    loaded = FeedforwardModel.load(path)
    assert loaded.joint_bias == {"torso_joint1": 0.3}
    np.testing.assert_allclose(loaded.bias_vector(["other", "torso_joint1"]), [0.0, 0.3])


def test_apply_bias_clips_to_joint_limits():
    model = FeedforwardModel(
        embodiment="rby1",
        calibrations=(FeedforwardCalibration(reference_q={"j": 1.0}, joint_bias={"j": 0.5}),),
    )
    biased = apply_bias([1.0], ["j"], model, lower=[0.0], upper=[1.2])
    np.testing.assert_allclose(biased, [1.2])


def test_executor_applies_feedforward_to_the_final_settle(monkeypatch):
    import emet.controller.manipulation.kinematic_pick_place as kpp
    import emet.motion.mujoco_arm_ik as arm_ik

    clock = {"now": 0.0}
    monkeypatch.setattr(kpp.time, "time", lambda: clock["now"])
    monkeypatch.setattr(arm_ik, "joint_qpos_addrs", lambda model, joints: list(range(len(joints))))
    monkeypatch.setattr(
        kpp,
        "solve_position_ik_multiseed",
        lambda *a, **k: SimpleNamespace(success=True, pos_error_m=0.0),
    )
    final_q = np.array([-0.25, 0.4])
    monkeypatch.setattr(
        kpp,
        "plan_arm_joint_path",
        lambda *a, **k: SimpleNamespace(
            success=True, planner="linear", detail=None, waypoints=[np.zeros(2), final_q]
        ),
    )

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    ex = object.__new__(KinematicPickPlaceExecutor)
    ex._model, ex._data = object(), SimpleNamespace(qpos=np.zeros(2))
    ex.joint_names = ("torso_joint1", "arm")
    ex.ee_body = "tool"
    ex.ik_tol_m, ex.ik_max_iters, ex.traj_dt, ex.traj_steps = 0.01, 1, 0.05, 2
    ex.manip_planner, ex.rrt_max_iter, ex._collision, ex.visualizer = "linear", 1, None, None
    ex._last_cmd_q = None
    ex._feedforward = FeedforwardModel(
        embodiment="rby1",
        calibrations=(
            FeedforwardCalibration(reference_q={"torso_joint1": -0.25}, joint_bias={"torso_joint1": 0.2}),
        ),
        samples=1,
    )
    ex._joint_position_bounds = lambda: (np.full(2, -np.inf), np.full(2, np.inf))
    ex._sync_qpos_from_robot = lambda: True
    ex._sleep = lambda seconds: clock.__setitem__("now", clock["now"] + seconds)
    ex._stream_arm_q = Mock()
    ex._fk_ee_path = lambda waypoints: []
    ex._wait_measured_ee = Mock(return_value=(True, 0.0))

    reached, error = ex._plan_and_execute_ee(np.array([0.0, 0.0, 0.8]))

    assert reached and error == 0.0
    expected = final_q + np.array([0.2, 0.0])
    for call in ex._stream_arm_q.call_args_list[2:]:
        np.testing.assert_allclose(call.args[0], expected)
    assert ex.last_attempt["reached"] is True

def test_bias_is_skipped_outside_the_calibrated_posture():
    model = FeedforwardModel(
        embodiment="rby1",
        calibrations=(FeedforwardCalibration(reference_q={"j": 1.0}, joint_bias={"j": 0.5}),),
        max_joint_distance=0.2,
    )
    np.testing.assert_allclose(apply_bias([1.1], ["j"], model), [1.6])
    np.testing.assert_allclose(apply_bias([2.0], ["j"], model), [2.0])

def test_posture_table_selects_the_nearest_calibration():
    model = FeedforwardModel(
        embodiment="rby1",
        calibrations=(
            FeedforwardCalibration(reference_q={"j": 0.0}, joint_bias={"j": 0.2}),
            FeedforwardCalibration(reference_q={"j": 1.0}, joint_bias={"j": 0.5}),
        ),
        max_joint_distance=0.15,
    )
    np.testing.assert_allclose(apply_bias([0.05], ["j"], model), [0.25])
    np.testing.assert_allclose(apply_bias([0.95], ["j"], model), [1.45])
    np.testing.assert_allclose(apply_bias([2.0], ["j"], model), [2.0])


def test_schema1_model_loads_as_one_calibration():
    model = FeedforwardModel.from_dict(
        {"schema": 1, "embodiment": "rby1", "joint_bias": {"j": 0.3}, "reference_q": {"j": 0.0}}
    )
    assert len(model.calibrations) == 1
    assert model.calibrations[0].joint_bias == {"j": 0.3}


def test_lift_outside_the_tight_envelope_does_not_inherit_pregrasp_bias():
    model = FeedforwardModel(
        embodiment="rby1",
        calibrations=(
            FeedforwardCalibration(
                reference_q={"torso_joint3": 0.6926},
                joint_bias={"torso_joint3": 0.3684},
            ),
        ),
    )
    # Measured pregrasp is 0.015 rad from the reference and gets the correction.
    np.testing.assert_allclose(
        apply_bias([0.7074], ["torso_joint3"], model), [1.0758], atol=1e-4
    )
    # The lift is 0.168 rad away; with the 0.12 envelope it must pass through.
    np.testing.assert_allclose(apply_bias([0.8605], ["torso_joint3"], model), [0.8605])


def test_executor_resolves_ik_from_the_measured_posture_when_tracking_fails(monkeypatch):
    import emet.controller.manipulation.kinematic_pick_place as kpp
    import emet.motion.mujoco_arm_ik as arm_ik

    clock = {"now": 0.0}
    monkeypatch.setattr(kpp.time, "time", lambda: clock["now"])
    monkeypatch.setattr(arm_ik, "joint_qpos_addrs", lambda model, joints: list(range(len(joints))))
    monkeypatch.setattr(
        kpp,
        "solve_position_ik_multiseed",
        lambda *a, **k: SimpleNamespace(success=True, pos_error_m=0.0),
    )
    monkeypatch.setattr(
        kpp,
        "plan_arm_joint_path",
        lambda *a, **k: SimpleNamespace(
            success=True, planner="linear", detail=None, waypoints=[np.zeros(2), np.zeros(2)]
        ),
    )
    solved = []

    def fake_live_ik(model, data, *, ee_body, joint_names, target_pos, **kwargs):
        solved.append((np.asarray(data.qpos).copy(), np.asarray(target_pos).copy()))
        return SimpleNamespace(success=True, qpos=np.array([0.5, 0.3]))

    monkeypatch.setattr(arm_ik, "solve_position_ik", fake_live_ik)

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    ex = object.__new__(KinematicPickPlaceExecutor)
    ex._model, ex._data = object(), SimpleNamespace(qpos=np.array([0.4, 0.2]))
    ex.joint_names = ("torso_joint1", "arm")
    ex.ee_body = "tool"
    ex.ik_tol_m, ex.ik_max_iters, ex.traj_dt, ex.traj_steps = 0.01, 1, 0.05, 2
    ex.manip_planner, ex.rrt_max_iter, ex._collision, ex.visualizer = "linear", 1, None, None
    ex._last_cmd_q = None
    ex._feedforward = None
    ex._joint_position_bounds = lambda: (np.full(2, -np.inf), np.full(2, np.inf))
    ex._sync_qpos_from_robot = lambda: True
    ex._sleep = lambda seconds: clock.__setitem__("now", clock["now"] + seconds)
    ex._stream_arm_q = Mock()
    ex._fk_ee_path = lambda waypoints: []
    ex._wait_measured_ee = Mock(side_effect=[(False, 0.5), (True, 0.01)])

    reached, error = ex._plan_and_execute_ee(np.array([0.0, 0.0, 0.8]))

    assert reached and error == 0.01
    assert len(solved) == 1
    # The live solve starts from the measured qpos, not the commanded plan.
    np.testing.assert_allclose(solved[0][0], [0.4, 0.2])
    np.testing.assert_allclose(ex._stream_arm_q.call_args_list[-1].args[0], [0.5, 0.3])
