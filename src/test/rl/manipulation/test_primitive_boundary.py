# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""An unloaded primitive cannot proceed into closure or loaded execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv, _Boundary, _bounded_path


@pytest.mark.parametrize("primitive,phases", [("pregrasp", ["pregrasp"]), ("grasp_pose", ["pregrasp", "grasp"])])
@pytest.mark.parametrize("success", [False, True])
def test_unwinds_before_closure_on_success_and_failed_endpoint(primitive, phases, success):
    calls = []
    result = SimpleNamespace(success=success, message="endpoint")

    def execute(phase, path):
        calls.append(phase)
        return result

    bounded = _bounded_path(execute, primitive)
    with pytest.raises(_Boundary) as boundary:
        for phase in ("pregrasp", "grasp", "close", "lift"):
            bounded(phase, [])
    assert calls == phases and boundary.value.result is result


@pytest.mark.parametrize("primitive", ["lift", "close", "grasp", "", None, True, [], 1])
def test_rejects_unsupported_boundary_before_spawning(primitive):
    with pytest.raises(ValueError, match="primitive must be"):
        TampPrimitiveEnv([{"id": "stub"}], primitive=primitive)


def test_default_boundary_remains_pregrasp():
    env = TampPrimitiveEnv([{"id": "stub"}])
    assert env.primitive == "pregrasp" and env.process is None
    env.close()


def test_rollout_propagates_boundary_and_keeps_early_failure(monkeypatch):
    from emet.rl.manipulation import tamp_primitive_env
    from emet.rl.manipulation.whole_body_training import primitive_rollout

    received = []

    class Rejected:
        def __init__(self, *args, **kwargs):
            received.append(kwargs["primitive"])

        def reset(self, **kwargs):
            raise RuntimeError("public_plan:planning_failed")

        def close(self):
            pass

    monkeypatch.setattr(tamp_primitive_env, "TampPrimitiveEnv", Rejected)
    (row,) = primitive_rollout([{"id": "test"}], primitive="grasp_pose")
    assert received == ["grasp_pose"]
    assert row["primitive"] == "grasp_pose" and not row["success"]
    assert row["initialization_failed"] and row["stop_confirmed"] is None
    assert row["code"] == "public_plan:planning_failed"


@pytest.mark.sim
def test_native_grasp_pose_stops_before_closure_and_preserves_tick_credit():
    from emet.rl.manipulation.whole_body_training import load_development

    suite, _ = load_development()
    env = TampPrimitiveEnv(
        suite["train_cases"][1:2], oracle=True, primitive="grasp_pose", wheel_feedback_current_velocity=True
    )
    try:
        _, info = env.reset()
        while True:
            _, _, done, truncated, info = env.step(info["oracle_action"])
            ticks = info["reward_group_ticks"]
            np.testing.assert_allclose(ticks.sum(0), info["reward_groups"], rtol=1e-5, atol=1e-6)
            assert ticks.shape == (info["elapsed_control_ticks"], 3)
            if done or truncated:
                break
        assert info["primitive"] == "grasp_pose" and info["boundary_reached"]
        assert info["primitive_success"] and info["code"] == "grasp_pose_complete"
        assert info["terminal_pose_in_contract"] and info["gripper_open"]
        assert info["terminal_pose"]["position_error_m"] <= 0.003
        assert info["terminal_pose"]["orientation_error_rad"] <= 0.02
        assert info["stop_confirmed"] and not info["full_pick_place_success"]
        commands = info["commands"]
        assert any(c.get("phase") == "grasp" for c in commands)
        assert not any(c.get("gripper") == "close" or c.get("phase") == "lift" for c in commands)
        terms = info["reward_terms"]
        final_tracking = 0.05 * (terms["hand_pose_score"] + terms["base_pose_score"])
        assert ticks[-1, 0] == pytest.approx(final_tracking + 2, abs=1e-6)
    finally:
        env.close()
