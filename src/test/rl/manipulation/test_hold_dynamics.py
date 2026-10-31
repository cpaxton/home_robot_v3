import numpy as np
import pytest

from emet.eval.mars_hold_dynamics import quiet_metrics
from emet.eval.mars_policy_audit import stationary_trial
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS


def sample():
    fields, width = history_layout(MARS)
    h = np.zeros((20, width))
    h[:, fields["hand_pose"]] = h[:, fields["hand_goal"]] = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    return {"actor_observation": {"history": h.tolist()}, "action_source": "policy", "failure": None}


def test_quiet_dwell_counts_all_six_dofs_and_never_fault_stops():
    trace = [sample() for _ in range(50)]
    assert quiet_metrics(trace, retention_context=False)["two_second_quiet_pose_hold"]
    fields, _ = history_layout(MARS)
    for i in range(6):
        trace = [sample() for _ in range(50)]
        for row in trace:
            row["actor_observation"]["history"][-1][fields["base_velocity"].start + i] = 0.04
        assert not quiet_metrics(trace, retention_context=False)["two_second_quiet_pose_hold"]
    trace = [sample() for _ in range(50)]
    trace[-1]["failure"] = "policy_inference_deadline"
    assert not quiet_metrics(trace, retention_context=False)["two_second_quiet_pose_hold"]
    with pytest.raises(ValueError, match="observations required"):
        quiet_metrics([{"action_source": "policy"}], retention_context=False)


def test_stationary_audit_rejects_ambiguous_or_nontracking_baselines():
    for kwargs in [
        {"selector": "typo"},
        {"selector": "explicit_hold", "joint_delta": [0] * 5},
        {"selector": "zero", "wheel_feedback_gain": float("nan")},
    ]:
        with pytest.raises(ValueError):
            stationary_trial({}, None, None, **kwargs)


def test_quiet_pose_bounds_are_explicit_and_can_match_the_tamp_contract():
    trace = [sample() for _ in range(50)]
    fields, _ = history_layout(MARS)
    for row in trace:
        row["actor_observation"]["history"][-1][fields["hand_goal"].start] = 0.004
    assert quiet_metrics(trace, retention_context=False)["two_second_quiet_pose_hold"]
    strict = quiet_metrics(trace, retention_context=False, position_tolerance_m=0.003, orientation_tolerance_rad=0.02)
    assert not strict["two_second_quiet_pose_hold"] and strict["position_tolerance_m"] == 0.003
    with pytest.raises(ValueError, match="positive"):
        quiet_metrics(trace, retention_context=False, position_tolerance_m=float("nan"))
