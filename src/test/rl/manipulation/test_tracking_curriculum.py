from dataclasses import replace

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv, running_cost
from emet.rl.manipulation.tracking import tracking_score
from emet.rl.manipulation.tracking_curriculum import tracking_updates


def passing_summary():
    return {
        "reach": {
            "initialization_failure_rate": 0,
            "unsafe_contact_rate": 0,
            "wheel_lift_rate": 0,
            "retention_lost_rate": 0,
            "ee_position_error_p95": 0.004,
            "ee_orientation_error_p95": 0.01,
            "base_contract_violation_p95": 0.001,
            "base_target_count": 20,
            "base_heading_error_rad_p95": 0.005,
        }
    }


def test_precision_and_motion_curriculum_wait_for_general_and_loaded_evidence():
    config = ManipulationConfig(
        robot="innate_mars",
        tracking_reward=True,
        tracking_precision=2,
        smoothness_curriculum=True,
        motion_penalty_scale=0.05,
    )
    summary = passing_summary()
    assert tracking_updates(config, summary, 19) == {}
    assert tracking_updates(config, summary, 20) == {"tracking_precision": 3, "motion_penalty_scale": 0.4}
    config = replace(
        config,
        grasp_lift_probability=0.5,
        trajectory_preview=True,
        retention_context=True,
        balanced_grasp_setup=True,
        fixed_base_contract=True,
        action_mode="direct_velocity",
    )
    assert tracking_updates(config, summary, 20) == {}
    failed = passing_summary()
    failed["reach"]["retention_lost_rate"] = 0.05
    assert tracking_updates(config, summary, 20, lift_summary=failed) == {}
    failed = passing_summary()
    failed["reach"]["base_heading_error_rad_p95"] = 0.017
    assert tracking_updates(config, summary, 20, lift_summary=failed) == {}
    assert tracking_updates(config, summary, 20, lift_summary=summary) == {
        "tracking_precision": 3,
        "motion_penalty_scale": 0.4,
    }
    config = replace(config, tracking_precision=4, motion_penalty_scale=0.4)
    assert tracking_updates(config, summary, 20, lift_summary=summary) == {"motion_penalty_scale": 1.0}


@pytest.mark.parametrize("scale", [0, 0.05, 1])
def test_motion_curriculum_never_scales_force_or_retention_cost(scale):
    values = {
        "dt": 0.05,
        "force": 5,
        "jerk": np.ones(5) * 10,
        "action": np.ones(8) * 0.3,
        "rotation": 0.01,
        "translation": 0.002,
        "motion_scale": scale,
    }
    baseline = running_cost(**values)
    assert running_cost(**{**values, "force": 15}) - baseline == pytest.approx(0.05 * 0.02 * 10)
    assert running_cost(**{**values, "rotation": 0.02}) - baseline == pytest.approx(0.05 * 2 * 0.01)
    assert running_cost(**{**values, "translation": 0.012}) - baseline == pytest.approx(0.05 * 5 * 0.01)


@pytest.mark.parametrize(
    "change",
    [
        {"motion_penalty_scale": -1},
        {"motion_penalty_scale": np.nan},
        {"hand_gated_base_reward": True},
        {"smoothness_curriculum": True},
        {"jerk_limited_lift_reference": True},
        {"hand_gated_base_reward": 1},
    ],
)
def test_invalid_curriculum_or_missing_tracking_contract_fails(change):
    with pytest.raises(ValueError):
        ManipulationConfig(**change)


@pytest.mark.sim
@pytest.mark.parametrize("mode", ["combined", "base"])
def test_native_base_credit_uses_scored_goal_tick_and_preserves_base_only(mode):
    config = ManipulationConfig(
        robot="innate_mars",
        scenario="reach",
        randomization_scale=0,
        action_mode="direct_velocity",
        tracking_reward=True,
        trajectory_preview=True,
        dynamic_references=True,
        reach_target_mode=mode,
        tracking_precision=2,
        pose_progress_weight=1,
    )
    legacy, gated = ManipulationEnv(config), ManipulationEnv(replace(config, hand_gated_base_reward=True))
    try:
        for env in [legacy, gated]:
            env.reset(seed=300001)
            env.control.set_pose_targets(
                env.control.base_pose(), np.ones(3, dtype=bool), env.control.hand_goal, mode != "base"
            )
        goal_moved = False
        for _ in range(20):
            before = gated.control.hand_goal.copy()
            _, a, done_a, trunc_a, info_a = legacy.step(np.zeros(8))
            _, b, done_b, trunc_b, info_b = gated.step(np.zeros(8))
            np.testing.assert_array_equal(legacy.data.qpos, gated.data.qpos)
            np.testing.assert_array_equal(legacy.data.qvel, gated.data.qvel)
            score = tracking_score(info_b["ee_position_error"], info_b["ee_orientation_error"], 2)
            expected = a - info_a["reward_terms"]["base_tracking_reward"] * (1 - score) if mode == "combined" else a
            assert b == pytest.approx(expected, abs=1e-12)
            assert info_b["reward_terms"]["base_tracking_reward"] == pytest.approx(
                info_a["reward_terms"]["base_tracking_reward"] * (score if mode == "combined" else 1)
            )
            goal_moved |= not np.allclose(before, gated.control.hand_goal, atol=1e-8)
            assert done_a == done_b and trunc_a == trunc_b
            if done_a or trunc_a:
                break
        assert goal_moved
    finally:
        legacy.close()
        gated.close()
