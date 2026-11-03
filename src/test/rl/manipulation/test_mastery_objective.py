"""Reward must distinguish contract success from merely reaching the pose."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.mastery_objective import (
    ContractObjective,
    ContractRewardConfig,
    TeacherContractReward,
    add_contract_reward_arguments,
    contract_reward_from_args,
    mastery_metrics,
    smoothness_budgets,
)


def suite():
    return {
        "gate": {
            "force_floor_N": 1.0,
            "base_linear_jerk_floor_m_s3": 0.1,
            "base_yaw_jerk_floor_rad_s3": 0.5,
            "joint_jerk_floor_rad_s3": 1.0,
            "smoothness_ratio": 1.25,
            "terminal_hold_s": 2,
            "hold_drift_m": 0.003,
        }
    }


def metrics():
    return {
        "peak_force_N": 0.5,
        "base_linear_jerk": 0.2,
        "base_yaw_jerk": 0.1,
        "joint_jerk": 0.5,
        "settled_seconds": 2.5,
        "hold_drift_m": 0.001,
    }


def info():
    return {
        "is_success": True,
        "reference_complete": True,
        "reward_terms": {"success_bonus": 10.0},
        "control_tick_jerk": {"base": [0, 0], "joint": [0] * 5},
    }


def test_exact_teacher_relative_budgets_with_physical_floors():
    assert smoothness_budgets(suite(), metrics()) == {
        "peak_force_N": 1.0,
        "base_linear_jerk": 0.25,
        "base_yaw_jerk": 0.5,
        "joint_jerk": 1.0,
    }


def test_bonus_requires_terminal_full_contract_and_is_paid_once():
    objective = ContractObjective(suite(), metrics())
    reward, terms = objective.apply(10.2, info(), metrics(), dt=0.05, terminal=False, path_failed=False)
    assert reward == pytest.approx(0.2)
    assert terms["contract_success_bonus"] == 0
    reward, terms = objective.apply(10.2, info(), metrics(), dt=0.05, terminal=True, path_failed=False)
    assert reward == pytest.approx(10.2) and terms["contract_success_bonus"] == 10
    assert objective.apply(10.2, info(), metrics(), dt=0.05, terminal=True, path_failed=False)[0] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "failure",
    [
        "wheel_lift",
        "unsafe_contact",
        "invalid_physics",
        "retention_lost",
        "initialization_failed",
        "not_settled",
        "drift",
        "path_failed",
        "unfinished",
    ],
)
def test_bad_contract_cannot_collect_completion_bonus(failure):
    observation, measured = info(), metrics()
    if failure == "not_settled":
        measured["settled_seconds"] = 1.9
    elif failure == "drift":
        measured["hold_drift_m"] = 0.004
    elif failure == "unfinished":
        observation["reference_complete"] = False
    elif failure != "path_failed":
        observation[failure] = True
    _, terms = ContractObjective(suite(), metrics()).apply(
        10.2, observation, measured, dt=0.05, terminal=True, path_failed=failure == "path_failed"
    )
    assert terms["contract_success_bonus"] == 0


def test_one_spike_is_charged_once_as_a_peak_and_blocks_later_success():
    objective = ContractObjective(suite(), metrics())
    measured = metrics()
    measured["base_linear_jerk"] = 0.5
    observation = info()
    observation["control_tick_jerk"]["base"][0] = 0.5
    _, terms = objective.apply(10.2, observation, measured, dt=0.05, terminal=False, path_failed=False)
    excess = np.sqrt(2) - 1  # jerk / budget == 2
    assert terms["contract_jerk_excess_cost"] == pytest.approx(0.05 * excess)
    assert terms["contract_peak_excess_cost"] == pytest.approx(2 * excess)
    observation["control_tick_jerk"]["base"][0] = 0
    reward, terms = objective.apply(10.2, observation, measured, dt=0.05, terminal=True, path_failed=False)
    assert terms["contract_peak_excess_cost"] == terms["contract_jerk_excess_cost"] == 0
    assert terms["contract_success_bonus"] == 0 and reward == pytest.approx(0.2)


def test_dense_cost_integrates_time_while_peak_cost_does_not_repeat():
    measured, observation = metrics(), info()
    measured["base_linear_jerk"] = observation["control_tick_jerk"]["base"][0] = 0.5
    one, two = ContractObjective(suite(), metrics()), ContractObjective(suite(), metrics())
    cost1 = one.apply(10, observation, measured, dt=0.1, terminal=False, path_failed=False)[0]
    cost2 = sum(two.apply(10, observation, measured, dt=0.05, terminal=False, path_failed=False)[0] for _ in range(2))
    assert cost1 == pytest.approx(cost2)


def test_benign_contact_is_free_but_over_budget_peak_blocks_success():
    measured = metrics()
    measured["peak_force_N"] = 0.9
    objective = ContractObjective(suite(), metrics())
    assert objective.apply(10.2, info(), measured, dt=0.05, terminal=True, path_failed=False)[0] == pytest.approx(10.2)
    measured["peak_force_N"] = 2
    _, terms = ContractObjective(suite(), metrics()).apply(
        10.2, info(), measured, dt=0.05, terminal=True, path_failed=False
    )
    assert terms["contract_peak_excess_cost"] > 0 and terms["contract_success_bonus"] == 0


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_reward_weights_rejected(weight):
    with pytest.raises(ValueError):
        ContractRewardConfig(peak_excess_weight=weight)


def test_cli_requires_explicit_reward_enable_and_records_tunable_weights():
    import argparse

    parser = argparse.ArgumentParser()
    add_contract_reward_arguments(parser)
    assert contract_reward_from_args(parser.parse_args([])) is None
    with pytest.raises(ValueError, match="require --contract-reward"):
        contract_reward_from_args(parser.parse_args(["--peak-excess-weight", "3"]))
    result = contract_reward_from_args(parser.parse_args(["--contract-reward", "--peak-excess-weight", "3"]))
    assert result == ContractRewardConfig(peak_excess_weight=3)


@pytest.mark.sim
def test_native_teacher_wrapping_preserves_physics_observations_and_delays_bonus():
    from emet.rl.manipulation.command_runtime import task_pose_teacher_action
    from emet.rl.manipulation.mastery import MasteryEnv, load_suite

    spec = load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_command_history_v1.json"
    )
    env = MasteryEnv(spec, stages=["base"], split="train")
    try:
        cid = "train:base:hold"
        env.reset(options={"case_id": cid})
        admission = deepcopy(env.mastery_admission)
        originals = []
        while True:
            obs, reward, done, truncated, result = env.step(task_pose_teacher_action(env.control))
            originals.append((obs, reward, done, truncated, result))
            if done or truncated:
                break
        measured = mastery_metrics(env, result)
        teacher = {"cases": [{"id": cid, "mastered": True, "admission": admission, "metrics": measured}]}
        wrapped = TeacherContractReward(env, teacher)
        wrapped.reset(options={"case_id": cid})
        bonuses, old_bonuses = [], []
        for old_obs, old_reward, old_done, old_truncated, old_info in originals:
            obs, reward, done, truncated, result = wrapped.step(task_pose_teacher_action(env.control))
            for key in obs:
                np.testing.assert_array_equal(obs[key], old_obs[key])
            assert (done, truncated) == (old_done, old_truncated)
            bonuses.append(result["reward_terms"]["contract_success_bonus"])
            old_bonuses.append(old_info["reward_terms"]["success_bonus"])
            assert reward == pytest.approx(old_reward - old_bonuses[-1] + bonuses[-1])
            assert result["reward_terms"]["contract_peak_excess_cost"] == 0
            assert result["reward_terms"]["contract_jerk_excess_cost"] == 0
        assert sum(bonuses) == sum(old_bonuses) == 10
        assert bonuses[-1] == 10 and old_bonuses[-1] == 0
        assert result["contract_reward"]["terminal_pass"] is True
        assert mastery_metrics(env, result) == measured
        teacher["cases"][0]["admission"]["reference_sha256"] = "wrong"
        with pytest.raises(ValueError, match="identity mismatch"):
            wrapped.reset(options={"case_id": cid})
    finally:
        env.close()
