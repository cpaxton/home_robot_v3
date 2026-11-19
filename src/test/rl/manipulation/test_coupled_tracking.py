"""Full-robot reference geometry, mask semantics and training observations."""

import copy
import json

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from emet.rl.manipulation.coupled_tracking import (
    COARSE_SUITE,
    CoupledReference,
    CoupledTrackingEnv,
    admit_reference,
    load_suite,
    spline_bounds,
    suite_contract,
)


def test_distribution_has_disjoint_ids_seeds_and_both_embodiments():
    suite = load_suite()
    rows = [c for cases in suite["robots"].values() for c in cases]
    assert len(rows) == len({c["seed"] for c in rows}) == len({c["id"] for c in rows}) == 20
    for cases in suite["robots"].values():
        assert {c["mode"] for c in cases if c["split"] == "train"} == {
            "hand_free_base",
            "base_only",
            "coupled",
            "fixed_world_hand",
        }
        assert any("retarget" in c for c in cases if c["split"] == "audit")


@pytest.mark.parametrize("fault", ["seed", "mask", "nan", "self_motion"])
def test_invalid_distribution_rejected_before_building_a_simulator(tmp_path, fault):
    suite = copy.deepcopy(load_suite())
    cases = suite["robots"]["stretch"]
    if fault == "seed":
        cases[1]["seed"] = cases[0]["seed"]
    elif fault == "mask":
        cases[1]["base_mask"] = [True] * 3
    elif fault == "nan":
        cases[0]["travel_m"] = float("nan")
    else:
        cases[5].pop("self_motion_scale")
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_suite(path)


def test_exact_spline_derivative_bounds_include_internal_extrema():
    spline = CubicSpline([0.0, 0.4, 1.0], [[0.0, 0.0], [1.0, -0.5], [0.0, 2.0]])
    bounds = spline_bounds(spline)
    for order, bound in enumerate(bounds, 1):
        dense = np.max(abs(spline(np.linspace(0, 1, 10001), order)), axis=0)
        assert np.all(dense <= bound * (1 + 1e-12))
        np.testing.assert_allclose(dense, bound, rtol=1e-4)


def test_coarse_distribution_keeps_precise_hand_and_requires_real_base_progress():
    old, coarse = load_suite(), load_suite(COARSE_SUITE)
    before, after = suite_contract(old), suite_contract(coarse)
    assert before["base_xy_tolerance"] == 0.02 and before["base_heading_tolerance"] == 0.04
    assert after["base_xy_tolerance"] == 0.05
    assert after["base_heading_tolerance"] == pytest.approx(np.deg2rad(5))
    for key in ("hand_position_tolerance", "hand_orientation_tolerance"):
        assert before[key] == after[key]
    all_old = [c for cases in old["robots"].values() for c in cases]
    all_new = [c for cases in coarse["robots"].values() for c in cases]
    assert {c["seed"] for c in all_old}.isdisjoint(c["seed"] for c in all_new)
    assert {c["id"] for c in all_old}.isdisjoint(c["id"] for c in all_new)
    for c in all_new:
        if c["mode"] == "fixed_world_hand":
            assert c["purpose"] == "hand_hold_nullspace_diagnostic"
        elif c["mode"] == "arm_only":
            assert c["travel_m"] == 0 and c["turn_rad"] == 0 and any(c["arm_delta"])
        elif any(c["base_mask"][:2]):
            assert abs(c["travel_m"]) > 2 * after["base_xy_tolerance"]


@pytest.mark.parametrize("fault", ["radius", "hand", "purpose", "trivial_base"])
def test_coarse_distribution_rejects_contract_drift_and_trivial_base_goals(tmp_path, fault):
    suite = load_suite(COARSE_SUITE)
    if fault == "radius":
        suite["acceptance"]["base_xy_tolerance"] = 0.15
    elif fault == "hand":
        suite["acceptance"]["hand_position_tolerance"] = 0.01
    elif fault == "purpose":
        suite["robots"]["innate_mars"][5]["purpose"] = "tracking"
    else:
        suite["robots"]["innate_mars"][0]["travel_m"] = 0.01
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_suite(path)


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_coarse_general_reward_has_no_in_region_heading_precision_pressure(robot):
    from emet.rl.manipulation.whole_body_reward import RewardMeter

    case = load_suite(COARSE_SUITE)["robots"][robot][0]
    env = CoupledTrackingEnv(robot, [case], suite_path=COARSE_SUITE)
    try:
        env.reset()
        e, c = env.unwrapped, env.unwrapped.control
        assert e.config.base_xy_tolerance == 0.05
        assert c.base_xy_tolerance == e.active_base_xy_tolerance == 0.05
        assert e.config.hand_position_tolerance == 0.003
        c.base_goal = c.base_pose().copy()
        action, motion = np.zeros(e.profile.action_size), np.zeros(9)
        meter = RewardMeter(e, base_heading_region=True)
        meter.tick(action, motion)
        original = meter.last_terms["base_pose_score"]
        c.base_goal[2] += np.deg2rad(4)
        meter.tick(action, motion)
        assert meter.last_terms["base_pose_score"] == original
        c.base_goal[2] += np.deg2rad(3)
        meter.tick(action, motion)
        assert meter.last_terms["base_pose_score"] < original
        # Frozen/default objective still distinguishes these headings.
        old = RewardMeter(e)
        c.base_goal = c.base_pose().copy()
        old.tick(action, motion)
        original = old.last_terms["base_pose_score"]
        c.base_goal[2] += np.deg2rad(4)
        old.tick(action, motion)
        assert old.last_terms["base_pose_score"] < original
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_arm_only_case_holds_base_with_fixed_base_tolerance(robot):
    """Arm-only cases must keep the fixed-base hold, not the 5 cm acceptance radius."""
    from emet.rl.manipulation.whole_body_reward import RewardMeter

    case = next(c for c in load_suite(COARSE_SUITE)["robots"][robot] if c["mode"] == "arm_only")
    env = CoupledTrackingEnv(robot, [case], suite_path=COARSE_SUITE)
    try:
        env.reset()
        e, c = env.unwrapped, env.unwrapped.control
        assert e.fixed_base_trial is True
        assert e.active_base_xy_tolerance == pytest.approx(e.config.fixed_base_xy_tolerance)
        assert c.base_xy_tolerance == e.active_base_xy_tolerance
        assert c.base_goal is not None and c.hand_mask
        # A 1 cm base drift is inside the 5 cm acceptance radius but outside the
        # fixed-base hold, so the reward must already be penalizing it.
        c.base_goal = c.base_pose().copy()
        action, motion = np.zeros(e.profile.action_size), np.zeros(9)
        meter = RewardMeter(e, base_heading_region=True)
        meter.tick(action, motion)
        settled = meter.last_terms["base_pose_score"]
        c.base_goal[:2] += [0.01, 0.0]
        meter.tick(action, motion)
        assert meter.last_terms["base_pose_score"] < settled
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_continuous_coupled_reference_preserves_physics_limits_and_masks(robot):
    cases = load_suite()["robots"][robot]
    env = CoupledTrackingEnv(robot, cases)
    try:
        obs, info = env.reset(options={"case_id": cases[1]["id"]})
        e = env.unwrapped
        reference = e.control.task_trajectory
        assert info["coupled_admission"]["accepted"]
        assert not e.control.base_mask.any() and e.control.hand_mask
        assert set(obs) == {"history", "preview", "motion", "critic"}
        assert env.observation_space.contains(obs)
        before = e.data.qpos.copy(), e.data.qvel.copy(), float(e.data.time)
        reference = CoupledReference(e.control, cases[3])
        assert admit_reference(e, reference)["accepted"]
        bounds = reference.certificate["derivative_bounds"]
        for measured, key in zip(bounds, ("speed", "acceleration", "jerk"), strict=True):
            assert np.all(np.array(measured) <= np.array(reference.certificate[key]) * (1 + 1e-9))
        # The correct axle point, rather than base_link, has zero lateral speed.
        for t in np.linspace(0.1, reference.end - 0.1, 20):
            a, b = reference.sample(t - 0.0001)[0], reference.sample(t + 0.0001)[0]

            def axle(pose):
                yaw = pose[2]
                r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                return pose[:2] + r @ e.control.axle_offset[:2]

            derivative = (axle(b) - axle(a)) / 0.0002
            yaw = reference.sample(t)[0][2]
            assert abs(derivative @ [-np.sin(yaw), np.cos(yaw)]) < 1e-7
        np.testing.assert_array_equal(e.data.qpos, before[0])
        np.testing.assert_array_equal(e.data.qvel, before[1])
        assert e.data.time == before[2]
        assert reference.record()["format"] == "coupled_fk_reference_v1"
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_fixed_world_hand_has_full_robot_witness_and_real_base_motion(robot):
    case = load_suite()["robots"][robot][5]
    env = CoupledTrackingEnv(robot, [case])
    try:
        env.reset()
        e = env.unwrapped
        ref = e.control.task_trajectory
        assert ref.fixed_hand is not None and env.admission["accepted"]
        assert env.admission["fixed_hand_witness_position_m"] < 0.0001
        assert env.admission["fixed_hand_witness_orientation_rad"] < 0.001
        assert np.linalg.norm(ref.base[-1] - ref.base[0]) > 1e-4
        for t in np.linspace(0, ref.end, 25):
            np.testing.assert_array_equal(ref.sample(t)[1], ref.fixed_hand)
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_training_reset_and_partial_base_mask_do_not_hide_actor_channels(robot):
    from emet.rl.manipulation.command_runtime import task_pose_teacher_action
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    cases = load_suite()["robots"][robot]
    env = CoupledTrackingEnv(robot, [cases[4]])
    try:
        env.reset()
        c = env.unwrapped.control
        path = c.task_trajectory
        original = task_pose_teacher_action(c)
        # Poison only inactive yaw. The declared active hand path is unchanged.
        base = path.base.copy()
        base[:, 2] += 1.5
        alternate = TaskPoseTrajectory(path.times, base, path.hand, path.gripper, base_mask=path.base_mask)
        c.set_task_trajectory(alternate)
        poisoned = task_pose_teacher_action(c)
        np.testing.assert_allclose(poisoned, original, atol=0.001)
        c.set_task_trajectory(path)
        before = env.unwrapped.data.time
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        assert env.unwrapped.data.time > before and info["elapsed_control_ticks"] == 1
        assert np.isfinite(info["reward_groups"]).all() and env.observation_space.contains(obs)
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_existing_ppo_can_collect_and_update_from_coupled_observations(robot):
    import torch

    from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig
    from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
    from emet.rl.manipulation.whole_body_training import collect_rollout, tensor_observation

    case = load_suite()["robots"][robot][1]
    env = CoupledTrackingEnv(robot, [case])
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            torch.set_num_threads(1)
            obs, _ = env.reset()
            model = MultiCriticPolicy(robot=robot)
            sensor = tensor_observation(obs)
            with torch.no_grad():
                expected = model.actor(sensor).clone()
                poisoned = {**sensor, "critic": sensor["critic"] + 100}
                torch.testing.assert_close(model.actor(poisoned), expected, rtol=0, atol=0)
            learner = MultiCriticPPO(model, PPOConfig(epochs=1, batch_size=8))
            rollout, _ = collect_rollout(env, model, obs, 8)
            assert np.all(rollout["ticks"] == 1)
            assert set(rollout["case_ids"]) == {case["id"]}
            assert np.isfinite(rollout["rewards"]).all()
            before = model.actor.gain.weight.detach().clone()
            learner.update(rollout)
            assert torch.isfinite(model.actor.gain.weight).all()
            assert not torch.equal(model.actor.gain.weight, before)
    finally:
        env.close()


def test_teacher_cli_summary_serializes_numpy_predicates():
    from emet.rl.manipulation.coupled_evaluation import readout_summary

    report = {
        "cases": [{"accepted": np.bool_(True), "teacher": {"tracking_success": np.bool_(True)}}, {"accepted": False}],
        "ready_for_training": np.bool_(False),
    }
    summary = readout_summary(report)
    assert json.loads(json.dumps(summary)) == {
        "cases": 2,
        "admitted": 1,
        "teacher_successes": 1,
        "ready_for_training": False,
    }


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["stretch", "innate_mars"])
def test_partial_base_goals_retain_coupled_authority_to_close_hand_error(robot):
    from emet.rl.manipulation.coupled_evaluation import teacher_episode

    case = load_suite()["robots"][robot][4]
    env = CoupledTrackingEnv(robot, [case])
    try:
        env.reset()
        result = teacher_episode(env, case)
        assert result["tracking_success"] and result["post_stop_pose_in_contract"]
        assert result["stop_confirmed"] and not any(result["safety_flags"].values())
        assert result["final_metrics"]["ee_position_error"] < 0.003
        assert result["final_metrics"]["ee_orientation_error"] < 0.02
    finally:
        env.close()


@pytest.mark.sim
def test_measured_state_retarget_rebuilds_reference_from_current_pose():
    case = {
        "id": "innate_mars:retarget:test:0",
        "seed": 35600999,
        "split": "audit",
        "mode": "hand_free_base",
        "travel_m": 0.03,
        "turn_rad": 0.0,
        "arm_delta": [0.0, 0.0, 0.0, 0.0, 0.0],
        "base_mask": [False, False, False],
        "retarget": {"at_fraction": 0.5, "travel_m": -0.02},
    }
    env = CoupledTrackingEnv("innate_mars", [case])
    try:
        env.reset()
        e, c = env.unwrapped, env.unwrapped.control
        original = c.task_trajectory
        assert env.retarget_at is not None and not env.retargeted
        for _ in range(500):
            pre_base = c.base_pose().copy()
            _, _, done, truncated, info = env.step(np.zeros(e.profile.action_size))
            assert not done and not truncated
            if info["retarget_delivered"]:
                replacement = c.task_trajectory
                assert replacement is not original
                # A measured-state rebuild starts at the pre-step measured pose,
                # not the original reference's planned endpoint.
                np.testing.assert_allclose(replacement.start_base, pre_base, rtol=1e-12, atol=1e-12)
                assert replacement.travel == -0.02
                assert info["coupled_admission"]["accepted"]
                break
        else:
            pytest.fail("measured-state retarget was not delivered")
    finally:
        env.close()
