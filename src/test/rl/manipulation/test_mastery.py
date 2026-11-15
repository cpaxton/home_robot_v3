"""Small-problem contracts: honest splits, continuous FK and full control authority."""

from copy import deepcopy

import numpy as np
import pytest

from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, segment
from emet.rl.manipulation.mastery import MasteryEnv, load_suite, select_cases, slow_path, validate_preflight


def hand_suite():
    from pathlib import Path

    return load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_hand_accuracy_v1.json"
    )


def test_hand_distribution_keeps_frozen_contract_and_validation_out_of_small_amplitudes():
    suite = hand_suite()
    frozen = load_suite()
    assert suite["config"]["hand_gated_base_reward"] is True
    for key in ("cases", "reference", "gate"):
        assert suite[key] == frozen[key]
    canonical = select_cases(suite, ["base", "arm"], "all")
    expanded = select_cases(suite, ["base", "arm"], "train", population="training")
    assert len([c for c in expanded if c["stage"] == "base"]) == 45
    arm = [c for c in expanded if c["stage"] == "arm"]
    assert len(arm) == 12
    assert {c["reset_seed"] for c in expanded}.isdisjoint(
        c["reset_seed"] for c in canonical if c["split"] == "validation"
    )
    assert all(c["split"] == "train" for c in arm)
    assert len(select_cases(suite, ["base", "arm"], "all", population="admission")) == 66
    assert select_cases(suite, ["arm"], "validation", population="admission") == select_cases(
        frozen, ["arm"], "validation"
    )


def test_stage_sampler_weights_are_independent_of_case_count_and_seeded():
    from collections import Counter

    from emet.rl.manipulation.mastery import TrainingCaseSampler

    suite = hand_suite()
    cases = select_cases(suite, ["base", "arm"], "train", population="training")
    sampler = TrainingCaseSampler(cases, suite["training_stage_weights"], seed=19)
    chosen = [sampler.next() for _ in range(10000)]
    counts = Counter(c["stage"] for c in chosen)
    assert 0.73 < counts["arm"] / len(chosen) < 0.77
    for stage, size in (("base", 45), ("arm", 12)):
        first_cycle = [c["id"] for c in chosen if c["stage"] == stage][:size]
        assert len(set(first_cycle)) == size
    sampler.reset(19)
    assert [sampler.next() for _ in range(100)] == chosen[:100]
    with pytest.raises(ValueError, match="cover every selected stage"):
        TrainingCaseSampler(select_cases(suite, ["loaded"], "train"), suite["training_stage_weights"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("arm_training_scales", [0, 1]),
        ("arm_training_scales", [0.5]),
        ("arm_training_scales", [True, 1]),
        ("training_stage_weights", {"arm": float("nan")}),
        ("training_stage_weights", {"task": 1}),
        ("training_stage_weights", {"arm": -1, "base": 2}),
    ],
)
def test_hand_sampling_rejects_invalid_recipe(tmp_path, key, value):
    import json

    suite = hand_suite()
    suite[key] = value
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_suite(path)


@pytest.mark.sim
def test_hand_velocity_audit_reads_measured_jacobian_without_mutating_control():
    from emet.rl.manipulation.hand_audit import hand_tracking_diagnostics

    env = MasteryEnv(hand_suite(), stages=["arm"], split="train")
    try:
        _, info = env.reset(options={"case_id": "train:arm:pitch_up"})
        assert not info["initialization_failed"]
        env.control.velocity[:] = 0.01
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.control.target.copy()]
        report = hand_tracking_diagnostics(env.control)
        for current, previous in zip(
            [env.data.qpos, env.data.qvel, env.data.ctrl, env.control.target], before, strict=True
        ):
            np.testing.assert_array_equal(current, previous)
        np.testing.assert_array_equal(report["slewed_joint_velocity_rad_s"], np.full(5, 0.01))
        assert np.isfinite(list(report["predicted_arm_linear_velocity_world_m_s"])).all()
        assert abs(report["measured_speed_toward_current_goal_m_s"]) < 0.01
    finally:
        env.close()


def crossed_suite():
    from pathlib import Path

    return load_suite(Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_crossed_v1.json")


def test_crossed_training_population_keeps_canonical_evaluation_and_disjoint_postures():
    suite = crossed_suite()
    canonical = select_cases(suite, ["base"], "all")
    assert canonical == select_cases(load_suite(), ["base"], "all")
    training = select_cases(suite, ["base"], "train", population="training")
    assert len(training) == len({c["id"] for c in training}) == 25
    originals = select_cases(suite, ["base"], "train")
    assert all(c in training for c in originals)
    assert {(c["reset_seed"], c["travel_m"], c["turn_rad"]) for c in training} == {
        (posture["reset_seed"], primitive["travel_m"], primitive["turn_rad"])
        for posture in originals
        for primitive in originals
    }
    assert {c["reset_seed"] for c in training}.isdisjoint(
        c["reset_seed"] for c in canonical if c["split"] == "validation"
    )
    assert len(select_cases(suite, ["base"], "all", population="admission")) == 30
    for stage in ["arm", "coupled", "loaded"]:
        assert select_cases(suite, [stage], "all", population="admission") == select_cases(suite, [stage], "all")
    with pytest.raises(ValueError):
        select_cases(suite, ["base"], "validation", population="training")


def test_crossed_preflight_requires_every_added_pair_and_success():
    suite = crossed_suite()
    report = {
        "complete": True,
        "controller": "task_pose_teacher",
        "suite_sha256": digest(suite),
        "source_hashes": source_hashes(),
        "cases": [{"id": c["id"], "mastered": True} for c in select_cases(suite, ["base"], "all")],
    }
    with pytest.raises(ValueError):
        validate_preflight(report, suite, ["base"])
    report["cases"] = [
        {"id": c["id"], "mastered": True} for c in select_cases(suite, ["base"], "all", population="admission")
    ]
    validate_preflight(report, suite, ["base"])
    report["cases"][-1]["mastered"] = False
    with pytest.raises(ValueError):
        validate_preflight(report, suite, ["base"])


@pytest.mark.parametrize("mutation", ["format", "cases", "gate", "population"])
def test_crossed_suite_rejects_weakened_contract_or_undeclared_population(tmp_path, mutation):
    import json

    suite = crossed_suite()
    if mutation == "format":
        suite["format"] = "mars-tracking-mastery-v1"
    elif mutation == "cases":
        suite["cases"][0]["reset_seed"] += 100
    elif mutation == "gate":
        suite["gate"]["terminal_hold_s"] = 1
    else:
        suite["training_population"]["validation_in_training"] = True
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_suite(path)


@pytest.mark.sim
def test_crossed_training_balances_every_pair_and_uses_measured_posture():
    suite = crossed_suite()
    env = MasteryEnv(suite, stages=["base"], split="train", population="training")
    try:
        ids = []
        for i in range(25):
            obs, info = env.reset(seed=19 if i == 0 else None)
            assert not info["initialization_failed"], info["mastery"]
            assert set(obs) == {"history", "preview"} and obs["history"].shape == (20, 101)
            ids.append(env.mastery_case["id"])
        assert len(set(ids)) == 25 and all(cid.startswith("train:") for cid in ids)
        env.reset(options={"case_id": "train:base:yaw_left@posture:yaw_right"})
        state = env.mastery_admission["initial_state_sha256"]
        reference = env.mastery_admission["reference_sha256"]
        env.reset(options={"case_id": "train:base:yaw_right"})
        assert env.mastery_admission["initial_state_sha256"] == state
        assert env.mastery_admission["reference_sha256"] != reference
    finally:
        env.close()


def test_suite_has_small_disjoint_nontrivial_case_sets():
    suite = load_suite()
    train = select_cases(suite, suite["stages"], "train")
    validation = select_cases(suite, suite["stages"], "validation")
    assert len(train) == len(validation) == 13
    assert {c["reset_seed"] for c in train}.isdisjoint(c["reset_seed"] for c in validation)
    assert len(select_cases(suite, ["base", "arm"], "all")) == 18
    assert any(c["travel_m"] > 0 for c in train) and any(c["travel_m"] < 0 for c in train)
    assert any(c["turn_rad"] > 0 for c in train) and any(c["turn_rad"] < 0 for c in train)
    assert max(abs(c["arm_delta"][4]) for c in train) >= 0.08


@pytest.mark.parametrize("mutation", ["overlap", "duplicate", "loose_accuracy", "nan_reference"])
def test_invalid_or_weakened_suite_rejected(tmp_path, mutation):
    import json

    suite = load_suite()
    if mutation == "overlap":
        suite["cases"][13]["reset_seed"] = suite["cases"][0]["reset_seed"]
    if mutation == "duplicate":
        suite["cases"][1]["id"] = suite["cases"][0]["id"]
    if mutation == "loose_accuracy":
        suite["config"]["hand_position_tolerance"] = 0.03
    if mutation == "nan_reference":
        suite["reference"]["minimum_duration_s"] = float("nan")
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_suite(path)


def test_uniform_slowdown_preserves_geometry_and_scales_each_derivative():
    part = segment(np.array([0.0, 0.1]), np.array([0.2, -0.1]), np.ones(2), np.ones(2))
    path = ArmTrajectory([part])
    original = part.coefficients.copy()
    slower = slow_path(path, 1.1)
    for t in np.linspace(0, path.end, 31):
        a, b = path.sample(t), slower.sample(t * 1.1)
        for order in range(4):
            np.testing.assert_allclose(b[order], a[order] / 1.1**order, atol=1e-10)
    np.testing.assert_array_equal(part.coefficients, original)
    for factor in [0, 0.9, np.nan, np.inf]:
        with pytest.raises(ValueError):
            slow_path(path, factor)


def test_teacher_preflight_requires_both_splits_exact_cases_and_current_sources():
    suite = load_suite()
    cases = select_cases(suite, ["base"], "all")
    report = {
        "complete": True,
        "controller": "task_pose_teacher",
        "suite_sha256": digest(suite),
        "source_hashes": source_hashes(),
        "cases": [{"id": c["id"], "mastered": True} for c in cases],
    }
    validate_preflight(report, suite, ["base"])
    for alteration in ["missing", "duplicate", "failure", "old_sources"]:
        bad = deepcopy(report)
        if alteration == "missing":
            bad["cases"].pop()
        if alteration == "duplicate":
            bad["cases"].append(bad["cases"][0])
        if alteration == "failure":
            bad["cases"][0]["mastered"] = False
        if alteration == "old_sources":
            bad["source_hashes"] = {}
        with pytest.raises(ValueError):
            validate_preflight(bad, suite, ["base"])
    with pytest.raises(ValueError):
        validate_preflight(report, suite, ["base", "arm"])


@pytest.mark.sim
def test_case_reset_repeats_real_physics_after_motion_and_does_not_leak_case_id():
    env = MasteryEnv(stages=["arm"], split="train")
    try:
        obs, info = env.reset(options={"case_id": "train:arm:pitch_up"})
        assert not info["initialization_failed"], info["mastery"]
        initial = env.mastery_admission["initial_state_sha256"]
        reference = env.mastery_admission["reference_sha256"]
        assert set(obs) == {"history", "preview"} and obs["history"].shape == (20, 101)
        assert env.control.hand_mask and env.control.base_mask.all()
        # Wheel commands remain real actions even in arm-focused cases.
        env.step(np.array([0.1, 0.1, 0, 0, 0, 0, 0, 0]))
        assert np.any(abs(env.control.base_command) > 0)
        _, again = env.reset(options={"case_id": "train:arm:pitch_up"})
        assert not again["initialization_failed"]
        assert env.mastery_admission["initial_state_sha256"] == initial
        assert env.mastery_admission["reference_sha256"] == reference
        before = env.data.qpos.copy()
        time = float(env.data.time)
        with pytest.raises(ValueError):
            env.reset(options={"qpos": before.tolist()})
        np.testing.assert_array_equal(env.data.qpos, before)
        assert env.data.time == time
    finally:
        env.close()


@pytest.mark.sim
def test_exact_fk_reference_has_a_nonholonomic_axle_witness_and_native_bounds():
    env = MasteryEnv(stages=["coupled"], split="train")
    try:
        _, info = env.reset(options={"case_id": "train:coupled:forward_turn"})
        assert not info["initialization_failed"], info["mastery"]
        ref = env.mastery_reference
        for t in np.linspace(0.001, ref.end - 0.001, 80):
            a, b = ref.coordinates(t - 1e-5)[0], ref.coordinates(t + 1e-5)[0]
            axle_a = a[:2] + ref.yaw_rotation(a[2]) @ ref.offset
            axle_b = b[:2] + ref.yaw_rotation(b[2]) @ ref.offset
            yaw = ref.coordinates(t)[0][2]
            assert abs((axle_b - axle_a) @ np.array([-np.sin(yaw), np.cos(yaw)])) < 1e-9
            base, q = ref.coordinates(t)
            np.testing.assert_allclose(ref.sample(t)[1], ref.fk(base, q), atol=1e-12)
            _, v, acc, j = ref.path.sample(t)
            assert np.all(abs(v) <= np.r_[env.profile.base_speed, env.profile.base_yaw_speed, env.profile.speed] + 1e-8)
            assert np.all(
                abs(acc)
                <= np.r_[env.profile.base_acceleration, env.profile.base_yaw_acceleration, env.profile.acceleration]
                + 1e-8
            )
            assert np.isfinite(j).all()
        assert not env.fixed_base_contract_active
        assert ref.record()["joint_reference_actor_input"] is False
    finally:
        env.close()


@pytest.mark.sim
def test_balanced_case_cycles_never_draw_validation_cases():
    env = MasteryEnv(stages=["base"], split="train")
    try:
        ids = []
        for i in range(10):
            _, info = env.reset(seed=19 if i == 0 else None)
            assert not info["initialization_failed"], info["mastery"]
            ids.append(env.mastery_case["id"])
        assert len(set(ids[:5])) == len(set(ids[5:])) == 5
        assert all(cid.startswith("train:") for cid in ids)
    finally:
        env.close()


@pytest.mark.sim
def test_measured_collision_stops_before_benchmark_can_claim_mastery(monkeypatch):
    env = MasteryEnv(stages=["base"], split="train")
    try:
        _, info = env.reset(options={"case_id": "train:base:hold"})
        assert not info["initialization_failed"], info["mastery"]
        monkeypatch.setattr(env.mastery_checker, "configuration_collides", lambda *args: True)
        _, _, done, truncated, info = env.step(np.zeros(8))
        assert done and not truncated and env.control.stopped
        assert info["code"] == "mastery_measured_collision" and not info["is_success"]
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("num_envs", [1, 2])
def test_real_mastery_trainer_updates_and_exports_post_update_bundle(tmp_path, monkeypatch, num_envs):
    import json
    from types import SimpleNamespace

    import torch

    from emet.rl import regression_gate
    from emet.rl.manipulation.mastery import evaluate_suite, save_report, train
    from emet.rl.manipulation.policy import ManipulationPolicy

    torch.set_num_threads(1)
    suite = deepcopy(load_suite())
    suite["cases"] = [c for c in suite["cases"] if c["stage"] == "base" and c["id"].endswith(":hold")]
    teacher = evaluate_suite(suite, stages=["base"])
    assert all(c["mastered"] for c in teacher["cases"])
    preflight = tmp_path / "preflight.json.gz"
    save_report(preflight, teacher)
    # This unit workflow tests PPO/export, not the independent regression runner.
    monkeypatch.setattr(regression_gate, "verify_regressions", lambda path: None)
    out = tmp_path / "train"
    args = SimpleNamespace(
        preflight=preflight,
        regression_receipt=tmp_path / "unused.receipt.json",
        output=out,
        stages=["base"],
        seed=5,
        steps=512 * num_envs,
        num_envs=num_envs,
        vec_backend="dummy",
        learning_rate=1e-4,
        discount=0.9975,
        base_std=0.01,
        arm_std=0.1,
        initialize_from=None,
        wandb_mode="disabled",
        eval_every=8192,
    )
    train(args, suite)
    assert json.loads((out / "complete.json").read_text())["steps"] == 512 * num_envs
    policy = ManipulationPolicy(out / "last")
    assert policy.manifest["steps"] == 512 * num_envs
    assert policy.model.policy.optimizer.state_dict()["state"]
    assert json.loads((out / "last/summary.json").read_text())["complete"]
    updates = [json.loads(line) for line in (out / "updates.jsonl").read_text().splitlines()]
    assert [row["timesteps"] for row in updates] == [256 * num_envs, 512 * num_envs]
    assert updates[0]["optimizer_epochs"] < updates[1]["optimizer_epochs"]
    for row in updates:
        assert {"train/approx_kl", "train/entropy_loss", "train/value_loss", "train/explained_variance"} <= row.keys()
        assert len(row["policy/action_std"]) == 8
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert sum(row["steps"] for row in metrics[-1]["rollout/cases"].values()) == 256 * num_envs
    assert sum(metrics[-1]["sampling/steps"].values()) == 512 * num_envs
    assert "reward_terms_sum" in next(iter(metrics[-1]["rollout/cases"].values()))
    env = MasteryEnv(suite, stages=["base"], split="validation")
    try:
        obs, _ = env.reset()
        assert policy(obs).shape == (8,) and np.isfinite(policy(obs)).all()
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize(
    "direction,index,sign", [("forward", 0, 1), ("reverse", 0, -1), ("yaw_left", 1, 1), ("yaw_right", 1, -1)]
)
def test_small_direct_commands_move_the_measured_base_in_requested_direction(direction, index, sign):
    env = MasteryEnv(stages=["base"], split="train")
    try:
        _, info = env.reset(options={"case_id": "train:base:" + direction})
        assert not info["initialization_failed"]
        origin = env.control.base_pose().copy()
        action = np.zeros(8)
        action[index] = sign * 0.03
        for _ in range(30):
            _, _, done, truncated, info = env.step(action)
            assert not done and not truncated, info["code"]
        measured = env.control.base_pose() - origin
        progress = measured[:2] @ np.array([np.cos(origin[2]), np.sin(origin[2])]) if index == 0 else measured[2]
        assert progress * sign > 0.001
        assert info["mastery"]["base_velocity_command"][index] * sign > 0
        assert info["mastery"]["base_velocity_measured"][index] * sign > 0
    finally:
        env.close()


def test_rollout_diagnostics_keeps_case_rewards_action_variance_clipping_and_stop_codes():
    from emet.rl.manipulation.mastery import RolloutDiagnostics

    audit = RolloutDiagnostics()
    info = {
        "reward_terms": {"base_tracking_reward": 0.1, "physical_cost": 0.02},
        "code": "running",
        "initialization_failed": False,
        "mastery": {"base_velocity_command": [-0.003, 0], "base_velocity_measured": [-0.002, 0.001]},
        "base_linear_velocity_m_s": -0.002,
        "base_yaw_velocity_rad_s": 0.001,
        "base_goal_error": 0.005,
        "base_heading_error_rad": 0.01,
    }
    audit.add("reverse", [-2, 0], [-1, 0], 0.08, info, False)
    audit.add("reverse", [0, 0], [0, 0], -20, {**info, "code": "mastery_tracking_envelope"}, True)
    other = {k: v for k, v in info.items() if k != "reward_terms"}
    audit.add(
        "rejected", [0, 0], [0, 0], -20, {**other, "initialization_failed": True, "code": "initialization_failed"}, True
    )
    row = audit.record()["reverse"]
    assert row["steps"] == 2
    assert row["reward_sum"] == pytest.approx(-19.92)
    assert row["reward_terms_sum"]["base_tracking_reward"] == pytest.approx(0.2)
    assert row["action_mean"] == [-1, 0] and row["action_std"] == [1, 0]
    assert row["action_clipped_fraction"] == [0.5, 0]
    assert row["terminal_codes"] == {"mastery_tracking_envelope": 1}
    assert row["base_velocity_measured_mean"] == [-0.002, 0.001]
    assert audit.record()["rejected"]["reward_terms_sum"] == {"initialization_penalty": 20}


def test_checkpoint_evaluation_cannot_reset_exploration_rng_even_on_failure():
    import random

    import torch

    from emet.rl.manipulation.mastery import preserve_training_rng

    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    with pytest.raises(RuntimeError):
        with preserve_training_rng():
            random.seed(100)
            np.random.seed(100)
            torch.manual_seed(100)
            raise RuntimeError("failed evaluator")
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_mastery_v1_hold_clock_matches_its_gate():
    suite = load_suite()
    assert suite["config"]["terminal_hold_seconds"] == suite["gate"]["terminal_hold_s"] == 2
