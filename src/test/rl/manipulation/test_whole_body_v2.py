"""Test actual isolation, derivative bounds, PPO returns and tick scheduling."""

import json
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from emet.rl.manipulation.base_navigation_feedback import BaseNavigationFeedback
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.jerk_reference import compile_joint_path, polynomial_peaks
from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig, combine_advantages, grouped_gae
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy, WholeBodyActor
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle, export_actor
from emet.rl.manipulation.whole_body_env import ControlStepMixture


def observation(robot="innate_mars", count=4):
    p = PROFILES[robot]
    f, width = history_layout(p, retention_context=True)
    pf, _ = preview_layout()
    h, preview = torch.zeros(count, 20, width), torch.zeros(count, 7, 18)
    identity = torch.tensor([1.0, 0, 0, 1, 0, 0])
    h[..., f["hand_pose"]][..., 3:] = identity
    h[..., f["hand_mask"]] = 1
    h[..., f["base_mask"]] = 1
    h[..., f["phase"]][..., 0] = 1
    preview[..., pf["hand_error"]][..., 3:] = identity
    preview[..., pf["hand_mask"]] = 1
    preview[..., pf["base_mask"]] = 1
    return {"history": h, "preview": preview, "motion": torch.zeros(count, 9), "critic": torch.zeros(count, 34)}


@pytest.mark.parametrize(
    "distance,speed,accel,jerk", [(1.0, 0.2, 1.0, 5.0), (0.01, 2.0, 4.0, 8.0), (0.5, 2.0, 0.5, 8.0)]
)
def test_s_curve_is_c2_obeys_governed_bounds_and_reaches_corners(distance, speed, accel, jerk):
    path, record = compile_joint_path(
        [[0.0], [distance], [-distance / 2]],
        [speed],
        [accel],
        [jerk],
        phase_acceleration=2,
        phase_jerk=8,
        profile="s_curve",
    )
    assert record["interpolation"] == "jerk_limited_s_curve_trapezoidal_velocity_c2"
    for i, part in enumerate(path.segments):
        v, a, j = polynomial_peaks(part)
        assert v[0] <= speed * (1 + 1e-8)
        assert (a + 2 * v)[0] <= accel * (1 + 1e-8)
        assert (j + 6 * a + 8 * v)[0] <= jerk * (1 + 1e-8)
        if i:
            for prior, current in zip(
                path.segments[i - 1].sample(part.start)[:3], part.sample(part.start)[:3], strict=True
            ):
                np.testing.assert_allclose(prior, current, atol=1e-8)
    np.testing.assert_allclose(path.sample(path.end)[0], [-distance / 2], atol=1e-8)
    np.testing.assert_allclose(path.sample(path.end)[1:3], 0, atol=1e-8)
    assert record["dynamic_feasibility_verified"] is False


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_actor_uses_history_without_privileged_inputs_and_preserves_equilibrium(robot):
    torch.manual_seed(21)
    actor = WholeBodyActor(robot)
    with torch.no_grad():
        actor.gain.weight.normal_(0, 0.1)
    obs = observation(robot)
    assert torch.count_nonzero(actor(obs)) == 0  # zero errors and velocities
    pf, _ = preview_layout()
    obs["preview"][..., pf["hand_error"]][..., 0] = 0.005
    first = actor(obs).detach().clone()
    obs["critic"].normal_(100, 10)
    torch.testing.assert_close(first, actor(obs))
    fields = actor.encoder.fields
    obs["history"][:, 0, fields["proprio/positions"]] += 0.3
    assert not torch.equal(first, actor(obs))  # oldest history sample has a path to the actor
    assert torch.count_nonzero(actor(obs)[:, -1]) == 0  # no release authority
    obs["history"][..., fields["hand_mask"]] = 0
    obs["history"][..., fields["base_mask"]] = 0
    assert torch.count_nonzero(actor(obs)) == 0


def test_critic_gradient_never_updates_actor():
    model = MultiCriticPolicy()
    obs = observation()
    before = model.values(obs).detach().clone()
    obs["critic"][:, 0] = 1
    assert not torch.equal(before, model.values(obs))
    model.values(obs).sum().backward()
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(any(p.grad is not None for p in critic.parameters()) for critic in model.critics)


def test_grouped_returns_bootstrap_truncation_and_do_not_cross_episode_boundaries():
    rewards = np.ones((3, 3))
    values = np.zeros((3, 3))
    following = np.ones((3, 3)) * 2
    a, returns = grouped_gae(
        rewards, values, following, [False, False, True], [False, True, True], [1, 2, 1], gamma=0.5, lam=1.0
    )
    np.testing.assert_allclose(a, [[2.75] * 3, [1.5] * 3, [1.0] * 3])
    np.testing.assert_array_equal(a, returns)
    for ticks in ([1, np.nan, 1], [1, 0.5, 1], [1, 0, 1]):
        with pytest.raises(ValueError):
            grouped_gae(rewards, values, following, [False] * 3, [False] * 3, ticks)


def test_separate_advantage_normalization_removes_reward_unit_scaling():
    torch.manual_seed(21)
    a = torch.randn(100, 3)
    first, _ = combine_advantages(a, normalization="per_head")
    second, _ = combine_advantages(a * torch.tensor([1.0, 1000.0, 0.02]), normalization="per_head")
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("base_scale", [1.0, 0.1])
def test_ppo_changes_actor_and_all_three_critics_with_finite_losses(base_scale):
    torch.manual_seed(21)
    model = MultiCriticPolicy(hidden=16, stationary_base_std_scale=base_scale)
    obs = observation(count=16)
    obs["preview"][:, :, 7] = 0.01
    with torch.no_grad():
        actions = model.distribution(obs).sample()
        log, _, values = model.evaluate(obs, actions)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rollout = {
        "observations": {k: v.numpy() for k, v in obs.items()},
        "actions": actions.numpy(),
        "log_prob": log.numpy(),
        "values": values.numpy(),
        "next_values": values.numpy(),
        "rewards": np.arange(48).reshape(16, 3) / 48,
        "terminated": [False] * 15 + [True],
        "episode_end": [False] * 15 + [True],
        "ticks": [1] * 16,
    }
    stats = MultiCriticPPO(model, PPOConfig(epochs=2, batch_size=8)).update(rollout)
    assert np.isfinite(stats["critic_losses"]).all()
    for prefix in ("actor.", "critics.0.", "critics.1.", "critics.2."):
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith(prefix))


@pytest.mark.parametrize("value", [0, -0.1, 1.01, float("nan"), float("inf"), True])
def test_base_exploration_rejects_invalid_scales(value):
    with pytest.raises(ValueError, match="exploration scale"):
        MultiCriticPolicy(stationary_base_std_scale=value)


def test_default_exploration_keeps_identical_seeded_samples_and_probabilities():
    model = MultiCriticPolicy(hidden=16)
    obs = observation()
    original = torch.distributions.Normal(model.actor(obs), model.log_std.clamp(-6, 0).exp())
    torch.manual_seed(51)
    expected = original.sample()
    torch.manual_seed(51)
    actual = model.distribution(obs).sample()
    assert torch.equal(actual, expected)
    log, entropy, _ = model.evaluate(obs, actual)
    mask = model.actor.action_mask(obs)
    assert torch.equal(log, (original.log_prob(actual) * mask).sum(-1))
    assert torch.equal(entropy, (original.entropy() * mask).sum(-1))


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_reference_conditioned_noise_preserves_authority_and_matches_ppo_likelihood(robot):
    model = MultiCriticPolicy(robot=robot, hidden=16, stationary_base_std_scale=0.1)
    with torch.no_grad():
        model.actor.gain.weight.normal_(0, 0.1)
    obs = observation(robot, count=8)
    f, pf = model.actor.encoder.fields, model.actor.encoder.preview_fields
    obs["preview"][:, :, pf["hand_error"]][:, :, 0] = 0.01
    obs["history"][1, :, f["hand_mask"]] = 0
    obs["history"][2, :, f["base_mask"]] = 0
    obs["motion"][3, 2] = 0.001
    obs["preview"][4, 4, pf["base_error"]] = 0.01  # upcoming motion, current velocity zero
    obs["history"][5, :, f["base_mask"]][:, 0] = 0
    obs["preview"][6, 4, pf["base_mask"]] = 0
    obs["preview"][7, :, pf["base_error"]] = 1  # far static goal still permits correction
    obs["history"][7, :, f["base_velocity"]] = 0.1  # reference stationarity is not measured rest
    obs["preview"][0, 0, pf["base_error"]] = 0.2  # past target does not gate future exploration
    assert model.base_exploration_condition(obs).tolist() == [True, False, False, False, False, False, False, True]
    mean = model.actor(obs).detach().clone()
    dist = model.distribution(obs)
    assert torch.equal(mean, dist.mean)
    expected = model.log_std.clamp(-6, 0).exp().expand_as(mean).clone()
    expected[[0, 7], :2] *= 0.1
    assert torch.equal(dist.scale, expected)
    assert torch.count_nonzero(mean[7, :2]) > 0  # no base authority mask
    action = dist.sample() * model.actor.action_mask(obs)
    log, entropy, _ = model.evaluate(obs, action)
    reference = torch.distributions.Normal(mean, expected)
    mask = model.actor.action_mask(obs)
    torch.testing.assert_close(log, (reference.log_prob(action) * mask).sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(entropy, (reference.entropy() * mask).sum(-1), rtol=0, atol=0)
    (-log.mean()).backward()
    assert model.log_std.grad is not None and torch.isfinite(model.log_std.grad).all()


@pytest.mark.parametrize("jitter", ["target", "motion"])
def test_reference_producer_jitter_retains_full_base_exploration(jitter):
    model = MultiCriticPolicy(hidden=16, stationary_base_std_scale=0.1)
    obs = observation(count=1)
    assert model.base_exploration_condition(obs).item()
    if jitter == "target":
        fields = model.actor.encoder.preview_fields
        obs["preview"][0, -1, fields["base_error"].start] = 1e-12
    else:
        obs["motion"][0, 0] = 1e-12
    assert not model.base_exploration_condition(obs).item()
    expected = model.log_std.clamp(-6, 0).exp()[None]
    torch.testing.assert_close(model.distribution(obs).scale, expected, rtol=0, atol=0)


def test_actor_only_bundle_roundtrip_and_rejects_privileged_inputs(tmp_path):
    cfg = replace(
        ManipulationConfig(),
        robot="innate_mars",
        retention_context=True,
        trajectory_preview=True,
        action_mode="direct_velocity",
    )
    actor = WholeBodyActor()
    destination = tmp_path / "actor"
    export_actor(actor, destination, cfg, "test-model", steps=0)
    bundle = WholeBodyBundle(destination, robot="innate_mars", fingerprint="test-model")
    weights = torch.load(destination / "policy.zip", weights_only=True)
    assert all(not k.startswith("critics") for k in weights["state_dict"])
    obs = {k: v[0].numpy() for k, v in observation().items()}
    with pytest.raises(ValueError, match="privileged"):
        bundle(obs)
    del obs["critic"]
    np.testing.assert_array_equal(bundle(obs), np.zeros(8))
    bundle.bind_reference(lambda: obs["motion"])
    np.testing.assert_array_equal(bundle({k: v for k, v in obs.items() if k != "motion"}), np.zeros(8))


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_base_rate_ablation_preserves_arm_feedback_and_reference_authority(robot):
    legacy = WholeBodyActor(robot)
    candidate = WholeBodyActor(robot, base_rate_feedback=False)
    with torch.no_grad():
        gains = legacy.gain.bias.reshape(-1, legacy.feedback_width)
        gains[:, 24 : 24 + legacy.encoder.profile.arm_size + 6] = 1
        gains[:, -9:] = 1
        candidate.load_state_dict(legacy.state_dict())
    obs = observation(robot)
    obs["history"][..., legacy.encoder.fields["proprio/velocities"]] = 0.01
    obs["history"][..., legacy.encoder.fields["base_velocity"]] = 0.01
    original, changed = legacy(obs), candidate(obs)
    assert torch.count_nonzero(original[:, :2]) == original.shape[0] * 2
    torch.testing.assert_close(changed[:, :2], torch.zeros_like(changed[:, :2]), rtol=0, atol=0)
    torch.testing.assert_close(changed[:, 2:], original[:, 2:], rtol=0, atol=0)
    assert torch.count_nonzero(changed[:, 2:-1]) > 0
    # Removing the direct rate path does not remove announced-motion authority.
    obs["motion"][:, 0] = 0.1
    assert torch.count_nonzero(candidate(obs)[:, :2]) == original.shape[0] * 2


@pytest.mark.parametrize("rate_feedback", [True, False])
def test_bundle_pins_base_feedback_behavior_and_rejects_manifest_only_change(tmp_path, rate_feedback):
    cfg = replace(ManipulationConfig(), retention_context=True, trajectory_preview=True, action_mode="direct_velocity")
    actor = WholeBodyActor(base_rate_feedback=rate_feedback)
    destination = tmp_path / "actor"
    export_actor(actor, destination, cfg, "test-model", steps=0)
    assert WholeBodyBundle(destination).actor.base_rate_feedback is rate_feedback
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["base_rate_feedback"] = not rate_feedback
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="base feedback contract"):
        WholeBodyBundle(destination)


def test_legacy_bundle_without_feedback_metadata_preserves_original_actor(tmp_path):
    import hashlib

    cfg = replace(ManipulationConfig(), retention_context=True, trajectory_preview=True, action_mode="direct_velocity")
    destination = tmp_path / "legacy"
    export_actor(WholeBodyActor(), destination, cfg, "test-model", steps=0)
    path = destination / "policy.zip"
    weights = torch.load(path, weights_only=True)
    weights.pop("base_rate_feedback")
    torch.save(weights, path)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("base_rate_feedback")
    manifest["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert WholeBodyBundle(destination).actor.base_rate_feedback is True


class TickEnv(gym.Env):
    action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
    observation_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

    def __init__(self, ticks):
        self.ticks = ticks

    def reset(self, **kwargs):
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 0.0, True, False, {"elapsed_control_ticks": self.ticks}


def test_mixture_balances_physical_ticks_with_unequal_episode_lengths():
    env = ControlStepMixture({"general": TickEnv(1), "tamp": TickEnv(10)})
    for _ in range(110):
        env.reset()
        env.step([0])
    assert env.ticks.tolist() == [100, 100]
    assert env.episodes.tolist() == [100, 10]


@pytest.mark.sim
def test_native_general_observations_are_finite_and_sensor_bundle_does_not_step_physics():
    from emet.rl.manipulation.mastery import MasteryEnv, load_suite
    from emet.rl.manipulation.whole_body_env import GeneralTrackingEnv

    env = GeneralTrackingEnv(
        MasteryEnv(
            load_suite("configs/benchmarks/mars_tracking_mastery_supported_v1.json"),
            stages=("arm",),
            split="train",
            population="training",
        )
    )
    try:
        obs, info = env.reset(seed=90)
        assert not info["initialization_failed"]
        assert env.observation_space.contains(obs)
        actor = WholeBodyActor()
        state = env.unwrapped.data.qpos.copy()
        time = float(env.unwrapped.data.time)
        action = actor({k: torch.from_numpy(v)[None] for k, v in obs.items()}).detach()[0].numpy()
        np.testing.assert_array_equal(state, env.unwrapped.data.qpos)
        assert env.unwrapped.data.time == time
        obs, reward, _, _, info = env.step(action)
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        assert info["reward_groups"].shape == (3,)
        from emet.rl.manipulation.whole_body_reward import current_contact_force

        current = current_contact_force(env.unwrapped)
        env.unwrapped.monitor.peak_force = 1e6
        assert current_contact_force(env.unwrapped) == current
    finally:
        env.close()


@pytest.mark.sim
def test_actual_public_tamp_pregrasp_with_s_curve_and_confirmed_stop():
    import json

    from emet.rl.manipulation.whole_body_training import load_development, primitive_rollout

    suite, _ = load_development()
    rows = primitive_rollout(suite["train_cases"][:1])
    assert len(rows) == 1 and rows[0]["success"] and rows[0]["stop_confirmed"]
    assert not rows[0]["full_pick_place_success"]
    assert rows[0]["inference_counts"]["policy_calls"] == 0
    assert rows[0]["inference_counts"]["reference_controller_calls"] > 0
    assert "jerk_limited_s_curve_trapezoidal_velocity_c2" in json.dumps(rows[0]["commands"])


def test_development_train_audit_and_canonical_cases_are_disjoint():
    import json
    from pathlib import Path

    from emet.rl.manipulation.whole_body_training import load_development

    suite, _ = load_development()
    frozen = json.loads(Path("configs/benchmarks/mars_public_tamp_whole_body_v2.json").read_text())
    train = {c["seed"] for c in suite["train_cases"]}
    audit = {c["seed"] for c in suite["audit_cases"]}
    canonical = {c["seed"] for c in frozen["cases"]}
    assert not train & audit and not (train | audit) & canonical


def test_tamp_experimental_gate_preserves_known_failure_but_rejects_new_failures_and_unsafe_stops():
    import copy

    from emet.rl.manipulation.whole_body_training import load_development, tamp_training_readiness

    suite, _ = load_development()
    rows = [
        {
            "id": c["id"],
            "success": c["id"] not in suite["teacher_expected_failures"],
            "code": suite["teacher_expected_failures"].get(c["id"], "pregrasp_complete"),
            "stop_confirmed": True,
            "initialization_failed": False,
            "commands": ["executed"],
            "decisions": 10,
            "safety_flags": dict.fromkeys(("wheel_lift", "invalid_physics", "unsafe_contact", "retention_lost"), False),
        }
        for c in suite["train_cases"] + suite["audit_cases"]
    ]
    assert tamp_training_readiness(rows, suite)
    assert sum(r["success"] for r in rows) == 5  # readiness is not all-success qualification
    for mutate in (
        lambda r: r[2].update(code="tracking_timeout"),
        lambda r: r[2]["safety_flags"].update(wheel_lift=True),
        lambda r: r[2].update(stop_confirmed=False),
        lambda r: r[2].update(initialization_failed=True),
    ):
        changed = copy.deepcopy(rows)
        mutate(changed)
        assert not tamp_training_readiness(changed, suite)


@pytest.mark.sim
def test_known_tamp_challenge_is_still_a_failure_with_confirmed_stop():
    from emet.rl.manipulation.whole_body_training import load_development, primitive_rollout

    suite, _ = load_development()
    (row,) = primitive_rollout(suite["train_cases"][2:3])
    assert not row["success"] and row["code"] == "object_changed_replan"
    assert row["stop_confirmed"] and not any(row["safety_flags"].values())
    assert row["object_rotation_rad"] > 0.05
    assert row["full_pick_place_success"] is False


def test_rollout_loop_and_ppo_update_run_native_observation_contract():
    """Exercise the real collector, clipping boundary, reset and update together."""
    from emet.rl.manipulation.whole_body_training import collect_rollout

    class Toy:
        def __init__(self):
            self.steps = 0

        def reset(self):
            obs = {k: v[0].numpy() for k, v in observation().items()}
            obs["preview"][:, 7] = 0.01
            return obs, {}

        def step(self, action):
            self.steps += 1
            obs, _ = self.reset()
            return (
                obs,
                0.0,
                self.steps % 3 == 0,
                False,
                {
                    "reward_groups": np.array([action[2], -np.sum(action**2), -0.1 * abs(action[0])]),
                    "elapsed_control_ticks": 1,
                },
            )

    torch.manual_seed(90)
    model = MultiCriticPolicy(hidden=16)
    env = Toy()
    obs, _ = env.reset()
    episodes = []
    rollout, obs = collect_rollout(env, model, obs, 32, episode_callback=episodes.append)
    assert len(episodes) == 10 and rollout["episode_end"].sum() == 10
    stats = MultiCriticPPO(model, PPOConfig(epochs=1, batch_size=16)).update(rollout)
    assert np.isfinite(stats["actor_loss"]) and np.isfinite(stats["critic_losses"]).all()


def test_reference_twist_uses_announced_retiming_masks_and_wrapped_yaw():
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    from emet.rl.manipulation.whole_body_observation import reference_velocity

    def sample(time):
        t = time * 0.25
        yaw = np.arctan2(np.sin(np.pi + 0.4 * t), np.cos(np.pi + 0.4 * t))
        rot = Rotation.from_rotvec([0, 0, 0.8 * t]).as_matrix()
        return np.array([0.1 * t, -0.2 * t, yaw]), np.r_[[0.3 * t, 0.4 * t, 0.5 * t], rot[:, :2].ravel()], 0.0

    control = SimpleNamespace(
        data=SimpleNamespace(time=0.0), preview_sampler=sample, base_mask=np.array([True, False, True]), hand_mask=True
    )
    before = dict(vars(control))
    expected = [0.025, 0.0, 0.1, 0.075, 0.1, 0.125, 0.0, 0.0, 0.2]
    np.testing.assert_allclose(reference_velocity(control), expected, atol=1e-7)
    assert vars(control).keys() == before.keys() and control.data.time == 0.0
    control.hand_mask = False
    np.testing.assert_allclose(reference_velocity(control)[3:], 0.0)
    control.preview_sampler = None
    np.testing.assert_array_equal(reference_velocity(control), np.zeros(9))


@pytest.mark.sim
def test_real_tamp_actor_rollout_updates_three_critics_without_teacher_actions():
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
    from emet.rl.manipulation.whole_body_training import collect_rollout, load_development

    suite, _ = load_development()
    env = TampPrimitiveEnv(suite["train_cases"][:1], oracle=False)
    torch.manual_seed(90)
    model = MultiCriticPolicy(hidden=16)
    try:
        obs, info = env.reset()
        assert "oracle_action" not in info
        rollout, _ = collect_rollout(env, model, obs, 8)
        assert rollout["actions"].shape == (8, 8)
        assert np.isfinite(rollout["rewards"]).all()
        stats = MultiCriticPPO(model, PPOConfig(epochs=1, batch_size=8)).update(rollout)
        assert np.isfinite(stats["critic_losses"]).all()
    finally:
        env.close()


@pytest.mark.sim
def test_exported_v2_actor_loads_through_source_bound_tamp_backend(tmp_path):
    import gzip
    import hashlib
    import json

    from emet.eval.mars_tamp_control import fixture
    from emet.rl.manipulation.carry_benchmark import source_hashes
    from emet.rl.manipulation.tamp_policy import CheckpointTampPolicy
    from emet.rl.manipulation.whole_body_training import load_development

    suite, _ = load_development()
    env, admission = fixture(suite["train_cases"][0])
    try:
        assert admission["accepted"]
        cfg = replace(env.config, retention_context=True, trajectory_preview=True, physical_wheel_slew=True)
        actor = tmp_path / "actor"
        export_actor(WholeBodyActor(), actor, cfg, env.fingerprint, steps=0)
        manifest = json.loads((actor / "manifest.json").read_text())
        raw = json.dumps(
            {"run": {"source_hashes": source_hashes()}, "checkpoints": {"test": {"manifest": manifest}}}
        ).encode()
        archive = tmp_path / "report.json.gz"
        archive.write_bytes(gzip.compress(raw))
        evidence = tmp_path / "summary.json"
        evidence.write_text(
            json.dumps({"full_report_archive": str(archive), "uncompressed_sha256": hashlib.sha256(raw).hexdigest()})
        )
        policy = CheckpointTampPolicy(actor, env, evidence=evidence)
        np.testing.assert_array_equal(policy(env.control.observe()), np.zeros(8))
        policy.verify_unchanged()
        assert policy.calls == 1 and policy.admission["simulation_only"]
    finally:
        env.close()


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
@pytest.mark.parametrize("mode", ["active_hand", "partial_base"])
@pytest.mark.parametrize("config", [BaseNavigationFeedback(), BaseNavigationFeedback(reference_tracking=True, heading_gain_scale=2)])
def test_navigation_structure_preserves_coupled_and_partial_mask_actions(robot, mode, config):
    torch.manual_seed(82)
    original = WholeBodyActor(robot, base_rate_feedback=False)
    structured = WholeBodyActor(robot, base_rate_feedback=False, base_navigation_feedback=config)
    with torch.no_grad():
        original.gain.weight.normal_(0, 0.1)
        original.gain.bias.normal_(0, 0.2)
        structured.load_state_dict(original.state_dict())
    obs = observation(robot)
    pf, f = original.encoder.preview_fields, original.encoder.fields
    obs["preview"][..., pf["base_error"]] = torch.tensor([0.1, 0.03, -0.2])
    obs["preview"][..., pf["hand_error"]][..., 0] = 0.01
    if mode == "partial_base":
        obs["history"][..., f["hand_mask"]] = 0
        obs["preview"][..., pf["hand_mask"]] = 0
        obs["history"][..., f["base_mask"]][..., 2] = 0
    torch.testing.assert_close(structured(obs), original(obs), rtol=0, atol=0)


def navigation_observation(actor):
    obs = observation(actor.robot, count=1)
    obs["history"][..., actor.encoder.fields["hand_mask"]] = 0
    obs["preview"][..., actor.encoder.preview_fields["hand_mask"]] = 0
    obs["preview"][..., actor.encoder.preview_fields["hand_error"]] = 0
    return obs


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_navigation_structure_corrects_heading_sign_and_holds_inside_declared_region(robot):
    actor = WholeBodyActor(robot, base_rate_feedback=False, base_navigation_feedback=BaseNavigationFeedback())
    with torch.no_grad():
        gain = actor.gain.bias.reshape(-1, actor.feedback_width)
        for offset in (0, 12):
            gain[0, offset] = -3
            gain[1, offset] = -20  # longitudinal-to-yaw leakage must disappear
            gain[1, offset + 1] = -4
            gain[1, offset + 2] = -2
    obs = navigation_observation(actor)
    error = actor.encoder.preview_fields["base_error"]
    obs["preview"][..., error] = torch.tensor([0.01, 0.001, 0.01])
    torch.testing.assert_close(actor(obs), torch.zeros(1, 8), rtol=0, atol=0)
    for sign in (-1, 1):
        obs["preview"][..., error] = torch.tensor([0.0, 0.0, sign * 0.2])
        assert actor(obs)[0, 1] * sign > 0
        assert actor(obs)[0, 0] == 0
        obs["preview"][..., error] = torch.tensor([sign * 0.1, 0.0, 0.0])
        assert actor(obs)[0, 0] * sign > 0
        assert actor(obs)[0, 1] == 0


def test_navigation_structure_retains_signed_lateral_correction_during_announced_motion():
    actor = WholeBodyActor(base_rate_feedback=False, base_navigation_feedback=BaseNavigationFeedback())
    with torch.no_grad():
        gain = actor.gain.bias.reshape(-1, actor.feedback_width)
        gain[1, 1] = gain[1, 13] = -4
    obs = navigation_observation(actor)
    obs["preview"][..., actor.encoder.preview_fields["base_error"]] = torch.tensor([0.0, 0.01, 0.0])
    for sign in (-1, 1):
        obs["motion"][0, 0] = sign * 0.1
        assert actor(obs)[0, 1] * sign > 0


def test_navigation_zero_initialized_gains_can_learn_and_state_derivatives_are_finite():
    actor = WholeBodyActor(base_rate_feedback=False, base_navigation_feedback=BaseNavigationFeedback())
    obs = navigation_observation(actor)
    obs["preview"][..., actor.encoder.preview_fields["base_error"]] = torch.tensor([0.1, 0.001, 0.2])
    actor(obs)[:, :2].sum().backward()
    gain_grad = actor.gain.bias.grad.reshape(-1, actor.feedback_width)
    assert gain_grad[0, 0] > 0 and gain_grad[1, 2] > 0
    with torch.no_grad():
        actor.gain.bias.normal_(0, 0.1)
    obs["preview"].requires_grad_(True)
    actor.zero_grad()
    actor(obs)[:, :2].sum().backward()
    assert torch.isfinite(obs["preview"].grad).all()
    assert torch.count_nonzero(obs["preview"].grad) > 0


@pytest.mark.parametrize("xy,yaw", [(0.0, 0.1), (True, 0.1), (float("nan"), 0.1), (0.05, float("inf")), (0.05, 4.0)])
def test_navigation_contract_rejects_invalid_units(xy, yaw):
    with pytest.raises(ValueError):
        BaseNavigationFeedback(xy, yaw)


@pytest.mark.parametrize("config", [BaseNavigationFeedback(), BaseNavigationFeedback(reference_tracking=True, heading_gain_scale=2)])
def test_navigation_bundle_pins_contract_and_rejects_incompatible_declared_tolerance(tmp_path, config):
    cfg = replace(
        ManipulationConfig(),
        retention_context=True,
        trajectory_preview=True,
        action_mode="direct_velocity",
        base_xy_tolerance=0.05,
        base_heading_tolerance=np.deg2rad(5),
    )
    actor = WholeBodyActor(base_rate_feedback=False, base_navigation_feedback=config)
    destination = tmp_path / "actor"
    export_actor(actor, destination, cfg, "test-model", steps=0)
    bundle = WholeBodyBundle(destination)
    assert bundle.actor.base_navigation_feedback == actor.base_navigation_feedback
    obs = navigation_observation(actor)
    obs["preview"][..., actor.encoder.preview_fields["base_error"]] = torch.tensor([0.1, 0.0, 0.2])
    expected = actor(obs)[0].detach().numpy()
    sensor = {k: v[0].numpy() for k, v in obs.items() if k != "critic"}
    np.testing.assert_array_equal(bundle(sensor), expected)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["base_navigation_feedback"]["xy_region_m"] = 0.06
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="navigation contract"):
        WholeBodyBundle(destination)
    with pytest.raises(ValueError, match="inner hold region"):
        export_actor(actor, tmp_path / "incompatible", replace(cfg, base_xy_tolerance=0.005), "test-model", steps=0)


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_reference_tracking_feedforward_uses_body_units_without_learned_rescaling(robot):
    actor = WholeBodyActor(robot, base_navigation_feedback=BaseNavigationFeedback(reference_tracking=True))
    obs = navigation_observation(actor)
    obs["history"][..., actor.encoder.fields["base_pose"]][..., 2] = np.pi / 2
    p = actor.encoder.profile
    obs["motion"][0, :3] = torch.tensor([0, 0.5 * p.base_speed, -0.2 * p.base_yaw_speed])
    with torch.no_grad():
        gain = actor.gain.bias.reshape(-1, actor.feedback_width)
        gain[:2, -9:] = 100  # would double-count the known velocity mapping
    torch.testing.assert_close(actor(obs)[0, :2], torch.tensor([0.5, -0.2]))
    obs["motion"][0, :3] *= 10
    torch.testing.assert_close(actor(obs)[0, :2], torch.tensor([1.0, -1.0]))
    obs["history"][..., actor.encoder.fields["hand_mask"]] = 1
    # The opt-in mapping is absent when the hand is active (the legacy map is
    # allowed to use its learned motion gains instead).
    actor.base_navigation_feedback = None
    expected = actor(obs)
    actor.base_navigation_feedback = BaseNavigationFeedback(reference_tracking=True)
    torch.testing.assert_close(actor(obs), expected, rtol=0, atol=0)


def test_reference_tracking_does_not_chase_inner_xy_error_during_a_turn():
    actor = WholeBodyActor(base_navigation_feedback=BaseNavigationFeedback(reference_tracking=True))
    obs = navigation_observation(actor)
    with torch.no_grad():
        gain = actor.gain.bias.reshape(-1, actor.feedback_width)
        gain[0, 0] = gain[0, 12] = gain[1, 1] = gain[1, 13] = 20
    obs["preview"][..., actor.encoder.preview_fields["base_error"]] = torch.tensor([0.01, 0.01, 0.0])
    obs["motion"][0, 2] = 0.1
    torch.testing.assert_close(actor(obs)[0, :2], torch.tensor([0.0, 0.2]))
    obs["preview"][..., actor.encoder.preview_fields["base_error"]] = torch.tensor([0.1, 0.0, 0.0])
    assert actor(obs)[0, 0] > 0


def test_heading_gain_calibration_is_explicit_and_does_not_change_other_rows():
    actor = WholeBodyActor(base_navigation_feedback=BaseNavigationFeedback())
    obs = navigation_observation(actor)
    obs["preview"][..., actor.encoder.preview_fields["base_error"]] = torch.tensor([0.0, 0.0, 0.2])
    with torch.no_grad():
        actor.gain.bias.reshape(-1, actor.feedback_width)[1, 2] = 1
    expected = actor(obs)
    actor.base_navigation_feedback = BaseNavigationFeedback(heading_gain_scale=2)
    torch.testing.assert_close(torch.atanh(actor(obs)[0, 1]), 2 * torch.atanh(expected[0, 1]))
    torch.testing.assert_close(actor(obs)[0, 2:], expected[0, 2:], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"reference_tracking": 1}, {"heading_gain_scale": True}, {"heading_gain_scale": 0}, {"heading_gain_scale": float("nan")}])
def test_reference_tracking_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BaseNavigationFeedback(**kwargs)


def test_reference_tracking_contract_is_versioned_and_fail_closed():
    legacy = BaseNavigationFeedback().record()
    assert legacy["format"] == "signed_base_navigation_v1" and len(legacy) == 3
    config = BaseNavigationFeedback(reference_tracking=True, heading_gain_scale=2)
    record = config.record()
    assert record["format"] == "signed_base_navigation_v2"
    assert BaseNavigationFeedback.from_record(record) == config
    for broken in ({**record, "format": "signed_base_navigation_v1"}, {k: v for k, v in record.items() if k != "heading_gain_scale"}, {**record, "unknown": True}):
        with pytest.raises(ValueError, match="feedback contract"):
            BaseNavigationFeedback.from_record(broken)


@pytest.mark.parametrize("navigation", [None, BaseNavigationFeedback()])
def test_configured_bundles_require_versioned_format_and_explicit_metadata(tmp_path, navigation):
    import hashlib

    cfg = replace(
        ManipulationConfig(),
        retention_context=True,
        trajectory_preview=True,
        action_mode="direct_velocity",
        base_xy_tolerance=0.05,
        base_heading_tolerance=np.deg2rad(5),
    )
    actor = WholeBodyActor(base_rate_feedback=False, base_navigation_feedback=navigation)
    path = tmp_path / "configured"
    export_actor(actor, path, cfg, "test-model", steps=0)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["format"] == "emet-whole-body-v3"  # v2-only runtimes reject it
    from emet.rl.manipulation.policy import load_control_policy

    assert load_control_policy(path).actor.base_navigation_feedback == navigation
    weights = torch.load(path / "policy.zip", weights_only=True)
    weights.pop("base_navigation_feedback")
    torch.save(weights, path / "policy.zip")
    manifest["weights_sha256"] = hashlib.sha256((path / "policy.zip").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicit feedback options"):
        WholeBodyBundle(path)


def test_navigation_feedback_cannot_be_silently_downgraded_to_legacy_format(tmp_path):
    import hashlib

    cfg = replace(
        ManipulationConfig(),
        retention_context=True,
        trajectory_preview=True,
        action_mode="direct_velocity",
        base_xy_tolerance=0.05,
        base_heading_tolerance=np.deg2rad(5),
    )
    path = tmp_path / "actor"
    export_actor(
        WholeBodyActor(base_rate_feedback=False, base_navigation_feedback=BaseNavigationFeedback()),
        path,
        cfg,
        "model",
        steps=0,
    )
    manifest = json.loads((path / "manifest.json").read_text())
    weights = torch.load(path / "policy.zip", weights_only=True)
    manifest["format"] = weights["format"] = "emet-whole-body-v2"
    torch.save(weights, path / "policy.zip")
    manifest["weights_sha256"] = hashlib.sha256((path / "policy.zip").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires the configured v3 format"):
        WholeBodyBundle(path)
