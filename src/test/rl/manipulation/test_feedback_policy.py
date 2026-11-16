from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.feedback_geometry import rotation_vector
from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.policy import ManipulationPolicy, export
from emet.rl.manipulation.profiles import MARS


class Commands(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {
            "history": gym.spaces.Box(-np.inf, np.inf, (20, 101), dtype=np.float32),
            "preview": gym.spaces.Box(-np.inf, np.inf, (7, 18), dtype=np.float32),
        }
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = equilibrium()
        return obs, {}

    def step(self, action):
        obs, _ = self.reset()
        obs["preview"][:, 1] = 0.05
        return obs, -float(np.sum((action - 0.1) ** 2)), False, False, {}


def equilibrium():
    rng = np.random.default_rng(14)
    obs = {"history": rng.normal(size=(20, 101)).astype(np.float32), "preview": np.zeros((7, 18), dtype=np.float32)}
    h, _ = history_layout(MARS, retention_context=True)
    p, _ = preview_layout()
    obs["history"][:, h["proprio/velocities"]] = 0
    obs["history"][:, h["base_velocity"]] = 0
    obs["preview"][:, p["base_mask"]] = 1
    obs["preview"][:, p["hand_mask"]] = 1
    obs["preview"][:, p["hand_error"].start + 3 : p["hand_error"].stop] = [1, 0, 0, 1, 0, 0]
    return obs


@pytest.fixture
def feedback():
    torch.set_num_threads(1)
    cfg = ManipulationConfig(
        robot="innate_mars",
        action_mode="direct_velocity",
        trajectory_preview=True,
        categorical_features_unscaled=True,
        feedback_policy=True,
        retention_context=True,
    )
    env = CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, norm_reward=False)
    model = PPO(
        FeedbackActorCriticPolicy,
        env,
        device="cpu",
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": GoalFeedbackFeatures,
            "features_extractor_kwargs": {"robot": "innate_mars", "retention_context": True},
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        },
    )
    yield cfg, env, model
    env.close()


def test_motion_equilibrium_is_independent_of_learned_gains_and_absolute_context(feedback):
    _, env, model = feedback
    with torch.no_grad():
        for parameter in model.policy.action_net.parameters():
            parameter.normal_(mean=0, std=0.3)
    obs = equilibrium()
    action = model.predict(env.normalize_obs(obs), deterministic=True)[0]
    np.testing.assert_array_equal(action[:-1], np.zeros(7))
    assert np.isfinite(action).all() and np.max(abs(action)) <= 1
    # Goal updates and actual motion can activate the learned feedback.
    p, _ = preview_layout()
    obs["preview"][4, p["base_error"].start] = 0.02
    moved = model.predict(env.normalize_obs(obs), deterministic=True)[0]
    assert np.any(abs(moved[:-1]) > 1e-5)
    obs = equilibrium()
    h, _ = history_layout(MARS, retention_context=True)
    obs["history"][-1, h["base_velocity"].start + 4] = 0.03
    assert np.any(abs(model.predict(env.normalize_obs(obs), deterministic=True)[0][:-1]) > 1e-5)


def test_feedback_policy_ppo_export_and_original_bundle_loader_roundtrip(feedback, tmp_path):
    cfg, env, model = feedback
    model.learn(16)
    assert model.policy.action_net.gain.weight.detach().abs().sum() > 0
    export(model, env, tmp_path, cfg, "feedback-test", steps=16)
    policy = ManipulationPolicy(tmp_path, robot="innate_mars", fingerprint="feedback-test")
    obs = equilibrium()
    np.testing.assert_allclose(policy(obs), model.predict(env.normalize_obs(obs), deterministic=True)[0], atol=1e-7)
    np.testing.assert_array_equal(policy(obs)[:-1], np.zeros(7))


def test_feedback_mode_rejects_running_command_normalization():
    with pytest.raises(ValueError, match="feedback policy"):
        ManipulationConfig(feedback_policy=True)


@pytest.mark.sim
def test_explicit_angular_units_transfer_preserves_actor_and_critic(feedback, tmp_path):
    from emet.rl.manipulation.initialization import _model, transfer_actor

    cfg, env, source = feedback
    with torch.no_grad():
        source.policy.action_net.gain.weight.normal_(std=0.03)
        source.policy.action_net.gain.bias.normal_(std=0.03)
    _, fingerprint = _model(cfg.robot, {})
    export(source, env, tmp_path, cfg, fingerprint, steps=0)
    new_cfg = replace(cfg, feedback_orientation_scale=0.15, wheel_feedback_gain=0.5)
    new_env = CommandVecNormalize(DummyVecEnv([Commands]), config=new_cfg, norm_reward=False)
    try:
        target = PPO(
            FeedbackActorCriticPolicy,
            new_env,
            device="cpu",
            n_steps=8,
            batch_size=4,
            policy_kwargs={
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {"robot": cfg.robot, "retention_context": True, "orientation_scale": 0.15},
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )
        details = transfer_actor(target, new_env, tmp_path, new_cfg, fingerprint)
        assert details["feedback_orientation_scale_change"]["function_preserving"]
        obs = equilibrium()
        p, _ = preview_layout()
        obs["preview"][:, p["hand_error"].start + 4] = 0.15
        old_obs, new_obs = env.normalize_obs(obs), new_env.normalize_obs(obs)
        np.testing.assert_allclose(
            source.predict(old_obs, deterministic=True)[0], target.predict(new_obs, deterministic=True)[0], atol=1e-6
        )
        with torch.no_grad():
            old_value = source.policy.predict_values(source.policy.obs_to_tensor(old_obs)[0])
            new_value = target.policy.predict_values(target.policy.obs_to_tensor(new_obs)[0])
        torch.testing.assert_close(old_value, new_value, atol=1e-6, rtol=1e-5)
        assert not target.policy.optimizer.state_dict()["state"]
    finally:
        new_env.close()


def test_yaw_audit_distinguishes_correct_error_from_negative_learned_gain(feedback, tmp_path):
    from emet.rl.manipulation.yaw_audit import feedback_terms

    cfg, env, model = feedback
    head = model.policy.action_net
    with torch.no_grad():
        head.zero_output()
        head.gain.bias[head.feedback_width + 2] = -1
        head.gain.bias[head.feedback_width + 14] = -1
    export(model, env, tmp_path, cfg, "feedback-test", steps=0)
    policy = ManipulationPolicy(tmp_path)
    obs = equilibrium()
    h, _ = history_layout(MARS, retention_context=True)
    obs["history"][:, h["base_pose"]] = 0
    obs["history"][:, h["base_goal"]] = [0, 0, 0.04]
    obs["preview"][:, 3] = 0.04
    action, row = feedback_terms(policy, obs)
    assert row["world_yaw_error_rad"] > 0 and row["normalized_yaw_error"] > 0
    assert row["yaw_gain_current"] < 0 and row["yaw_gain_preview"] < 0
    assert action[1] < 0
    np.testing.assert_allclose(action, policy(obs), atol=1e-7)


def make_model(config, env):
    return PPO(
        FeedbackActorCriticPolicy,
        env,
        device="cpu",
        seed=17,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "restoring_yaw_feedback": config.restoring_yaw_feedback,
            "turn_xy_tolerance": config.base_xy_tolerance,
            "features_extractor_class": GoalFeedbackFeatures,
            "features_extractor_kwargs": GoalFeedbackFeatures.kwargs_for(config),
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        },
    )


def test_so3_log_matches_scipy_at_small_angles_and_half_turns():
    vectors = np.r_[
        np.zeros((1, 3)),
        np.eye(3) * 1e-9,
        np.eye(3) * np.pi,
        [[1, -2, 3] / np.linalg.norm([1, -2, 3]) * (np.pi - 1e-7)],
        Rotation.random(100, random_state=7).as_rotvec(),
    ]
    matrices = torch.tensor(Rotation.from_rotvec(vectors).as_matrix(), dtype=torch.float64, requires_grad=True)
    actual = rotation_vector(matrices)
    np.testing.assert_allclose(Rotation.from_rotvec(actual.detach()).as_matrix(), matrices.detach(), atol=1e-7)
    np.testing.assert_allclose(actual.detach()[:4], vectors[:4], atol=1e-12)
    actual.sum().backward()
    assert torch.isfinite(matrices.grad).all()


def test_reference_velocity_has_physical_units_current_frames_and_wrapped_yaw(feedback):
    cfg, env, _ = feedback
    cfg = replace(cfg, feedback_reference_velocity=True, feedback_orientation_error="rotvec")
    features = GoalFeedbackFeatures(env.observation_space, **GoalFeedbackFeatures.kwargs_for(cfg))
    obs = equilibrium()
    p, _ = preview_layout()
    # Non-identity hand frame and reference: catches reversed products and
    # accidental angular velocities expressed in the reference frame.
    current = Rotation.from_euler("xyz", [0.4, -0.3, 0.7]).as_matrix()
    reference = Rotation.from_euler("xyz", [-0.2, 0.5, 0.3]).as_matrix()
    linear, angular = np.array([0.03, -0.02, 0.01]), np.array([0.2, -0.1, 0.3])
    for index, offset in enumerate((-0.05, 0, 0.05, 0.1, 0.2, 0.5, 1.0)):
        rotation = current.T @ Rotation.from_rotvec(angular * offset).as_matrix() @ reference
        obs["preview"][index, p["hand_error"]] = np.r_[current.T @ (linear * offset), rotation[:, :2].ravel()]
        obs["preview"][index, p["base_error"]] = [
            0.02 * offset,
            -0.01 * offset,
            np.arctan2(np.sin(3.13 + offset), np.cos(3.13 + offset)),
        ]
    tensors = {k: torch.tensor(v[None]) for k, v in env.normalize_obs(obs).items()}
    signals = features(tensors)[:, -features.feedback_width :]
    np.testing.assert_allclose(
        signals[0, 24:33], np.r_[0.02, -0.01, 1.0, current.T @ linear, current.T @ angular], atol=1.5e-5
    )
    np.testing.assert_allclose(signals[0, 6:9], Rotation.from_matrix(current.T @ reference).as_rotvec(), atol=1e-6)
    np.testing.assert_array_equal(signals[0, 9:12], 0)
    # Repeated targets yield no feedforward even with nonzero tracking error.
    obs["preview"][:] = obs["preview"][1]
    tensors = {k: torch.tensor(v[None]) for k, v in env.normalize_obs(obs).items()}
    np.testing.assert_array_equal(
        features(tensors)[0, -features.feedback_width + 24 : -features.feedback_width + 33], 0
    )
    # Inactive targets are zeroed under preview_v1, including rotation columns.
    obs["preview"][:, p["hand_mask"]] = 0
    obs["preview"][:, p["hand_error"]] = 0
    tensors = {k: torch.tensor(v[None]) for k, v in env.normalize_obs(obs).items()}
    assert torch.isfinite(features(tensors)).all()


@pytest.mark.parametrize("orientation", ["rotation6", "rotvec"])
@pytest.mark.parametrize("velocity", [False, True])
def test_feedback_ablation_ppo_export_and_stationary_equilibrium(feedback, tmp_path, orientation, velocity):
    cfg, _, _ = feedback
    cfg = replace(cfg, feedback_orientation_error=orientation, feedback_reference_velocity=velocity)
    env = CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, norm_reward=False)
    try:
        model = make_model(cfg, env)
        model.learn(16)
        assert model.policy.action_net.gain.weight.detach().abs().sum() > 0
        with torch.no_grad():
            for parameter in model.policy.action_net.parameters():
                parameter.normal_(std=0.2)
        export(model, env, tmp_path, cfg, "feedback-test", steps=16)
        policy = ManipulationPolicy(tmp_path)
        np.testing.assert_array_equal(policy(equilibrium())[:-1], 0)
        obs = equilibrium()
        p, _ = preview_layout()
        obs["preview"][2, p["hand_error"].start] = 0.003
        np.testing.assert_allclose(policy(obs), model.predict(env.normalize_obs(obs), deterministic=True)[0], atol=1e-7)
        # A manifest cannot relabel an already trained feature representation.
        import json

        path = tmp_path / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["environment"]["feedback_reference_velocity"] = not velocity
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="feature options"):
            ManipulationPolicy(tmp_path)
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("restoring", [False, True])
def test_reference_velocity_transfer_preserves_actor_and_critic(feedback, tmp_path, restoring):
    from emet.rl.manipulation.initialization import _model, transfer_actor

    cfg, env, _ = feedback
    cfg = replace(cfg, restoring_yaw_feedback=restoring)
    source = make_model(cfg, env)
    with torch.no_grad():
        for parameter in source.policy.action_net.parameters():
            parameter.normal_(std=0.03)
    _, fingerprint = _model(cfg.robot, {})
    export(source, env, tmp_path, cfg, fingerprint, steps=0)
    cfg = replace(cfg, feedback_reference_velocity=True, feedback_orientation_scale=0.15)
    new_env = CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, norm_reward=False)
    try:
        target = make_model(cfg, new_env)
        details = transfer_actor(target, new_env, tmp_path, cfg, fingerprint)
        assert details["feedback_reference_velocity_added"]
        obs = equilibrium()
        p, _ = preview_layout()
        obs["preview"][2, p["hand_error"].start] = 0.004
        obs["preview"][:, p["base_error"].start + 2] = 0.03
        rotation = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
        obs["preview"][:, p["hand_error"].start + 3 : p["hand_error"].stop] = rotation[:, :2].ravel()
        # Include an eligible base-only turn to exercise the restoring branch.
        for hand in [True, False]:
            trial = {k: v.copy() for k, v in obs.items()}
            if not hand:
                trial["preview"][:, p["hand_mask"]] = 0
                trial["preview"][:, p["hand_error"]] = 0
            before, after = env.normalize_obs(trial), new_env.normalize_obs(trial)
            np.testing.assert_allclose(
                source.predict(before, deterministic=True)[0], target.predict(after, deterministic=True)[0], atol=1e-6
            )
            with torch.no_grad():
                torch.testing.assert_close(
                    source.policy.predict_values(source.policy.obs_to_tensor(before)[0]),
                    target.policy.predict_values(target.policy.obs_to_tensor(after)[0]),
                    atol=1e-6,
                    rtol=1e-5,
                )
        assert not target.policy.optimizer.state_dict()["state"]
    finally:
        new_env.close()


@pytest.mark.sim
def test_reference_velocity_respects_executor_progress_and_completion(feedback):
    from emet.rl.manipulation.env import ManipulationEnv
    from emet.rl.manipulation.tracking import TaskPoseTrajectory, encode_preview

    cfg, normalizer, _ = feedback
    cfg = replace(cfg, feedback_reference_velocity=True)
    env = ManipulationEnv(cfg)
    try:
        env.reset(seed=42)
        c = env.control
        hand = c.ee_pose()
        target = hand.copy()
        target[:3] += [0.02, 0, 0]
        trajectory = TaskPoseTrajectory(
            [0, 1], [c.base_pose()] * 2, [hand, target], [c.grip_reference] * 2, base_mask=[True] * 3
        )
        c.set_task_trajectory(trajectory)
        features = GoalFeedbackFeatures(normalizer.observation_space, **GoalFeedbackFeatures.kwargs_for(cfg))
        measured = []
        for elapsed, rate in [(0.25, 1), (0.25, 0.5), (0.25, 0), (1, 1)]:
            c.set_trajectory_progress(elapsed, rate=rate)
            obs = equilibrium()
            obs["preview"] = encode_preview(c, c.preview_sampler)
            preview = torch.tensor(normalizer.normalize_obs(obs)["preview"][None])
            measured.append(features.reference_velocity_features(preview).detach().numpy())
        np.testing.assert_allclose(measured[1], measured[0] * 0.5, atol=1e-6)
        assert np.linalg.norm(measured[0][0, 3:6]) > 0.019
        np.testing.assert_allclose(measured[2:], 0, atol=1e-6)
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("change", ["orientation", "remove_velocity"])
def test_feedback_transfer_rejects_semantic_changes(feedback, tmp_path, change):
    from emet.rl.manipulation.initialization import _model, transfer_actor

    cfg, _, _ = feedback
    cfg = replace(cfg, feedback_reference_velocity=change == "remove_velocity")
    env = CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, norm_reward=False)
    new_cfg = replace(
        cfg,
        feedback_reference_velocity=False,
        feedback_orientation_error="rotvec" if change == "orientation" else "rotation6",
    )
    target_env = CommandVecNormalize(DummyVecEnv([Commands]), config=new_cfg, norm_reward=False)
    try:
        source = make_model(cfg, env)
        _, fingerprint = _model(cfg.robot, {})
        export(source, env, tmp_path, cfg, fingerprint, steps=0)
        target = make_model(new_cfg, target_env)
        with pytest.raises(ValueError, match="representation changes|cannot remove"):
            transfer_actor(target, target_env, tmp_path, new_cfg, fingerprint)
    finally:
        env.close()
        target_env.close()


@pytest.mark.parametrize(
    "options",
    [
        {"feedback_reference_velocity": True},
        {"feedback_reference_velocity": "true"},
        {"feedback_orientation_error": "rotvec"},
        {"feedback_orientation_error": "euler"},
    ],
)
def test_feedback_ablation_requires_valid_feedback_config(options):
    with pytest.raises(ValueError, match="feedback|orientation"):
        ManipulationConfig(**options)
