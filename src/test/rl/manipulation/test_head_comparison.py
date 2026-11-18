"""Head diagnostics preserve masks, initialization, backbone and source weights."""

import numpy as np
import pytest
import torch

from emet.rl.manipulation.head_comparison import DirectMotionActor, fit_head, matched_heads
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.whole_body_actor import WholeBodyActor


def observations(robot):
    fields, width = history_layout(PROFILES[robot], retention_context=True)
    preview, _ = preview_layout()
    history = np.zeros((8, 20, width), np.float32)
    future = np.zeros((8, 7, 18), np.float32)
    history[..., fields["hand_pose"]][..., 3:] = [1, 0, 0, 1, 0, 0]
    history[..., fields["hand_mask"]] = 1
    history[..., fields["base_mask"]] = 1
    future[..., preview["hand_mask"]] = 1
    future[..., preview["base_mask"]] = 1
    future[..., preview["hand_error"]][..., 3:] = [1, 0, 0, 1, 0, 0]
    return {"history": history, "preview": future, "motion": np.zeros((8, 9), np.float32)}


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_shared_initial_means_frozen_backbone_and_independent_head_learning(robot):
    torch.set_num_threads(1)
    source = WholeBodyActor(robot)
    original = {k: v.clone() for k, v in source.state_dict().items()}
    heads = matched_heads(source)
    obs = observations(robot)
    tensor = {k: torch.from_numpy(v) for k, v in obs.items()}
    for actor in heads.values():
        np.testing.assert_array_equal(actor(tensor).detach(), 0)
        for block in (actor.gru, actor.context, actor.grip):
            assert all(not p.requires_grad for p in block.parameters())
    target = np.zeros((8, PROFILES[robot].action_size), np.float32)
    target[:, 2] = 0.1
    direct = heads["direct"]
    initial = {k: v.clone() for k, v in direct.state_dict().items()}
    fit_head(direct, obs, target, updates=10)
    assert not torch.equal(direct(tensor), heads["gain"](tensor))
    assert torch.count_nonzero(direct(tensor)[:, -1]) == 0
    for key, value in direct.state_dict().items():
        if not key.startswith("motion_head."):
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_direct_head_remains_bounded_and_cannot_act_with_all_targets_inactive():
    actor = DirectMotionActor(WholeBodyActor())
    with torch.no_grad():
        actor.motion_head[-1].bias.fill_(100)
    obs = observations("innate_mars")
    tensor = {k: torch.from_numpy(v) for k, v in obs.items()}
    assert torch.all(actor(tensor)[:, :-1] == 1)
    fields, _ = history_layout(PROFILES["innate_mars"], retention_context=True)
    tensor["history"][..., fields["hand_mask"]] = 0
    tensor["history"][..., fields["base_mask"]] = 0
    assert torch.count_nonzero(actor(tensor)) == 0


def test_source_base_feedback_mode_is_gain_only_and_preserves_declared_feedback():
    from emet.rl.manipulation.base_navigation_feedback import BaseNavigationFeedback

    source = WholeBodyActor(
        base_rate_feedback=False,
        base_navigation_feedback=BaseNavigationFeedback(reference_tracking=True, heading_gain_scale=2.0),
    )
    heads = matched_heads(source, source_base_feedback=True)
    assert set(heads) == {"gain"}
    gain = heads["gain"]
    assert gain.base_rate_feedback is False
    assert gain.base_navigation_feedback == source.base_navigation_feedback
    assert gain.base_navigation_feedback is not source.base_navigation_feedback
    assert torch.count_nonzero(gain.gain.weight) == torch.count_nonzero(gain.gain.bias) == 0
    for k, v in source.state_dict().items():
        if not k.startswith("gain."):
            torch.testing.assert_close(v, gain.state_dict()[k], rtol=0, atol=0)
    comparison = matched_heads(source)
    assert comparison["gain"].base_rate_feedback is True
    assert comparison["gain"].base_navigation_feedback is None


def test_direct_head_cannot_be_published_as_an_existing_gain_actor_bundle(tmp_path):
    from emet.rl.manipulation.env import ManipulationConfig
    from emet.rl.manipulation.whole_body_bundle import export_actor

    actor = DirectMotionActor(WholeBodyActor())
    config = ManipulationConfig(action_mode="direct_velocity", trajectory_preview=True, retention_context=True)
    destination = tmp_path / "actor"
    with pytest.raises(RuntimeError, match="state_dict"):
        export_actor(actor, destination, config, "diagnostic-model", steps=0, training={})
    assert not destination.exists()


def test_trainable_backbone_mode_is_explicit_and_keeps_initialization_rng_isolated():
    source = WholeBodyActor()
    rng = torch.get_rng_state().clone()
    heads = matched_heads(source, backbone_frozen=False)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert all(p.requires_grad for p in heads["direct"].gru.parameters())
    assert all(not p.requires_grad for p in heads["direct"].grip.parameters())
    before = {k: v.clone() for k, v in heads["direct"].gru.state_dict().items()}
    obs = observations("innate_mars")
    target = np.zeros((8, 8), np.float32)
    target[:, 2] = 0.1
    fit_head(heads["direct"], obs, target, updates=10)
    assert any(not torch.equal(v, before[k]) for k, v in heads["direct"].gru.state_dict().items())


def test_weighted_fitting_excludes_zero_weight_examples_and_rejects_invalid_mixtures():
    torch.set_num_threads(1)
    actor = matched_heads(WholeBodyActor())["direct"]
    obs = observations("innate_mars")
    targets = np.zeros((8, 8), np.float32)
    targets[:, 2] = -0.2
    targets[0, 2] = 0.2
    fit_head(actor, obs, targets, updates=40, sample_weights=[1, 0, 0, 0, 0, 0, 0, 0])
    tensor = {k: torch.from_numpy(v) for k, v in obs.items()}
    assert torch.all(actor(tensor)[:, 2] > 0.1)
    for weights in ([1], [0] * 8, [-1] + [1] * 7, [float("nan")] * 8, [1e308] * 8):
        with pytest.raises(ValueError, match="sample weights"):
            fit_head(actor, obs, targets, updates=1, sample_weights=weights)


@pytest.mark.sim
def test_acquisition_targets_do_not_teleport_or_edit_live_actuator_state():
    from emet.rl.manipulation.arm_splines import ArmSplineTrackingEnv, motion_manifest
    from emet.rl.manipulation.head_comparison import recovery_reference

    spec = motion_manifest()
    case = spec["cases"][0]
    env = ArmSplineTrackingEnv([case], timing="bounded_v2", finish_after_hold=True)
    try:
        env.reset()
        e = env.unwrapped
        before = {k: getattr(e.data, k).copy() for k in ("qpos", "qvel", "ctrl")}
        now = float(e.data.time)
        _, info = recovery_reference(env, seed=case["seed"] + 900_000_000)
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(e.data, key), value)
        assert float(e.data.time) == now
        assert info["initial_hand_error_m"] > 0.003
        assert env.reference.initial_hold_s == 1.5
        assert info["admission"]["accepted"]
        assert not e.control.stopped
    finally:
        env.close()


@pytest.mark.sim
def test_virtual_goal_labels_preserve_live_physics_history_and_command():
    from emet.rl.manipulation.arm_splines import ArmSplineTrackingEnv, motion_manifest
    from emet.rl.manipulation.head_comparison import label_nearby_goals

    env = ArmSplineTrackingEnv([motion_manifest()["cases"][0]], timing="bounded_v2", finish_after_hold=True)
    try:
        env.reset()
        e, c = env.unwrapped, env.unwrapped.control
        before = {k: getattr(e.data, k).copy() for k in ("qpos", "qvel", "ctrl")}
        history, goal = c.history.copy(), c.hand_goal.copy()
        trajectory, sampler, time = c.task_trajectory, c.preview_sampler, float(e.data.time)
        labels, _ = label_nearby_goals(env, seed=31, count=2)
        assert len(labels) == 2
        for label in labels:
            assert set(label["observation"]) == {"history", "preview", "motion"}
            np.testing.assert_array_equal(label["observation"]["motion"], 0)
            assert label["hand_error_m"] > 0.001
            assert np.isfinite(label["action"]).all() and np.max(abs(label["action"])) <= 1
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(e.data, key), value)
        np.testing.assert_array_equal(c.history, history)
        np.testing.assert_array_equal(c.hand_goal, goal)
        assert c.task_trajectory is trajectory and c.preview_sampler is sampler
        assert float(e.data.time) == time
    finally:
        env.close()
