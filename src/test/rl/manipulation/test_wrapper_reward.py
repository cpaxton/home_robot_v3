"""Reward shape and physical-tick integration, independently of policy learning."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.reward_config import WholeBodyRewardConfig, hand_tracking_terms

PROFILE = Path("configs/benchmarks/whole_body_reward_contract_v1.json")


def test_versioned_profile_and_legacy_defaults():
    cfg = WholeBodyRewardConfig()
    assert WholeBodyRewardConfig.from_record(cfg.record()) == cfg
    assert WholeBodyRewardConfig.from_record(json.loads(PROFILE.read_text())) == cfg.contract_tracking()
    pose, velocity, gate = hand_tracking_terms(cfg, 0.01, 0.02, np.zeros(6), 0.003, 0.1)
    assert pose == pytest.approx(np.exp(-1.2))
    assert velocity == 0.5 and gate == 1
    assert cfg.failure_penalty == 2 and cfg.tamp_failure_penalty == 10


@pytest.mark.parametrize(
    "change",
    [
        {"position_scale_m": 0},
        {"rotation_scale_rad": float("nan")},
        {"hand_pose_weight": True},
        {"fine_pose_weight": 1.1},
        {"stability_scale": -1},
        {"progress_weight": float("inf")},
        {"pose_gated_velocity": 1},
        {"guard_penalty": 0},
    ],
)
def test_invalid_settings_rejected(change):
    with pytest.raises(ValueError):
        WholeBodyRewardConfig(**change)


@pytest.mark.parametrize("change", ["missing", "unknown", "version"])
def test_complete_configuration_required(change):
    record = WholeBodyRewardConfig().record()
    if change == "missing":
        del record["jerk_weight"]
    elif change == "unknown":
        record["typo"] = 1
    else:
        record["format"] = "future"
    with pytest.raises(ValueError):
        WholeBodyRewardConfig.from_record(record)


def test_fine_hand_priority_and_pose_gated_velocity():
    cfg = WholeBodyRewardConfig.contract_tracking()
    scores = [hand_tracking_terms(cfg, p, 0, np.zeros(6), 0.003, 0.1) for p in (0, 0.003, 0.005, 0.01, 0.02)]
    assert all(a[0] > b[0] and a[1] > b[1] for a, b in zip(scores, scores[1:], strict=False))
    assert scores[3][1] < 0.02  # quiet while 10 mm wrong earns little velocity credit
    assert scores[0][:2] == (4, 0.5)
    assert scores[2][0] - scores[3][0] > 0.3
    angular = [hand_tracking_terms(cfg, 0, r, np.zeros(6), 0.003, 0.1)[0] for r in (0, 0.02, 0.1, 0.2)]
    assert all(a > b for a, b in zip(angular, angular[1:], strict=False))
    assert hand_tracking_terms(cfg, 0, 0, np.ones(6), 0.003, 0.1)[1] < scores[0][1]


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, np.zeros(6), 0.003, 0.1),
        (0, 0, np.zeros(5), 0.003, 0.1),
        (0, 0, np.zeros(6), 0, 0.1),
        (0, float("nan"), np.zeros(6), 0.003, 0.1),
    ],
)
def test_malformed_tracking_inputs_fail_closed(args):
    with pytest.raises(ValueError):
        hand_tracking_terms(WholeBodyRewardConfig(), *args)


@pytest.fixture
def native():
    from emet.rl.manipulation.mastery import MasteryEnv, load_suite

    env = MasteryEnv(
        load_suite("configs/benchmarks/mars_tracking_mastery_supported_v1.json"),
        stages=("arm",),
        split="train",
        population="training",
    )
    env.reset(seed=90)
    try:
        yield env.unwrapped
    finally:
        env.close()


@pytest.mark.sim
def test_progress_tracks_measured_motion_not_goal_motion(native, monkeypatch):
    from emet.rl.manipulation.whole_body_reward import RewardMeter

    c = native.control
    measured = c.ee_pose().copy()
    monkeypatch.setattr(c, "ee_pose", lambda: measured.copy())
    c.hand_mask = True
    meter = RewardMeter(native, config=WholeBodyRewardConfig.contract_tracking())
    action, motion = np.zeros(c.profile.action_size), np.zeros(9)

    def tick():
        meter.tick(action, motion)
        return meter.last_terms["hand_progress_reward"]

    with pytest.raises(ValueError, match="before_step"):
        tick()
    meter.before_step()
    c.hand_goal = measured.copy()
    c.hand_goal[0] += 0.02
    assert tick() == pytest.approx(0)  # reference jumps; robot did not move
    meter.before_step()
    measured[0] += 0.005
    assert tick() > 0
    meter.before_step()
    measured[0] -= 0.005
    assert tick() < 0
    meter.before_step()
    measured[3:] = (
        Rotation.from_rotvec([0, 0, 0.02]).as_matrix() @ np.asarray(c.hand_goal[3:]).reshape(3, 2)[:, :2]
    ).ravel()
    assert tick() < 0
    c.hand_mask = False
    meter.before_step()
    measured[0] += 0.005
    assert tick() == 0
    meter.before_step()
    c.hand_mask = True
    measured[0] += 0.005
    assert tick() == 0  # activation cannot earn previous inactive movement


@pytest.mark.sim
def test_stability_curriculum_preserves_contact_guard(native, monkeypatch):
    import emet.rl.manipulation.whole_body_reward as module

    monkeypatch.setattr(module, "current_contact_force", lambda e: 6.0)
    native.monitor.wheel_lift = True
    meter = module.RewardMeter(native, config=replace(WholeBodyRewardConfig(), stability_scale=0))
    groups = meter.tick(np.zeros(native.profile.action_size), np.zeros(9))
    assert groups[1] == 0
    assert groups[2] == pytest.approx(-10 - native.profile.dt * 0.1)


@pytest.mark.sim
@pytest.mark.parametrize("kind", ["general", "coupled"])
def test_wrappers_report_and_apply_explicit_reward(kind):
    cfg = WholeBodyRewardConfig.contract_tracking()
    if kind == "general":
        from emet.rl.manipulation.whole_body_training import make_environment

        env = make_environment(
            "configs/benchmarks/mars_whole_body_v2_development.json",
            basic_cases=Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json"),
            reward_config=cfg,
        )
    else:
        from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv

        env = CoupledTrackingEnv("innate_mars", reward_config=cfg)
    try:
        _, info = env.reset(seed=90)
        assert info["reward_contract"]["configuration"] == cfg.record()
        _, reward, _, _, info = env.step(np.zeros(8))
        assert np.isfinite(reward)
        assert info["reward_contract"]["configuration"] == cfg.record()
        assert info["reward_terms"]["hand_pose_component"] <= 4
    finally:
        env.close()


@pytest.mark.sim
def test_spawned_tamp_preserves_contract_and_bundled_tick_rewards():
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
    from emet.rl.manipulation.whole_body_training import load_development

    suite, _ = load_development()
    cfg = replace(WholeBodyRewardConfig.contract_tracking(), success_bonus=7)
    env = TampPrimitiveEnv(suite["train_cases"][:1], oracle=True, reward_config=cfg.record())
    bundled = False
    try:
        _, info = env.reset()
        for _ in range(2000):
            assert info["reward_contract"]["configuration"] == cfg.record()
            _, _, done, truncated, info = env.step(info["oracle_action"])
            ticks = info["reward_group_ticks"]
            np.testing.assert_allclose(ticks.sum(axis=0), info["reward_groups"], atol=1e-5)
            bundled |= len(ticks) > 1
            if done or truncated:
                break
        else:
            pytest.fail("bounded pregrasp did not finish")
        assert info["primitive_success"] and info["stop_confirmed"] and bundled
        from emet.rl.manipulation.profiles import MARS

        terms = info["reward_terms"]
        last_tracking = MARS.dt * (terms["hand_pose_score"] + terms["base_pose_score"]) + terms["hand_progress_reward"]
        assert ticks[-1, 0] == pytest.approx(last_tracking + cfg.success_bonus, abs=1e-6)
    finally:
        env.close()


def test_paired_runs_reject_reward_changes_before_rollout_comparison():
    from emet.rl.manipulation.whole_body_training import verify_exploration_pair, verify_initial_pair

    for check in (verify_initial_pair, verify_exploration_pair):
        with pytest.raises(ValueError, match="reward configurations differ"):
            check(
                {"reward_configuration": WholeBodyRewardConfig().record()},
                {"reward_configuration": WholeBodyRewardConfig.contract_tracking().record()},
            )
