# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Contract-native Gym environment and PPO checkpoint tests."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from emet.controller.manipulation.wbc_contract import (
    ArmObservation,
    ArmSpec,
    BaseSpec,
    ContactObservation,
    EmbodimentProfile,
    GripperObservation,
    GripperSpec,
    JointGroupSpec,
    JointSpec,
    WbcObservation,
    active_action_mask,
)
from emet.controller.manipulation.wbc_profiles import sourccey_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv, encode_observation, wbc_observation_layout
from emet.rl.manipulation.wbc_training import (
    SCHEMA_VERSION,
    WbcActorCritic,
    WbcPPO,
    WbcPPOConfig,
    collect_teacher_data,
    evaluate_policy,
)


def _fake_profile() -> EmbodimentProfile:
    arm = JointGroupSpec(
        name="arm",
        role="arm",
        joints=(
            JointSpec(
                "j1",
                (-1.0, 1.0),
                1.0,
                1.0,
                kp_eff=1.0,
                max_lead=0.2,
            ),
            JointSpec(
                "j2",
                (-1.0, 1.0),
                1.0,
                1.0,
                kp_eff=1.0,
                max_lead=0.2,
            ),
        ),
    )
    return EmbodimentProfile(
        name="fake_contract_robot",
        base=BaseSpec(name="base", kind="planar", velocity_limits=(0.5, 1.0)),
        joint_groups=(arm,),
        arms=(
            ArmSpec(
                name="primary",
                joint_group="arm",
                ee_link="ee",
                gripper=GripperSpec(
                    name="gripper",
                    joint_names=("grip",),
                    command_limits=((-1.0, 1.0),),
                ),
            ),
        ),
    )


class _FakeBackend:
    profile = _fake_profile()

    def __init__(self) -> None:
        self.q = np.zeros(2)
        self.base = np.zeros(3)
        self.target = (0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0)

    def reset(self, *, seed=None, options=None):
        self.q = np.zeros(2)
        self.base = np.zeros(3)
        return self._observation()

    def step(self, command, dt):
        self.q = np.asarray([command.joint_positions["j1"], command.joint_positions["j2"]], dtype=np.float64)
        self.base[0] += float(command.base_velocity[0]) * dt
        self.base[2] = math.atan2(
            math.sin(self.base[2] + float(command.base_velocity[1]) * dt),
            math.cos(self.base[2] + float(command.base_velocity[1]) * dt),
        )
        return self._observation(), False, {"contact_force": 0.0}

    def teacher_action(self):
        return np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)

    def close(self):
        return None

    def _observation(self) -> WbcObservation:
        return WbcObservation(
            base_pose=tuple(float(v) for v in self.base),
            base_twist=(0.0, 0.0),
            arms=(
                ArmObservation(
                    name="primary",
                    joint_position=tuple(float(v) for v in self.q),
                    joint_velocity=(0.0, 0.0),
                    joint_torque=(0.0, 0.0),
                    ee_pose=(float(self.q[0]), float(self.q[1]), 0.0, 1.0, 0.0, 0.0, 0.0),
                    ee_target=self.target,
                    target_mask=(True, True),
                    gripper=GripperObservation(position=(0.0,), velocity=(0.0,)),
                ),
            ),
            occupancy=None,
        )


def test_layout_encodes_targets_and_presence_flags():
    profile = _fake_profile()
    observation = _FakeBackend()._observation()
    layout = wbc_observation_layout(profile)
    vector = encode_observation(observation, profile, layout=layout)
    assert vector.shape == (layout.size,)
    assert np.isfinite(vector).all()
    assert layout.slices["has_base_target"].stop - layout.slices["has_base_target"].start == 1
    assert not vector[layout.slices["has_base_target"]][0]
    assert vector[layout.slices["arm.primary.has_target"]][0]
    assert np.allclose(vector[layout.slices["arm.primary.ee_target"]], observation.arm("primary").ee_target)

    with_target = replace(observation, base_target=(0.5, 0.0, 0.1))
    encoded = encode_observation(with_target, profile, layout=layout)
    assert encoded[layout.slices["has_base_target"]][0] == 1.0
    assert encoded[layout.slices["base_target"]][2] == pytest.approx(0.1)


def test_layout_encodes_measured_contact_and_gripper_force():
    profile = _fake_profile()
    observation = _FakeBackend()._observation()
    layout = wbc_observation_layout(profile)
    contact = ContactObservation(
        in_contact=True,
        wrench=(10.0, -20.0, 30.0, 1.0, 2.0, 3.0),
        source="ft_sensor",
    )
    arm = replace(
        observation.arms[0],
        contact=contact,
        gripper=GripperObservation(position=(0.0,), velocity=(0.0,), force=(10.0,)),
    )
    with_contact = replace(observation, arms=(arm,))
    vector = encode_observation(with_contact, profile, layout=layout)
    assert vector[layout.slices["arm.primary.has_contact_state"]][0] == 1.0
    assert vector[layout.slices["arm.primary.contact_flag"]][0] == 1.0
    assert np.allclose(
        vector[layout.slices["arm.primary.contact_wrench"]],
        (0.2, -0.4, 0.6, 0.1, 0.2, 0.3),
    )
    assert vector[layout.slices["arm.primary.gripper_force"]][0] == pytest.approx(0.5)

    missing = encode_observation(observation, profile, layout=layout)
    assert missing[layout.slices["arm.primary.has_contact_state"]][0] == 0.0
    assert missing[layout.slices["arm.primary.contact_flag"]][0] == 0.0
    assert np.all(missing[layout.slices["arm.primary.contact_wrench"]] == 0.0)


def test_layout_includes_occupancy_when_requested():
    profile = _fake_profile()
    layout = wbc_observation_layout(profile, occupancy_shape=(2, 2, 2))
    observation = _FakeBackend()._observation()
    missing = encode_observation(observation, profile, layout=layout)
    assert missing[layout.slices["has_occupancy"]][0] == 0
    assert np.all(missing[layout.slices["occupancy"]] == 0)

    occupied = replace(observation, occupancy=np.ones((2, 2, 2), dtype=np.uint8))
    vector = encode_observation(occupied, profile, layout=layout)
    assert vector[layout.slices["has_occupancy"]][0] == 1
    assert np.allclose(vector[layout.slices["occupancy"]], 1.0)

    with pytest.raises(ValueError, match="occupancy"):
        encode_observation(occupied, profile)


def test_contract_env_reset_step_and_action_mask():
    env = WbcGymEnv(_FakeBackend(), horizon_s=2.0, control_dt=0.1, success_hold_s=0.1)
    observation, info = env.reset(seed=0)
    assert observation.shape == env.observation_space.shape
    assert info["profile_fingerprint"] == _fake_profile().fingerprint
    mask = active_action_mask(_FakeBackend()._observation(), _fake_profile())
    assert mask.tolist() == [True, True, True, True, True]

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[2] = 1.0  # move j1 toward the target
    next_observation, reward, terminated, truncated, info = env.step(action)
    assert next_observation.shape == observation.shape
    assert np.isfinite(reward)
    assert "reward_terms" in info
    assert info["reward_terms"]["pos_err"] > 0
    assert info["action_mask"].tolist() == [True, True, True, True, True]
    assert not terminated
    assert not truncated


def test_env_hold_base_ignores_base_actions():
    env = WbcGymEnv(_FakeBackend(), horizon_s=2.0, control_dt=0.1, success_hold_s=0.1, hold_base=True)
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    action[1] = 1.0
    _, _, _, _, info = env.step(action)
    assert np.allclose(env.backend.base, 0.0)
    assert info["action_mask"][:2].tolist() == [False, False]


def test_contract_env_terminates_after_success_hold():
    env = WbcGymEnv(_FakeBackend(), horizon_s=5.0, control_dt=0.1, success_hold_s=0.1)
    env.reset(seed=0)
    # The adapter commands relative to its measured observation, so place both
    # the plant and the env's held observation at the target before holding.
    env.backend.q = np.array([0.5, 0.5])
    env._observation = env.backend._observation()
    _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    assert info["success"]
    assert terminated
    assert not truncated


@pytest.mark.sim
def test_mujoco_backend_reaches_sampled_arm_target_with_pd():
    from emet.rl.manipulation.wbc_mujoco import MuJoCoWbcBackend

    backend = MuJoCoWbcBackend(seed=0, base_goal_probability=0.0)
    env = WbcGymEnv(backend, horizon_s=3.0, control_dt=0.05, success_hold_s=0.1)
    env.reset(seed=0)
    assert backend._target_q is not None
    info = {"success": False}
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(backend.teacher_action())
        if terminated or truncated:
            break
    assert info["success"], info
    env.close()


@pytest.mark.sim
def test_mujoco_teacher_reaches_forward_base_and_arm_target():
    from emet.rl.manipulation.wbc_mujoco import MuJoCoWbcBackend

    backend = MuJoCoWbcBackend(seed=2, base_goal_probability=1.0, base_goal_mode="forward")
    env = WbcGymEnv(backend, horizon_s=5.0, control_dt=0.05, success_hold_s=0.1)
    env.reset(seed=2)
    info = {"success": False}
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(backend.teacher_action())
        if terminated or truncated:
            break
    assert info["success"], info
    env.close()


@pytest.mark.sim
def test_mujoco_backend_base_velocity_moves_the_planar_base():
    from emet.rl.manipulation.wbc_mujoco import MuJoCoWbcBackend

    env = WbcGymEnv(MuJoCoWbcBackend(seed=0), horizon_s=2.0, control_dt=0.05, success_hold_s=0.1)
    env.reset(seed=0)
    start = env.backend.control.base_pose().copy()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0  # bounded forward base velocity
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    end = env.backend.control.base_pose()
    assert end[0] - start[0] > 0.03
    assert abs(end[1] - start[1]) < 0.02
    env.close()


@pytest.mark.sim
def test_adapter_matched_mujoco_backend_smoke():
    from emet.rl.manipulation.wbc_mujoco import MuJoCoWbcBackend

    env = WbcGymEnv(MuJoCoWbcBackend(seed=0), horizon_s=0.5, control_dt=0.05, success_hold_s=0.1)
    observation, info = env.reset(seed=0)
    assert observation.shape == env.observation_space.shape
    next_observation, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    assert next_observation.shape == observation.shape
    assert np.isfinite(reward)
    assert "reward_terms" in info
    env.close()


def test_ppo_updates_and_checkpoint_round_trip(tmp_path):
    profile = _fake_profile()
    env = WbcGymEnv(_FakeBackend(), horizon_s=2.0, control_dt=0.1, success_hold_s=0.1)
    config = WbcPPOConfig(
        hidden=16,
        rollout_steps=8,
        epochs=1,
        batch_size=4,
        seed=0,
        bc_kl_coefficient=0.1,
    )
    trainer = WbcPPO(
        WbcActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden=16),
        config,
    )
    teacher_observations, teacher_actions = collect_teacher_data(env, steps=8, seed=0)
    teacher_metrics = trainer.pretrain(teacher_observations, teacher_actions, epochs=1, batch_size=4)
    assert np.isfinite(teacher_metrics["bc_loss"])

    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics = trainer.train(
        env,
        total_steps=16,
        checkpoint_path=checkpoint_path,
        checkpoint_every=8,
        metrics_path=metrics_path,
        checkpoint_metadata={"contract": {"horizon_s": 2.0}},
    )
    assert np.isfinite(list(metrics.values())).all()
    assert checkpoint_path.is_file()
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 2
    periodic, periodic_metadata = WbcPPO.load(checkpoint_path, profile)
    assert periodic_metadata["contract"] == {"horizon_s": 2.0}
    assert periodic.steps_trained >= 8

    path = trainer.save(tmp_path / "policy.pt", env, metadata={"test": True})
    loaded, metadata = WbcPPO.load(path, profile)
    assert metadata == {"test": True}
    assert loaded.policy.action_size == env.action_space.shape[0]
    assert loaded.policy.observation_size == env.observation_space.shape[0]
    assert loaded.steps_trained == 16
    loaded.train(env, total_steps=8)
    assert loaded.steps_trained == 24
    evaluation = evaluate_policy(loaded.policy, env, episodes=1, seed=100)
    assert evaluation["episodes"] == 1.0
    assert np.isfinite([evaluation["mean_return"], evaluation["success_rate"], evaluation["mean_steps"]]).all()

    changed = replace(profile, name="fake_contract_robot_changed")
    with pytest.raises(ValueError, match="fingerprint"):
        WbcPPO.load(path, changed)


def test_load_migrates_v2_observation_layout(tmp_path):
    profile = sourccey_sim_profile()
    old_layout = wbc_observation_layout(profile, include_contact=False)
    new_layout = wbc_observation_layout(profile)
    assert old_layout.size < new_layout.size
    policy = WbcActorCritic(old_layout.size, profile.action_dim, hidden=8)
    with torch.no_grad():
        policy.actor[0].weight.zero_()
        policy.critic[0].weight.zero_()
        policy.actor[0].weight[0, old_layout.slices["arm.right.joint_position"]] = 1.0
        policy.actor[0].weight[0, old_layout.slices["arm.right.gripper_force"]] = 1.0
    payload = {
        "format": WbcPPO.format,
        "schema_version": SCHEMA_VERSION,
        "observation_layout_version": 2,
        "profile": profile.to_dict(),
        "profile_fingerprint": profile.fingerprint,
        "observation_size": old_layout.size,
        "action_size": profile.action_dim,
        "config": asdict(WbcPPOConfig(hidden=8)),
        "steps_trained": 0,
        "updates": 0,
        "metadata": {"test": True},
        "state_dict": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
        "optimizer_state": None,
        "reference_actor_state": None,
        "reference_log_std": None,
    }
    path = tmp_path / "v2_policy.pt"
    torch.save(payload, path)
    loaded, metadata = WbcPPO.load(path, profile)
    assert loaded.policy.observation_size == new_layout.size
    assert metadata["observation_layout_migrated_from"] == 2
    weight = loaded.policy.actor[0].weight
    assert torch.allclose(
        weight[0, new_layout.slices["arm.right.joint_position"]],
        torch.ones(5),
    )
    assert torch.allclose(weight[0, new_layout.slices["arm.right.gripper_force"]], torch.tensor([20.0]))
    assert torch.count_nonzero(weight[0, new_layout.slices["arm.right.has_contact_state"]]) == 0
    assert torch.count_nonzero(weight[0, new_layout.slices["arm.left.joint_position"]]) == 0


def test_ppo_odd_total_steps_are_consumed():
    env = WbcGymEnv(_FakeBackend(), horizon_s=2.0, control_dt=0.1, success_hold_s=0.1)
    config = WbcPPOConfig(hidden=8, rollout_steps=2, epochs=1, batch_size=2, seed=0)
    trainer = WbcPPO(
        WbcActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden=8),
        config,
    )
    trainer.train(env, total_steps=3)
    assert trainer.steps_trained == 3
    env.close()
