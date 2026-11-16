"""Native masked feedback law, versioned export, and mask handoff invariants."""

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.wheel_feedback import execution_record
from emet.rl.manipulation.whole_body_actor import WholeBodyActor
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle, export_actor


def config(**kwargs):
    return replace(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            action_mode="direct_velocity",
            wheel_feedback_gain=0.5,
            wheel_feedback_current_velocity=True,
            physical_wheel_slew=True,
            wheel_feedback_base_only_gains=(0.0, 0.5),
            retention_context=True,
            trajectory_preview=True,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("value", [True, [0], [0, 3], [0, np.nan], [True, 0.5], [0, "0.5"], [0, 0.5, 1]])
def test_invalid_scoped_settings_rejected_before_simulation(value):
    with pytest.raises(ValueError):
        config(wheel_feedback_base_only_gains=value)


@pytest.mark.parametrize(
    "kwargs", [{"robot": "stretch"}, {"physical_wheel_slew": False}, {"wheel_feedback_current_velocity": False}]
)
def test_scoped_mode_requires_validated_native_execution_boundary(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_versioned_bundle_binds_environment_to_weights_and_cannot_be_downgraded(tmp_path):
    actor = WholeBodyActor(hidden=16)
    directory = tmp_path / "actor"
    cfg = config()
    export_actor(actor, directory, cfg, "model", steps=0)
    manifest = json.loads((directory / "manifest.json").read_text())
    weights = torch.load(directory / "policy.zip", weights_only=True)
    assert manifest["format"] == weights["format"] == "emet-whole-body-v4"
    assert manifest["wheel_feedback"] == weights["wheel_feedback"] == execution_record(cfg)
    loaded = WholeBodyBundle(directory)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(loaded.actor.state_dict()[key], value, rtol=0, atol=0)
    variants = []
    for field in ("wheel_feedback",):
        m = copy.deepcopy(manifest)
        m.pop(field)
        variants.append((m, weights))
        w = copy.deepcopy(weights)
        w.pop(field)
        variants.append((manifest, w))
    m = copy.deepcopy(manifest)
    m["environment"]["wheel_feedback_base_only_gains"] = [0.2, 0.5]
    variants.append((m, weights))
    m = copy.deepcopy(manifest)
    m["environment"]["wheel_feedback_current_velocity"] = False
    variants.append((m, weights))
    m = copy.deepcopy(manifest)
    w = copy.deepcopy(weights)
    m["format"] = w["format"] = "emet-whole-body-v3"
    variants.append((m, w))
    for m, w in variants:
        torch.save(w, directory / "policy.zip")
        m = copy.deepcopy(m)
        m["weights_sha256"] = hashlib.sha256((directory / "policy.zip").read_bytes()).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(m))
        with pytest.raises(ValueError):
            WholeBodyBundle(directory)


@pytest.mark.sim
def test_mask_switch_uses_current_mask_and_preserves_physics_history_and_fault_hold():
    env = ManipulationEnv(config())
    try:
        env.reset(seed=35874000)
        c = env.control
        c.set_pose_targets(c.base_pose(), [True] * 3, c.ee_pose(), False)
        np.testing.assert_array_equal(c.wheel_feedback_gains(), [0, 0.5])
        state = (env.data.qpos.copy(), env.data.qvel.copy(), c.history.copy(), c.target.copy(), c.grip_target)
        c.set_pose_targets(c.base_pose(), [True] * 3, c.ee_pose(), True)
        assert c.wheel_feedback_gains() == 0.5
        for a, b in zip((env.data.qpos, env.data.qvel, c.history, c.target, c.grip_target), state, strict=True):
            np.testing.assert_array_equal(a, b)
        for mask in ([True, True, False], [False, False, False], [False, False, True]):
            c.set_pose_targets(c.base_pose(), mask, c.ee_pose(), False)
            assert c.wheel_feedback_gains() == 0.5
        c.set_pose_targets(c.base_pose(), [True] * 3, c.ee_pose(), False)
        c.base_command[:] = [0.02, 0.03]
        c._base_feedback_velocity = lambda: np.array([0.01, -0.02])
        previous = c.applied_base_velocity.copy()
        env.data.time += env.model.opt.timestep
        interval = float(env.data.time) - c._applied_base_time
        c.apply()
        assert np.all(
            abs(c.applied_base_velocity - previous)
            <= np.array([c.profile.base_acceleration, c.profile.base_yaw_acceleration]) * interval + 1e-9
        )
        c.stop()
        hold = (c.target.copy(), c.grip_target)
        for active in (True, False, True):
            c.set_pose_targets(c.base_pose(), [True] * 3, c.ee_pose(), active)
            c.apply()
            assert c.stopped
            np.testing.assert_array_equal(c.applied_base_velocity, 0)
            np.testing.assert_array_equal(c.target, hold[0])
            assert c.grip_target == hold[1]
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("mask,hand", [([True] * 3, True), ([True, True, False], False)])
def test_ineligible_native_execution_is_bitwise_identical_to_scalar_feedback(mask, hand):
    legacy = ManipulationEnv(config(wheel_feedback_base_only_gains=None))
    scoped = ManipulationEnv(config())
    try:
        for env in (legacy, scoped):
            env.reset(seed=35874002)
            env.control.set_pose_targets(env.control.base_pose(), mask, env.control.ee_pose(), hand)
        action = np.array([0.08, -0.03, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        for _ in range(4):
            for env in (legacy, scoped):
                env._tick(action, score=False)
            for field in ("qpos", "qvel", "ctrl", "actuator_force"):
                np.testing.assert_array_equal(getattr(legacy.data, field), getattr(scoped.data, field))
    finally:
        legacy.close()
        scoped.close()


def test_scoped_execution_cannot_be_exported_for_a_different_actor_robot(tmp_path):
    with pytest.raises(ValueError, match="execution robot"):
        export_actor(WholeBodyActor("stretch", hidden=16), tmp_path / "actor", config(), "model", steps=0)
    assert not (tmp_path / "actor").exists()
