import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env

from emet.rl.base import ACCELERATION, BASE, BASE_CONTRACT, MarsBaseEnv
from emet.rl.server import ReactiveController

pytestmark = pytest.mark.sim


@pytest.fixture
def env():
    e = MarsBaseEnv()
    yield e
    e.close()


def test_base_gym_contract_and_reset(env):
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=42)
    assert env.action_space.shape == (2,)
    assert obs["history"].shape == (10, BASE_CONTRACT["history_width"])
    np.testing.assert_array_equal(obs["history"][0], obs["history"][-1])
    with pytest.raises(ValueError, match="base-only"):
        env.set_goal({"arm": {"joint1": 0.2}})
    with pytest.raises(ValueError, match="two finite"):
        env.step(np.ones(7))


def test_acceleration_limits_reversal_and_retarget_hold_arm(env):
    env.reset(seed=42)
    c = env.control
    arm = c.targets.copy()
    previous = np.zeros(2)
    for action in ([1, 1], [1, 1], [-1, -1], [-1, -1]):
        _, _, _, _, info = env.step(action)
        applied = c.last_action[:2] * [BASE.max_v, BASE.max_w]
        assert np.all(np.abs(applied - previous) <= ACCELERATION * BASE.control_dt + 1e-9)
        np.testing.assert_allclose(c.targets, arm, atol=1e-9)
        previous = applied.copy()
        before = c.last_action.copy()
        env.set_goal({"base": [0.4, -0.2, 0.1]})
        np.testing.assert_array_equal(c.last_action, before)
        assert np.isfinite(info["measured_jerk"]).all()


def test_baseline_settles_accurately(env):
    env.reset(seed=42)
    for _ in range(200):
        _, _, term, trunc, info = env.step(env.control.baseline_action())
        if term or trunc:
            break
    assert info["is_success"], info
    assert info["xy_error"] <= 0.02 and info["yaw_error"] <= 0.035
    assert np.linalg.norm(env.control.base_velocity()[:2]) < 0.03
    assert info["settling_time"] is not None


def test_runtime_matches_base_env_and_rejects_arm(env):
    class Policy:
        task = "base"

        def __call__(self, obs):
            assert obs["history"].shape == (10, BASE_CONTRACT["history_width"])
            return np.array([0.2, -0.1])

    env.reset(seed=42)
    d = mujoco.MjData(env.model)
    d.qpos[:] = env.data.qpos
    mujoco.mj_forward(env.model, d)
    runtime = ReactiveController(env.model, d, policy=Policy())
    runtime.start(env.control.goal.payload())
    with pytest.raises(ValueError, match="base-only"):
        runtime.update({"arm": {"joint1": 0.2}})
    for _ in range(25):
        runtime.step()
    obs, *_ = env.step([0.2, -0.1])
    live, _ = runtime.control.observe()
    np.testing.assert_allclose(obs["history"], live["history"], atol=1e-6)
    np.testing.assert_allclose(env.data.qpos, d.qpos, atol=1e-6)


def test_base_bundle_export_reload(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from emet.rl.policy import PPOPolicy, export_policy

    torch.set_num_threads(1)
    vec = VecNormalize(DummyVecEnv([MarsBaseEnv]), norm_reward=False)
    try:
        model = sb3.PPO(
            "MultiInputPolicy",
            vec,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            policy_kwargs={"net_arch": [32, 32]},
            device="cpu",
        )
        model.learn(32)
        obs = vec.reset()
        expected = model.predict(obs, deterministic=True)[0][0]
        raw = {k: v[0] for k, v in vec.get_original_obs().items()}
        export_policy(model, vec, tmp_path, {"env": {"task": "base"}})
        policy = PPOPolicy(tmp_path)
        assert policy.task == "base"
        np.testing.assert_allclose(policy(raw), expected, atol=1e-6)
    finally:
        vec.close()


def test_delayed_dynamics_are_seeded_and_reset():
    from emet.rl.env import EnvConfig

    e = MarsBaseEnv(EnvConfig(task="base", stage=1))
    try:
        traces = []
        for _ in range(2):
            e.reset(seed=17)
            trace = []
            for _ in range(6):
                _, reward, _, _, _ = e.step([0.5, -0.2])
                trace.append(np.r_[e.data.qpos, e.control.last_action, reward])
            traces.append(trace)
        np.testing.assert_array_equal(traces[0], traces[1])
    finally:
        e.close()


def test_base_baseline_runtime_acceleration(env):
    env.reset(seed=42)
    runtime = ReactiveController(env.model, env.data, base_only=True)
    runtime.start({"base": [0.5, 0, 0]})
    runtime.step()
    command = runtime.control.last_action[:2] * [BASE.max_v, BASE.max_w]
    assert np.all(np.abs(command) <= ACCELERATION * BASE.control_dt + 1e-9)
