import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env

from emet.rl.control import Goal, MarsControl, baseline_action
from emet.rl.env import EnvConfig, MarsReachEnv
from emet.rl.model import build_model, model_fingerprint
from emet.rl.server import ReactiveController

pytestmark = pytest.mark.sim


@pytest.fixture
def env():
    result = MarsReachEnv()
    yield result
    result.close()


def test_gym_contract_and_seeded_trajectory(env):
    check_env(env, skip_render_check=True)
    traces = []
    for _ in range(2):
        obs, _ = env.reset(seed=17)
        trace = []
        for _ in range(10):
            obs, reward, terminated, truncated, _ = env.step(np.zeros(7))
            trace.append(np.r_[env.data.qpos, reward, terminated, truncated])
        traces.append(trace)
    np.testing.assert_array_equal(traces[0], traces[1])
    assert env.data.time == pytest.approx(0.5)
    assert model_fingerprint() == model_fingerprint()


def test_reaching_excludes_gripper_mimic_and_masks_inactive_base(env):
    env.reset(seed=42)
    before = env.control.base_pose()
    gripper = env.data.qpos[env.model.jnt_qposadr[env.model.joint("joint6").id]]
    env.set_goal({"arm": {"joint1": 0.1}})
    for _ in range(5):
        env.step(np.ones(7))
    assert env.control.last_action[:2].tolist() == [0, 0]
    assert env.control.last_action[3:].tolist() == [0, 0, 0, 0]
    np.testing.assert_allclose(env.control.base_pose(), before, atol=0.002)
    q6 = env.data.qpos[env.model.jnt_qposadr[env.model.joint("joint6").id]]
    q6m = env.data.qpos[env.model.jnt_qposadr[env.model.joint("joint6M").id]]
    assert q6 == pytest.approx(gripper, abs=0.001)
    assert q6m == pytest.approx(-q6, abs=0.001)
    with pytest.raises(ValueError, match="unknown reaching"):
        Goal.parse({"arm": {"joint6": 0.2}}, env.control)


def test_clipping_and_invalid_actions(env):
    env.reset(seed=1)
    old = env.control.targets.copy()
    env.control.accept_action(np.full(7, 100))
    assert np.max(np.abs(env.control.targets - old)) <= 0.03000001
    for bad in (np.zeros(8), np.full(7, np.nan)):
        with pytest.raises(ValueError, match="seven finite"):
            env.control.accept_action(bad)
    with pytest.raises(ValueError, match="outside limits"):
        env.set_goal({"arm": {"joint1": 100}})


def test_baseline_completes_coordinated_reach(env):
    env.reset(seed=42)
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(baseline_action(env.control))
        if terminated or truncated:
            break
    assert info["is_success"] and not info["collision"]
    assert info["xy_error"] < 0.05
    assert info["joint_error"] < 0.05


def test_collision_geometry_and_obstacle_observation():
    env = MarsReachEnv(EnvConfig(stage=3, obstacle_count=1))
    try:
        env.reset(seed=2)
        base = env.control.base_pose()
        env.data.mocap_pos[0] = [base[0] + 0.4, base[1], 0.2]
        mujoco.mj_forward(env.model, env.data)
        far, far_info = env.control.observe()
        assert far["obstacles"].reshape(-1, 5)[0, 4] == 1
        env.data.mocap_pos[0, :2] = base[:2]
        mujoco.mj_forward(env.model, env.data)
        _, near = env.control.observe()
        assert near["collision"]
        assert near["obstacle_clearance"] < far_info["obstacle_clearance"]
    finally:
        env.close()


def test_transformed_base_axes_map_forward_velocity():
    model = build_model()
    parent = model.body("base_root").id
    model.body_quat[parent] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    control = MarsControl(model, data)
    control.accept_action([1, 0, 0, 0, 0, 0, 0])
    control.apply()
    np.testing.assert_allclose(data.ctrl[control.base_act], [0.3, 0, 0], atol=1e-7)
    assert control.base_pose()[2] == pytest.approx(np.pi / 2)


def test_training_and_runtime_use_identical_observations_and_actions(env):
    env.reset(seed=42)
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.data.qpos
    data.qvel[:] = env.data.qvel
    mujoco.mj_forward(env.model, data)
    runtime = ReactiveController(env.model, data, policy=lambda obs: np.full(7, 0.1))
    runtime.start(env.control.goal.payload())
    for _ in range(25):
        runtime.step()
    obs, *_ = env.step(np.full(7, 0.1))
    live, _ = runtime.control.observe()
    for key in obs:
        np.testing.assert_allclose(live[key], obs[key], atol=1e-6)


def test_runtime_goal_update_is_applied_at_next_tick_and_failure_stops(env):
    env.reset(seed=42)
    runtime = ReactiveController(env.model, env.data, policy=lambda obs: np.full(7, np.nan))
    runtime.start(env.control.goal.payload())
    runtime.update({"arm": {"joint1": 0.25}})
    runtime.step()
    assert runtime.control.goal.arm[0] == 0.25
    assert runtime.outcome[0] == "failed"
    assert runtime.stopping
    assert not runtime.control.last_action.any()


def test_time_limit_and_goal_change_reward_rebase(env):
    env.reset(seed=42)
    env.set_goal({"base": [100, 0, 0]})
    _, reward, _, _, _ = env.step(np.zeros(7))
    assert reward < 0  # No spurious positive progress from target replacement.
    env.steps = 199
    _, _, terminated, truncated, _ = env.step(np.zeros(7))
    assert not terminated and truncated
