import mujoco
import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.grasp_setup import plan_setup, support_margin
from emet.rl.manipulation.tracking import TaskFrameFK


def fixture():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="pregrasp", contact_cone="elliptic"))
    env.payload_name = "object1"
    env.payload = env.model.body("object1").id
    env.model.body_mass[env.payload] = 0.1
    env.model.body_inertia[env.payload] *= 5
    env.control.initialize(np.array([0, -0.4, 0.2, 0.2, 0]), env.profile.gripper_open)
    center = np.mean(
        [env.data.body(n).xpos + env.data.body(n).xmat.reshape(3, 3) @ [0.03, 0, 0] for n in env.profile.finger_bodies],
        axis=0,
    )
    address = env.model.jnt_qposadr[env.model.body("object1").jntadr[0]]
    env.data.qpos[address : address + 3] = center
    env.model.body_pos[env.model.body("table").id] = center - [0, 0, 0.13]
    env.model.geom_size[env.model.body_geomadr[env.model.body("table").id], :2] = 0.04
    mujoco.mj_forward(env.model, env.data)
    return env


@pytest.mark.sim
def test_loaded_setup_changes_base_but_preserves_world_hand_and_live_state():
    env = fixture()
    try:
        qpos, qvel, ctrl = (x.copy() for x in (env.data.qpos, env.data.qvel, env.data.ctrl))
        positions, sizes = env.model.body_pos.copy(), env.model.geom_size.copy()
        time = env.data.time
        assert support_margin(env.model, env.data, env.profile, env.payload)["margin_m"] < 0
        assert plan_setup(env, azimuths=(0.0,)) is None
        plan = plan_setup(env)
        assert plan is not None and abs(plan["azimuth_offset_rad"]) > 0
        assert plan["minimum_sampled_margin_m"] >= 0.005
        fk = TaskFrameFK(env.model, env.data, env.control)
        np.testing.assert_allclose(fk(plan["base"], plan["arm"]), plan["hand"], atol=1e-6)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
        np.testing.assert_array_equal(env.model.body_pos, positions)
        np.testing.assert_array_equal(env.model.geom_size, sizes)
        assert env.data.time == time
    finally:
        env.close()


@pytest.mark.sim
def test_maximum_margin_option_does_not_prefer_a_marginal_nominal_stance():
    from emet.eval.mars_tamp_control import fixture as public_fixture
    from emet.rl.manipulation.whole_body_training import load_development

    development, _ = load_development()
    env, admission = public_fixture(development["train_cases"][1], wheel_feedback_current_velocity=True)
    robust_env = None
    try:
        assert admission["accepted"]
        robust_env, robust_admission = public_fixture(
            development["train_cases"][1], wheel_feedback_current_velocity=True, grasp_setup_prefer_nominal=False
        )
        assert robust_admission["accepted"]
        # Compare selection before the procedural pregrasp approach and settling;
        # asking plan_setup() to lift again afterward is a different reach task.
        nominal = env.randomization["grasp_setup"]
        robust = robust_env.randomization["grasp_setup"]
        assert nominal["selection_reason"] == "nominal_path_margin"
        assert robust["selection_reason"] == "maximum_margin"
        assert robust["minimum_sampled_margin_m"] > nominal["minimum_sampled_margin_m"]
        np.testing.assert_allclose(robust["hand"], nominal["hand"], atol=1e-6)
        assert robust_env.config.grasp_setup_prefer_nominal is False
        assert env.config.grasp_setup_prefer_nominal is True
    finally:
        env.close()
        if robust_env is not None:
            robust_env.close()
