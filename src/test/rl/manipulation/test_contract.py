from dataclasses import replace

import mujoco
import numpy as np
import pytest

from emet.rl.manipulation.control import ManipulationControl
from emet.rl.manipulation.env import (
    DISTURBANCE_FORCE_N,
    DISTURBANCE_TORQUE_NM,
    ManipulationConfig,
    ManipulationEnv,
    sample_disturbance,
)
from emet.rl.manipulation.fixtures import evidence, load_case
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.profiles import (
    STRETCH,
    STRETCH_CLOSED_SLIDE_M,
    TRACKING_TRIM_GAIN,
    TRACKING_TRIM_LIMIT,
    ModelAdmissionError,
    stretch_finger_to_slide,
)
from emet.rl.manipulation.safety import ContactMonitor, retention_error

pytestmark = pytest.mark.sim


def test_stretch_closed_gripper_reference_is_derived_from_archived_coupling():
    assert STRETCH.gripper_closed == pytest.approx(STRETCH_CLOSED_SLIDE_M)
    assert stretch_finger_to_slide(-0.376) == pytest.approx(-0.020)
    assert stretch_finger_to_slide(0.560) == pytest.approx(0.040)


def test_disturbance_randomizes_world_frame_end_effector_wrench():
    wrench = sample_disturbance(np.random.default_rng(31))
    assert wrench.shape == (6,)
    assert np.isfinite(wrench).all()
    assert DISTURBANCE_FORCE_N[0] <= np.linalg.norm(wrench[:3]) <= DISTURBANCE_FORCE_N[1]
    assert DISTURBANCE_TORQUE_NM[0] <= np.linalg.norm(wrench[3:]) <= DISTURBANCE_TORQUE_NM[1]
    mild = sample_disturbance(np.random.default_rng(31), difficulty=0)
    assert 0.5 <= np.linalg.norm(mild[:3]) <= 1.0
    assert 0.01 <= np.linalg.norm(mild[3:]) <= 0.04


def test_payload_friction_randomization_does_not_change_support_traction():
    env = ManipulationEnv(ManipulationConfig(scenario="reach"))
    try:
        _, info = env.reset(seed=17)
        payload_geom = env.model.body_geomadr[env.payload]
        expected = env.friction.copy()
        expected[payload_geom, 0] *= info["randomization"]["friction_factor"]
        np.testing.assert_allclose(env.model.geom_friction, expected)
    finally:
        env.close()


def test_disturbance_is_a_ramped_wrench_on_the_hand_not_the_payload():
    env = ManipulationEnv(ManipulationConfig(scenario="disturbance"))
    try:
        _, info = env.reset(seed=0)
        assert not info["initialization_failed"]
        env.disturbance_start = 0
        env.disturbance_steps = 6
        env.disturbance_wrench = np.array([2, 0, 0, 0, 0.1, 0])
        env.step(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(env.data.xfrc_applied[env.control.ee], 0)
        env.step(np.zeros(env.action_space.shape))
        np.testing.assert_allclose(env.data.xfrc_applied[env.control.ee], env.disturbance_wrench * 0.5)
        env.step(np.zeros(env.action_space.shape))
        np.testing.assert_allclose(env.data.xfrc_applied[env.control.ee], env.disturbance_wrench)
        np.testing.assert_array_equal(env.data.xfrc_applied[env.payload], 0)
    finally:
        env.close()


def test_frozen_goals_and_failures_unchanged():
    left, right = evidence("left"), evidence("right")
    assert left["rotation_limit_rad"] == 0.1
    assert left["translation_limit_m"] == 0.02
    assert left["goals"] == right["goals"]
    failure = left["events"][-1]
    assert failure["rotation_residual_rad"] > left["rotation_limit_rad"]
    assert right["events"][-1]["contacts"][0]["distance_m"] == pytest.approx(-0.0012670796046189946)
    assert not left["complete_physics_snapshot"]


def test_mars_missing_support_geometry_blocks_training():
    with pytest.raises(ModelAdmissionError, match="wheel/caster"):
        build_manipulation_model("innate_mars")


def test_mars_gripper_uses_rectangular_urdf_contact_jaws():
    model, _ = build_manipulation_model("innate_mars", diagnostics=True)
    for name in ("link61_contact", "link62_contact"):
        geom = model.geom(name)
        assert model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_BOX
        np.testing.assert_allclose(model.geom_size[geom.id], [0.024, 0.008, 0.01945])
        np.testing.assert_allclose(model.geom_pos[geom.id], [0.024, 0, 0])
        assert model.geom_contype[geom.id] == 1
        assert model.geom_conaffinity[geom.id] == 1


def test_grip_corrections_are_bounded_and_release_requires_authorization():
    model, _ = build_manipulation_model("stretch")
    data = mujoco.MjData(model)
    control = ManipulationControl(model, data, STRETCH)
    control.initialize([0.55, 0.15, 0, -0.26, 0], 0)
    control.set_reference(control.positions(), phase="carry", grip=0)
    for _ in range(100):
        control.accept(np.array([0, 0, 0, 0, 0, 1]))
    assert control.grip_target <= 0.05 * 0.06 + 1e-12
    with pytest.raises(ValueError, match="authorization"):
        control.set_reference(control.positions(), phase="release")
    control.grip_target = control.grip_reference + 0.001
    control.stop()
    assert control.grip_target == control.grip_reference
    assert not control.release_authorized
    control.accept(np.ones(6))
    assert control.grip_target == control.grip_reference
    control.set_reference(control.positions())
    assert control.stopped  # A new goal cannot clear a safety latch.


def test_fault_during_authorized_release_freezes_measured_gripper_position():
    model, _ = build_manipulation_model("stretch")
    data = mujoco.MjData(model)
    control = ManipulationControl(model, data, STRETCH)
    control.initialize([0.55, 0.15, 0, -0.26, 0], STRETCH.gripper_closed)
    control.set_reference(control.positions(), phase="release", release=True)
    measured = (STRETCH.gripper_closed + STRETCH.gripper_open) / 2
    data.qpos[control.grip_q] = measured
    control.stop()
    assert control.grip_target == pytest.approx(measured)
    assert not control.release_authorized


def test_tracking_trim_is_slow_and_bounded_to_compensate_load_error():
    model, _ = build_manipulation_model("stretch")
    data = mujoco.MjData(model)
    control = ManipulationControl(model, data, STRETCH)
    control.initialize([0.55, 0.15, 0, -0.26, 0], 0)
    pitch = 3
    control.target[pitch] += 0.03
    control._trim_time = float(data.time) - 0.1
    control.apply()
    assert control.tracking_integral[pitch] == pytest.approx(0.003)
    assert data.ctrl[control.act[pitch]] == pytest.approx(control.target[pitch] + TRACKING_TRIM_GAIN * 0.003)
    control.tracking_integral[pitch] = TRACKING_TRIM_LIMIT / TRACKING_TRIM_GAIN
    control.apply()
    assert data.ctrl[control.act[pitch]] <= control.target[pitch] + TRACKING_TRIM_LIMIT


def test_support_detector_observes_lift_without_impacts():
    model = load_case("left")
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    data = mujoco.MjData(model)
    monitor = ContactMonitor(model, STRETCH)
    # A free-base vertical displacement is test setup, not an execution action.
    base_joint = model.body(STRETCH.base_body).jntadr[0]
    data.qpos[model.jnt_qposadr[base_joint] + 2] += 0.1
    mujoco.mj_forward(model, data)
    monitor.update(data)
    assert monitor.metrics()["wheel_lift"]
    assert min(monitor.metrics()["max_wheel_gap"]) > 0.09


def test_retention_boundary_is_not_relaxed():
    reference = (np.zeros(3), np.eye(3))
    theta = 0.10013630133085095
    current = (
        np.array([0.0052749, 0, 0]),
        np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]),
    )
    translation, rotation = retention_error(reference, current)
    assert translation < 0.02 and rotation > 0.1


def test_arm_floor_loading_physically_lifts_wheels_and_is_detected():
    # A small diagnostic mechanism tests the detector, not robot-model accuracy.
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".002"/>
      <worldbody><geom type="plane" size="2 2 .1"/>
        <body name="base" pos="0 0 .151"><freejoint/>
          <geom type="box" size=".12 .12 .04" mass="1"/>
          <body name="left" pos="0 .1 -.1"><geom type="sphere" size=".05" mass=".1"/></body>
          <body name="right" pos="0 -.1 -.1"><geom type="sphere" size=".05" mass=".1"/></body>
          <body name="foot" pos="0 0 .2"><joint name="arm" type="slide" axis="0 0 1" range="-.5 .1"/>
            <geom type="sphere" size=".03" mass=".1"/>
            <body name="tip"/>
          </body>
        </body>
      </worldbody>
      <actuator><position joint="arm" kp="1000" kv="30"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    profile = replace(STRETCH, base_body="base", wheel_bodies=("left", "right"), finger_bodies=("foot", "tip"))
    monitor = ContactMonitor(model, profile)
    for _ in range(500):
        mujoco.mj_step(model, data)
    monitor.update(data)
    assert not monitor.wheel_lift
    monitor.reset()
    data.ctrl[0] = -0.5
    for _ in range(500):
        mujoco.mj_step(model, data)
        monitor.update(data)
    assert monitor.wheel_lift
    assert monitor.peak_force > 0
    assert max(monitor.max_wheel_gap) > 0.01
