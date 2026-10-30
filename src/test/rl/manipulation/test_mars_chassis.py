"""Physical Mars chassis regression: support, wheel drive and arm-induced tip."""

import mujoco
import numpy as np
import pytest

from emet.rl.manipulation.control import ManipulationControl
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.safety import ContactMonitor
from emet.robots.innate_mars.chassis import WHEEL_RADIUS, WHEEL_SEPARATION, source_geometry

pytestmark = pytest.mark.sim


@pytest.fixture
def chassis():
    model, _ = build_manipulation_model("innate_mars")
    model.body_pos[model.body("table").id] = [4, 4, 0.1]
    data = mujoco.MjData(model)
    control = ManipulationControl(model, data, MARS)
    control.initialize([0, -0.4, 0.2, 0.2, 0], MARS.gripper_open)
    for _ in range(1000):
        control.apply()
        mujoco.mj_step(model, data)
    return model, data, control, ContactMonitor(model, MARS)


def test_support_dimensions_and_mass_come_from_pinned_cad():
    shell, wheels, caster, density = source_geometry()
    assert sum(p.volume * density for p in (shell, *wheels, caster)) == pytest.approx(0.89)
    separation = wheels[0].bounds.mean(axis=0)[1] - wheels[1].bounds.mean(axis=0)[1]
    assert separation == pytest.approx(WHEEL_SEPARATION, abs=1e-6)
    for wheel in wheels:
        assert wheel.extents[2] / 2 == pytest.approx(WHEEL_RADIUS, abs=2e-5)
    assert shell.bounds[0, 2] > 0.016  # Chassis hull cannot support the robot at wheel height.


def test_resting_support_loads_balance_weight_without_false_lift(chassis):
    model, data, control, monitor = chassis
    mass = sum(model.body_mass[list(monitor.robot_bodies)])
    loads = []
    for _ in range(100):
        control.apply()
        mujoco.mj_step(model, data)
        evidence = monitor.update(data)
        assert not monitor.wheel_lift
        assert np.all(evidence["wheel_load"] > 0.05)
        loads.append(np.sum(evidence["wheel_load"]))
    # Instantaneous load includes vertical acceleration from contact settling;
    # time-averaged support must balance weight without hiding large oscillation.
    assert np.mean(loads) == pytest.approx(mass * 9.81, rel=0.01)
    assert max(abs(np.asarray(loads) - mass * 9.81)) < 0.1 * mass * 9.81
    assert monitor.metrics()["wheel_support_evaluated"]


def test_wheel_actuators_drive_chassis_through_ground_contact(chassis):
    model, data, control, monitor = chassis
    start = control.base_pose().copy()
    control.base_command[:] = [0.06, 0]
    for _ in range(500):
        control.apply()
        mujoco.mj_step(model, data)
        monitor.update(data)
    delta = control.base_pose() - start
    assert delta[0] > 0.025
    assert abs(delta[1]) < 0.02
    assert not monitor.wheel_lift
    assert np.isfinite(data.qpos).all()


def test_feedback_cannot_bypass_final_actuator_acceleration_limit(chassis):
    model, data, control, monitor = chassis
    control.base_command[:] = [0.1, 0.4]
    control.set_physical_wheel_slew(True)
    control.set_base_bias_compensation(True)
    previous = control.applied_base_velocity.copy()
    limits = np.array([MARS.base_acceleration, MARS.base_yaw_acceleration]) * model.opt.timestep
    for _ in range(100):
        mujoco.mj_step(model, data)
        control.apply()
        actual = control.applied_base_velocity.copy()
        assert np.all(abs(actual - previous) <= limits + 1e-10)
        # Repeated observe/apply calls at one physics instant cannot integrate
        # another ramp step or command a faster motor target.
        commands = data.ctrl[control.wheel_act].copy()
        control.apply()
        np.testing.assert_array_equal(data.ctrl[control.wheel_act], commands)
        previous = actual
    assert np.linalg.norm(previous) > 0
    control.stop()
    np.testing.assert_array_equal(control.applied_base_velocity, 0)
    np.testing.assert_array_equal(data.ctrl[control.wheel_act], 0)


def test_hand_load_can_physically_unload_and_lift_caster(chassis):
    model, data, control, monitor = chassis
    monitor.update(data)
    assert not monitor.wheel_lift
    # A force is applied to the real arm body; no base pose or support load is injected.
    data.xfrc_applied[control.ee, 2] = -20
    for _ in range(100):
        control.apply()
        mujoco.mj_step(model, data)
        monitor.update(data)
        if monitor.wheel_lift:
            break
    assert monitor.wheel_lift
    assert monitor.max_wheel_gap[2] > 0.0001
    assert monitor.min_wheel_load[2] < 0.01


def test_caster_swivels_passively_while_driven_wheels_turn(chassis):
    model, data, control, monitor = chassis
    start = control.base_pose().copy()
    swivel = model.joint("mars_caster_swivel").qposadr[0]
    initial_swivel = float(data.qpos[swivel])
    control.base_command[:] = [0.04, 0.35]
    for _ in range(500):
        control.apply()
        mujoco.mj_step(model, data)
        monitor.update(data)
    assert control.base_pose()[2] - start[2] > 0.15
    assert abs(data.qpos[swivel] - initial_swivel) > 0.02
    assert not monitor.wheel_lift
