# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Real FK/collision regressions for bounded release refinement."""

from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult, PhysicalPickPlaceExecutor
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker
from emet.motion.release import (
    ReleasedFallChecker,
    ReleaseLimits,
    body_bounds,
    plan_release,
    refine_release,
    wait_native_open,
)


def rig(extra=""):
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
      <body name="base_link" pos="0 0 .025">
        <joint name="height" type="slide" axis="0 0 1" range="0 .2"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <body name="rubber_tip_left" pos="-.03 0 0">
          <joint name="left" type="slide" axis="-.7071 0 -.7071" range="0 .02"/>
          <geom type="sphere" size=".012" mass=".1"/>
        </body>
        <body name="rubber_tip_right" pos=".03 0 0">
          <joint name="right" type="slide" axis=".7071 0 -.7071" range="0 .02"/>
          <geom type="sphere" size=".012" mass=".1"/>
        </body>
      </body>
      <body name="payload" pos="0 0 .025"><freejoint/>
        <geom type="box" size=".02 .02 .02" mass=".1"/>
      </body>
      <body name="table" pos="0 0 -.05"><geom type="box" size=".3 .3 .05"/></body>
      """
        + extra
        + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    checker = MujocoSceneCollisionChecker(
        model, robot_body="base_link", allowed_pairs=[("payload", n) for n in ("rubber_tip_left", "rubber_tip_right")]
    )
    checker.set_payload(model, data, "payload", "base_link")
    ex = PhysicalPickPlaceExecutor(
        SimpleNamespace(),
        model=model,
        data=data,
        ee_body="base_link",
        joint_names=("height",),
        collision=checker,
        synchronize=lambda _: None,
        command_joints=Mock(),
        input_mode="privileged",
        gripper_open_configuration={"left": 0.02, "right": 0.02},
        position_tolerance_m=0.0002,
    )
    ex.payload_body = "payload"
    point = data.body("base_link").xpos.copy()
    ex.place_targets = [(point, np.eye(3)), (point, np.eye(3)), (point + [0, 0, 0.03], np.eye(3))]
    return ex


def search(ex, drop=0.04):
    return plan_release(
        ex,
        support="table",
        region=np.array([[-0.1, -0.1], [0.1, 0.1]]),
        limits=ReleaseLimits(0.04, drop, height_joint="height"),
        initial_z=0.025,
    )


def test_opening_collision_finds_bounded_alternative_without_mutating_state():
    ex = rig()
    reference = ex.collision.payload_transform.copy()
    _, error = ex.plan_gripper_open()
    assert error == "gripper_open_path_failed:invalid_goal:collision"
    before = [a.copy() for a in (ex.data.qpos, ex.data.qvel, ex.data.qacc, ex.data.contact.dist)]
    plan, error = search(ex)
    assert error is None and plan is not None
    assert 0.025 < plan.target_z <= 0.035 and 0 < plan.gap_m <= 0.04
    assert len(plan.prefix) == 2
    for old, new in zip(before, (ex.data.qpos, ex.data.qvel, ex.data.qacc, ex.data.contact.dist), strict=True):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(reference, ex.collision.payload_transform)
    ex.command_joints.assert_not_called()


def test_drop_limit_rejects_all_clearance_alternatives_without_commands():
    ex = rig()
    plan, _ = search(ex, drop=0.006)
    assert plan is None
    result = refine_release(ex, support="table", region=None, limits=ReleaseLimits(0.04, 0.006, height_joint="height"))
    assert not result.success and result.phase == "release"
    ex.command_joints.assert_not_called()


def test_retreat_obstacle_is_checked_with_open_fingers():
    ex = rig('<body name="obstacle" pos="-.045 0 .05"><geom type="box" size=".008 .02 .008"/></body>')
    plan, error = search(ex)
    assert plan is None and "collision" in error
    ex.command_joints.assert_not_called()


def test_measured_tracking_reserve_selects_more_clearance_without_relaxing_limits():
    ex = rig()
    ordinary, error = search(ex)
    assert ordinary is not None and error is None
    reserved, error = plan_release(
        ex,
        support="table",
        region=None,
        limits=ReleaseLimits(0.04, 0.04, height_joint="height"),
        initial_z=0.025,
        tracking_reserve_m=0.008,
    )
    assert reserved is not None and error is None
    assert reserved.target_z > ordinary.target_z
    assert reserved.gap_m <= 0.04
    ex.command_joints.assert_not_called()


def test_nominal_opening_is_checked_as_well_as_lowered_reserve():
    ex = rig('<body name="overhang" pos="-.045 0 .069"><geom type="box" size=".003 .02 .0015"/></body>')
    for name in ("left", "right"):
        ex.model.jnt_range[ex.model.joint(name).id, 1] = 0.05
    ex.gripper_open_configuration = {"left": 0.045, "right": 0.045}
    ex.data.qpos[0] = 0.04
    addr = ex.model.jnt_qposadr[ex.model.body("payload").jntadr[0]]
    ex.data.qpos[addr + 2] = 0.065
    mujoco.mj_forward(ex.model, ex.data)
    ex.collision.set_payload(ex.model, ex.data, "payload", "base_link")
    point = ex.data.body(ex.ee_body).xpos.copy()
    ex.place_targets = [(point, np.eye(3)), (point, np.eye(3)), (point + [0, 0, 0.03], np.eye(3))]
    # The lowered probe alone clears this overhang. The actual opening arc
    # intersects it, so the measured pose must never be accepted as-is.
    plan, _ = plan_release(
        ex, support="table", region=None, limits=ReleaseLimits(0.04, 0.06, height_joint="height"), initial_z=0.065
    )
    assert plan is None or len(plan.prefix) > 1
    ex.command_joints.assert_not_called()


def test_tilted_cylinder_support_uses_shape_extent_and_still_rejects_deep_contact():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="robot" pos="2 0 0"><geom size=".01"/></body>
      <body name="payload"><freejoint/><geom type="cylinder" size=".02 .04" mass=".1"/></body>
      <body name="table" pos="0 0 -.05"><geom type="box" size=".3 .3 .05"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    angle = 0.12
    data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2) / np.sqrt(2), np.sin(angle / 2) / np.sqrt(2), 0]
    data.qpos[2] = 0.04 * np.cos(angle) + 0.02 * np.sin(angle) - 0.0002
    mujoco.mj_forward(model, data)
    gid, bid = model.body("payload").geomadr[0], model.body("payload").id
    exaggerated = data.geom_xpos[gid, 2] - np.abs(data.geom_xmat[gid].reshape(3, 3)[2]) @ model.geom_aabb[gid, 3:]
    assert exaggerated < -0.001
    bounds = body_bounds(model, data, {bid})
    np.testing.assert_allclose(bounds[0, 2], -0.0002, atol=1e-12)
    base = MujocoSceneCollisionChecker(model, robot_body="robot")
    checker = ReleasedFallChecker(
        model, base, "payload", "table", np.array([[-0.3, -0.3, 0], [0.3, 0.3, 0]]), None, ReleaseLimits(0.04, 0.04)
    )
    before = data.qpos.copy()
    assert not checker.configuration_collides(model, data)
    assert frozenset(("payload", "table")) not in checker.allowed_pairs
    np.testing.assert_array_equal(before, data.qpos)
    data.qpos[2] -= 0.002
    assert checker.configuration_collides(model, data)


def test_observed_input_never_uses_privileged_release_geometry(monkeypatch):
    ex = rig()
    ex.input_mode = "observed"
    monkeypatch.setattr("emet.motion.release.placements_from_mujoco_model", Mock(side_effect=AssertionError))
    plan, error = search(ex)
    assert plan is None and error == "native_execution_unavailable"


def test_falling_payload_volume_rejects_an_obstacle_below_it():
    ex = rig('<body name="peg" pos="0 0 .01"><geom type="box" size=".005 .005 .01"/></body>')
    # Hold the payload well clear of the peg at the initial position. Its
    # future descent into the peg must still be rejected during retreat.
    addr = ex.model.body("payload").jntadr[0]
    ex.data.qpos[ex.model.jnt_qposadr[addr] + 2] = 0.05
    for name, value in [("height", 0.04), ("left", 0.02), ("right", 0.02)]:
        ex.data.qpos[ex.model.joint(name).qposadr[0]] = value
    mujoco.mj_kinematics(ex.model, ex.data)
    checker = ReleasedFallChecker(
        ex.model,
        ex.collision,
        "payload",
        "table",
        np.array([[-0.3, -0.3, 0], [0.3, 0.3, 0]]),
        None,
        ReleaseLimits(0.04, 0.04),
    )
    before = ex.data.qpos.copy()
    assert checker.configuration_collides(ex.model, ex.data)
    assert any("peg" in c["bodies"] for c in checker.last_contacts)
    np.testing.assert_array_equal(before, ex.data.qpos)


@pytest.mark.parametrize("lingering_contact", (False, True))
def test_open_ack_requires_measured_joint_arrival_and_contact_release(monkeypatch, lingering_contact):
    ex = rig()
    ex.use_sim_time = True
    ex.released_collision = SimpleNamespace(obj="payload", support="table", configuration_collides=lambda *args: False)
    ticks = [0]

    def synchronize(data):
        ticks[0] += 1
        data.time = ticks[0] * 0.05
        if ticks[0] >= 4:
            data.qpos[ex.model.joint("left").qposadr[0]] = 0.02
            data.qpos[ex.model.joint("right").qposadr[0]] = 0.02
            data.qpos[ex.model.jnt_qposadr[ex.model.body("payload").jntadr[0]] + 2] = 0.0199
        mujoco.mj_kinematics(ex.model, data)

    ex.synchronize = synchronize
    monkeypatch.setattr("emet.motion.release.time.sleep", lambda _: None)
    if lingering_contact:
        # Even arrived opening joints cannot certify release if a finger is
        # still contacting the payload (modelled by this retained contact row).
        monkeypatch.setattr("emet.motion.release.mujoco.mj_collision", lambda model, data: None)
        # A new MjData's contact buffer is empty; copy the actual closed contact
        # buffer into it, rather than inventing a controller ACK as evidence.
        copy_data = mujoco.mj_copyData

        def copy_contacts(target, model, source):
            copy_data(target, model, source)
            assert target.ncon > 0

        monkeypatch.setattr("emet.motion.release.mujoco.mj_copyData", copy_contacts)
    result = wait_native_open(ex)
    assert result.success is not lingering_contact
    assert ticks[0] >= 4
    ex.command_joints.assert_not_called()


def test_height_joint_bounds_remain_distinct_from_collisions():
    ex = rig()
    ex.data.qpos[ex.model.joint("left").qposadr[0]] = -0.005
    # Existing opening-path validation retains the joint-bounds code. Search
    # must not clip the measured finger into the valid interval.
    plan, error = search(ex)
    assert plan is None and "joint_bounds" in error
    assert ex.data.qpos[ex.model.joint("left").qposadr[0]] == -0.005


@pytest.mark.parametrize("failure_at", ("refinement", "opening_verification"))
def test_shared_place_executor_stops_before_next_command_on_release_failure(failure_at):
    ex = rig()
    ex.robot.open_gripper = Mock(return_value=True)
    ex.place_paths = [[np.zeros(1)]] * 3
    ex.transport = lambda: PhysicalMotionResult(True, "ok")
    ex._execute_path = Mock(return_value=PhysicalMotionResult(True, "ok"))
    ex._replan_at_measured_pose = Mock(return_value=([[np.zeros(1)]] * 3, None))
    failed = PhysicalMotionResult(False, "no_plan_within_budget", "release")
    if failure_at == "refinement":
        ex.release_refiner = lambda: failed
    else:
        ex.release_verifier = lambda: failed
    assert ex.place_only("table", object_gt_body="payload") is failed
    assert [c.args[0] for c in ex._execute_path.call_args_list] == ["preplace", "place"]
    assert ex.robot.open_gripper.call_count == (failure_at == "opening_verification")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_drop_m": float("nan")},
        {"max_raise_m": -1},
        {"max_adjustments": True},
        {"offsets_m": (0.003, 0.003)},
        {"offsets_m": ()},
    ],
)
def test_invalid_limits_rejected(kwargs):
    values = {"max_raise_m": 0.04, "max_drop_m": 0.04}
    values.update(kwargs)
    with pytest.raises(ValueError, match="invalid_release_limits"):
        ReleaseLimits(**values)
