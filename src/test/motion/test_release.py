# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Real FK/collision regressions for bounded release refinement."""

from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult, PhysicalPickPlaceExecutor
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker, body_subtree
from emet.motion.release import (
    ReleasedFallChecker,
    ReleaseLimits,
    body_bounds,
    held_checker,
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


def rig_xy(extra=""):
    """Same release rig with an added lateral slide so the arm can shift in x."""
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
      <body name="base_link" pos="0 0 .025">
        <joint name="height" type="slide" axis="0 0 1" range="0 .2"/>
        <joint name="lateral" type="slide" axis="1 0 0" range="-.1 .1"/>
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
        joint_names=("height", "lateral"),
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


def test_opening_collision_finds_bounded_alternative_without_mutating_state():
    ex = rig()
    reference = ex.collision.payload_transform.copy()
    robot_ids, allowed_pairs, geom_pairs = (set(ex.collision.robot_ids), set(ex.collision.allowed_pairs),
                                            set(ex.collision.allowed_geom_pairs))
    _, error = ex.plan_gripper_open()
    assert error == "gripper_open_path_failed:invalid_goal:collision"
    before = [a.copy() for a in (ex.data.qpos, ex.data.qvel, ex.data.qacc, ex.data.contact.dist)]
    plan, error = search(ex)
    assert error is None and plan is not None
    assert 0.025 < plan.target_z <= 0.045 and 0 < plan.gap_m <= 0.04
    assert len(plan.prefix) == 2
    for old, new in zip(before, (ex.data.qpos, ex.data.qvel, ex.data.qacc, ex.data.contact.dist), strict=True):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(reference, ex.collision.payload_transform)
    assert (ex.collision.robot_ids, ex.collision.allowed_pairs, ex.collision.allowed_geom_pairs) == (robot_ids, allowed_pairs, geom_pairs)
    ex.command_joints.assert_not_called()


def test_drop_limit_rejects_all_clearance_alternatives_without_commands():
    ex = rig()
    plan, _ = search(ex, drop=0.006)
    assert plan is None
    result = refine_release(ex, support="table", region=None, limits=ReleaseLimits(0.04, 0.006, height_joint="height"))
    assert not result.success and result.phase == "release"
    ex.command_joints.assert_not_called()


@pytest.mark.parametrize('block_release', [False, True])
def test_additional_map_guard_survives_held_opening_and_released_fall_clones(block_release):
    ex = rig()
    calls = []
    original = ex.collision
    def guard(owner, model, data):
        assert owner is not original
        assert model is ex.model and data is not ex.data
        calls.append(type(owner).__name__)
        return block_release and isinstance(owner, ReleasedFallChecker)
    original.additional_scene_checks = (guard,)
    plan, error = search(ex)
    assert 'HeldPoseChecker' in calls and 'ReleasedFallChecker' in calls
    if block_release:
        assert plan is None and 'collision' in error
    else:
        assert plan is not None and error is None
    ex.command_joints.assert_not_called()


def test_retreat_obstacle_is_checked_with_open_fingers():
    ex = rig('<body name="obstacle" pos="-.045 0 .05"><geom type="box" size=".008 .02 .008"/></body>')
    ids = set(ex.collision.robot_ids)
    plan, error = search(ex)
    assert plan is None and "collision" in error
    assert ex.collision.robot_ids == ids
    ex.command_joints.assert_not_called()


def test_held_checker_has_independent_exclusions_and_nested_departure_state():
    ex = rig()
    ex.collision._support_departure = {frozenset((0, 1)): {'height_m': 1.}}
    clone = held_checker(ex.collision, ex.collision.payload_transform)
    clone.robot_ids.clear()
    clone.allowed_pairs.clear()
    clone.allowed_geom_pairs.add(frozenset((1, 2)))
    clone._support_departure[frozenset((0, 1))]['height_m'] = 2.
    assert ex.collision.robot_ids and ex.collision.allowed_pairs
    assert not ex.collision.allowed_geom_pairs
    assert ex.collision._support_departure[frozenset((0, 1))]['height_m'] == 1.


@pytest.mark.parametrize('anchor,reserve', [(np.nan, 0.), (np.inf, 0.), (.025, np.nan), (.025, -.001)])
def test_release_numeric_limits_cannot_be_disabled_by_bad_anchor_or_reserve(anchor, reserve):
    ex = rig()
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(.04, .04, height_joint='height'), initial_z=anchor, tracking_reserve_m=reserve)
    assert plan is None and error == 'native_execution_unavailable'


def test_unrelated_vertical_slide_cannot_certify_ee_lowered_clearance():
    ex = rig('<body name="other" pos="1 0 0"><joint name="other_height" type="slide" axis="0 0 1"/><geom size=".01"/></body>')
    ex.joint_names = (*ex.joint_names, 'other_height')
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(.04, .04, height_joint='other_height'), initial_z=.025)
    assert plan is None and error == 'native_execution_unavailable'


def test_ik_tolerance_cannot_overshoot_actual_raise_limit(monkeypatch):
    ex = rig()
    def overshoot(self, point, rotation):
        start = self.data.qpos[self.qadr].copy()
        self.data.qpos[self.qadr] = point[2]-.025+.002
        mujoco.mj_kinematics(self.model, self.data)
        return [start, self.data.qpos[self.qadr].copy()], None
    monkeypatch.setattr(PhysicalPickPlaceExecutor, 'plan_pose', overshoot)
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(.01, .04, offsets_m=(.01,), height_joint='height'), initial_z=.025)
    assert plan is None and error == 'release_raise_limit'
    ex.command_joints.assert_not_called()


def test_last_adjustment_still_requires_fresh_current_only_verification(monkeypatch):
    ex = rig()
    ex.payload_retained = Mock(return_value=True)
    ex._execute_path = Mock(return_value=PhysicalMotionResult(True, 'ok'))
    plans = [SimpleNamespace(prefix=[np.array([0.]), np.array([value])], checker=ex.collision,
        released_checker=object(), target_z=.025+value, gap_m=.01,
        target_point=np.array([0., 0., .025+value])) for value in (.01, .02)]
    plans.append(SimpleNamespace(prefix=[np.array([.02])], checker=ex.collision,
        released_checker=object(), target_z=.045, gap_m=.01, target_point=np.array([0., 0., .045])))
    planner = Mock(side_effect=[(plan, None) for plan in plans])
    monkeypatch.setattr('emet.motion.release.plan_release', planner)
    result = refine_release(ex, support='table', region=None,
        limits=ReleaseLimits(.04, .04, height_joint='height', max_adjustments=2))
    assert result.success and ex._execute_path.call_count == 2
    assert [call.kwargs['allow_adjustment'] for call in planner.call_args_list] == [True, True, False]
    assert all(call.kwargs['initial_z'] == .025 for call in planner.call_args_list)


@pytest.mark.parametrize('fault', ['targets', 'support', 'height', 'payload', 'parent', 'empty_open'])
def test_release_missing_preconditions_reject_without_commands(fault):
    ex = rig()
    support, limits = 'table', ReleaseLimits(.04, .04, height_joint='height')
    if fault == 'targets':
        ex.place_targets = []
    elif fault == 'support':
        support = 'missing'
    elif fault == 'height':
        limits = ReleaseLimits(.04, .04, height_joint='missing')
    elif fault == 'payload':
        ex.collision.payload_body = 'another'
    elif fault == 'parent':
        ex.collision.payload_parent = 'table'
    else:
        ex.gripper_open_configuration = {}
    plan, error = plan_release(ex, support=support, region=None, limits=limits, initial_z=.025)
    assert plan is None and error == 'native_execution_unavailable'
    ex.command_joints.assert_not_called()


def test_zero_adjustment_budget_does_not_search_unexecutable_raise(monkeypatch):
    ex = rig()
    ex.plan_pose = Mock(side_effect=AssertionError('No alternative IK permitted'))
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(.04, .04, height_joint='height', max_adjustments=0),
        initial_z=.025, allow_adjustment=False)
    assert plan is None and error.startswith('gripper_open_path_failed')
    ex.plan_pose.assert_not_called()


def test_missing_release_checker_or_opening_geometry_is_typed_failure():
    ex = rig()
    assert wait_native_open(ex).message == 'native_execution_unavailable'
    ex.released_collision = object()
    ex.gripper_open_configuration = {}
    assert wait_native_open(ex).message == 'native_execution_unavailable'


@pytest.mark.parametrize('depth,rejected', [(.0009, False), (.0011, True)])
def test_resting_support_keeps_the_unchanged_one_mm_penetration_gate(depth, rejected):
    ex = rig()
    for name, value in [('height', .05), ('left', .02), ('right', .02)]:
        ex.data.qpos[ex.model.joint(name).qposadr[0]] = value
    address = int(ex.model.jnt_qposadr[ex.model.body('payload').jntadr[0]])
    ex.data.qpos[address+2] = .02-depth
    mujoco.mj_kinematics(ex.model, ex.data)
    checker = ReleasedFallChecker(ex.model, ex.collision, 'payload', 'table',
        np.array([[-.3, -.3, 0.], [.3, .3, 0.]]), None, ReleaseLimits(.04, .04))
    assert checker.configuration_collides(ex.model, ex.data) is rejected
    if rejected:
        assert checker.last_contacts[0]['reason'] == 'support_penetration'


def test_measured_tracking_reserve_preserves_both_delivery_limits():
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
    # Candidate ranking need not be monotone on the shifted discrete grid.
    # The complete delivered interval must still obey both support limits.
    assert reserved.gap_m - (.008 + .003) >= -.001
    assert reserved.gap_m <= 0.04
    ex.command_joints.assert_not_called()


def test_large_measured_shortfall_is_included_in_candidate_height_without_widening_limits():
    ex = rig()
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(.04, .04, height_joint='height'), initial_z=.025, tracking_reserve_m=.019)
    assert plan is not None, error
    assert .025+.02 < plan.target_z <= .025+.04
    assert plan.gap_m <= .04
    ex.command_joints.assert_not_called()


@pytest.mark.parametrize('maximum_raise,accepted', [(.04, True), (.01, False)])
def test_release_raises_from_delivered_pose_above_nominal_without_ratchet(maximum_raise, accepted):
    ex = rig('<body name="opening_obstacle" pos="-.045 0 .06"><geom type="box" size=".002 .01 .002"/></body>')
    ex.data.qpos[ex.model.joint('height').qposadr[0]] = .05
    address = int(ex.model.jnt_qposadr[ex.model.body('payload').jntadr[0]])
    ex.data.qpos[address+2] = .075
    mujoco.mj_kinematics(ex.model, ex.data)
    ex.collision.set_payload(ex.model, ex.data, 'payload', ex.ee_body)
    # Original nominal release was .025: all its old +offsets are below the
    # delivered pose. A correction may only rise vertically from measured .075.
    before = ex.data.qpos.copy()
    ex.place_targets[2] = (np.array([0., 0., .125]), np.eye(3))
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(maximum_raise, .08, height_joint='height'), initial_z=.075)
    assert (plan is not None) is accepted, error
    if plan is not None:
        assert .075 < plan.target_z <= .075+maximum_raise
    np.testing.assert_array_equal(ex.data.qpos, before)
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
@pytest.mark.parametrize("initial_clock", (0.0, 20.0))
def test_open_ack_requires_measured_joint_arrival_and_contact_release(monkeypatch, lingering_contact, initial_clock):
    ex = rig()
    ex.use_sim_time = True
    ex.released_collision = SimpleNamespace(
        obj="payload", support="table", limits=ReleaseLimits(0.04, 0.04), configuration_collides=lambda *args: False
    )
    ticks = [0]

    def synchronize(data):
        ticks[0] += 1
        data.time = initial_clock + ticks[0] * 0.05
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
        {"lateral_directions": ()},
    ],
)
def test_invalid_limits_rejected(kwargs):
    values = {"max_raise_m": 0.04, "max_drop_m": 0.04}
    values.update(kwargs)
    with pytest.raises(ValueError, match="invalid_release_limits"):
        ReleaseLimits(**values)


@pytest.mark.parametrize("timeout,success", [(3.0, False), (12.0, True)])
def test_open_support_wait_keeps_motion_limits_until_late_settling(monkeypatch, timeout, success):
    ex = rig()
    ex.use_sim_time = True
    ex.released_collision = SimpleNamespace(
        obj="payload",
        support="table",
        limits=ReleaseLimits(0.04, 0.04, open_settle_timeout_s=timeout),
        configuration_collides=lambda *args: False,
    )
    dof = int(ex.model.jnt_dofadr[ex.model.body("payload").jntadr[0]])
    addr = int(ex.model.jnt_qposadr[ex.model.body("payload").jntadr[0]])

    def synchronize(data):
        data.time += 0.05
        for name in ("left", "right"):
            data.qpos[ex.model.joint(name).qposadr[0]] = 0.02
        data.qpos[addr + 2] = 0.0199
        data.qvel[dof] = 0.02 if data.time < 5.8 else 0.0
        mujoco.mj_kinematics(ex.model, data)

    ex.synchronize = synchronize
    monkeypatch.setattr("emet.motion.release.time.sleep", lambda _: None)
    result = wait_native_open(ex)
    assert result.success is success
    if success:
        assert ex.data.time >= 5.9
    ex.command_joints.assert_not_called()


def test_release_candidate_diagnostics_identify_opening_and_acceptance_without_dispatch():
    ex = rig()
    ex.event = Mock()
    plan, error = search(ex)
    assert plan is not None and error is None
    rows = [call.kwargs for call in ex.event.call_args_list if call.kwargs.get('phase') == 'release_candidate']
    assert any(row['stage'] == 'opening' and row['reason'].startswith('gripper_open_path_failed') for row in rows)
    assert rows[-1]['stage'] == 'accepted'
    ex.command_joints.assert_not_called()


def test_release_bounds_rejection_records_geometry_without_dispatch():
    ex = rig()
    ex.event = Mock()
    plan, error = plan_release(ex, support='table', region=np.array([[.1,.1],[.2,.2]]),
        limits=ReleaseLimits(.04,.04,height_joint='height'),initial_z=.025)
    assert plan is None
    rows = [call.kwargs for call in ex.event.call_args_list if call.kwargs.get('stage') == 'support_bounds']
    assert rows and np.asarray(rows[0]['bounds']).shape == (2,3)
    assert rows[0]['region'] == [[.1,.1],[.2,.2]]
    ex.command_joints.assert_not_called()


def test_adjustment_prefers_support_slack_over_first_barely_feasible_pose():
    ex = rig()
    ex.event = Mock()
    plan, error = search(ex)
    assert plan is not None and error is None
    rows = [call.kwargs for call in ex.event.call_args_list]
    feasible = [row for row in rows if row.get('stage') == 'feasible']
    selected = [row for row in rows if row.get('stage') == 'accepted'][-1]
    assert len(feasible) >= 2
    assert selected['support_slack_m'] == max(row['support_slack_m'] for row in feasible)
    assert selected['support_slack_m'] > feasible[0]['support_slack_m']
    assert plan.target_z > feasible[0]['target_z']
    assert plan.target_z <= .025 + .04 and plan.gap_m <= .04
    ex.command_joints.assert_not_called()


def test_lateral_fallback_shifts_destination_when_vertical_opening_stays_blocked():
    ex = rig_xy('<body name="wall" pos=".05 0 .075"><geom type="box" size=".005 .1 .075"/></body>')
    reference = ex.collision.payload_transform.copy()
    # A tall wall to +x blocks the whole-hand opening at the measured pose and
    # at every vertical correction, but not a lateral shift away from it.
    plan, error = plan_release(ex, support="table", region=None,
        limits=ReleaseLimits(.04, .04, height_joint="height"), initial_z=.025)
    assert plan is not None, error
    assert plan.target_point[0] < -0.005
    assert abs(plan.target_point[1]) < 1e-9
    np.testing.assert_array_equal(reference, ex.collision.payload_transform)
    ex.command_joints.assert_not_called()


def test_disabled_lateral_fallback_preserves_vertical_only_verdict():
    ex = rig_xy('<body name="wall" pos=".05 0 .075"><geom type="box" size=".005 .1 .075"/></body>')
    limits = ReleaseLimits(.04, .04, height_joint="height", lateral_offsets_m=(), lateral_directions=())
    plan, error = plan_release(ex, support="table", region=None, limits=limits, initial_z=.025)
    assert plan is None and "collision" in error
    ex.command_joints.assert_not_called()


def test_lateral_candidates_remain_inside_the_declared_region():
    ex = rig_xy('<body name="wall" pos=".05 0 .075"><geom type="box" size=".005 .1 .075"/></body>')
    region = np.array([[-0.06, -0.02], [0.03, 0.02]])
    # The only clear hand-opening destination is to -x; it must stay inside the
    # requested region rather than escaping through an unreachable/outside pose.
    plan, error = plan_release(ex, support="table", region=region,
        limits=ReleaseLimits(.04, .04, height_joint="height"), initial_z=.025)
    assert plan is not None, error
    probe = mujoco.MjData(ex.model)
    mujoco.mj_copyData(probe, ex.model, ex.data)
    probe.qpos[ex.qadr] = plan.prefix[-1]
    assert not plan.checker.configuration_collides(ex.model, probe)
    bounds = body_bounds(ex.model, probe, body_subtree(ex.model, 'payload'))
    assert np.all(bounds[0, :2] >= region[0] - .005)
    assert np.all(bounds[1, :2] <= region[1] + .005)
    ex.command_joints.assert_not_called()


def test_no_lateral_alternative_may_escape_a_tight_goal_region():
    ex = rig_xy('<body name="wall" pos=".05 0 .075"><geom type="box" size=".005 .1 .075"/></body>')
    plan, error = plan_release(ex, support='table', region=np.array([[-.022, -.022], [.022, .022]]),
        limits=ReleaseLimits(.04, .04, height_joint='height'), initial_z=.025)
    assert plan is None and 'collision' in error
    ex.command_joints.assert_not_called()


def test_lateral_limit_remains_anchored_to_original_release_position():
    ex = rig_xy('<body name="wall" pos="-.05 0 .075"><geom type="box" size=".005 .1 .075"/></body>')
    events = []
    ex.event = lambda **row: events.append(row)
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(.04, .04, height_joint='height'), initial_z=.025, initial_xy=[-.03, 0])
    assert plan is None and 'collision' in error
    assert any(r.get('stage') == 'lateral_limit' for r in events)
    ex.command_joints.assert_not_called()


def test_successful_lateral_adjustment_shifts_the_executed_retreat(monkeypatch):
    ex = rig_xy()
    before = [p.copy() for p, _ in ex.place_targets]
    ex.payload_retained = Mock(return_value=True)
    ex._execute_path = Mock(return_value=PhysicalMotionResult(True, 'ok'))
    adjusted = SimpleNamespace(prefix=[np.zeros(2), np.array([.01, -.02])], checker=ex.collision,
        released_checker=object(), target_z=.035, target_point=np.array([-.02, 0, .035]), lateral=(-.02, 0.))
    verified = SimpleNamespace(prefix=[np.zeros(2)], checker=ex.collision,
        released_checker=object(), target_z=.035, target_point=np.array([-.02, 0, .035]))
    planner = Mock(side_effect=[(adjusted, None), (verified, None)])
    monkeypatch.setattr('emet.motion.release.plan_release', planner)
    result = refine_release(ex, support='table', region=None, limits=ReleaseLimits(.04, .04, height_joint='height'))
    assert result.success
    ex._execute_path.assert_called_once()
    np.testing.assert_allclose(ex.release_lateral_shift, [-.02, 0])
    for (point, _), old in zip(ex.place_targets, before, strict=True):
        np.testing.assert_allclose(point, old + [-.02, 0, 0])
    np.testing.assert_array_equal(planner.call_args_list[1].kwargs['initial_xy'], [0, 0])


@pytest.mark.parametrize('unknown_lateral_fall', [False, True])
def test_lateral_fallback_keeps_map_guards_on_the_released_object(unknown_lateral_fall):
    ex = rig_xy('<body name="wall" pos=".05 0 .075"><geom type="box" size=".005 .1 .075"/></body>')
    before = ex.data.qpos.copy()
    reference = ex.collision.payload_transform.copy()
    fall_queries = []

    def map_guard(owner, model, data):
        assert data is not ex.data
        if isinstance(owner, ReleasedFallChecker) and data.qpos[1] < -.005:
            fall_queries.append(float(data.qpos[1]))
            return unknown_lateral_fall
        return False

    ex.collision.additional_scene_checks = (map_guard,)
    plan, error = plan_release(ex, support='table', region=None,
        limits=ReleaseLimits(.04, .04, height_joint='height'), initial_z=.025)
    assert fall_queries
    assert (plan is None) is unknown_lateral_fall
    if unknown_lateral_fall:
        assert 'collision' in error
    np.testing.assert_array_equal(ex.data.qpos, before)
    np.testing.assert_array_equal(ex.collision.payload_transform, reference)
    ex.command_joints.assert_not_called()


def test_rejected_held_start_skips_all_release_adjustment_ik():
    ex = rig_xy()
    ex.collision.additional_scene_checks = (lambda *_: True,)
    ex.plan_pose = Mock(side_effect=AssertionError("invalid held start cannot reach adjustment IK"))
    plan, error = plan_release(
        ex, support="table", region=None, limits=ReleaseLimits(.04, .04, height_joint="height"), initial_z=.025
    )
    assert plan is None
    assert error == "arm_path_failed:invalid_start:collision"
    ex.plan_pose.assert_not_called()
    ex.command_joints.assert_not_called()
