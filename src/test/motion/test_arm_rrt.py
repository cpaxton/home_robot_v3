# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for joint-space arm RRT-Connect (voxel collision + linear fallback)."""

from __future__ import annotations

import numpy as np

from emet.motion.algo import get_planner
from emet.motion.arm_rrt import joint_limits_from_model, plan_arm_joint_path, resolve_agent_manip_planner
from emet.motion.base import ConfigurationSpace
from emet.motion.voxel_arm_collision import link_samples_collide_2d


def test_resolve_agent_manip_planner_default():
    assert resolve_agent_manip_planner(config_mode=None) == "rrt_connect"
    assert resolve_agent_manip_planner(config_mode="linear") == "linear"
    assert resolve_agent_manip_planner(config_mode="bogus") == "rrt_connect"


def test_rrt_connect_2d_avoids_obstacle():
    """Classic 2D RRT-Connect: start/goal free, wall blocks straight line."""
    obs = np.zeros((20, 20), dtype=bool)
    # Vertical wall in the middle (cells x=9..10, y=0..19) — leave a gap at bottom
    obs[9:11, 2:] = True
    origin = np.array([0.0, 0.0])
    res = 0.1

    def validate(q: np.ndarray) -> bool:
        x, y = float(q[0]), float(q[1])
        if not (0.0 <= x <= 1.9 and 0.0 <= y <= 1.9):
            return False
        return not link_samples_collide_2d(obs, grid_origin=origin, resolution=res, sample_xy=[(x, y)], inflate_cells=0)

    space = ConfigurationSpace(2, np.array([0.0, 0.0]), np.array([1.9, 1.9]), step_size=0.15)
    start = np.array([0.2, 1.0])
    goal = np.array([1.7, 1.0])
    assert validate(start) and validate(goal)
    # Straight line through wall must be invalid
    mid = 0.5 * (start + goal)
    assert not validate(mid)

    np.random.seed(0)
    planner = get_planner("rrt_connect", space, validate, max_iter=800, goal_tolerance=0.12)
    result = planner.plan(start, goal, verbose=False)
    assert result.success, "RRT-Connect should find a path around the wall"
    assert result.trajectory is not None and len(result.trajectory) >= 2
    for node in result.trajectory:
        assert validate(node.state)


def test_plan_arm_joint_path_linear_explicit():
    """With planner=linear and no collision, returns interpolated path."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Use only first 2 arm joints for a tiny plan (faster / simpler)
    joints = RBY1_LEFT_ARM_JOINTS[:2]
    from emet.motion.mujoco_arm_ik import joint_qpos_addrs

    qadr = joint_qpos_addrs(model, joints)
    q0 = np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)
    q1 = q0 + np.array([0.2, -0.15])
    # Clamp into limits
    for i, name in enumerate(joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        q1[i] = float(np.clip(q1[i], lo, hi))

    plan = plan_arm_joint_path(
        model,
        data,
        joint_names=joints,
        q_start=q0,
        q_goal=q1,
        collision=None,
        planner="linear",
        linear_steps=5,
        linear_fallback=False,
    )
    assert plan.success
    assert plan.planner == "linear"
    assert len(plan.waypoints) == 6


def test_plan_arm_rrt_connect_no_collision():
    """RRT-Connect in joint space with no obstacle map should reach a nearby goal."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS, joint_qpos_addrs
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    joints = RBY1_LEFT_ARM_JOINTS[:3]
    qadr = joint_qpos_addrs(model, joints)
    for i, name in enumerate(joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        data.qpos[qadr[i]] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    q0 = np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)
    q1 = q0.copy()
    q1[0] = float(
        np.clip(
            q0[0] + 0.35,
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[0])][0],
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[0])][1],
        )
    )

    np.random.seed(1)
    plan = plan_arm_joint_path(
        model,
        data,
        joint_names=joints,
        q_start=q0,
        q_goal=q1,
        collision=None,
        planner="rrt_connect",
        max_iter=200,
        linear_fallback=True,
        shortcut=False,
    )
    assert plan.success, f"expected success, got {plan}"
    assert plan.planner in ("rrt_connect", "linear")
    assert len(plan.waypoints) >= 2


def test_tiny_measured_joint_stop_overshoot_can_return_inside_without_clipping():
    from types import SimpleNamespace

    import mujoco

    from emet.motion.arm_rrt import arm_config_violation, make_arm_validate_fn

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
        <joint name="slide" type="slide" range="0 .34"/><geom size=".01"/>
        </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    seen = []
    collision = SimpleNamespace(configuration_collides=lambda _m, d: seen.append(float(d.qpos[0])) or False)
    check = make_arm_validate_fn(model, data, ['slide'], collision)
    assert check(np.array([-5e-5]))
    assert seen[-1] == -5e-5
    assert arm_config_violation(model, data, ['slide'], np.array([-5e-5]), collision) is None
    assert not check(np.array([-2e-4]))
    assert arm_config_violation(model, data, ['slide'], np.array([-2e-4])).startswith('joint_bounds:')
    plan = plan_arm_joint_path(model, data, joint_names=['slide'], q_start=np.array([-5e-5]),
                               q_goal=np.array([.002]), collision=collision, planner='linear')
    assert plan.success
    assert plan.waypoints[0][0] == -5e-5
    assert np.isclose(plan.waypoints[-1][0], .002)


def test_invalid_start_distinguishes_bounds_from_collision():
    """invalid_start carries a detail naming which predicate failed."""
    from types import SimpleNamespace

    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS, joint_qpos_addrs
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    joints = RBY1_LEFT_ARM_JOINTS[:2]
    qadr = joint_qpos_addrs(model, joints)
    mujoco.mj_forward(model, data)
    q0 = np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)

    lo, _hi = joint_limits_from_model(model, joints)
    out_of_bounds = q0.copy()
    out_of_bounds[0] = lo[0] - 1.0
    plan = plan_arm_joint_path(
        model,
        data,
        joint_names=joints,
        q_start=out_of_bounds,
        q_goal=q0,
        collision=None,
        planner="linear",
    )
    assert not plan.success
    assert plan.reason == "invalid_start"
    assert plan.detail is not None and plan.detail.startswith("joint_bounds:"), plan.detail

    colliding = SimpleNamespace(configuration_collides=lambda _m, _d: True)
    plan = plan_arm_joint_path(
        model,
        data,
        joint_names=joints,
        q_start=q0,
        q_goal=q0 + 0.01,
        collision=colliding,
        planner="linear",
    )
    assert not plan.success
    assert plan.reason == "invalid_start"
    assert plan.detail == "collision", plan.detail


def test_voxel_checker_rejects_colliding_config():
    obs = np.zeros((10, 10), dtype=bool)
    obs[5, 5] = True
    assert link_samples_collide_2d(obs, grid_origin=np.array([0.0, 0.0]), resolution=0.1, sample_xy=[(0.55, 0.55)])
    assert not link_samples_collide_2d(obs, grid_origin=np.array([0.0, 0.0]), resolution=0.1, sample_xy=[(0.1, 0.1)])


def test_departure_search_samples_around_blocked_direct_exit():
    import mujoco

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0" range="-.1 .1"/>
      <joint name="y" type="slide" axis="0 1 0" range="-.1 .1"/>
      <geom size=".001" mass="1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)

    class Collision:
        def configuration_collides(self, model, data):
            x, y = data.qpos
            return x < .02 or (x < .025 and y < .015)

    class Departure:
        def sample_separating(self, rng):
            return np.array([0., .03])

        def validate_edge(self, start, end):
            for q in np.linspace(start, end, 30):
                if np.linalg.norm(q) > .06 or (q[0] > .005 and q[1] < .015):
                    return False
            return True

    start, goal = np.zeros(2), np.array([.04, .03])
    collision, departure = Collision(), Departure()
    ordinary = plan_arm_joint_path(model, data, joint_names=['x', 'y'], q_start=start,
                                   q_goal=goal, collision=collision)
    assert ordinary.reason == 'invalid_start'
    recovered = plan_arm_joint_path(model, data, joint_names=['x', 'y'], q_start=start,
                                   q_goal=goal, collision=collision, departure=departure, step_size=.005)
    assert recovered.success, recovered
    assert recovered.departure is departure
    np.testing.assert_allclose(recovered.waypoints[0], start)
    np.testing.assert_allclose(recovered.waypoints[-1], goal)
    assert max(q[1] for q in recovered.waypoints if q[0] <= .005) >= .015


def test_collision_departure_cannot_bypass_joint_limits_or_planning_deadline():
    import time

    import mujoco

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" range="0 .1"/><geom size=".01" mass="1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    invalid = plan_arm_joint_path(model, data, joint_names=['x'], q_start=[.2], q_goal=[.05], departure=object())
    assert invalid.reason == 'invalid_start' and invalid.detail.startswith('joint_bounds')

    class SlowClearance:
        def configuration_collides(self, model, data):
            time.sleep(.002)
            return False

    limited = plan_arm_joint_path(model, data, joint_names=['x'], q_start=[0.], q_goal=[.05],
                                 collision=SlowClearance(), max_time_s=.0001)
    assert not limited.success and limited.reason == 'no_plan_within_budget'


def test_posture_detour_certifies_both_legs_and_exact_goal():
    import mujoco

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
      <joint name="y" type="slide" axis="0 1 0" limited="true" range="-2 2"/>
      <geom type="sphere" size=".01" mass="1"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)

    class Wall:
        def configuration_collides(self, model, data):
            return bool(abs(data.qpos[0] - .5) < .12 and abs(data.qpos[1]) < .2)

    checker = Wall()
    result = plan_arm_joint_path(
        model, data, joint_names=("x", "y"), q_start=np.array([0., 0.]), q_goal=np.array([1., 0.]),
        collision=checker, planner="linear", step_size=.025, via_configurations=([.5, .6],),
    )
    assert result.success and result.planner == "posture_detour"
    assert any(np.allclose(q, [.5, .6]) for q in result.waypoints)
    assert np.array_equal(result.waypoints[-1], [1., 0.])
    for first, second in zip(result.waypoints, result.waypoints[1:], strict=False):
        for t in np.linspace(0, 1, 10):
            data.qpos[:] = (1-t)*first + t*second
            assert not checker.configuration_collides(model, data)
    multi = plan_arm_joint_path(
        model, data, joint_names=("x", "y"), q_start=np.array([0., 0.]), q_goal=np.array([1., 0.]),
        collision=checker, planner="linear", step_size=.025,
        via_configurations=([ [.1, .5], [.9, .5] ],),
    )
    assert multi.success and multi.planner == "posture_detour"
    assert any(np.allclose(q, [.1, .5]) for q in multi.waypoints)
    assert any(np.allclose(q, [.9, .5]) for q in multi.waypoints)
    blocked = plan_arm_joint_path(
        model, data, joint_names=("x", "y"), q_start=np.array([0., 0.]), q_goal=np.array([1., 0.]),
        collision=checker, planner="linear", step_size=.025, via_configurations=([.5, 0.], [3., .6]),
    )
    assert not blocked.success


def test_escape_probes_vertical_slide_before_gradient_or_random_search():
    import mujoco
    import pytest

    from emet.motion.contact_departure import ContactDeparture, DepartureMeasurement

    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody><body>
      <joint name="wrist" type="hinge" range="-1 1"/>
      <joint name="lift" type="slide" axis="0 0 1" range="0 .1"/>
      <geom type="sphere" size=".01" mass="1"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)

    def query(q):
        return DepartureMeasurement({'floor': float(q[1]-.005)}, np.array([[0., 0., q[1]]]))

    policy = ContactDeparture(np.zeros(2), query)
    # A long guide must not consume the search budget before simple escape.
    policy.guide_configurations = ((np.array([.2, .02]),),)
    policy.sample_separating = lambda rng: pytest.fail('Coordinate lift should solve this escape before gradient search')

    class Floor:
        def configuration_collides(self, model, data):
            return data.qpos[1] < .01

    result = plan_arm_joint_path(model, data, joint_names=('wrist', 'lift'), q_start=np.zeros(2),
        q_goal=np.array([.3, .03]), collision=Floor(), departure=policy, shortcut=False,
        rng=np.random.default_rng(0))
    assert result.success and result.planner == 'departure_rrt'
    prefix = result.waypoints[:result.waypoints.escape_end_index+1]
    assert all(q[0] == 0. for q in prefix)
    assert prefix[-1][1] >= .01


def test_explicit_linear_joint_limit_recovery_checks_actual_start_and_preserves_goal_limits():
    import mujoco

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
        <joint name="finger" type="slide" range="0 .04"/><geom size=".01"/>
        </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    options = {'joint_names': ('finger',), 'planner': 'linear', 'step_size': .001}
    start = np.array([.040105])
    assert not plan_arm_joint_path(model, data, q_start=start, q_goal=np.array([.012]), **options).success
    seen = []

    class Geometry:
        def configuration_collides(self, model, data):
            seen.append(float(data.qpos[0]))
            return False

    result = plan_arm_joint_path(model, data, q_start=start, q_goal=np.array([.012]),
        collision=Geometry(), start_bound_slack=.0005, **options)
    assert result.success
    assert seen[0] == start[0]  # Actual geometry, not a clipped start.
    assert result.waypoints[0][0] == start[0]
    assert np.all(np.diff(np.asarray(result.waypoints)[:, 0]) <= 0.)
    for begin, end in ((.041, .012), (.040105, .041)):
        assert not plan_arm_joint_path(model, data, q_start=np.array([begin]), q_goal=np.array([end]),
            start_bound_slack=.0005, **options).success

    class Blocked:
        def configuration_collides(self, model, data):
            return True

    assert not plan_arm_joint_path(model, data, q_start=start, q_goal=np.array([.012]),
        collision=Blocked(), start_bound_slack=.0005, **options).success
