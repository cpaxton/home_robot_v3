"""Physics tests distinguish free tracking, blocked motion, and contact recovery."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker
from emet.simulation.teleport_collision import teleport_endpoint_contacts


def slider(wall):
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".002" gravity="0 0 0" integrator="implicitfast"/>
    <worldbody><body name="tool"><joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
    <geom type="sphere" size=".05" mass="1"/></body>
    {'<geom name="wall" type="box" pos=".35 0 0" size=".05 .3 .3"/>' if wall else ''}
    </worldbody><actuator><position joint="slide" kp="100" kv="20" forcerange="-20 20"/>
    </actuator></mujoco>''')


def advance(model, data, goal):
    data.ctrl[0] = goal
    force = 0.0
    for _ in range(1500):
        mujoco.mj_step(model, data)
        if data.ncon:
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, 0, contact_force)
            force = max(force, contact_force[0])
    mujoco.mj_forward(model, data)
    return force


@pytest.mark.parametrize('wall', [False, True])
def test_free_and_blocked_tracking_use_actual_arrival(wall):
    model = slider(wall)
    data = mujoco.MjData(model)
    contact_force = advance(model, data, .6)
    ex = object.__new__(KinematicPickPlaceExecutor)
    ex._data = data
    ex._model = model
    ex.ee_body = 'tool'
    ex.ik_tol_m = .035
    ex._sync_qpos_from_robot = lambda: True
    ex._joint_tracking_evidence = lambda: {}
    arrived, error = ex._wait_measured_ee(np.array([.6, 0, 0]), timeout_s=0)
    assert arrived == (not wall)
    if wall:
        assert contact_force > 1 and error > .3
        # A known obstacle must be rejected by the planner before actuation.
        checker = MujocoSceneCollisionChecker(model, robot_body='tool')
        plan = plan_arm_joint_path(model, data, joint_names=['slide'], q_start=[0], q_goal=[.6],
                                   collision=checker, planner='linear', linear_steps=2)
        assert not plan.success
        # Release from contact must restore tracking, without disabling contact.
        assert advance(model, data, 0) >= 0
        assert abs(data.qpos[0]) < .001 and data.ncon == 0
    else:
        assert contact_force == 0 and error < .001


def test_teleport_wall_rejection_preserves_all_live_state():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="wall" type="box" pos="1 0 0" size=".1 1 1"/>
    <body name="base_link"><freejoint/><geom type="sphere" size=".2" mass="1"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:] = .1
    mujoco.mj_forward(model, data)
    q, v, position = data.qpos.copy(), data.qvel.copy(), data.xpos.copy()
    spec = SimpleNamespace(base_link_name='base_link')
    assert teleport_endpoint_contacts(model, data, spec, [1, 0, 0])
    assert not teleport_endpoint_contacts(model, data, spec, [.4, 0, 0])
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(data.qvel, v)
    np.testing.assert_array_equal(data.xpos, position)


def test_contact_deviation_is_the_next_planning_start(monkeypatch):
    import emet.controller.manipulation.kinematic_pick_place as module
    model = slider(False)
    data = mujoco.MjData(model)
    ex = object.__new__(KinematicPickPlaceExecutor)
    ex._model, ex._data = model, data
    ex.joint_names, ex.ee_body = ('slide',), 'tool'
    ex._last_cmd_q = np.array([.8])
    ex.ik_tol_m, ex.ik_max_iters = .035, 150
    ex.manip_planner, ex.rrt_max_iter, ex.traj_steps = 'linear', 10, 15
    ex._collision = None
    def measured():
        data.qpos[0] = .2
        return True
    ex._sync_qpos_from_robot = measured
    def ik(*args, **kwargs):
        assert kwargs['tol_m'] <= ex.ik_tol_m * .25
        data.qpos[0] = .6
        return SimpleNamespace(success=True, pos_error_m=0)
    def plan(*args, **kwargs):
        np.testing.assert_allclose(kwargs['q_start'], [.2])
        return SimpleNamespace(success=False, planner='linear', reason='test_stop')
    monkeypatch.setattr(module, 'solve_position_ik_multiseed', ik)
    monkeypatch.setattr(module, 'plan_arm_joint_path', plan)
    assert not ex._plan_and_execute_ee(np.array([.6, 0, 0]))[0]
    ex._sync_qpos_from_robot = lambda: False
    assert not ex._plan_and_execute_ee(np.array([.6, 0, 0]))[0]
    assert ex._last_motion_failure == 'missing_joint_state'


def test_server_rejects_collision_before_teleport_or_attachment(monkeypatch):
    import threading

    from emet.simulation.robosuite_server import RobosuiteZmqServer
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="box" pos="1 0 0" size=".1 1 1"/>
    <body name="base_link"><freejoint/><geom type="sphere" size=".2" mass="1"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    before = data.qpos.copy()
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel, server._mjdata = model, data
    server._mj_lock = threading.RLock()
    server._spec = SimpleNamespace(base_link_name='base_link')
    server._initial_xyt = np.zeros(3)
    server._is_molmospaces_session = lambda: False
    server._resolve_nav_goal_world_xyt = lambda action, raw, init: (*raw, {})
    server._log_nav_action = lambda *a, **kw: None
    server._teleport_base_world_xyt = lambda *a: pytest.fail('mutated live pose before rejection')
    server._snap_kinematic_attachments = lambda: pytest.fail('moved payload before rejection')
    with pytest.raises(RuntimeError, match='teleport_endpoint_collision'):
        server.handle_action({'xyt': [1, 0, 0], 'nav_world': True, 'nav_teleport': True})
    assert not server._at_goal
    np.testing.assert_array_equal(data.qpos, before)


@pytest.mark.parametrize('x,blocked', [(0, False), (.7995, False), (.85, True)])
def test_touching_is_not_deep_wall_penetration(x, blocked):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="box" pos="1 0 0" size=".1 1 1"/>
    <body name="base_link"><joint name="x" type="slide" axis="1 0 0"/>
    <joint name="y" type="slide" axis="0 1 0"/><joint name="yaw" axis="0 0 1"/>
    <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    spec = SimpleNamespace(base_link_name='base_link', planar_base_joint_names=('x', 'y', 'yaw'))
    assert bool(teleport_endpoint_contacts(model, data, spec, [x, 0, 0])) == blocked
    np.testing.assert_array_equal(data.qpos, [0, 0, 0])


@pytest.mark.parametrize('support', ['fixed', 'server_weld'])
@pytest.mark.parametrize('target', [
    [.32505, 1.86830, -1.52349, 2.00108, 2.83443, .02506, -.46815, .07608, 1.65766, 0],
    [.45296, 1.85549, -1.38002, 2.09136, 2.80404, .05577, -.68676, .12459, 1.65070, 0],
    [.15808, 1.88262, -1.67427, 1.66889, 2.86444, .00779, -.32871, .01386, 1.65810, 0],
])
def test_rby1_fixed_base_tracks_recorded_targets_without_contact(target, support):
    """r7 potato/apple/kettle targets: isolate arm dynamics from the base-hold mechanism.

    This fixture removes the free joint, not gravity, limits or actuator forces.
    It is a controller regression, not evidence that the furnished approach is valid.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from emet.motion.arm_manip_profile import ArmManipProfile
    from emet.robots import get_robot_spec
    path = Path(get_robot_spec('rby1').mjcf_path)
    root = ET.parse(path).getroot()
    root.find('compiler').set('assetdir', str(path.parent / 'meshes'))
    body = root.find(".//body[@name='base_link']")
    if support == 'fixed':
        body.remove(body.find('freejoint'))
        root.remove(root.find('equality'))
    root.remove(root.find('keyframe'))  # its qpos layout includes the removed free joint
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    profile = ArmManipProfile.for_robot('rby1')
    for name, value in zip(profile.actuator_names, profile.home_cmd, strict=True):
        aid = model.actuator(name).id
        data.ctrl[aid] = value
        data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]] = value
    qa = [int(model.joint(n).qposadr[0]) for n in profile.joint_names]
    aids = [model.actuator(n.replace('_joint', '')).id for n in profile.joint_names]
    start = data.qpos[qa].copy()
    if support == 'server_weld':
        from emet.simulation.robosuite_server import RobosuiteZmqServer
        server = object.__new__(RobosuiteZmqServer)
        server._mjmodel, server._mjdata = model, data
        server._spec = get_robot_spec('rby1')
        server._nav_goal_world = None
        server._stationary_base_freejoint_qpos = data.qpos[:7].copy()
        server._passive_base_support = False
        server._base_freejoint_addrs = lambda: (0, 0)
    target = np.asarray(target)
    for step in range(4000):
        data.ctrl[aids] = start + min(1, step * model.opt.timestep / 2) * (target - start)
        if support == 'server_weld':
            server._hold_stationary_base_freejoint_if_idle()
        mujoco.mj_step(model, data)
        assert data.ncon == 0
    assert np.max(np.abs(data.qpos[qa] - target)) < .005

    if support == 'server_weld':
        weld = model.equality('emet_stationary_base').id
        assert data.eq_active[weld]
        server._nav_goal_world = np.array([1, 0, 0])
        server._hold_stationary_base_freejoint_if_idle()
        assert not data.eq_active[weld]
        server._nav_goal_world = None
        server._passive_base_support = True
        server._hold_stationary_base_freejoint_if_idle()
        assert not data.eq_active[weld]


def test_base_teleport_preserves_attachment_offset_without_reregistering():
    import threading

    from emet.simulation.robosuite_server import RobosuiteZmqServer

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base_link"><freejoint/><geom type="sphere" size=".1"/>
        <body name="tool" pos=".3 0 .5"><geom type="sphere" size=".02"/></body>
      </body>
      <body name="load" pos=".4 0 .5"><freejoint/><geom type="box" size=".03 .03 .03"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel, server._mjdata, server._mj_lock = model, data, threading.RLock()
    server._teleport_planar_base_world_xyt = lambda *args: False
    server._base_freejoint_addrs = lambda: (0, 0)
    server._patch_emet_session_body_pos = lambda *args: None
    server._kinematic_attachments = {"load": {"ee_body": "tool", "offset_local": [.1, 0., 0.]}}
    assert server._teleport_base_world_xyt(1., 2., np.pi / 2)
    mujoco.mj_forward(model, data)
    ee = data.body("tool")
    observed_offset = ee.xmat.reshape(3, 3).T @ (data.body("load").xpos - ee.xpos)
    np.testing.assert_allclose(observed_offset, [.1, 0, 0], atol=1e-9)
    assert server._kinematic_attachments["load"]["offset_local"] == [.1, 0, 0]


@pytest.mark.parametrize("ratio,blocked,accepted", [(1., False, False), (.2, False, True), (.2, True, False)])
def test_measured_arrival_scales_slow_sim_budget_without_accepting_blockage(monkeypatch, ratio, blocked, accepted):
    from types import SimpleNamespace

    from emet.controller.manipulation import kinematic_pick_place as module

    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda dt: clock.__setitem__(0, clock[0] + dt)))
    ex = object.__new__(KinematicPickPlaceExecutor)
    ex.robot = SimpleNamespace(_state={"sim_to_real_ratio": ratio})
    ex.ee_body, ex.ik_tol_m = "tool", .035
    ex._data = SimpleNamespace(body=lambda _: SimpleNamespace(
        xpos=np.array([1. if clock[0] >= 5 and not blocked else 0., 0., 0.])))
    ex._sync_qpos_from_robot = lambda: True
    ex._joint_tracking_evidence = lambda: {}
    ok, error = ex._wait_measured_ee(np.array([1., 0., 0.]))
    assert ok is accepted
    assert ex.last_ee_verification["timeout_wall_s"] == 3 / ratio
    assert clock[0] <= 3 / ratio + .05
    assert (error <= .035) is accepted
