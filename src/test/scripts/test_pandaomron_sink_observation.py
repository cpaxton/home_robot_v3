# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_pickup_probe_excludes_the_counter_inside_the_payload_selection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from eval_pandaomron_sink_observation import pickup_probe

    y, x = np.indices((20, 20))
    world = np.stack((x * 0.005, y * 0.005, np.full(x.shape, 0.9)), axis=-1)
    world[7:14, 7:14, 2] = 1.0
    patch = SimpleNamespace(plane=np.array([0.0, 0.0, 0.9]), measurement_margin_m=0.005)
    selected = {"polygon": [(1, 1), (18, 1), (18, 18), (1, 18)], "height_offset_m": -0.025}

    target, yaws, evidence = pickup_probe(np.ones(x.shape), world, selected, patch)

    np.testing.assert_allclose(target, [0.05, 0.05, 0.975])
    assert len(yaws) == 4
    assert evidence["points"] == 49
    assert evidence["central_96_percent_bounds_m"][0][2] == pytest.approx(1.0)


def test_transfer_score_requires_verified_tool_receipt_physics_and_native_actions():
    from eval_pandaomron_sink_agent import score_transfer

    physics = {'initial_success': False, 'upstream_success': True, 'basin_contains_tongs': True,
        'lift_with_two_pads_m': .08, 'final_object_speed_mps': 0., 'unintended_contact_frames': 0,
        'unintended_self_contact_frames': 0}
    receipt = {'execute_pick_place_plan': {'status': 'ok', 'code': 'ok'}}
    actions = ['{"accepted": true, "forbidden": []}']
    assert score_transfer(physics, receipt, actions) == (True, True, True)
    assert score_transfer(physics, {}, actions) == (True, False, False)
    assert not score_transfer({**physics, 'unintended_self_contact_frames': 1}, receipt, actions)[2]
    assert not score_transfer(physics, receipt, [actions[0], '{"forbidden": ["teleport"]}'])[2]
    assert not score_transfer(physics, receipt, [])[2]


def test_serial_benchmark_compilation_preserves_physics_and_assets():
    from xml.etree import ElementTree

    from emet.simulation.pandaomron_environment import serial_mesh_compiler

    xml = '<mujoco><compiler angle="radian"/><option timestep=".002"/><asset><mesh name="part" file="part.obj"/></asset></mujoco>'
    result = ElementTree.fromstring(serial_mesh_compiler(xml))
    assert result.find('compiler').attrib == {'angle': 'radian', 'usethread': 'false'}
    assert result.find('option').get('timestep') == '.002'
    assert result.find('asset/mesh').get('file') == 'part.obj'


@pytest.mark.parametrize('fail_reset', [False, True])
def test_native_absolute_reset_holds_native_posture_without_reinterpreting_episode_commands(fail_reset):
    from emet.simulation.pandaomron_environment import preserve_absolute_reset_posture

    goals = []
    q = np.array([0., -1., 0., -2.2, 0., 1.5, .7, .1])
    robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7), _ref_torso_joint_pos_indexes=np.array([7]),
        create_action_vector=lambda parts: np.r_[parts['right'], parts['torso'], 0.])
    env = SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=SimpleNamespace(qpos=q)),
                          _pre_action=lambda action, policy_step: goals.append(np.array(action)))

    def reset():
        env._pre_action(np.zeros(9), True)
        env._pre_action(np.zeros(9), False)
        if fail_reset:
            raise RuntimeError('reset_failed')

    env._reset_internal = reset
    preserve_absolute_reset_posture(env)
    if fail_reset:
        with pytest.raises(RuntimeError, match='reset_failed'):
            env._reset_internal()
    else:
        env._reset_internal()
    for goal in goals:
        np.testing.assert_array_equal(goal[:8], q)
    env._pre_action(np.zeros(9), True)
    np.testing.assert_array_equal(goals[-1], np.zeros(9))


def test_robot_only_compilation_uses_the_same_serial_geometry_policy():
    from emet.simulation.pandaomron_environment import compile_serial_robot_model

    model = compile_serial_robot_model(SimpleNamespace(get_xml=lambda: '<mujoco><option timestep=".004"/><worldbody><geom size=".03"/></worldbody></mujoco>'))
    assert model.ngeom == 1 and model.opt.timestep == pytest.approx(.004)
    assert model.geom_size[0, 0] == pytest.approx(.03)


def test_serial_robot_loader_restores_compiler_after_failed_model_load(monkeypatch):
    import sys

    from emet.simulation.pandaomron_environment import serial_robot_loader

    class XML:
        def get_model(self, mode='mujoco'):
            return 'original'

        def get_xml(self):
            return '<mujoco><worldbody><geom size=".03"/></worldbody></mujoco>'

    original = XML.get_model
    monkeypatch.setitem(sys.modules, 'robosuite.models.base', SimpleNamespace(MujocoXML=XML))

    def load():
        assert XML().get_model().ngeom == 1
        raise RuntimeError('load_failed')

    env = serial_robot_loader(SimpleNamespace(_load_model=load))
    with pytest.raises(RuntimeError, match='load_failed'):
        env._load_model()
    assert XML.get_model is original


def test_native_bridge_owns_auxiliary_rendering_instead_of_generic_stereo(monkeypatch):
    from emet.simulation.pandaomron_server import PandaOmronZmqServer
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    specs = []

    def intercept(self, *, robot_spec, **kwargs):
        specs.append(robot_spec)
        raise RuntimeError('base_constructor_intercepted')

    monkeypatch.setattr(RobosuiteZmqServer, '__init__', intercept)
    with pytest.raises(RuntimeError, match='base_constructor_intercepted'):
        PandaOmronZmqServer(env=None, environment={}, scene_camera='robot0_agentview_right')
    # The generic second-camera path captures a different physics state and
    # overwrites the timestamp used by the bridge's synchronized RGB-D views.
    assert specs[0].camera_names == ['robot0_agentview_right']


def test_initial_settle_holds_measured_robot_posture_and_opens_gripper():
    from emet.simulation.pandaomron_environment import settle_absolute_environment

    q = np.array([-.1, -.8, 0., -2.1, 0., 1.5, .7, .002])
    commands = []
    robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7), _ref_torso_joint_pos_indexes=np.array([7]),
        create_action_vector=lambda parts: np.r_[parts['right'], parts['torso'], parts['right_gripper']])
    def step(action):
        commands.append(action.copy())
        q[:] += .001  # Measured drift must not become a moving hold target.
    env = SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=SimpleNamespace(qpos=q)), step=step)
    original = q.copy()
    settle_absolute_environment(env, steps=3)
    assert len(commands) == 3
    for action in commands:
        np.testing.assert_allclose(action[:-1], original)
        assert action[-1] == -1.
    with pytest.raises(ValueError, match='invalid_initial_settle_steps'):
        settle_absolute_environment(env, steps=-1)
