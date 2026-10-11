"""Online approach selection checks live geometry and remains read-only until execution."""
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.task.tamp import task_search
from emet.core.command_client import send_command
from emet.core.command_runtime import CommandRuntime


def test_ring_translates_with_object_and_preserves_side_arm_yaw():
    first = task_search.approach_candidates_for_object_xy([0, 0])
    second = task_search.approach_candidates_for_object_xy([2, -1], mode='side')
    assert len(first) == 16
    np.testing.assert_allclose(np.linalg.norm(np.asarray(first)[:, :2], axis=1), .55)
    np.testing.assert_allclose(np.asarray(second)[:, :2] - np.asarray(first)[:, :2], [[2, -1]] * 16)
    np.testing.assert_allclose(first[0], [0, .55, -np.pi / 2], atol=1e-10)
    assert second[0][2] == pytest.approx(np.pi / 2)


def fake_robot(clear):
    def query(poses):
        return {'poses': np.asarray(poses).tolist(), 'clear': clear[:len(poses)]}
    return SimpleNamespace(
        _state={'sim_base_pose_query': True}, check_base_poses=query,
        get_emet_session=lambda: {'sim_object_placements': {'object': {'pos': [2, 1, .1]}}},
        move_base_to=lambda *a, **kw: pytest.fail('planning must not move'),
    )


def plan(robot, **kwargs):
    return task_search.plan_pick_place(robot, object_query='object', receptacle_query='bin',
                                      object_gt_body='object', grasp_poses=[np.eye(4)], **kwargs)


def test_selection_skips_collisions_then_ik_failure(monkeypatch):
    robot = fake_robot([False] * 3 + [True] * 13)
    seen = []
    executor = SimpleNamespace(_ensure_model=lambda: True, _model=None, _data=None,
                               ee_body='tool', joint_names=['arm'])
    monkeypatch.setattr(task_search, '_sync_executor_base_to_xyt', lambda ex, pose: seen.append(pose.copy()))
    monkeypatch.setattr(task_search, 'rank_grasps_by_ik',
                        lambda *a, **kw: [(0, .001 if len(seen) > 1 else 1., len(seen) > 1)])
    result = plan(robot, executor=executor)
    assert result.success
    expected = task_search.approach_candidates_for_object_xy([2, 1])[4]
    np.testing.assert_allclose(result.steps[0].args['xyt'], expected)
    assert len(seen) == 2
    assert any('approach[0] collision' in s for s in result.expanded_nodes)


def test_all_blocked_and_query_failure_fail_without_motion():
    robot = fake_robot([False] * 16)
    assert not plan(robot).success
    def failed(poses):
        raise TimeoutError('no receipt')
    robot.check_base_poses = failed
    result = plan(robot)
    assert not result.success and result.message.startswith('approach_validation_failed:')


def test_explicit_pose_and_route_validator_are_honored():
    robot = fake_robot([True] * 16)
    explicit = [4, 3, .2]
    result = plan(robot, approach_pose=explicit)
    assert result.steps[0].args['xyt'] == explicit
    assert not plan(robot, approach_validator=lambda p: False).success


def test_simulator_query_reports_contacts_without_changing_state():
    from emet.simulation.robosuite_server import RobosuiteZmqServer
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="box" pos="1 0 0" size=".1 1 1"/>
    <body name="base_link"><freejoint/><geom type="sphere" size=".2" mass="1"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel, server._mjdata = model, data
    server._mj_lock = threading.RLock()
    server._spec = SimpleNamespace(base_link_name='base_link')
    before = data.qpos.copy()
    result = server.check_base_poses([[1, 0, 0], [0, 0, 0]])
    assert result['clear'] == [False, True] and result['scope'] == 'base_endpoint_only'
    np.testing.assert_array_equal(data.qpos, before)
    with pytest.raises(ValueError):
        server.check_base_poses([[0, 0, 0]] * 33)


def test_read_only_query_result_crosses_command_protocol():
    from emet.controller.generic_zmq_client import GenericZmqClient
    class Server(CommandRuntime):
        def __init__(self):
            self.initialize_commands()
            self._last_step = -1
        def check_base_poses(self, poses):
            return {'poses': poses, 'clear': [False, True]}
        def handle_action(self, action):
            pytest.fail('query was dispatched as actuation')
    server = Server()
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0,
                             _state=server.command_message({'sim_base_pose_query': True}))
    def send(action):
        server.dispatch_command(action)
        client._state = server.command_message({'sim_base_pose_query': True})
    client.send_message = send
    client.send_action = lambda payload, **kwargs: send_command(client, payload, **kwargs)
    result = GenericZmqClient.check_base_poses(client, [[1, 0, 0], [0, 0, 0]])
    assert result['clear'] == [False, True]
    assert server.command_tracker.snapshot()[-1]['status'] == 'succeeded'
    client._state['sim_base_pose_query'] = False
    with pytest.raises(RuntimeError, match='does not advertise'):
        GenericZmqClient.check_base_poses(client, [[0, 0, 0]])
