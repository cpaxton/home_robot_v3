"""Stretch endpoint queries use measured scene state and never move live bodies."""
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.simulation.mujoco_server_stretch import MujocoZmqServer


def server_fixture():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base_link" pos="0 0 .2"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      <body name="obstacle" pos="2 0 .2"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    server = object.__new__(MujocoZmqServer)
    server._articulation_model = model
    server._articulation_lock = threading.RLock()
    server._status = SimpleNamespace(scene_qpos=data.qpos.tolist())
    server.get_robot_spec = lambda: SimpleNamespace(base_link_name="base_link", planar_base_joint_names=None)
    return server, model, data


def test_checks_measured_obstacle_not_xml_pose_and_preserves_state():
    server, model, data = server_fixture()
    # The obstacle has moved away from its XML default of x=2.
    data.qpos[7] = 1
    server._status.scene_qpos = data.qpos.tolist()
    before = data.qpos.copy()
    result = server.check_base_poses([[1, 0, 0], [2, 0, 0]])
    assert result["clear"] == [False, True]
    assert result["scope"] == "base_endpoint_only"
    assert result["contacts"][0]
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_array_equal(server._status.scene_qpos, before)


@pytest.mark.parametrize("values", [None, [], [float('nan')] * 14])
def test_missing_or_invalid_measurement_never_falls_back_to_defaults(values):
    server, _, _ = server_fixture()
    server._status.scene_qpos = values
    assert server._base_pose_query_qpos() is None
    with pytest.raises(RuntimeError, match="measured simulator scene unavailable"):
        server.check_base_poses([[0, 0, 0]])


@pytest.mark.parametrize("poses", [[], [[0, 0]], [[0, 0, float('nan')]], [[0, 0, 0]] * 33])
def test_rejects_invalid_query(poses):
    server, _, _ = server_fixture()
    with pytest.raises(ValueError, match="finite world XYT"):
        server.check_base_poses(poses)


def test_advertisement_requires_measurement_and_supported_model():
    server, model, _ = server_fixture()
    server._emet_session = {}
    server._refresh_articulation_state = lambda: None
    assert server._attach_emet_session({})['sim_base_pose_query'] is True
    server._status.scene_qpos = None
    assert server._attach_emet_session({})['sim_base_pose_query'] is False
    # A fixed-base custom model must not advertise this free-base endpoint.
    fixed = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="base_link"><geom size=".1"/></body></worldbody></mujoco>')
    server._articulation_model = fixed
    server._status.scene_qpos = []
    assert server._attach_emet_session({})['sim_base_pose_query'] is False
