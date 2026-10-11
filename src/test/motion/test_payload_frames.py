"""Rigid payload transport and coherent geometry, including welded mesh children."""
import threading

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.simulation.sim_manipulation import set_free_body_pose
from emet.simulation.sim_object_placements import measured_held_object_geometry


def scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base"><freejoint/><geom size=".02"/>
        <body name="tool" pos=".3 0 .5" euler=".2 .3 .1"><geom size=".02"/></body>
      </body>
      <body name="root" pos=".4 .1 .5" euler=".1 .2 .3"><freejoint/>
        <geom type="box" size=".02 .03 .04"/>
        <body name="load" pos=".05 -.02 .01" euler=".3 -.2 .4">
          <geom type="box" size=".06 .02 .03"/>
        </body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize('body', ['root', 'load'])
def test_free_body_pose_sets_named_body_orientation_and_position(body):
    model, data = scene()
    rotation = Rotation.from_euler('xyz', [1.1, -.5, 2.]).as_matrix()
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    assert set_free_body_pose(model, data, body, [1., 2., 3.], quat)
    np.testing.assert_allclose(data.body(body).xpos, [1., 2., 3.], atol=1e-12)
    np.testing.assert_allclose(data.body(body).xmat.reshape(3, 3), rotation, atol=1e-12)


@pytest.mark.parametrize('body', ['root', 'load'])
def test_attached_geometry_survives_transport_and_detects_real_slip(body):
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    model, data = scene()
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel, server._mjdata, server._mj_lock = model, data, threading.RLock()
    server._patch_emet_session_body_pos = lambda *args: None
    ee, obj = data.body('tool'), data.body(body)
    rotation = ee.xmat.reshape(3, 3)
    offset = rotation.T @ (obj.xpos - ee.xpos)
    relative_rotation = rotation.T @ obj.xmat.reshape(3, 3)
    server._kinematic_attachments = {body: {
        'ee_body': 'tool', 'offset_local': offset, 'rotation_local': relative_rotation}}
    original = np.array(measured_held_object_geometry(model, data, body, 'tool')['vertices_ee'])
    # Multiple noncommuting rotations, with real physics between attachment snaps.
    for angles in ([.1, .5, 1.], [-.4, .2, -2.]):
        quat = Rotation.from_euler('xyz', angles).as_quat()[[3, 0, 1, 2]]
        assert set_free_body_pose(model, data, 'base', [1., -.4, .2], quat)
        mujoco.mj_step(model, data)
        server._snap_kinematic_attachments()
        measured = measured_held_object_geometry(model, data, body, 'tool')
        np.testing.assert_allclose(measured['vertices_ee'], original, atol=1e-10)
        np.testing.assert_allclose(
            data.body('tool').xmat.reshape(3, 3).T @ data.body(body).xmat.reshape(3, 3),
            relative_rotation, atol=1e-10)
    # The published vertices must follow measured slip, not the saved offset.
    assert set_free_body_pose(model, data, body, data.body(body).xpos.copy() + [.03, 0, 0])
    slipped = np.array(measured_held_object_geometry(model, data, body, 'tool')['vertices_ee'])
    assert not np.allclose(slipped, original, atol=.01, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(slipped - original, axis=1), .03, atol=1e-10)
    # The root geometry is included even when the semantic name selects a child.
    assert original.shape == (8, 3)


def test_placement_uses_coherent_payload_and_rejects_measured_slip():
    from types import SimpleNamespace

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.motion.placement_geometry import PlacementCollisionChecker

    model, data = scene()
    held = measured_held_object_geometry(model, data, 'load', 'tool')
    session = {'sim_held_object_geometry': {'load': held}}
    executor = object.__new__(KinematicPickPlaceExecutor)
    executor.manip_collision = 'ground_truth'
    executor.ee_body, executor._model, executor._data = 'tool', model, data
    executor.robot = SimpleNamespace(get_emet_session=lambda: session,
                                     _spec=SimpleNamespace(base_link_name='base'))
    executor.profile = SimpleNamespace(gripper_contact_bodies=lambda: ())
    executor._sync_qpos_from_robot = lambda: True
    support = [[-1., -1., 0.], [1., 1., 0.]]
    executor._placement_support_bounds = np.array([support])
    executor._placements = lambda: {'load': {'bounds': [[99]*3, [100]*3]},
        'support': {'bounds': [[-1, -1, -.1], [1, 1, 0]], 'support_surfaces': [support]}}
    scene_geometry, payload, _ = executor._placement_geometry('load', 'support')
    # Neither the unrelated AABB nor a newer local EE pose may alter this snapshot.
    assert set_free_body_pose(model, data, 'base', [2., 3., 4.])
    _, repeated, _ = executor._placement_geometry('load', 'support')
    np.testing.assert_array_equal(payload.vertices_ee, repeated.vertices_ee)
    checker = PlacementCollisionChecker(model, robot_body='base', ee_body='tool',
                                         scene=scene_geometry, payload=payload, contact_bodies=('tool',))
    held['vertices_ee'] = (np.array(held['vertices_ee']) + [.03, 0, 0]).tolist()
    with pytest.raises(ValueError, match='placement_attachment_changed'):
        executor._refresh_placement_checker('load', 'support', checker)
    session.clear()
    with pytest.raises(ValueError, match='placement_payload_geometry_missing'):
        executor._placement_geometry('load', 'support')


def test_mesh_payload_bounds_use_vertices_not_inflated_principal_axis_box():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="tetra" vertex="0 0 0 .2 0 0 0 .1 0 0 0 .3"/>
      </asset><worldbody><body name="tool"><geom size=".01"/></body>
      <body name="load" pos="1 2 3"><freejoint/><geom type="mesh" mesh="tetra"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    geometry = measured_held_object_geometry(model, data, 'load', 'tool')
    vertices = np.array(geometry['vertices_ee'])
    np.testing.assert_allclose(vertices.min(axis=0), [1, 2, 3], atol=1e-7)
    np.testing.assert_allclose(vertices.max(axis=0), [1.2, 2.1, 3.3], atol=1e-7)
