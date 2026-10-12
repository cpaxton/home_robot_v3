"""Planar placement must converge from any yaw without corrupting failed probes."""
import mujoco
import numpy as np
import pytest

from emet.simulation.spawn_planar import write_planar_base_xyt


def fixture(*, singular=False, limited=False):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
      <worldbody><body pos="1 -2 0" euler="0 0 .7"><body name="anchor">
        <joint name="x" type="slide" axis="1 0 0" {('limited="true" range="-.1 .1"' if limited else '')}/>
        <joint name="y" type="slide" axis="{('1 0 0' if singular else '0 1 0')}"/>
        <joint name="yaw" type="hinge" axis="0 0 1"/>
        <geom size=".1" mass="1"/><body name="base" pos=".2 -.1 0"><geom size=".05"/></body>
      </body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[:] = [.4, -.3, 1.57]
    data.qvel[:] = [1, 2, 3]
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize('yaw', [0, np.pi/2, np.pi, -np.pi/2, .8])
def test_world_pose_with_yawed_ancestor_and_offset_base(yaw):
    model, data = fixture()
    for x, y in [(2, 1), (-1, -.5), (2, 1)]:
        assert write_planar_base_xyt(model, data, joint_names=('x', 'y', 'yaw'),
                                     base_body_name='base', world_x=x, world_y=y, world_yaw=yaw)
        np.testing.assert_allclose(data.body('base').xpos[:2], [x, y], atol=1e-6)
        r = data.body('base').xmat.reshape(3, 3)
        assert abs(np.arctan2(np.sin(np.arctan2(r[1, 0], r[0, 0])-yaw),
                             np.cos(np.arctan2(r[1, 0], r[0, 0])-yaw))) < 1e-6
        np.testing.assert_array_equal(data.qvel, 0)


@pytest.mark.parametrize('case', ['singular', 'limited', 'nonfinite', 'missing'])
def test_failed_candidate_preserves_live_state(case):
    model, data = fixture(singular=case == 'singular', limited=case == 'limited')
    before = [v.copy() for v in (data.qpos, data.qvel, data.qacc, data.xpos, data.contact.dist)]
    assert not write_planar_base_xyt(model, data,
        joint_names=('missing' if case == 'missing' else 'x', 'y', 'yaw'),
        base_body_name='base', world_x=np.nan if case == 'nonfinite' else 3., world_y=2., world_yaw=1.)
    for actual, expected in zip((data.qpos, data.qvel, data.qacc, data.xpos, data.contact.dist), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_exhausted_search_does_not_leave_last_rejected_candidate(monkeypatch):
    from emet.simulation import spawn_planar
    model, data = fixture()
    before = data.qpos.copy()
    monkeypatch.setattr(spawn_planar, 'collision_scene_xy_clip_rect', lambda *a, **kw: None)
    monkeypatch.setattr(spawn_planar, 'scene_collision_centroid_xy', lambda *a, **kw: None)
    monkeypatch.setattr(spawn_planar, 'iter_annulus_xy_candidates', lambda **kw: [(2., 1.)])
    monkeypatch.setattr(spawn_planar, '_ithor_occupancy_priority_xy', lambda *a, **kw: ([], None))
    monkeypatch.setattr(spawn_planar, 'walkable_floor_z_at_xy', lambda *a, **kw: None)
    assert spawn_planar.find_planar_base_xyt(model, data, base_body_name='base',
        joint_names=('x', 'y', 'yaw')) is None
    np.testing.assert_array_equal(data.qpos, before)
