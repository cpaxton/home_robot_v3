"""A held payload may lift from shallow soft support, never deepen it."""

import mujoco
import numpy as np
import pytest

from emet.motion.mujoco_collision import MujocoSceneCollisionChecker

pytestmark = pytest.mark.sim


def scene(depth=0.0012, *, dynamic_support=False):
    joint = "<freejoint/>" if dynamic_support else ""
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <body name="table">{joint}<geom type="box" size="1 1 .05"/></body>
      <body name="robot" pos="3 0 0"><freejoint/><geom size=".02"/></body>
      <body name="payload" pos="0 0 {0.1 - depth}"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    checker = MujocoSceneCollisionChecker(model, robot_body="robot")
    checker.set_payload(model, data, "payload", "robot")
    z = int(model.jnt_qposadr[model.body("robot").jntadr[0]]) + 2
    return model, data, checker, z


def test_departure_allows_only_original_compression_and_clears_on_exit():
    m, d, c, z = scene()
    assert c.configuration_collides(m, d)
    before = d.qpos.copy()
    with c.payload_support_departure(m, d) as contacts:
        assert contacts and not c.configuration_collides(m, d)
        np.testing.assert_array_equal(d.qpos, before)
        d.qpos[z] += 0.0001
        assert not c.configuration_collides(m, d)
        d.qpos[z] -= 0.0002
        assert c.configuration_collides(m, d)  # deepening by 0.1 mm is rejected
        d.qpos[:] = before
        assert not c.configuration_collides(m, d)
    assert c.configuration_collides(m, d)


@pytest.mark.parametrize("depth,dynamic", [(0.0021, False), (0.0012, True)])
def test_deep_or_moving_support_does_not_qualify(depth, dynamic):
    m, d, c, _ = scene(depth, dynamic_support=dynamic)
    with c.payload_support_departure(m, d) as contacts:
        assert not contacts
        assert c.configuration_collides(m, d)


def test_support_permission_does_not_allow_robot_table_collision():
    m, d, c, z = scene()
    with c.payload_support_departure(m, d):
        d.qpos[z - 2 : z + 1] = [0.5, 0, 0.04]
        assert c.configuration_collides(m, d)
        assert any("robot" in row["bodies"] for row in c.last_contacts)


def test_permission_clears_after_failure():
    m, d, c, _ = scene()
    with pytest.raises(RuntimeError, match="abort"):
        with c.payload_support_departure(m, d):
            raise RuntimeError("abort")
    assert not c._support_departure and c.configuration_collides(m, d)


def test_contacts_inside_normal_tolerance_receive_no_extra_permission():
    m, d, c, z = scene(0.0002)
    with c.payload_support_departure(m, d) as contacts:
        assert not contacts and not c.configuration_collides(m, d)
        d.qpos[z] -= 0.0004
        assert not c.configuration_collides(m, d)
        d.qpos[z] -= 0.0005
        assert c.configuration_collides(m, d)
