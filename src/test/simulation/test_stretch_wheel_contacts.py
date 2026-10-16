# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Check authored wheel contacts without loading room assets or running a policy."""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("floor_friction", ["1 .005 .0001", ".9 .9 .001"])
def test_wheel_material_survives_room_contact_mixing(side, floor_friction):
    source = ET.parse(Path(__file__).resolve().parents[2] / "emet/assets/robot/stretch.xml")
    root = ET.Element("mujoco")
    root.append(deepcopy(source.getroot().find("default")))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="2 2 .1", friction=floor_friction)
    body = ET.SubElement(world, "body", pos="0 0 .049", quat="1 1 0 0", childclass="stretch")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "inertial", mass=".15", pos="0 0 0", diaginertia=".001 .001 .001")
    wheel = deepcopy(source.find(f".//body[@name='link_{side}_wheel']/geom[@class='wheel_collision']"))
    wheel.set("pos", "0 0 0")
    wheel.set("name", "wheel")
    body.append(wheel)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.geom("wheel").priority == 1
    assert data.ncon > 0
    for contact in data.contact:
        assert contact.dim == 6
        np.testing.assert_allclose(contact.friction, [1, 1, 0.005, 0.0001, 0.0001])
        np.testing.assert_allclose(contact.solref, [0.005, 1])


def test_full_bundled_scene_keeps_wheel_material_against_high_friction_floor():
    """Compile the shipped room and robot, including their real default classes."""
    scene = Path(__file__).resolve().parents[2] / "emet/assets/robot/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.geom("floor").friction[:] = [100, 10, 1]
    data = mujoco.MjData(model)
    base_joint = int(model.body("base_link").jntadr[0])
    base_qpos = int(model.jnt_qposadr[base_joint])
    data.qpos[base_qpos + 2] -= 0.002
    mujoco.mj_forward(model, data)
    wheel_bodies = {model.body(f"link_{side}_wheel").id for side in ("left", "right")}
    wheel_contacts = [
        contact for contact in data.contact if any(int(model.geom_bodyid[g]) in wheel_bodies for g in contact.geom)
    ]
    assert wheel_contacts, "Expected real wheel-floor contacts in the bundled scene"
    for contact in wheel_contacts:
        np.testing.assert_allclose(contact.friction, [1, 1, 0.005, 0.0001, 0.0001])
        assert contact.dim == 6
