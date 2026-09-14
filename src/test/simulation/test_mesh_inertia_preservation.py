# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from emet.simulation.stretch_mujoco.utils import ensure_mesh_inertia, preserve_body_inertias

SCENE = """<mujoco>
  <compiler inertiafromgeom="true"/>
  <asset><mesh name="solid" vertex="0 0 0  .1 0 0  0 .1 0  0 0 .1"
    face="0 2 1  0 1 3  0 3 2  1 2 3"/></asset>
  <worldbody><body name="object" pos="1 2 3">
    <freejoint/><geom type="mesh" mesh="solid" density="100"/>
  </body></worldbody>
</mujoco>"""


def test_shell_compatibility_preserves_source_mass_com_and_inertia():
    source = mujoco.MjModel.from_xml_string(SCENE)
    broken = mujoco.MjModel.from_xml_string(ensure_mesh_inertia(SCENE))
    assert broken.body("object").mass[0] > 10 * source.body("object").mass[0]
    xml = preserve_body_inertias(SCENE, source)
    result = mujoco.MjModel.from_xml_string(ensure_mesh_inertia(xml))
    for key in ("body_mass", "body_ipos", "body_iquat", "body_inertia", "body_pos", "body_quat"):
        np.testing.assert_allclose(getattr(result, key), getattr(source, key), atol=1e-12)
    assert ET.fromstring(xml).find(".//geom").get("density") == "100"
    # No collision or appearance changes accompany the inertia repair.
    for key in ("geom_type", "geom_contype", "geom_conaffinity", "geom_rgba"):
        np.testing.assert_array_equal(getattr(result, key), getattr(source, key))


def test_preserving_inertials_is_idempotent_and_honors_source_not_stale_xml():
    source = mujoco.MjModel.from_xml_string(SCENE)
    stale = SCENE.replace("<freejoint/>", '<inertial mass="99" pos="0 0 0" diaginertia="1 1 1"/><freejoint/>')
    once = preserve_body_inertias(stale, source)
    assert preserve_body_inertias(once, source) == once
    assert len(ET.fromstring(once).findall(".//inertial")) == 1


@pytest.mark.parametrize("name", [None, "missing"])
def test_unmatched_bodies_fail_instead_of_silently_reinferring_mass(name):
    source = mujoco.MjModel.from_xml_string(SCENE)
    root = ET.fromstring(SCENE)
    body = root.find(".//body")
    if name is None:
        del body.attrib["name"]
    else:
        body.set("name", name)
    with pytest.raises((ValueError, KeyError)):
        preserve_body_inertias(ET.tostring(root, encoding="unicode"), source)
