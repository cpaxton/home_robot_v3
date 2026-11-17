# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Collision semantics survive contact-count changes without bulk field views."""

import mujoco
import pytest

from emet.motion.mujoco_collision import MujocoSceneCollisionChecker


def fixture():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body name="robot" pos="0 0 .4"><freejoint/><geom type="sphere" size=".1"/></body>
      <body name="unrelated" pos=".6 0 .05"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>""")
    return model, mujoco.MjData(model), MujocoSceneCollisionChecker(model, robot_body="robot")


def test_collision_copies_scalar_contacts_without_bulk_field_access(monkeypatch):
    model, data, checker = fixture()

    def forbidden(_):
        raise AssertionError("bulk contact accessor must not be used")

    monkeypatch.setattr(type(data.contact), "geom", property(forbidden))
    monkeypatch.setattr(type(data.contact), "dist", property(forbidden))
    mujoco.mj_forward(model, data)
    monkeypatch.setattr(type(data.contact[0]), "geom", property(forbidden))
    scratch = mujoco.MjData(model)
    for _ in range(100):
        for height, expected in ((0.4, False), (0.05, True), (0.0995, False), (0.095, True)):
            data.qpos[2] = height
            mujoco.mj_forward(model, data)
            mujoco.mj_copyData(scratch, model, data)
            assert checker.configuration_collides(model, scratch) is expected
            if expected:
                assert all(set(c["bodies"]) == {"world", "robot"} for c in checker.last_contacts)
                assert all(c["distance_m"] < -0.001 for c in checker.last_contacts)


@pytest.mark.parametrize("kind", ["body", "geom"])
def test_allowed_pairs_remain_exact_after_contact_snapshots(kind):
    model, data, checker = fixture()
    data.qpos[2] = 0.05
    if kind == "body":
        checker.allowed_pairs.add(frozenset(("world", "robot")))
    else:
        checker.allowed_geom_pairs.add(frozenset((0, 1)))
    assert not checker.configuration_collides(model, data)
    assert checker.last_contacts == []


def test_empty_contact_set_clears_previous_collision():
    model, data, checker = fixture()
    data.qpos[2] = 0.05
    assert checker.configuration_collides(model, data)
    data.qpos[2] = data.qpos[9] = 0.4
    assert not checker.configuration_collides(model, data)
    assert data.ncon == 0
    assert checker.last_contacts == []
