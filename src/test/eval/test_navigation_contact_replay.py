# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Sampled contact replay must not masquerade as continuous safety acceptance."""

import importlib.util
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
spec = importlib.util.spec_from_file_location(
    "contact_replay", Path(__file__).resolve().parents[3] / "scripts/replay_navigation_contacts.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="ground" type="plane" size="2 2 .1"/>
      <geom name="obstacle" type="sphere" size=".1" pos=".18 0 .15"/>
      <body name="robot" pos="0 0 .15"><freejoint/>
        <geom name="base" type="sphere" size=".03"/>
        <body name="hand" pos=".04 0 0"><geom name="hand_geom" type="sphere" size=".08"/></body>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    row = {key: getattr(data, key).tolist() for key in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")}
    row["sim_time"] = 0
    return model, row


def test_contact_includes_robot_descendants_and_retains_provenance():
    model, row = fixture()
    result = module.replay(model, [row], "robot", ["ground"])
    assert result["status"] == "contacts_found"
    assert any("hand" in contact["bodies"] for contact in result["contacts"])
    assert result["continuous_safety_verified"] is False
    assert result["sample_count"] == 1


def test_absence_of_sampled_contacts_does_not_certify_safety():
    model, row = fixture()
    row["qpos"][0] = -1
    result = module.replay(model, [row], "robot", ["ground"])
    assert result["status"] == "no_contacts_at_sampled_states"
    assert result["continuous_safety_verified"] is False
    assert module.replay(model, [], "robot", ["ground"])["status"] == "missing_states"


def test_invalid_state_or_allowlist_fails_closed():
    model, row = fixture()
    with pytest.raises(KeyError):
        module.replay(model, [row], "robot", ["typo"])
    with pytest.raises(ValueError):
        module.replay(model, [row], "world", [])
    row["qpos"] = []
    with pytest.raises(ValueError, match="Invalid qpos"):
        module.replay(model, [row], "robot", [])
