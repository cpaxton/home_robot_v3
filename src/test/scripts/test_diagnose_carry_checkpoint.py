# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Check command profiles without loading a simulator or privileged scene."""

import runpy
from pathlib import Path

import numpy as np
import pytest

helpers = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/diagnose_carry_checkpoint.py"))


@pytest.mark.parametrize("t,expected", [(-20, 0), (0, 0), (1, 0.5), (2, 1), (5, 1), (7, 0.5), (8, 0), (35, 0)])
def test_bounded_drive_profile(t, expected):
    assert helpers["drive_envelope"](t) == expected


@pytest.mark.parametrize("mode", ["hold", "straight", "turn", "brake"])
def test_wheel_profiles_stop_and_apply_transmission_gearing(mode):
    args = {"linear_speed": 0.05, "angular_speed": 0.2, "radius": 0.0508, "separation": 0.3153}
    control = helpers["wheel_controls"]
    np.testing.assert_allclose(control(mode, 35, **args, gears=np.array([3.0, 3.0])), 0)
    peak = control(mode, 3, **args, gears=np.array([3.0, 3.0]))
    np.testing.assert_allclose(peak, 3 * control(mode, 3, **args, gears=np.ones(2)))
    if mode == "turn":
        assert peak[0] < 0 < peak[1]
    if mode in {"straight", "brake"}:
        assert peak[0] == peak[1] > 0


def test_braking_ablation_changes_only_deceleration_phase():
    assert helpers["drive_envelope"](6.25, brake_s=0.25) == 0
    assert helpers["drive_envelope"](6.25, brake_s=2) == 0.875


@pytest.mark.parametrize("speed", [0.2, 0.5])
def test_feedback_turn_reaches_same_angle_with_bounded_rate_and_acceleration(speed):
    angle, rate, dt = 0.0, 0.0, 0.01
    for _ in range(2500):
        previous = rate
        rate = helpers["bounded_turn_rate"](1.5 - angle, rate, dt, speed)
        assert abs(rate) <= speed + 1e-9
        assert abs(rate - previous) <= 0.25 * dt + 1e-9
        angle += 0.65 * rate * dt  # Simulated imperfect wheel tracking, not a physics test.
    assert angle == pytest.approx(1.5, abs=0.01)


@pytest.mark.parametrize("yaw", [-2.0, 0.0, 1.5])
def test_measured_yaw_uses_mujoco_wxyz(yaw):
    qpos = np.array([0.0, 0.0, 0.0, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    assert helpers["base_yaw"](qpos) == pytest.approx(yaw)


@pytest.mark.parametrize(
    "options",
    [
        ["--noslip-iterations", "-1"],
        ["--modes", "release"],
        ["--turn-angle", "nan"],
        ["--turn-angle", "4"],
        ["--sample-period", "0"],
    ],
)
def test_invalid_ablation_rejected_before_loading_scene(monkeypatch, options):
    monkeypatch.setattr(
        "sys.argv",
        ["diagnose", "--scene", "missing.xml", "--trace", "missing.jsonl", "--out", "unused", "--time", "80", *options],
    )
    with pytest.raises(SystemExit) as exc:
        helpers["main"]()
    assert exc.value.code == 2


def test_local_contact_coordinates_include_rotation_and_translation():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(helpers["local_point"]([1, 4, 3], [1, 2, 3], rotation), [2, 0, 0])


def test_contact_evidence_includes_external_contacts_without_mutating_state():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body name="object" pos="0 0 .09"><freejoint/>
        <geom name="ball" type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    contacts = helpers["object_contacts"](model, data, {model.body("object").id}, set())
    assert len(contacts) == 1
    assert not contacts[0]["gripper_contact"]
    assert contacts[0]["distance_m"] == pytest.approx(-0.01)
    assert contacts[0]["wrench_contact_frame"][0] > 0
    np.testing.assert_allclose(contacts[0]["object_center_other_geom"], [0, 0, 0.09])
    np.testing.assert_array_equal(data.qpos, before)
