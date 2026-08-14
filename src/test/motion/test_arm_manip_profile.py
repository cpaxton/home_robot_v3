# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ArmManipProfile registry + end-to-end kinematic pick verification.

The end-to-end tests fake grasp *physics*: we do not simulate attachment/force.
Instead we assert that, after IK drives the EE to an object target, an annotated
gripper contact body (finger/jaw/EE) comes within contact range of the object. This
catches mis-wired profiles (wrong EE body, reversed arm chain, bad actuator map)
without needing a live robot or EGL server.
"""

from __future__ import annotations

import numpy as np
import pytest

from emet.motion.arm_manip_profile import ArmManipProfile, home_arm_q_array, resolve_manip_mode_for_robot


def test_rby1_and_galaxea_share_profile():
    p1 = ArmManipProfile.for_robot("rby1", arm="left")
    p2 = ArmManipProfile.for_robot("galaxea_r1", arm="left")
    assert p1.ee_body == p2.ee_body
    assert p1.joint_names == p2.joint_names
    assert len(home_arm_q_array(p1)) == len(p1.joint_names)


def test_unknown_robot_raises():
    with pytest.raises(KeyError, match="stretch"):
        ArmManipProfile.for_robot("stretch")


def test_robot_id_from_client_requires_id():
    from emet.motion.arm_manip_profile import robot_id_from_client

    class _Bare:
        pass

    with pytest.raises(ValueError, match="cannot resolve"):
        robot_id_from_client(_Bare())


def test_home_cmd_matches_actuator_count():
    from emet.robots.galaxea_r1 import R1_ACTUATOR_NAMES

    p = ArmManipProfile.for_robot("rby1", arm="left")
    assert len(p.home_cmd) == len(R1_ACTUATOR_NAMES)
    assert len(p.actuator_names) == len(R1_ACTUATOR_NAMES)


class _FakeRobot:
    def __init__(self, caps: dict):
        self._caps = caps

    def get_emet_session(self):
        return {"capabilities": self._caps}


def test_resolve_manip_mode_auto():
    assert resolve_manip_mode_for_robot(_FakeRobot({"kinematic_manip": True}), manip_mode="auto") == "kinematic"
    assert resolve_manip_mode_for_robot(_FakeRobot({"sim_set_body_pose": True}), manip_mode="auto") == "teleport"


# ---------------------------------------------------------------------------
# End-to-end: discovery -> IK -> gripper contact (no physics, no EGL)
# ---------------------------------------------------------------------------

# Every (robot, arm) we must be able to pick with. Robots with a shared profile
# (rby1 / galaxea_r1) are tested once for the arm they have in common.
PICK_ROBOTS = [
    pytest.param("sourccey", "left", id="sourccey-left"),
    pytest.param("sourccey", "right", id="sourccey-right"),
    pytest.param("xlerobot", "left", id="xlerobot-left"),
    pytest.param("xlerobot", "right", id="xlerobot-right"),
    pytest.param("rby1", "left", id="rby1-left"),
    pytest.param("innate_mars", "left", id="innate_mars-left"),
    pytest.param("franka_fr3", "left", id="franka_fr3-left"),
]


def _load_robot(robot_id: str):
    from emet.robots import get_robot_spec

    pytest.importorskip("mujoco")
    spec = get_robot_spec(robot_id)
    assert spec is not None and spec.mjcf_path, f"{robot_id} needs a vendored MJCF"
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("robot_id,arm", PICK_ROBOTS)
def test_profile_discovery_resolves(robot_id, arm):
    """Every supported pick robot gets a fully wired profile."""
    profile = ArmManipProfile.for_robot(robot_id, arm=arm)
    assert profile.joint_names, "must discover arm joints"
    assert profile.ee_body, "must pick an EE body"
    assert profile.gripper_contact_bodies(), "must have a contact body (gripper or EE)"
    model, _ = _load_robot(robot_id)
    import mujoco

    for body in (profile.ee_body, *profile.gripper_contact_bodies()):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0, f"missing body {body}"
    for joint in profile.joint_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0, f"missing joint {joint}"


@pytest.mark.parametrize("robot_id,arm", PICK_ROBOTS)
def test_ik_reaches_target_and_gripper_contacts(robot_id, arm):
    """Fake-grasp: IK to a target object; a gripper contact body must come within
    contact range of it (physics is not simulated)."""
    from emet.motion.mujoco_arm_ik import solve_position_ik_multiseed

    profile = ArmManipProfile.for_robot(robot_id, arm=arm)
    model, data = _load_robot(robot_id)
    import mujoco

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.ee_body)
    assert ee_id >= 0
    home_ee = np.array(data.body(ee_id).xpos, dtype=np.float64).copy()

    # Target a short distance straight ahead of the current EE (reachable everywhere).
    target = home_ee.copy()
    target[1] += 0.03

    result = solve_position_ik_multiseed(
        model,
        data,
        ee_body=profile.ee_body,
        joint_names=list(profile.joint_names),
        target_pos=target,
        tol_m=0.05,
        max_iters=200,
    )
    assert result.success, f"IK failed for {robot_id}/{arm} (err {result.pos_error_m:.4f} m)"
    assert result.pos_error_m <= 0.06

    # Fake grasp contact: the object must lie within the gripper's grasp region. For
    # finger-style grippers (rby1/galaxea) the fingers stay ~0.15 m apart in the open
    # pose, so the object sits between them; "contact" means the object is inside that
    # reach, not that a specific finger touched it.
    contact_radius_m = {
        "sourccey": 0.06,
        "xlerobot": 0.06,
        "rby1": 0.16,
        "galaxea_r1": 0.16,
        "innate_mars": 0.06,
        "franka_fr3": 0.06,
    }.get(robot_id, 0.06)
    min_dist = float("inf")
    for body in profile.gripper_contact_bodies():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        assert bid >= 0
        min_dist = min(min_dist, float(np.linalg.norm(np.array(data.body(bid).xpos) - target)))
    assert min_dist <= contact_radius_m, (
        f"gripper {profile.gripper_contact_bodies()} too far from object for {robot_id}/{arm} "
        f"({min_dist:.4f} m > {contact_radius_m} m)"
    )
