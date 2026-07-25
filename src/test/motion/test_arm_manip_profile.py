# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ArmManipProfile registry tests."""

from __future__ import annotations

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
