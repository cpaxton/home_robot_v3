# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""RoboCasa planar spawn clearance must be declared by every merged robot."""

from __future__ import annotations


def test_robocasa_planar_robots_declare_first_clearance():
    from emet.robots import get_robot_spec

    for name in ("rby1", "stretch", "sourccey"):
        spec = get_robot_spec(name)
        assert spec is not None, name
        assert spec.planar_spawn_robocasa_first_clearance_m is not None, name
