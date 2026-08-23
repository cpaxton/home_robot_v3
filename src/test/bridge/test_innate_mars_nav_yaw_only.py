# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Nav2 yaw-only helper (no ROS msgs — runs without Humble on the workstation)."""

import math

import pytest
from innate_mars_bridge.nav_helpers import is_yaw_only_relative


@pytest.mark.parametrize(
    "xyt,expected",
    [
        ([0.0, 0.0, math.pi / 4], True),
        ([0.0, 0.0, -0.5], True),
        ([0.0, 0.0, 0.0], False),
        ([0.1, 0.0, 0.5], False),
        ([0.0, 0.05, 0.5], False),
        ([1e-4, -1e-4, 0.2], True),
    ],
)
def test_is_yaw_only_relative(xyt, expected):
    assert is_yaw_only_relative(xyt) is expected
