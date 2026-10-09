# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Base planning must not initialize another simulator's OpenGL stack."""

import subprocess
import sys

import numpy as np
import pytest

from emet.motion.grid_coordinates import world_xy_to_grid


def test_astar_import_does_not_load_simulator_or_opengl():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import emet.motion.algo.a_star; "
            "assert not any(m == 'mujoco' or m.startswith('OpenGL') for m in sys.modules)",
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("convention, expected", [("grid_params", (12, -3)), ("world_offset", (-18, 3))])
def test_shared_grid_conventions(convention, expected):
    assert (
        world_xy_to_grid(1.2, -0.2, grid_origin=np.array([10, -2]), resolution=0.5, convention=convention) == expected
    )


def test_arm_import_keeps_compatibility_alias():
    from emet.motion.voxel_arm_collision import world_xy_to_grid as arm_conversion

    assert arm_conversion is world_xy_to_grid
