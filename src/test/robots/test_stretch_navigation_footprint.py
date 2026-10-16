# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import numpy as np

from emet.motion.kinematics import HelloStretchKinematics
from emet.robots import get_robot_spec
from emet.robots.footprint import Footprint


def test_stretch_spec_and_kinematics_share_stowed_envelope():
    spec = get_robot_spec("stretch").footprint
    model = HelloStretchKinematics.get_footprint()
    for field in ("width", "length", "width_offset", "length_offset"):
        assert getattr(spec, field) == getattr(model, field)
    lower = np.array([model.length_offset - model.length / 2, model.width_offset - model.width / 2])
    upper = np.array([model.length_offset + model.length / 2, model.width_offset + model.width / 2])
    # Robot-only collision mesh envelope, including 1 cm arm extension.
    assert np.all(lower <= [-0.297, -0.232])
    assert np.all(upper >= [0.095, 0.243])


def test_stowed_wrist_collision_is_not_outside_navigation_footprint():
    old = Footprint(width=0.34, length=0.33, length_offset=-0.1)
    new = HelloStretchKinematics.get_footprint()
    # A fixture at wrist height projects just outside the old base rectangle.
    # Physical navigation must check this cell even though wheels clear it.
    origin = np.array([50, 50])
    collision_cell = (48, 28)  # base-frame (-0.02, -0.22) m, 1 cm grid
    old_cells = {tuple(cell) for cell in old.grid_cells(0.01, [0, 0, 0], origin)}
    new_cells = {tuple(cell) for cell in new.grid_cells(0.01, [0, 0, 0], origin)}
    assert collision_cell not in old_cells
    assert collision_cell in new_cells
