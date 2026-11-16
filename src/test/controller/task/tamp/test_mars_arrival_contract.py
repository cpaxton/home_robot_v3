# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""A task arrival region cannot change reference admission or tracking bounds."""

from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.task.tamp.mars_pose_executor import arrival_tolerances


def test_default_and_explicit_arrival_preserve_planning_tolerances():
    executor = SimpleNamespace(position_tolerance_m=0.003, orientation_tolerance_rad=0.02)
    assert arrival_tolerances(executor) == (0.003, 0.02)
    assert arrival_tolerances(executor, (0.004, 0.04)) == (0.004, 0.04)
    assert vars(executor) == {"position_tolerance_m": 0.003, "orientation_tolerance_rad": 0.02}


@pytest.mark.parametrize(
    "requested",
    [
        (0, 0.04),
        (-0.001, 0.04),
        (0.004, 0),
        (0.011, 0.04),
        (0.004, 0.11),
        (np.nan, 0.04),
        (0.004, np.inf),
        (),
        (0.004,),
        (0.004, 0.04, 1),
        ["0.004", "0.04"],
        (True, 0.04),
        (0.004, False),
        (0.004j, 0.04),
    ],
)
def test_arrival_cannot_disable_the_independent_tracking_tube(requested):
    executor = SimpleNamespace(position_tolerance_m=0.003, orientation_tolerance_rad=0.02)
    with pytest.raises(ValueError, match="arrival tolerances"):
        arrival_tolerances(executor, requested)


@pytest.mark.parametrize("preference,balanced", [(0, True), ("false", True), (False, False)])
def test_robust_reset_is_explicit_and_requires_balanced_setup(preference, balanced):
    from emet.rl.manipulation.env import ManipulationConfig

    with pytest.raises(ValueError, match="maximum-margin setup"):
        ManipulationConfig(robot="innate_mars", balanced_grasp_setup=balanced, grasp_setup_prefer_nominal=preference)
