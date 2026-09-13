# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from emet.controller.operations.query_observation import observe_query_points, trim_depth_outliers


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_manipulation_geometry_requires_fresh_unique_semantics(fresh, valid, monkeypatch):
    monkeypatch.delenv("EMET_EQA_EPISODE_DIR", raising=False)
    world = np.ones((4, 4, 3))
    robot = Mock()
    robot._seq_id = 1
    obs = SimpleNamespace(rgb=np.zeros((4, 4, 3)), depth=np.ones((4, 4)), get_xyz_in_world_frame=lambda: world)
    robot.get_observation.return_value = obs
    agent = Mock()
    agent.ground_vlm_frame.return_value = (
        SimpleNamespace(instance=np.zeros((4, 4))),
        [],
        [0],
        {"valid": valid, "reason": "candidate budget exhausted"},
    )
    with patch(
        "emet.controller.operations.query_observation.wait_post_motion_obs",
        side_effect=lambda *a, **k: setattr(robot, "_seq_id", 2 if fresh else 1),
    ):
        if fresh and valid:
            result_obs, points = observe_query_points(agent, robot, "mug", stage="place_alignment")
            assert result_obs is obs
            assert points.shape == (16, 3)
        else:
            with pytest.raises(ValueError, match="candidate budget exhausted" if fresh else "Fresh"):
                observe_query_points(agent, robot, "mug", stage="place_alignment")
    if not fresh:
        agent.ground_vlm_frame.assert_not_called()
    agent.prepare_query_target.assert_not_called()


def test_minority_background_depth_does_not_define_object_extent():
    depth = np.linspace(0.8, 0.85, 100).reshape(10, 10)
    depth[-1] = 1.2
    mask, audit = trim_depth_outliers(depth, np.ones((10, 10), dtype=bool))
    assert mask.sum() == 90
    assert not mask[-1].any()
    assert audit["input_pixels"] == 100
    assert audit["retained_pixels"] == 90


def test_heavily_contaminated_mask_abstains_instead_of_selecting_tiny_support():
    depth = np.ones((10, 10))
    depth[:3] = 2
    with pytest.raises(ValueError, match="coherent depth"):
        trim_depth_outliers(depth, np.ones((10, 10), dtype=bool))


def test_planar_depth_and_small_sensor_noise_keep_supported_pixels():
    depth = np.ones((10, 10))
    depth[0] += 0.005
    mask, audit = trim_depth_outliers(depth, np.ones((10, 10), dtype=bool))
    assert mask.all()
    assert audit["band_m"] == 0.02
