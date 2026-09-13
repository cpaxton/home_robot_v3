# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from emet.controller.operations.query_observation import observe_query_points


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
    agent.ground_vlm_frame.return_value = (SimpleNamespace(instance=np.zeros((4, 4))), [], [0], {"valid": valid})
    with patch(
        "emet.controller.operations.query_observation.wait_post_motion_obs",
        side_effect=lambda *a, **k: setattr(robot, "_seq_id", 2 if fresh else 1),
    ):
        if fresh and valid:
            result_obs, points = observe_query_points(agent, robot, "mug", stage="place_alignment")
            assert result_obs is obs
            assert points.shape == (16, 3)
        else:
            with pytest.raises(ValueError):
                observe_query_points(agent, robot, "mug", stage="place_alignment")
    if not fresh:
        agent.ground_vlm_frame.assert_not_called()
    agent.prepare_query_target.assert_not_called()
