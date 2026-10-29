# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""An unavailable viewing pose must not crash a valid uncertain voxel answer."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from emet.controller.dynamem.eqa import run_eqa_one_iter


@pytest.mark.parametrize("sample,theta", [(None, None), (np.array([0.3, 0.2, 0.7]), 0.7)])
def test_missing_prefacing_viewpoint_leaves_failure_handling_to_navigation(sample, theta):
    point = np.array([0.1, 0.1, 0.8])
    navigate = Mock(return_value=SimpleNamespace(finished=False))
    agent = SimpleNamespace(
        _realtime_updates=True,
        robot=SimpleNamespace(get_base_pose=lambda: np.zeros(3)),
        _planning_base_xyt=np.asarray,
        planner=object(),
        voxel_map=SimpleNamespace(
            query_answer=lambda *args: ("not visible", "Unknown", False, "need another view", point, []),
            get_2d_map=lambda: (torch.zeros((5, 5), dtype=torch.bool), torch.ones((5, 5), dtype=torch.bool)),
            xy_to_grid_coords=lambda xy: (1, 1),
        ),
        space=SimpleNamespace(sample_navigation=lambda *args: sample),
        _rerun_refresh_monologue_panel=lambda: None,
        update=lambda: None,
        navigate_to_target_pose=navigate,
    )
    result = run_eqa_one_iter(agent, "Where is the object?", max_movement_step=1)
    assert result[0] == "Unknown" and result[3] is False
    assert agent._last_eqa_answer == "Unknown"
    assert navigate.call_count == 1
    assert navigate.call_args.args[2] == theta
