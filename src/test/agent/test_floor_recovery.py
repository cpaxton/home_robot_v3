# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Coverage recovery remains observation-only until fresh mapping succeeds."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tools import Tool, get_tools
from emet.controller.dynamem import look


@pytest.mark.parametrize("status", ["insufficient_floor_coverage", "no_reachable_workspace", "workspace_obstructed"])
@pytest.mark.parametrize("payload_state", ["empty", "unknown"])
def test_pick_failure_exposes_reason_and_only_safe_recovery(status, payload_state):
    executor = Mock(return_value=True)
    executor._last_exec_ok = False
    executor.visual_servo = True
    executor.agent = SimpleNamespace(query_driven_memory=True)
    executor.last_query_manipulation = {
        "reason": "workspace rejected",
        "phase": "pre_grasp",
        "payload_state": payload_state,
        "observed_after_action": True,
        "navigation": {"status": status, "reachable_cells": 28},
    }
    tools = {tool.name: tool for tool in get_tools({"executor": executor})}
    later = Mock()
    tools["later"] = Tool("later", "Must not run", {}, later)
    recovery = []
    ok, results, failed = _dispatch_tool_calls(
        [{"name": "pick_place", "arguments": {"object_name": "cup", "receptacle_name": "table"}}, {"name": "later"}],
        tools,
        executor,
        recovery_tools=recovery,
    )
    assert ok and failed
    later.assert_not_called()
    assert status in results[0] and '"reachable_cells": 28' in results[0]
    assert recovery == (["observe_floor"] if payload_state == "empty" and status != "workspace_obstructed" else [])


@pytest.mark.parametrize("failure", [None, "stale", "pose", "depth", "map", "unsupported", "sequence"])
def test_floor_observation_requires_fresh_measured_capture(monkeypatch, failure):
    robot = SimpleNamespace(_seq_id=1, head_to=Mock(return_value=True))
    robot.get_pan_tilt = Mock(side_effect=[(0.2, -0.5), (0.2, 0 if failure == "pose" else -1)])
    robot.get_observation = Mock(
        return_value=SimpleNamespace(
            rgb=np.ones((2, 2, 3)),
            depth=None if failure == "depth" else np.ones((2, 2)),
            get_xyz_in_world_frame=lambda: np.ones((2, 2, 3)),
        )
    )
    agent = SimpleNamespace(robot=robot, voxel_map=SimpleNamespace(observations=[]))
    agent.update = Mock(side_effect=lambda **kw: None if failure == "map" else agent.voxel_map.observations.append(1))

    def receive(robot, timeout):
        if failure != "stale":
            robot._seq_id += 1

    monkeypatch.setattr(look, "wait_post_motion_obs", receive)
    if failure == "unsupported":
        robot.head_to = None
    if failure == "sequence":
        robot._seq_id = None
    result = look.observe_floor(agent)
    assert result["ok"] is (failure is None)
    if failure not in (None, "map"):
        agent.update.assert_not_called()
    if failure is None:
        robot.head_to.assert_called_once_with(0.2, -1.0, blocking=True)
        agent.update.assert_called_once_with(full_perception=True)
