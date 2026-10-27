# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Deterministic cross-tool cases; no model, simulator or fabricated sensor evidence."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.dynamem.look import observe_blocked_floor
from emet.controller.dynamem.relative_motion import execute_relative_motion


def make_agent(reason):
    state = {"mapped": False}
    pose = np.zeros(3)
    detail = {"reason": reason, "checked_pose": pose.tolist(), "unknown_footprint_cells": 1}
    agent = SimpleNamespace(
        robot=SimpleNamespace(_seq_id=1, head_to=Mock(), get_pan_tilt=Mock(), get_observation=Mock()),
        space=SimpleNamespace(obstacle_map_mode="physical", last_validity=detail),
        planner=SimpleNamespace(reset=Mock()),
        _current_planning_xyt=lambda: pose.copy(),
        _find_phase_nav_timeout=lambda: 5,
        _last_nav_plan={"footprint": {"checked_pose": [99, 99, 0]}},
    )

    def validate(p):
        np.testing.assert_allclose(p, pose)
        return state["mapped"] and reason != "occupied_footprint"

    def filter_route(route, **kwargs):
        agent._last_nav_sweep_failure = detail
        return route, None if validate(pose) else f"rejected_swept_footprint:{reason}", None

    def execute(route, timeout):
        pose[:] = route[-1]
        return True, None

    agent.space.is_valid = Mock(side_effect=validate)
    agent._filter_unsafe_nav_traj = filter_route
    agent._execute_validated_waypoints = Mock(side_effect=execute)
    return agent, state


@pytest.mark.parametrize("tool,value", [("move_forward", 0.2), ("rotate_base", 30)])
def test_unknown_floor_handoff_rechecks_same_pose_and_requires_replanning(tool, value):
    agent, state = make_agent("unobserved_footprint")
    old_plan = agent._last_nav_plan
    rejected = execute_relative_motion(agent, tool, value)
    assert rejected["code"] == "motion_rejected"
    assert agent._last_nav_plan is not old_plan
    agent._execute_validated_waypoints.assert_not_called()
    # Only explicit test mapping progress changes validity. The recovery tool
    # must not clear unknown cells, and sensing is not movement authorization.
    state["mapped"] = True
    recovery = observe_blocked_floor(agent)
    assert recovery["status"] == "checked_pose_valid_replan_required"
    agent._execute_validated_waypoints.assert_not_called()
    assert execute_relative_motion(agent, tool, value)["code"] == "ok"


def test_real_obstruction_is_not_cleared_by_floor_recovery():
    agent, state = make_agent("occupied_footprint")
    assert execute_relative_motion(agent, "move_forward", 0.2)["code"] == "motion_rejected"
    state["mapped"] = True
    assert observe_blocked_floor(agent)["status"] == "no_unknown_floor_target"
    assert execute_relative_motion(agent, "move_forward", 0.2)["code"] == "motion_rejected"
    agent._execute_validated_waypoints.assert_not_called()


def test_rejection_receipt_cannot_mutate_recovery_plan():
    agent, _ = make_agent("unobserved_footprint")
    out = execute_relative_motion(agent, "move_forward", 0.2)
    out["data"]["footprint"]["checked_pose"][0] = 99
    assert agent._last_nav_plan["footprint"]["checked_pose"] == [0, 0, 0]


@pytest.mark.parametrize("present,depth", [(True, 3.3), (False, 1.0)])
def test_visible_far_or_absent_target_does_not_create_geometry(present, depth):
    from emet.memory.vlm_region_grounding import ground_vlm_region

    reply = (
        '{"verified":true,"box":[200,200,800,800],"point":[500,500]}'
        if present
        else '{"verified":false,"reason":"not visible"}'
    )
    frame = SimpleNamespace(
        rgb=np.zeros((40, 40, 3), dtype=np.uint8), depth=np.full((40, 40), depth), full_world_xyz=np.ones((40, 40, 3))
    )
    client = Mock(return_value=reply)
    _, detections, ids, audit = ground_vlm_region(
        frame, "target", "target", client=client, min_depth=0.25, max_depth=2.5
    )
    assert not audit["valid"]
    assert detections == [] and ids == []
    if not present:
        client.assert_called_once()  # No forced retry to turn semantic abstention into acceptance.


@pytest.mark.parametrize("present,status", [(True, "VISUAL_PRESENT"), (False, "VISUAL_ABSENT"), (None, "NOT_ASSESSED")])
def test_retrieval_miss_cannot_override_visual_evidence(present, status):
    from emet.memory.graph_eqa.agentic.verify import verification_feedback

    feedback = verification_feedback(
        "ABSENT", {"ok": True, "present": present}, {"view_obs_id": 7, "ok": False, "reason": "no supported surface"}, 7
    )
    assert feedback["status"] == status
    assert feedback["retrieval"]["status"] == "ABSENT"
    assert feedback["geometry"]["status"] == "not_admitted"


def test_range_feedback_reaches_model_without_becoming_visual_absence():
    from emet.memory.graph_eqa.agentic.verify import verification_feedback

    support = {"semantics": "unverified proposals, not localized targets", "max_depth_m": 2.5}
    grounding = {"view_obs_id": 7, "ok": False, "support_reason": "no supported surfaces", "depth_support": support}
    feedback = verification_feedback("ABSENT", {"ok": True, "present": True}, grounding, 7)
    assert feedback["status"] == "VISUAL_PRESENT"
    assert feedback["geometry"]["depth_support"] == support
    assert feedback["geometry"]["support_reason"] == "no supported surfaces"
    assert feedback["geometry"]["status"] == "not_admitted"
    assert verification_feedback("ABSENT", {"ok": True, "present": True}, grounding, 8)["geometry"] == {
        "status": "not_attempted"
    }


def test_stale_geometry_is_not_attached_to_current_assessment():
    from emet.memory.graph_eqa.agentic.verify import verification_feedback

    assert (
        verification_feedback("PRESENT", None, {"view_obs_id": 6, "ok": True}, 7)["geometry"]["status"]
        == "not_attempted"
    )


@pytest.mark.parametrize("tool", ["rotate_base", "move_forward"])
def test_failed_relative_motion_requires_new_model_decision(tool):
    from emet.memory.graph_eqa.agentic.run import requires_motion_replan

    assert requires_motion_replan(tool, {"ok": False, "code": "motion_rejected"})
    assert not requires_motion_replan(tool, {"ok": True})
