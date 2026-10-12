# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Semantic, explicitly assisted articulation actions for CHAT composition."""

from __future__ import annotations

import time
import uuid

from emet.controller.task.tamp.api import response
from emet.simulation.articulation import articulation_for_body


def set_receptacle_state(robot, context, task_ref, state, *, timeout_s=10.0):
    from emet.controller.task.tamp.agent_bridge import robot_session_key
    from emet.core.zmq_protocol import build_sim_set_joint_qpos_action
    from emet.simulation.physical_execution import physical_execution_enabled

    tool = "set_receptacle_state"
    evidence = {
        "assistance": ["joint_teleport"],
        "collision_scope": "none",
        "requested_state": state,
        "verified": False,
    }

    def result(code, *, partial=False):
        return response(tool, code=code, status="partial" if partial else None, data=evidence)

    if state not in {"open", "closed"}:
        return result("invalid_articulation_state")
    if robot is None or not callable(getattr(robot, "get_emet_session", None)):
        return result("robot_not_connected")
    session = robot.get_emet_session()
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return result("not_simulation")
    caps = session.get("capabilities") or {}
    if physical_execution_enabled() or not (caps.get("sim_set_joint_qpos") and caps.get("sim_articulation_state")):
        return result("articulation_unsupported")
    key = robot_session_key(robot)
    if not key or not key[-1] or key != context.get("_tamp_scene_key"):
        return result("scene_changed_replan")
    task = context.get("_tamp_task_refs", {}).get(str(task_ref))
    if task is None:
        return result("unknown_task_ref")
    group = articulation_for_body(session, task.receptacle_body)
    if not group or not group.get("supported"):
        return result("articulation_unsupported")
    request_id = uuid.uuid4().hex
    action = build_sim_set_joint_qpos_action(
        max(1, int(getattr(robot, "_last_step", -1)) + 1), group["joint"], group[state]
    )
    action["sim_set_joint_qpos"]["request_id"] = request_id
    # An uncertain command can have partial effects. Never reuse old grounded plans.
    context["_tamp_plans"] = {}
    try:
        robot.send_action(action, reliable=True)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            current = robot.get_emet_session()
            if robot_session_key(robot) != key:
                return result("scene_changed_replan", partial=True)
            receipt = current.get("articulation_result") or {}
            measured = articulation_for_body(current, task.receptacle_body)
            if receipt.get("request_id") == request_id and receipt.get("joint") == group["joint"]:
                if not receipt.get("applied"):
                    return result("articulation_command_failed", partial=True)
                if measured and measured.get("state") == state:
                    evidence.update(verified=True, observed_state=state, convention=measured.get("convention"))
                    return result("ok")
            time.sleep(0.05)
    except (RuntimeError, TimeoutError, ValueError, KeyError, TypeError):
        return result("articulation_command_failed", partial=True)
    return result("articulation_verification_timeout", partial=True)
