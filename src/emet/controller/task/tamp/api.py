"""Versioned, semantic-only results shared by all agent-facing TAMP tools."""

from __future__ import annotations

import functools
import json
import math
from typing import Any

SCHEMA_VERSION = 1
TAMP_TOOLS = frozenset({"scene_tasks", "plan_pick_place", "execute_pick_place_plan", "pick_place"})


def response(
    tool: str,
    *,
    code: str = "ok",
    status: str | None = None,
    data: dict[str, Any] | None = None,
    recovery: str | None = None,
) -> dict[str, Any]:
    if recovery is None:
        recovery = (
            "none"
            if code == "ok"
            else "rediscover"
            if code
            in {
                "unknown_task_ref",
                "ambiguous_object",
                "ambiguous_receptacle",
                "metadata_unavailable",
                "object_not_found",
                "receptacle_not_found",
            }
            else "replan"
            if code in {"scene_changed_replan", "unknown_plan", "approach_changed_replan"}
            else "inspect"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "status": status or ("ok" if code == "ok" else "error"),
        "code": code,
        "message": code.replace("_", " "),
        "data": data or {},
        "recovery": recovery,
    }


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(v) for v in value]
    return value


def json_tool(func):
    """Serialize once, with the same exception contract on every tool path."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs) -> str:
        try:
            result = func(*args, **kwargs)
            if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("invalid tool result")
            return json.dumps(_finite(result), allow_nan=False)
        except ValueError as exc:
            code = str(exc) if str(exc) in {"server_identity_missing", "invalid pose"} else "invalid_input"
            return json.dumps(response(func.__name__, code=code.replace(" ", "_")))
        except Exception:
            # Exceptions may contain private body IDs or paths. Keep them out of tool results.
            return json.dumps(response(func.__name__, code="internal_error", recovery="inspect"))

    return wrapped


def plan_data(plan, mode: str | None, plan_ref: str | None = None) -> dict[str, Any]:
    return {
        "plan_ref": plan_ref,
        "mode": mode,
        "assistance": ["object_latch", "object_placement"]
        if mode == "kinematic"
        else ["object_placement"]
        if mode == "teleport"
        else [],
        "base_motion": None,
        "collision_scope": "base_endpoint_only",
        "planned_ops": [s.op for s in plan.steps] if plan else [],
        "completed_ops": list(plan.completed_ops) if plan else [],
        "failed_stage": plan.failed_op if plan else None,
        "measurements": getattr(plan, "measurements", None) if plan else None,
    }


def failure_code(message: str) -> str:
    """Translate internal machine reasons; never expose their free-form details."""
    message = {
        "ambiguous_object_use_scene_tasks": "ambiguous_object",
        "ambiguous_receptacle_use_scene_tasks": "ambiguous_receptacle",
    }.get(message, message)
    parts = message.split(":")
    for part in reversed(parts):
        if part in {"joint_bounds", "collision", "stale_observation"}:
            return part
        for suffix, code in (
            ("stale_observation", "stale_observation"),
            ("missing_joint_state", "stale_observation"),
            ("joint_bounds", "joint_bounds"),
            ("collision", "collision"),
            ("tracking_failed", "tracking_timeout"),
            ("ik_failed", "ik_failed"),
            ("planning_failed", "planning_failed"),
            ("verify_failed", "verification_failed"),
        ):
            if part.endswith(suffix):
                return code
    allowed = {
        "unknown_plan",
        "scene_changed_replan",
        "invalid_plan",
        "not_simulation",
        "kinematic_capability_missing",
        "teleport_capability_missing",
        "server_identity_missing",
        "approach_changed_replan",
        "approach_validation_failed",
        "approach_collision",
        "no_live_scene",
        "object_not_found",
        "receptacle_not_found",
        "object_not_in_live_scene",
        "receptacle_not_in_live_scene",
        "no_reachable_grasp",
        "no_grasp_candidates",
        "object_not_in_gt",
        "execution_error",
        "manipulation_unavailable",
        "unknown_task_ref",
        "ambiguous_object",
        "ambiguous_receptacle",
        "provide_task_ref_or_object_and_receptacle",
        "planner_error",
        "approach_failed",
    }
    return parts[0] if parts[0] in allowed else "operation_failed"
