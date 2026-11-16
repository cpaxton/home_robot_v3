# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Strict stage-specific teacher readout, separate from a training receipt."""

from emet.rl.manipulation.tamp_primitive_env import PRIMITIVE_PHASE


def unloaded_stage_readiness(rows, cases, primitive):
    """Require every declared case; old pregrasp exceptions cannot admit grasp.

    This checks the stage outcomes only. Source-bound regressions, general-task
    admission and a separately versioned training protocol remain prerequisites.
    """
    if type(primitive) is not str or primitive not in PRIMITIVE_PHASE:
        raise ValueError("unsupported unloaded primitive")
    expected = [case["id"] for case in cases]
    if not expected or len(set(expected)) != len(expected) or [r.get("id") for r in rows] != expected:
        return False
    allowed = {None, "approach_clearance", "pregrasp"}
    if primitive == "grasp_pose":
        allowed.add("grasp")
    for row in rows:
        if (
            row.get("primitive") != primitive
            or row.get("success") is not True
            or row.get("code") != primitive + "_complete"
            or row.get("boundary_reached") is not True
            or row.get("terminal_pose_in_contract") is not True
            or row.get("gripper_open") is not True
            or row.get("stop_confirmed") is not True
            or row.get("initialization_failed") is not False
            or row.get("full_pick_place_success") is not False
            or row.get("decisions", 0) <= 0
            or not row.get("commands")
            or row.get("safety_flags")
            != dict.fromkeys(("wheel_lift", "invalid_physics", "unsafe_contact", "retention_lost"), False)
            or any(value is not False for value in row["safety_flags"].values())
            or row.get("inference_counts", {}).get("policy_calls") != 0
            or row.get("inference_counts", {}).get("reference_controller_calls", 0) <= 0
            or any(c.get("gripper") == "close" or c.get("phase") not in allowed for c in row["commands"])
        ):
            return False
    return True
