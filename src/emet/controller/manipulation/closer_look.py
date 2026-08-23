# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Closer-look / wrist-aim helper for the chat ``aim_arm_at`` tool.

Tries kinematic EE motion toward a localized object when the robot/manip stack
supports it; otherwise returns a structured ``not_implemented`` / ``localize_failed``
result for the attempt ledger. Does not claim success without moving the arm.

Successful aims are recorded on the agent/context so ``take_ee_picture`` may capture
the wrist camera only after a real aim (one capture consumes the aim grant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Shared key on agent + chat tool context after a successful ``aim_arm_at``.
LAST_CLOSER_LOOK_AIM_KEY = "_last_closer_look_aim"
# Last aim attempt (success or failure) — used to soft-allow EE when aim is not_implemented.
LAST_CLOSER_LOOK_ATTEMPT_KEY = "_last_closer_look_attempt"


@dataclass
class CloserLookResult:
    ok: bool
    status_code: str
    note: str
    xyz: tuple[float, float, float] | None = None
    phrase: str = ""

    def to_tool_outcome(self):
        from emet.agent.tool_outcome import ToolOutcome

        return ToolOutcome(
            ok=bool(self.ok),
            status=self.status_code,
            note=self.note,
            tool="aim_arm_at",
            payload={
                "phrase": self.phrase,
                "action_kind": "closer_look",
                "xyz": list(self.xyz) if self.xyz is not None else None,
            },
        )


def _clear_attr(agent: Any, context: dict[str, Any] | None, key: str) -> None:
    if agent is not None and hasattr(agent, key):
        try:
            delattr(agent, key)
        except Exception:
            setattr(agent, key, None)
    if context is not None:
        context.pop(key, None)


def clear_closer_look_aim(agent: Any = None, context: dict[str, Any] | None = None) -> None:
    """Drop any outstanding aim grant (failed aim or after EE capture)."""
    _clear_attr(agent, context, LAST_CLOSER_LOOK_AIM_KEY)


def clear_closer_look_attempt(agent: Any = None, context: dict[str, Any] | None = None) -> None:
    """Drop the last aim attempt marker."""
    _clear_attr(agent, context, LAST_CLOSER_LOOK_ATTEMPT_KEY)


def record_closer_look_aim(
    result: CloserLookResult,
    *,
    agent: Any = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Record the aim attempt; grant one EE capture only on success.

    Failed aims clear the grant. ``not_implemented`` is remembered so
    ``take_ee_picture`` can soft-allow once on stacks without kinematic aim.
    """
    attempt = {
        "ok": bool(result.ok),
        "phrase": str(result.phrase or ""),
        "status_code": str(result.status_code or ""),
        "xyz": list(result.xyz) if result.xyz is not None else None,
    }
    if agent is not None:
        setattr(agent, LAST_CLOSER_LOOK_ATTEMPT_KEY, attempt)
    if context is not None:
        context[LAST_CLOSER_LOOK_ATTEMPT_KEY] = attempt

    if not result.ok:
        clear_closer_look_aim(agent, context)
        return
    payload = {
        "ok": True,
        "phrase": str(result.phrase or ""),
        "status_code": str(result.status_code or "ok"),
        "xyz": list(result.xyz) if result.xyz is not None else None,
    }
    if agent is not None:
        setattr(agent, LAST_CLOSER_LOOK_AIM_KEY, payload)
    if context is not None:
        context[LAST_CLOSER_LOOK_AIM_KEY] = payload


def get_closer_look_aim(agent: Any = None, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the outstanding successful aim grant, if any."""
    for src in (agent, context):
        if src is None:
            continue
        raw = (
            getattr(src, LAST_CLOSER_LOOK_AIM_KEY, None)
            if not isinstance(src, dict)
            else src.get(LAST_CLOSER_LOOK_AIM_KEY)
        )
        if isinstance(raw, dict) and raw.get("ok"):
            return raw
    return None


def get_closer_look_attempt(agent: Any = None, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the last aim attempt payload, if any."""
    for src in (agent, context):
        if src is None:
            continue
        raw = (
            getattr(src, LAST_CLOSER_LOOK_ATTEMPT_KEY, None)
            if not isinstance(src, dict)
            else src.get(LAST_CLOSER_LOOK_ATTEMPT_KEY)
        )
        if isinstance(raw, dict) and raw.get("status_code"):
            return raw
    return None


def consume_closer_look_aim_for_ee_picture(
    *,
    agent: Any = None,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Allow one wrist capture after a successful aim; consume the grant.

    Soft-allows once when the last ``aim_arm_at`` returned ``not_implemented``
    (real Mars / stacks without kinematic aim) so wrist dogfood is not bricked.

    Returns ``(allowed, note, aim_payload)``.
    """
    aim = get_closer_look_aim(agent, context)
    if aim is not None:
        clear_closer_look_aim(agent, context)
        clear_closer_look_attempt(agent, context)
        phrase = str(aim.get("phrase") or "").strip()
        note = (
            f"Capturing wrist camera after aim at {phrase!r}." if phrase else "Capturing wrist camera after aim_arm_at."
        )
        return True, note, aim

    attempt = get_closer_look_attempt(agent, context)
    if attempt is not None and str(attempt.get("status_code") or "") == "not_implemented":
        clear_closer_look_attempt(agent, context)
        phrase = str(attempt.get("phrase") or "").strip()
        note = "Capturing wrist camera (aim not available on this stack"
        if phrase:
            note += f"; localized {phrase!r}"
        note += ")."
        return True, note, attempt

    return (
        False,
        (
            "take_ee_picture requires aim_arm_at first "
            "(successful aim grants one capture; not_implemented aim soft-allows once). "
            "Prefer face_toward + describe_scene (head camera) when the wrist stream is dark."
        ),
        None,
    )


def _localize_xyz(agent: Any, phrase: str) -> np.ndarray | None:
    if agent is None:
        return None
    # Prefer graph / memory localize when available.
    for attr in ("_localize_point_from_graph_memory", "localize_object"):
        fn = getattr(agent, attr, None)
        if callable(fn):
            try:
                pt = fn(phrase)
            except Exception:
                pt = None
            if pt is not None:
                arr = np.asarray(pt, dtype=float).reshape(-1)
                if arr.size >= 3:
                    return arr[:3]
    vm = None
    get_vm = getattr(agent, "get_voxel_map", None)
    if callable(get_vm):
        vm = get_vm()
    vm = vm or getattr(agent, "voxel_map", None)
    if vm is not None and hasattr(vm, "localize_text"):
        try:
            result = vm.localize_text(phrase, return_debug=True)
            point = result[0] if isinstance(result, (list, tuple)) else result
        except Exception:
            point = None
        if point is not None:
            arr = np.asarray(point, dtype=float).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
    return None


def _kinematic_aim_available(robot: Any, *, manip_mode: str, visual_servo: bool) -> bool:
    try:
        from emet.simulation.sim_manipulation import prefer_kinematic_manip

        return bool(prefer_kinematic_manip(robot, manip_mode=manip_mode, visual_servo=visual_servo))
    except Exception:
        return False


def aim_wrist_at_phrase(
    *,
    agent: Any,
    robot: Any,
    phrase: str,
    manip_mode: str = "teleport",
    visual_servo: bool = False,
    manip_collision: str = "none",
    manip_planner: str = "rrt_connect",
) -> CloserLookResult:
    """Localize ``phrase`` and, when kinematic manip is available, aim EE near it."""
    label = str(phrase or "").strip()
    if not label:
        return CloserLookResult(False, "empty_phrase", "aim_arm_at requires an object_label.", phrase=label)

    xyz = _localize_xyz(agent, label)
    if xyz is None:
        return CloserLookResult(
            False,
            "localize_failed",
            f"Could not localize {label!r} in memory/map — try describe_scene or explore first.",
            phrase=label,
        )
    xyz_t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    if not _kinematic_aim_available(robot, manip_mode=manip_mode, visual_servo=visual_servo):
        return CloserLookResult(
            False,
            "not_implemented",
            (
                f"Localized {label!r} at ({xyz_t[0]:.2f}, {xyz_t[1]:.2f}, {xyz_t[2]:.2f}) but "
                "wrist aim needs kinematic manip mode (not available on this robot/stack). "
                "Use face_toward + describe_scene (head camera) for a closer look."
            ),
            xyz=xyz_t,
            phrase=label,
        )

    try:
        from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

        voxel_map = None
        get_vm = getattr(agent, "get_voxel_map", None)
        if callable(get_vm):
            voxel_map = get_vm()
        exe = KinematicPickPlaceExecutor(
            robot,
            manip_collision=manip_collision,
            manip_planner=manip_planner,
            voxel_map=voxel_map,
        )
        # Aim slightly above the object so the wrist looks down at it.
        look_xyz = np.asarray(xyz_t, dtype=float) + np.array([0.0, 0.0, 0.20])
        plan_fn = getattr(exe, "_plan_and_execute_ee", None)
        if not callable(plan_fn):
            return CloserLookResult(
                False,
                "not_implemented",
                "Kinematic executor has no EE plan path for closer look.",
                xyz=xyz_t,
                phrase=label,
            )
        ok, err = plan_fn(look_xyz)
        if not ok:
            return CloserLookResult(
                False,
                "ik_unreachable",
                f"IK/plan failed aiming at {label!r} (err={err}). Try face_toward + describe_scene.",
                xyz=xyz_t,
                phrase=label,
            )
        return CloserLookResult(
            True,
            "ok",
            f"Aimed wrist near {label!r} at ({xyz_t[0]:.2f}, {xyz_t[1]:.2f}, {xyz_t[2]:.2f}). "
            "You can take_ee_picture now.",
            xyz=xyz_t,
            phrase=label,
        )
    except Exception as e:
        return CloserLookResult(
            False,
            "aim_failed",
            f"Closer-look aim failed for {label!r}: {e}",
            xyz=xyz_t,
            phrase=label,
        )
