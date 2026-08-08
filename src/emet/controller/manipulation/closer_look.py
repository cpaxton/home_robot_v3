# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Closer-look / wrist-aim helper for the chat ``aim_arm_at`` tool.

Tries kinematic EE motion toward a localized object when the robot/manip stack
supports it; otherwise returns a structured ``not_implemented`` / ``localize_failed``
result for the attempt ledger. Does not claim success without moving the arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
