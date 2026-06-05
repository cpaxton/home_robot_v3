# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Ground-truth object placements for sim ZMQ session metadata (Robocasa, default table, MolmoSpaces)."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

# Canonical default-table scene (scene_environment.xml); used when no Robocasa wizard dict exists.
DEFAULT_TABLE_SCENE_PLACEMENTS: dict[str, dict[str, Any]] = {
    "table": {"cat": "table", "pos": [0.0, -1.0, 0.24], "quat": [1.0, 0.0, 0.0, 0.0]},
    "object1": {"cat": "blue cube", "pos": [-0.02, -0.55, 0.6], "quat": [1.0, 0.0, 0.0, 0.0]},
    "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6], "quat": [1.0, 0.0, 0.0, 0.0]},
}

_EMET_INTERNAL_KEYS = frozenset({"_emet_spawn_hint_xyt"})

# Bodies we never treat as scene-graph objects when scanning a merged MJCF.
_SKIP_BODY_RE = re.compile(
    r"(?i)^(world|floor|ground|sky|light|camera|visual|collision|robot|stretch|"
    r"base_link|link\d|wheel|caster|gripper|ee_|head_|d405|d435|"
    r"worldbody|default)$"
)


def _jsonify_placement_entry(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "cat" in info:
        out["cat"] = str(info["cat"])
    pos = info.get("pos")
    if pos is not None:
        out["pos"] = [float(x) for x in np.asarray(pos, dtype=np.float64).reshape(-1)[:3]]
    quat = info.get("quat")
    if quat is not None:
        out["quat"] = [float(x) for x in np.asarray(quat, dtype=np.float64).reshape(-1)[:4]]
    return out


def placements_to_session_dict(placements: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """Convert Robocasa / MuJoCo placement dicts to JSON-safe lists for ``emet_session``."""
    if not placements:
        return None
    out: dict[str, dict[str, Any]] = {}
    for body_name, info in placements.items():
        if body_name in _EMET_INTERNAL_KEYS or not isinstance(info, dict):
            continue
        entry = _jsonify_placement_entry(info)
        if entry.get("pos"):
            out[str(body_name)] = entry
    return out or None


def placements_from_objects_info(objects_info: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """Robocasa wizard ``object_placements_info`` → session placements."""
    return placements_to_session_dict(objects_info)


def _body_in_robot_subtree(model: mujoco.MjModel, body_id: int, robot_root_id: int) -> bool:
    if robot_root_id < 0:
        return False
    b = body_id
    for _ in range(model.nbody + 2):
        if b < 0:
            return False
        if b == robot_root_id:
            return True
        b = int(model.body_parentid[b])
    return False


def _label_from_body_name(body_name: str) -> str:
    if body_name == "object1":
        return "blue cube"
    if body_name == "object2":
        return "red cylinder"
    if body_name == "table":
        return "table"
    # MolmoSpaces / iTHOR: collapse instance suffix (Apple_1_2 → Apple_0_2 style token).
    parts = body_name.split("_")
    if len(parts) >= 2 and parts[-2].isdigit():
        parts[-2] = "0"
        body_name = "_".join(parts)
    return body_name.replace("_", " ").strip()


_DEFAULT_TABLE_KINDS = frozenset({None, "stretch_default_scene", "default_table"})


def is_default_table_environment(environment_kind: str | None) -> bool:
    """True for packaged default-table scenes (Stretch and Robosuite runtimes)."""
    return environment_kind in _DEFAULT_TABLE_KINDS


def assert_default_table_gt(placements: dict[str, dict[str, Any]] | None) -> None:
    """Raise AssertionError if default-table GT (table, blue cube, red cylinder) is missing."""
    if not placements:
        raise AssertionError("expected default-table sim_object_placements, got None")
    cats = " ".join(str(info.get("cat", "")).lower() for info in placements.values())
    keys = " ".join(placements.keys()).lower()
    if "table" not in cats and "table" not in keys:
        raise AssertionError(f"expected table in GT placements, got keys={list(placements.keys())!r}")
    if "blue" not in cats and "object1" not in keys:
        raise AssertionError(f"expected blue cube in GT placements, got cats={cats!r}")
    if "red" not in cats and "object2" not in keys:
        raise AssertionError(f"expected red cylinder in GT placements, got cats={cats!r}")


def placements_from_mujoco_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_root_name: str = "base_link",
    max_bodies: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Scan MuJoCo bodies for scene objects (MolmoSpaces merged MJCF, generic merges).

    Skips the robot kinematic subtree and bodies whose names look like structure / sensors.
    """
    robot_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name)
    out: dict[str, dict[str, Any]] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or name == "world":
            continue
        if _body_in_robot_subtree(model, bid, robot_root_id):
            continue
        if _SKIP_BODY_RE.match(name):
            continue
        # Require at least one non-visual geom on this body (skip empty grouping bodies).
        has_geom = False
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            has_geom = True
            break
        if not has_geom:
            continue
        pos = np.asarray(data.body(bid).xpos, dtype=np.float64).reshape(3)
        quat = np.asarray(data.body(bid).xquat, dtype=np.float64).reshape(4)
        out[name] = {
            "cat": _label_from_body_name(name),
            "pos": pos,
            "quat": quat,
        }
        if max_bodies is not None and len(out) >= max_bodies:
            break
    return out


def build_sim_object_placements_for_session(
    *,
    objects_info: dict[str, Any] | None,
    environment_kind: str | None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    robot_root_name: str = "base_link",
    max_scan_bodies: int | None = 48,
) -> dict[str, dict[str, Any]] | None:
    """
    Pick ground-truth placements for ``emet_session["sim_object_placements"]``.

    Priority: Robocasa wizard dict → default table constants → MuJoCo body scan (MolmoSpaces).
    """
    from_wizard = placements_from_objects_info(objects_info)
    if from_wizard:
        return from_wizard
    if is_default_table_environment(environment_kind):
        return placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS)
    if model is not None and data is not None:
        cap = max_scan_bodies if environment_kind == "molmospaces" else None
        scanned = placements_from_mujoco_model(model, data, robot_root_name=robot_root_name, max_bodies=cap)
        return placements_to_session_dict(scanned)
    return None


def attach_sim_object_placements_to_session(
    session: dict[str, Any],
    *,
    objects_info: dict[str, Any] | None,
    environment_kind: str | None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    robot_root_name: str = "base_link",
    max_scan_bodies: int | None = 48,
) -> None:
    """Set ``session["sim_object_placements"]`` when GT is available (mutates *session*)."""
    gt = build_sim_object_placements_for_session(
        objects_info=objects_info,
        environment_kind=environment_kind,
        model=model,
        data=data,
        robot_root_name=robot_root_name,
        max_scan_bodies=max_scan_bodies,
    )
    if gt:
        session["sim_object_placements"] = gt
