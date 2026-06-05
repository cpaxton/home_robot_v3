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

from emet.simulation.molmo_occupancy._geom_aabb import geom_aabb

# All ``sim_object_placements[*].pos`` values are absolute MuJoCo world XYZ (meters).
SIM_OBJECT_PLACEMENTS_FRAME = "mujoco_world"

# Canonical default-table scene (scene_environment.xml); used when no Robocasa wizard dict exists.
DEFAULT_TABLE_SCENE_PLACEMENTS: dict[str, dict[str, Any]] = {
    "table": {"cat": "table", "pos": [0.0, -1.0, 0.24], "quat": [1.0, 0.0, 0.0, 0.0]},
    "object1": {"cat": "blue cube", "pos": [-0.02, -0.55, 0.6], "quat": [1.0, 0.0, 0.0, 0.0]},
    "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6], "quat": [1.0, 0.0, 0.0, 0.0]},
}

_EMET_INTERNAL_KEYS = frozenset({"_emet_spawn_hint_xyt"})

# Bodies we never treat as scene-graph objects when scanning a merged MJCF (per-body mode).
_SKIP_BODY_RE = re.compile(
    r"(?i)^(world|floor|ground|sky|light|camera|visual|collision|robot|stretch|"
    r"base_link|link\d|wheel|caster|gripper|ee_|head_|d405|d435|"
    r"worldbody|default)$"
)

# Robocasa fixture groups we skip (room shell / utilities, not semantic map labels).
_FIXTURE_GROUP_SKIP_RE = re.compile(r"(?i)^(wall|floor|ground|sky|light|outlet|light_switch|world|default|worldbody)")


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
    bounds = info.get("bounds")
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        out["bounds"] = [[float(x) for x in b[0]], [float(x) for x in b[1]]]
    else:
        aabb_min = info.get("aabb_min")
        aabb_max = info.get("aabb_max")
        if aabb_min is not None and aabb_max is not None:
            mn = np.asarray(aabb_min, dtype=np.float64).reshape(3)
            mx = np.asarray(aabb_max, dtype=np.float64).reshape(3)
            out["bounds"] = [[float(x) for x in mn], [float(x) for x in mx]]
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


def wizard_category_map(objects_info: dict[str, Any] | None) -> dict[str, str]:
    """Map MuJoCo body / group keys to Robocasa ``cat`` labels from the wizard dict."""
    if not objects_info:
        return {}
    out: dict[str, str] = {}
    for key, info in objects_info.items():
        if key in _EMET_INTERNAL_KEYS or not isinstance(info, dict):
            continue
        cat = info.get("cat")
        if cat:
            out[str(key)] = str(cat)
    return out


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


def fixture_group_key(body_name: str) -> str | None:
    """
    Robocasa / merged MJCF: collapse bodies into one semantic entity key.

    Examples: ``sink_main_group_basin_main`` → ``sink``; ``obj_main`` → ``obj_main``.
    """
    if not body_name or body_name == "world":
        return None
    if _FIXTURE_GROUP_SKIP_RE.match(body_name):
        return None
    if "_main_group_" in body_name:
        return body_name.split("_main_group_")[0]
    if body_name.endswith("_main"):
        return body_name
    return None


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


def _label_from_fixture_group(group_key: str, wizard_cats: dict[str, str]) -> str:
    """Human-readable label from the MuJoCo fixture group key (wizard ``cat`` wins for manipulables)."""
    if group_key in wizard_cats:
        return wizard_cats[group_key]
    token = str(group_key)
    if token.lower().startswith("distr_"):
        token = token[6:]
    if token.endswith("_main"):
        token = token[:-5]
    return token.replace("_", " ").strip() or group_key


def _collect_fixture_groups(
    model: mujoco.MjModel,
    *,
    robot_root_name: str = "base_link",
) -> dict[str, list[int]]:
    robot_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name)
    groups: dict[str, list[int]] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        if _body_in_robot_subtree(model, bid, robot_root_id):
            continue
        gk = fixture_group_key(name)
        if gk is None:
            continue
        groups.setdefault(gk, []).append(bid)
    return groups


def _geom_ids_for_bodies(model: mujoco.MjModel, body_ids: list[int]) -> list[int]:
    body_set = frozenset(body_ids)
    gids: list[int] = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid not in body_set:
            continue
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        gids.append(gid)
    return gids


def _world_aabb_for_geom_ids(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(center, bounds_2x3, quat_wxyz)`` in MuJoCo world frame, or ``None`` if empty."""
    if not geom_ids:
        return None
    center, size = geom_aabb(model, data, geom_ids, tight_mesh=True)
    half = np.asarray(size, dtype=np.float64).reshape(3) / 2.0
    if float(np.max(half)) <= 1e-6:
        return None
    bounds = np.stack([center - half, center + half], axis=0)
    # Use the first body's orientation as a nominal object frame (axis-aligned box in Rerun).
    bid = int(model.geom_bodyid[geom_ids[0]])
    quat = np.asarray(data.body(bid).xquat, dtype=np.float64).reshape(4)
    return np.asarray(center, dtype=np.float64).reshape(3), bounds, quat


def _placement_entry_from_geom_ids(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
    *,
    cat: str,
) -> dict[str, Any] | None:
    aabb = _world_aabb_for_geom_ids(model, data, geom_ids)
    if aabb is None:
        return None
    center, bounds, quat = aabb
    return {
        "cat": cat,
        "pos": center,
        "quat": quat,
        "bounds": bounds,
    }


def placements_from_mujoco_fixture_groups(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_root_name: str = "base_link",
    wizard_cats: dict[str, str] | None = None,
    max_groups: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Robocasa-style scan: one GT entry per fixture (sink, counter, cabinets, …) plus manipulable ``*_main`` bodies.

    Aggregates mesh/collision geoms across all bodies in each fixture group and stores world AABB ``bounds``.
    """
    wizard_cats = wizard_cats or {}
    _mj_forward(model, data)
    groups = _collect_fixture_groups(model, robot_root_name=robot_root_name)
    out: dict[str, dict[str, Any]] = {}
    for group_key in sorted(groups.keys()):
        if max_groups is not None and len(out) >= max_groups:
            break
        geom_ids = _geom_ids_for_bodies(model, groups[group_key])
        cat = _label_from_fixture_group(group_key, wizard_cats)
        entry = _placement_entry_from_geom_ids(model, data, geom_ids, cat=cat)
        if entry is not None:
            out[group_key] = entry
    return out


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
    wizard_cats: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Scan MuJoCo bodies for scene objects (MolmoSpaces merged MJCF, generic merges).

    Skips the robot kinematic subtree and bodies whose names look like structure / sensors.
    """
    wizard_cats = wizard_cats or {}
    robot_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name)
    _mj_forward(model, data)
    out: dict[str, dict[str, Any]] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or name == "world":
            continue
        if _body_in_robot_subtree(model, bid, robot_root_id):
            continue
        if _SKIP_BODY_RE.match(name):
            continue
        geom_ids = _geom_ids_for_bodies(model, [bid])
        if not geom_ids:
            continue
        cat = wizard_cats.get(name) or _label_from_body_name(name)
        entry = _placement_entry_from_geom_ids(model, data, geom_ids, cat=cat)
        if entry is None:
            continue
        out[name] = entry
        if max_bodies is not None and len(out) >= max_bodies:
            break
    return out


def merge_scene_placements(
    wizard: dict[str, dict[str, Any]] | None,
    scanned: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """
    Merge wizard manipulable-object labels with a full MJCF scan.

    Wizard ``cat`` (and optional pose hints) win on key collision; scan adds fixtures not in the wizard.
    """
    merged: dict[str, dict[str, Any]] = {}
    if scanned:
        merged.update(scanned)
    if wizard:
        for key, info in wizard.items():
            if key in _EMET_INTERNAL_KEYS or not isinstance(info, dict):
                continue
            base = dict(merged.get(key, {}))
            base.update(info)
            if base.get("pos"):
                merged[str(key)] = base
    return merged or None


def _mj_forward(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_forward(model, data)


def overlay_live_mujoco_body_poses(
    placements: dict[str, dict[str, Any]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_root_name: str = "base_link",
    wizard_cats: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """
    Refresh each placement ``pos``/``quat``/``bounds`` from live MuJoCo state.

    Uses fixture-group aggregation when the key matches a Robocasa group; otherwise falls back to a single body.
    """
    if not placements:
        return None
    _mj_forward(model, data)
    wizard_cats = wizard_cats or {}
    groups = _collect_fixture_groups(model, robot_root_name=robot_root_name)
    raw: dict[str, dict[str, Any]] = {}
    for key, info in placements.items():
        if key in _EMET_INTERNAL_KEYS or not isinstance(info, dict):
            continue
        entry = dict(info)
        cat = str(entry.get("cat") or wizard_cats.get(key) or _label_from_fixture_group(key, wizard_cats))
        if key in groups:
            geom_ids = _geom_ids_for_bodies(model, groups[key])
            live = _placement_entry_from_geom_ids(model, data, geom_ids, cat=cat)
            if live:
                entry.update(live)
        else:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(key))
            if bid >= 0:
                geom_ids = _geom_ids_for_bodies(model, [bid])
                live = _placement_entry_from_geom_ids(model, data, geom_ids, cat=cat)
                if live:
                    entry.update(live)
                else:
                    entry["pos"] = np.asarray(data.body(bid).xpos, dtype=np.float64).reshape(3)
                    entry["quat"] = np.asarray(data.body(bid).xquat, dtype=np.float64).reshape(4)
        merged = _jsonify_placement_entry(entry)
        if merged.get("pos"):
            raw[str(key)] = merged
    return raw or None


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

    Robocasa: full fixture scan (sink, counter, cabinets, …) merged with wizard manipulable objects.
    Default table: packaged constants. MolmoSpaces / generic: per-body MJCF scan.
    When ``model``/``data`` are available, refresh poses and world AABBs from live MuJoCo state.
    """
    wizard = placements_from_objects_info(objects_info)
    wizard_cats = wizard_category_map(objects_info)
    base: dict[str, dict[str, Any]] | None = None

    if model is not None and data is not None:
        if environment_kind == "robocasa" or wizard:
            scanned = placements_from_mujoco_fixture_groups(
                model,
                data,
                robot_root_name=robot_root_name,
                wizard_cats=wizard_cats,
                max_groups=max_scan_bodies,
            )
            base = merge_scene_placements(wizard, scanned)
        elif is_default_table_environment(environment_kind):
            base = merge_scene_placements(
                placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS),
                placements_from_mujoco_model(
                    model,
                    data,
                    robot_root_name=robot_root_name,
                    wizard_cats=wizard_cats,
                ),
            )
        else:
            cap = max_scan_bodies if environment_kind == "molmospaces" else None
            scanned = placements_from_mujoco_model(
                model,
                data,
                robot_root_name=robot_root_name,
                max_bodies=cap,
                wizard_cats=wizard_cats,
            )
            base = merge_scene_placements(wizard, scanned)
    elif wizard:
        base = wizard
    elif is_default_table_environment(environment_kind):
        base = placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS)

    if base and model is not None and data is not None:
        live = overlay_live_mujoco_body_poses(
            base,
            model,
            data,
            robot_root_name=robot_root_name,
            wizard_cats=wizard_cats,
        )
        if live:
            return live
    return base


def mujoco_model_data_for_gt_scan(robot_sim: Any) -> tuple[mujoco.MjModel | None, mujoco.MjData | None]:
    """
    Resolve ``(model, data)`` for GT MJCF scan on any sim server.

    Stretch runs MuJoCo in a subprocess and exposes the loaded scene as ``robot_sim.model`` (not
    ``mjmodel``). Robosuite uses in-process ``mjmodel`` / ``mjdata``. When only a model is available,
    builds a forward-passed ``MjData`` at ``qpos0`` (sufficient for static fixture GT at load).
    """
    model = getattr(robot_sim, "mjmodel", None)
    if model is None:
        model = getattr(robot_sim, "model", None)
    if model is None:
        return None, None
    data = getattr(robot_sim, "mjdata", None)
    if data is None:
        data = mujoco.MjData(model)
        _mj_forward(model, data)
    return model, data


def apply_navigation_origin_to_session(
    session: dict[str, Any],
    initial_xyt_world: np.ndarray | list[float] | tuple[float, ...],
) -> None:
    """
    Record spawn-world SE(2) in ``emet_session`` so nav-relative ``gps``/``compass`` compose to MuJoCo world.

    Matches Robosuite ZMQ contract: ``sim_object_placements`` stay absolute world; clients use
    :func:`emet.utils.geometry.nav_xyt_to_world_xyt` for Rerun / sanity checks.
    """
    ixy = np.asarray(initial_xyt_world, dtype=np.float64).reshape(-1)[:3]
    session["navigation_origin_xyt"] = [float(ixy[0]), float(ixy[1]), float(ixy[2])]
    if session.get("sim_object_placements"):
        session["sim_object_placements_note"] = (
            "pos is MuJoCo world XYZ; ZMQ gps/compass are episode-relative to navigation_origin_xyt"
        )


def attach_sim_object_placements_to_session(
    session: dict[str, Any],
    *,
    objects_info: dict[str, Any] | None,
    environment_kind: str | None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    robot_root_name: str = "base_link",
    max_scan_bodies: int | None = None,
) -> None:
    """Set ``session["sim_object_placements"]`` when GT is available (mutates *session*)."""
    scan_cap = max_scan_bodies
    if scan_cap is None and environment_kind == "molmospaces":
        scan_cap = 128
    gt = build_sim_object_placements_for_session(
        objects_info=objects_info,
        environment_kind=environment_kind,
        model=model,
        data=data,
        robot_root_name=robot_root_name,
        max_scan_bodies=scan_cap,
    )
    if gt:
        session["sim_object_placements"] = gt
        session["sim_object_placements_frame"] = SIM_OBJECT_PLACEMENTS_FRAME
        if session.get("navigation_origin_xyt") is not None:
            session["sim_object_placements_note"] = (
                "pos is MuJoCo world XYZ; ZMQ gps/compass are episode-relative to navigation_origin_xyt"
            )
