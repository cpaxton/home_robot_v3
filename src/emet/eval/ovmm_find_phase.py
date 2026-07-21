# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""OVMM-inspired find-phase metrics (FindObj / FindRec) for emet sim benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from emet.utils.config import resolve_config_yaml_path

MemoryBackendName = Literal["dynamem", "graph_eqa", "dynagraph", "ground_truth"]
ManipMode = Literal["skip", "oracle", "sim", "attempt"]
PlanarFrame = Literal["mujoco_xy", "habitat_xz"]
LocalizeSource = Literal[
    "voxel",
    "graph_near_recep",
    "graph_near_anchor",
    "graph_habitat_node",
    "memory_localize_text_graph",
    "memory_localize_text_voxel",
    "memory_check_graph",
    "memory_check_voxel",
    "memory_list_objects",
]


@dataclass(frozen=True)
class FindPhaseEpisode:
    """One OVMM-style find-phase episode (memory localization only)."""

    id: str
    tier: str
    sim: str
    object: str
    start_recep: str
    goal_recep: str
    success_radius_m: float = 0.75
    explore_steps: int = 0
    object_gt_body: str | None = None


@dataclass
class FindPhaseRunConfig:
    """Runtime overrides for one benchmark run."""

    backend: MemoryBackendName = "dynagraph"
    merge_xy_m: float | None = None
    staleness_horizon: int | None = None
    compare_to_gt: bool = False
    cpu_only: bool = False
    port_offset: int = 0
    not_rotate: bool = False
    perfect_depth: bool = True
    seed: int | None = None
    use_sensor_perception: bool = False
    prefer_voxel: bool = True
    manip_mode: ManipMode = "skip"
    nav_step_timeout_s: float | None = None


def resolve_find_phase_nav_step_timeout(
    *,
    cpu_only: bool,
    sim_kind: str,
    override: float | None = None,
) -> float:
    """ZMQ nav/obs wait budget for find-phase mapping (rotate + explore)."""
    if override is not None:
        return float(override)
    if cpu_only:
        return 45.0
    if sim_kind in ("robocasa", "molmospaces"):
        return 30.0
    return 15.0


def load_find_phase_episodes(path: str | Path) -> list[FindPhaseEpisode]:
    """Load episodes from ``configs/ovmm/find_phase_episodes.yaml``."""
    full = Path(resolve_config_yaml_path(str(path)))
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    rows = raw.get("episodes") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"expected list under 'episodes' in {full}")
    out: list[FindPhaseEpisode] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            FindPhaseEpisode(
                id=str(row["id"]),
                tier=str(row.get("tier", "")),
                sim=str(row["sim"]),
                object=str(row["object"]),
                start_recep=str(row["start_recep"]),
                goal_recep=str(row["goal_recep"]),
                success_radius_m=float(row.get("success_radius_m", 0.75)),
                explore_steps=int(row.get("explore_steps", 0)),
                object_gt_body=(str(row["object_gt_body"]) if row.get("object_gt_body") else None),
            )
        )
    return out


def resolve_object_query(
    episode: FindPhaseEpisode,
    placements: dict[str, dict[str, Any]] | None,
) -> str:
    """Resolve memory/GT object query; optional ``object_gt_body`` overrides from sim GT."""
    if episode.object_gt_body and placements and episode.object_gt_body in placements:
        return str(placements[episode.object_gt_body].get("cat") or episode.object)
    return episode.object


def category_matches(query: str, cat: str | None) -> bool:
    """Case-insensitive substring match between query and GT category label."""
    q = str(query or "").strip().lower()
    c = str(cat or "").strip().lower()
    if not q or not c:
        return False
    return q in c or c in q


def bodies_matching_category(
    placements: dict[str, dict[str, Any]],
    query: str,
) -> list[str]:
    """Return body names whose ``cat`` matches ``query``."""
    return [body for body, info in placements.items() if category_matches(query, str(info.get("cat") or body))]


def pick_find_object_gt_body(
    placements: dict[str, dict[str, Any]],
    object_query: str,
    start_recep: str,
    *,
    object_gt_body: str | None = None,
) -> str | None:
    """
    Choose the GT body for FindObj.

    When ``object_gt_body`` is set and present in ``placements``, use it directly.
    Otherwise match ``object_query`` on ``cat``; if multiple bodies match, prefer the
    one nearest any ``start_recep`` body in XY (OVMM start-receptacle disambiguation).
    """
    if object_gt_body and object_gt_body in placements:
        return object_gt_body
    obj_bodies = bodies_matching_category(placements, object_query)
    if not obj_bodies:
        return None
    start_bodies = bodies_matching_category(placements, start_recep)
    if not start_bodies:
        return sorted(obj_bodies)[0]

    def min_dist_to_start(body: str) -> float:
        frame = str(placements[body].get("frame") or "mujoco_xy")
        planar: PlanarFrame = "habitat_xz" if frame == "habitat_yup" else "mujoco_xy"
        pos_h = gt_horizontal_coords(placements[body], frame=planar)
        return min(
            float(np.linalg.norm(pos_h - gt_horizontal_coords(placements[s], frame=planar))) for s in start_bodies
        )

    return sorted(obj_bodies, key=min_dist_to_start)[0]


def horizontal_coords(
    xyz: np.ndarray | list,
    *,
    frame: PlanarFrame = "mujoco_xy",
) -> np.ndarray:
    """Return horizontal-plane coordinates for distance checks (meters)."""
    arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if frame == "habitat_xz":
        if arr.size >= 3:
            return np.array([float(arr[0]), float(arr[2])], dtype=np.float64)
        return np.array([float(arr[0]), float(arr[1] if arr.size > 1 else 0.0)], dtype=np.float64)
    return arr[:2]


def gt_horizontal_coords(
    placement: dict[str, Any],
    *,
    frame: PlanarFrame = "mujoco_xy",
) -> np.ndarray:
    """Horizontal coords for one placement entry (center or bounds clamp)."""
    pos = np.asarray(placement["pos"], dtype=np.float64).reshape(3)
    return horizontal_coords(pos, frame=frame)


def distance_to_placement_xy(
    pred_xyz: np.ndarray | list,
    placement: dict[str, Any],
    *,
    frame: PlanarFrame = "mujoco_xy",
) -> float:
    """XY/XZ distance from prediction to GT center, or to nearest point on ``bounds`` when present."""
    pred_h = horizontal_coords(pred_xyz, frame=frame)
    bounds = placement.get("bounds")
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        if frame == "habitat_xz":
            mn = np.array([float(b[0, 0]), float(b[0, 2])], dtype=np.float64)
            mx = np.array([float(b[1, 0]), float(b[1, 2])], dtype=np.float64)
        else:
            mn = b[0, :2]
            mx = b[1, :2]
        clamped = np.clip(pred_h, mn, mx)
        return float(np.linalg.norm(pred_h - clamped))
    gt_h = gt_horizontal_coords(placement, frame=frame)
    return float(np.linalg.norm(pred_h - gt_h))


def _pred_xyz_array(pred_xyz: np.ndarray | list | None) -> np.ndarray | None:
    if pred_xyz is None:
        return None
    arr = np.asarray(pred_xyz, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    if arr.size == 2:
        return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)
    return arr[:3]


def pred_xyz_to_json_list(pred_xyz: np.ndarray | list | None) -> list[float] | None:
    """Serialize a predicted XYZ for JSON metrics artifacts."""
    arr = _pred_xyz_array(pred_xyz)
    if arr is None:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def localization_pred_fields(
    obj_pred_xyz: np.ndarray | list | None,
    recep_pred_xyz: np.ndarray | list | None,
) -> dict[str, list[float] | None]:
    """Pred XYZ fields included in find-phase run JSON."""
    return {
        "pred_obj_xyz": pred_xyz_to_json_list(obj_pred_xyz),
        "pred_recep_xyz": pred_xyz_to_json_list(recep_pred_xyz),
    }


def set_find_phase_run_seed(seed: int) -> None:
    """Best-effort RNG seeding for repeatable perception/mapping runs."""
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _query_variants(query: str, placements: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Expand a text query with substring tokens and matching GT category strings."""
    base = str(query or "").strip()
    variants: list[str] = []
    if base:
        variants.append(base)
    low = base.lower()
    for token in low.replace("_", " ").split():
        if len(token) >= 2 and token not in {v.lower() for v in variants}:
            variants.append(token)
    if placements:
        for info in placements.values():
            cat = str(info.get("cat") or "").strip()
            if not cat:
                continue
            cat_low = cat.lower()
            if low and (low in cat_low or cat_low in low):
                variants.append(cat)
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        key = v.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def localize_point_to_world_xy(
    point_xyz: np.ndarray | list | None,
    session: dict[str, Any] | None,
) -> np.ndarray | None:
    """
    Map a memory localization XYZ to MuJoCo world XY for GT scoring.

    Graph nodes and detection outputs are already world-frame; episode-relative nav
    points (Robocasa ``gps`` frame) are converted when ``navigation_origin_xyt`` is
    present in the ZMQ session.
    """
    pred = _pred_xyz_array(point_xyz)
    if pred is None:
        return None
    if session is None or session.get("navigation_origin_xyt") is None:
        return pred
    from emet.utils.geometry import nav_xyt_to_world_xyt

    world_xyt = nav_xyt_to_world_xyt(np.array([pred[0], pred[1], 0.0], dtype=np.float64), session)
    return np.array([float(world_xyt[0]), float(world_xyt[1]), float(pred[2])], dtype=np.float64)


def _graph_nodes_matching(memory: Any, query: str) -> list[Any]:
    graph = getattr(memory, "_graph", None)
    if graph is None:
        return []
    out = []
    for node in graph.get_nodes():
        labels = getattr(node, "labels", None) or []
        if any(category_matches(query, str(lbl)) for lbl in labels):
            out.append(node)
    return out


def _pick_graph_xyz_near_recep(
    memory: Any,
    query: str,
    placements: dict[str, dict[str, Any]],
    near_recep: str,
) -> np.ndarray | None:
    """Pick the graph node matching ``query`` nearest ``near_recep`` GT bodies (OVMM disambiguation)."""
    nodes = _graph_nodes_matching(memory, query)
    if not nodes:
        return None
    ref_bodies = bodies_matching_category(placements, near_recep)
    if not ref_bodies:
        xyz = np.asarray(nodes[0].xyz, dtype=np.float64).reshape(3)
        return xyz
    frame: PlanarFrame = (
        "habitat_xz" if any(str(placements[b].get("frame")) == "habitat_yup" for b in ref_bodies) else "mujoco_xy"
    )
    ref_xy = [gt_horizontal_coords(placements[b], frame=frame) for b in ref_bodies]

    def _min_dist_to_recep(node) -> float:
        nxy = horizontal_coords(node.xyz, frame=frame)
        return min(float(np.linalg.norm(nxy - rxy)) for rxy in ref_xy)

    best = min(nodes, key=_min_dist_to_recep)
    return np.asarray(best.xyz, dtype=np.float64).reshape(3)


def _pick_graph_xyz_near_point(
    memory: Any,
    query: str,
    anchor_xyz: np.ndarray,
    *,
    frame: PlanarFrame = "mujoco_xy",
) -> np.ndarray | None:
    """Pick matching graph node nearest an anchor XYZ (optional disambiguation)."""
    nodes = _graph_nodes_matching(memory, query)
    if not nodes:
        return None
    anchor = horizontal_coords(anchor_xyz, frame=frame)

    def _dist(node) -> float:
        nxy = horizontal_coords(node.xyz, frame=frame)
        return float(np.linalg.norm(nxy - anchor))

    best = min(nodes, key=_dist)
    return np.asarray(best.xyz, dtype=np.float64).reshape(3)


def _voxel_localize(
    voxel_map: Any,
    query: str,
    *,
    placements: dict[str, dict[str, Any]] | None,
    session: dict[str, Any] | None,
) -> tuple[np.ndarray | None, str]:
    """Voxel-map localization only (preferred for find-phase; avoids merged-graph centroid drift)."""
    for q in _query_variants(query, placements):
        result = voxel_map.localize_text(q, debug=False, return_debug=True)
        target = result[0] if isinstance(result, (list, tuple)) else result
        if target is not None:
            xyz = localize_point_to_world_xy(target, session)
            if xyz is not None:
                return xyz, q
    return None, query


def _memory_localize_source(memory: Any, query: str) -> LocalizeSource:
    """Infer graph vs voxel for adapter localize/check (GraphEQA tries graph first)."""
    return "memory_localize_text_graph" if _graph_nodes_matching(memory, query) else "memory_localize_text_voxel"


def _memory_check_source(memory: Any, query: str) -> LocalizeSource:
    return "memory_check_graph" if _graph_nodes_matching(memory, query) else "memory_check_voxel"


def query_find_phase_localization(
    memory: Any,
    query: str,
    *,
    placements: dict[str, dict[str, Any]] | None = None,
    session: dict[str, Any] | None = None,
    near_recep: str | None = None,
    anchor_xyz: np.ndarray | None = None,
    voxel_map: Any | None = None,
    convert_nav_to_world: bool = False,
    prefer_voxel: bool = True,
    planar_frame: PlanarFrame = "mujoco_xy",
) -> tuple[np.ndarray | None, bool, str, LocalizeSource | None]:
    """
    Query memory for FindObj/FindRec localization with query variants and fallbacks.

    Returns:
        ``(world_xyz, success, query_used, localize_source)`` where ``world_xyz`` is suitable
        for GT distance checks and ``localize_source`` names the winning code path (``None`` on miss).
    """
    sess = session if convert_nav_to_world else None

    if prefer_voxel and voxel_map is not None and hasattr(voxel_map, "localize_text"):
        xyz, q_used = _voxel_localize(voxel_map, query, placements=placements, session=sess)
        if xyz is not None:
            return xyz, True, q_used, "voxel"

    if near_recep and placements is not None and _graph_nodes_matching(memory, query):
        xyz = _pick_graph_xyz_near_recep(memory, query, placements, near_recep)
        if xyz is not None:
            converted = localize_point_to_world_xy(xyz, sess)
            xyz = converted if converted is not None else xyz
            return xyz, True, query, "graph_near_recep"
    if anchor_xyz is not None and _graph_nodes_matching(memory, query):
        xyz = _pick_graph_xyz_near_point(memory, query, anchor_xyz, frame=planar_frame)
        if xyz is not None:
            converted = localize_point_to_world_xy(xyz, sess)
            xyz = converted if converted is not None else xyz
            return xyz, True, query, "graph_near_anchor"
    for q in _query_variants(query, placements):
        if planar_frame == "habitat_xz":
            nodes = _graph_nodes_matching(memory, q)
            if nodes:
                xyz = np.asarray(nodes[0].xyz, dtype=np.float64).reshape(3)
                return xyz, True, q, "graph_habitat_node"
        loc = memory.localize_text(q)
        if loc.success and loc.point_xyz is not None:
            xyz = localize_point_to_world_xy(loc.point_xyz, sess)
            if xyz is not None and planar_frame == "habitat_xz" and xyz.size >= 3:
                # Graph adapter forces z=1.0; keep Habitat Y-up XYZ from graph nodes when possible.
                nodes = _graph_nodes_matching(memory, q)
                if nodes:
                    xyz = np.asarray(nodes[0].xyz, dtype=np.float64).reshape(3)
            if xyz is not None:
                return xyz, True, q, _memory_localize_source(memory, q)
        check = memory.check_memory_for_object(q)
        if check.confidence > 0 and check.location_xyz is not None:
            xyz = localize_point_to_world_xy(check.location_xyz, sess)
            if xyz is not None:
                return xyz, True, q, _memory_check_source(memory, q)
    for label in memory.list_objects():
        if category_matches(query, label):
            loc = memory.localize_text(label)
            if loc.success and loc.point_xyz is not None:
                xyz = localize_point_to_world_xy(loc.point_xyz, sess)
                if xyz is not None:
                    return xyz, True, label, "memory_list_objects"
    return None, False, query, None


def score_find_object(
    pred_xyz: np.ndarray | list | None,
    placements: dict[str, dict[str, Any]] | None,
    object_query: str,
    start_recep: str,
    *,
    radius_m: float,
    object_gt_body: str | None = None,
    frame: PlanarFrame = "mujoco_xy",
) -> dict[str, Any]:
    """Score FindObj: predicted XYZ within ``radius_m`` of chosen GT object body."""
    if not placements:
        return {
            "find_object_success": False,
            "localization_err_obj_m": None,
            "gt_object_body": None,
        }
    gt_body = pick_find_object_gt_body(
        placements,
        object_query,
        start_recep,
        object_gt_body=object_gt_body,
    )
    pred = _pred_xyz_array(pred_xyz)
    if gt_body is None or pred is None:
        return {
            "find_object_success": False,
            "localization_err_obj_m": None,
            "gt_object_body": gt_body,
        }
    err_xy = distance_to_placement_xy(pred, placements[gt_body], frame=frame)
    return {
        "find_object_success": err_xy <= float(radius_m),
        "localization_err_obj_m": err_xy,
        "gt_object_body": gt_body,
    }


def score_find_recep(
    pred_xyz: np.ndarray | list | None,
    placements: dict[str, dict[str, Any]] | None,
    goal_recep: str,
    *,
    radius_m: float,
    frame: PlanarFrame = "mujoco_xy",
) -> dict[str, Any]:
    """Score FindRec: predicted XYZ within ``radius_m`` of any GT body matching ``goal_recep``."""
    if not placements:
        return {
            "find_recep_success": False,
            "localization_err_recep_m": None,
            "gt_recep_bodies": [],
        }
    recep_bodies = bodies_matching_category(placements, goal_recep)
    pred = _pred_xyz_array(pred_xyz)
    if not recep_bodies or pred is None:
        return {
            "find_recep_success": False,
            "localization_err_recep_m": None,
            "gt_recep_bodies": recep_bodies,
        }
    errors = [distance_to_placement_xy(pred, placements[body], frame=frame) for body in recep_bodies]
    best_err = min(errors)
    return {
        "find_recep_success": best_err <= float(radius_m),
        "localization_err_recep_m": best_err,
        "gt_recep_bodies": recep_bodies,
    }


def compute_find_phase_metrics(
    *,
    obj_pred_xyz: np.ndarray | list | None,
    recep_pred_xyz: np.ndarray | list | None,
    placements: dict[str, dict[str, Any]] | None,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    radius_m: float,
    object_gt_body: str | None = None,
    frame: PlanarFrame = "mujoco_xy",
) -> dict[str, Any]:
    """Combine FindObj / FindRec scores and OVMM-style partial success (mean of two phases)."""
    obj = score_find_object(
        obj_pred_xyz,
        placements,
        object_query,
        start_recep,
        radius_m=radius_m,
        object_gt_body=object_gt_body,
        frame=frame,
    )
    rec = score_find_recep(recep_pred_xyz, placements, goal_recep, radius_m=radius_m, frame=frame)
    partial = 0.5 * (float(obj["find_object_success"]) + float(rec["find_recep_success"]))
    return {
        **obj,
        **rec,
        "find_partial_success": partial,
        "success_radius_m": float(radius_m),
    }


def voxel_explored_cell_count(voxel_map: Any) -> int:
    """Count explored cells from a DynaMem voxel map."""
    if voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
        return 0
    obstacles, explored = voxel_map.get_2d_map()
    if explored is None:
        return 0
    if hasattr(explored, "cpu"):
        explored = explored.cpu().numpy()
    return int(np.asarray(explored).sum())


def voxel_explored_area_m2(voxel_map: Any) -> float:
    """Approximate explored floor area (m²) from voxel map 2D grid."""
    n_cells = voxel_explored_cell_count(voxel_map)
    if n_cells == 0 or voxel_map is None:
        return 0.0
    vs = float(getattr(voxel_map, "resolution", 0.05) or 0.05)
    return float(n_cells) * vs * vs


def graph_node_count(agent: Any) -> int:
    gm = getattr(agent, "graph_memory", None)
    if gm is None:
        return 0
    nodes = gm.get_nodes() if hasattr(gm, "get_nodes") else []
    return len(nodes)


def collect_scaling_diagnostics(
    agent: Any,
    placements: dict[str, dict[str, Any]] | None,
    *,
    episode_wall_s: float,
    n_controller_steps: int = 0,
) -> dict[str, Any]:
    """Export scaling diagnostics for paper plots."""
    vm = getattr(agent, "voxel_map", None)
    return {
        "n_graph_nodes": graph_node_count(agent),
        "n_voxel_explored_cells": voxel_explored_cell_count(vm),
        "n_voxel_explored_area_m2": voxel_explored_area_m2(vm),
        "n_placements": len(placements) if placements else 0,
        "episode_wall_s": float(episode_wall_s),
        "n_controller_steps": int(n_controller_steps),
    }


def apply_backend_parameters(
    parameters: Any,
    backend: MemoryBackendName,
    *,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
) -> Any:
    """Configure dynagraph merge/staleness for backend comparison runs."""
    from emet.eval.benchmark_dynagraph import apply_ovmm_backend_dynagraph

    return apply_ovmm_backend_dynagraph(
        parameters,
        backend,
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
    )


def create_find_phase_agent(
    robot: Any,
    parameters: dict[str, Any],
    backend: MemoryBackendName,
    *,
    cpu_only: bool = False,
    compare_to_gt: bool = False,
    use_sensor_perception: bool = False,
):
    """Instantiate the controller for a memory backend."""
    from emet.eval.benchmark_dynagraph import harness_controller_kwargs

    harness_kw = harness_controller_kwargs(parameters, harness="ovmm_find_phase", method=str(backend))
    use_instance_graph = bool(
        harness_kw.get("use_instance_graph", backend in ("graph_eqa", "dynagraph", "ground_truth"))
    )
    manipulation_only = bool(harness_kw.get("manipulation_only", False))
    if backend == "dynamem":
        from emet.controller.controller_dynamem import DynamemController

        agent = DynamemController(
            robot,
            parameters,
            save_rerun=False,
            use_instance_memory=True,
            cpu_only=cpu_only,
            eqa=False,
            defer_eqa_vllm=True,
        )
    elif backend == "graph_eqa":
        from emet.controller.controller_graph_eqa import GraphEQAController

        agent = GraphEQAController(
            robot,
            parameters,
            save_rerun=False,
            use_instance_graph=use_instance_graph,
            cpu_only=cpu_only,
            use_sensor_perception=use_sensor_perception,
            manipulation_only=manipulation_only,
        )
    elif backend == "dynagraph":
        from emet.controller.controller_dynagraph import DynagraphController

        agent = DynagraphController(
            robot,
            parameters,
            save_rerun=False,
            cpu_only=cpu_only,
            use_instance_graph=use_instance_graph,
            use_sensor_perception=use_sensor_perception,
            manipulation_only=manipulation_only,
            visualize_ground_truth=compare_to_gt,
        )
    elif backend == "ground_truth":
        from emet.controller.controller_dynagraph import DynagraphController

        agent = DynagraphController(
            robot,
            parameters,
            save_rerun=False,
            cpu_only=cpu_only,
            use_instance_graph=use_instance_graph,
            use_sensor_perception=False,
            manipulation_only=manipulation_only,
            ground_truth_mode=True,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")
    agent.start()
    return agent


def get_memory_backend_for_agent(agent: Any, backend: MemoryBackendName):
    """Return unified memory backend for find-phase queries."""
    from emet.memory.backend import get_memory_backend

    if backend == "dynamem":
        return get_memory_backend("dynamem", voxel_map=agent.voxel_map)
    return get_memory_backend(
        "graph_eqa",
        graph_memory=agent.graph_memory,
        voxel_map=getattr(agent, "voxel_map", None),
    )


def run_mapping_protocol(
    agent: Any,
    *,
    explore_steps: int,
    not_rotate: bool,
) -> int:
    """Rotate in place and optionally run frontier explore steps."""
    steps = 0
    if not not_rotate:
        agent.rotate_in_place()
        steps += 1
    for _ in range(max(0, int(explore_steps))):
        agent.execute_action("")
        steps += 1
    if backend_uses_ground_truth(agent):
        refresh = getattr(agent, "refresh_ground_truth", None)
        if callable(refresh):
            refresh()
    return steps


def backend_uses_ground_truth(agent: Any) -> bool:
    return bool(getattr(agent, "ground_truth_mode", False))


@dataclass
class EpisodeRunResult:
    """Metrics from one episode × backend run."""

    episode_id: str
    tier: str
    backend: str
    metrics: dict[str, Any] = field(default_factory=dict)


def run_episode_find_phase(
    episode: FindPhaseEpisode,
    run_cfg: FindPhaseRunConfig,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Run one find-phase episode: sim subprocess, mapping, memory queries, GT scoring.

    Caller should ensure sim dependencies are available for the episode tier.
    """
    import os
    import socket
    import subprocess
    import sys
    from dataclasses import replace

    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.core.parameters import get_parameters
    from emet.memory.graph_eqa.sim_ground_truth_graph import (
        gt_graph_completeness,
        instance_gt_association_recall,
        read_sim_object_placements,
    )
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
    from emet.utils.process_tree import popen_session, terminate_process_tree

    if run_cfg.seed is not None:
        set_find_phase_run_seed(int(run_cfg.seed))

    repo = repo_root or Path(__file__).resolve().parents[3]
    sim_cfg = load_sim_launch_config_from_path(episode.sim)
    port_offset = int(run_cfg.port_offset)
    recv_port = 4401 + port_offset

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    if run_cfg.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""

    sim_kind = str(getattr(sim_cfg, "kind", ""))
    nav_timeout = resolve_find_phase_nav_step_timeout(
        cpu_only=run_cfg.cpu_only,
        sim_kind=sim_kind,
        override=run_cfg.nav_step_timeout_s,
    )

    sim_cfg = replace(sim_cfg, port_offset=port_offset, headless=True)
    server_argv = prepare_mujoco_server_argv(sim_cfg)
    server_cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *server_argv]

    def wait_port(port: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=2):
                    return True
            except OSError:
                time.sleep(0.5)
        return False

    robot = None
    agent = None
    server = None
    t0 = time.monotonic()
    init_wall_s = 0.0
    mapping_wall_s = 0.0
    query_wall_s = 0.0
    try:
        server = popen_session(
            server_cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        bind_timeout = 180.0 if sim_kind in ("molmospaces", "robocasa") else 120.0
        if not wait_port(recv_port, bind_timeout):
            err_tail = ""
            if server.stderr and server.poll() is not None:
                err_tail = server.stderr.read() if hasattr(server.stderr, "read") else ""
            elif server.stdout and server.poll() is not None:
                err_tail = server.stdout.read() if hasattr(server.stdout, "read") else ""
            raise RuntimeError(
                f"sim server did not bind port {recv_port}" + (f": {err_tail[-500:]}" if err_tail else "")
            )
        settle = 25.0 if sim_kind in ("molmospaces", "robocasa") else 15.0
        if run_cfg.cpu_only:
            settle += 15.0
        time.sleep(settle)

        robot_kind = str(getattr(sim_cfg, "robot", "stretch"))
        robot = create_robot_client_from_cli(
            robot_kind,
            "127.0.0.1",
            port_offset=port_offset,
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=True,
        )
        robot.move_to_nav_posture()
        robot.set_velocity(v=30.0, w=15.0)

        parameters = apply_backend_parameters(
            get_parameters("dynav_config.yaml"),
            run_cfg.backend,
            merge_xy_m=run_cfg.merge_xy_m,
            staleness_horizon=run_cfg.staleness_horizon,
        )
        parameters["encoder"] = None
        if run_cfg.perfect_depth:
            parameters["debug_perfect_sensor_depth"] = True
        parameters["find_phase_nav_step_timeout_s"] = nav_timeout

        t_init0 = time.monotonic()
        agent = create_find_phase_agent(
            robot,
            parameters,
            run_cfg.backend,
            cpu_only=run_cfg.cpu_only,
            compare_to_gt=run_cfg.compare_to_gt,
            use_sensor_perception=run_cfg.use_sensor_perception,
        )
        init_wall_s = time.monotonic() - t_init0
        if run_cfg.backend == "ground_truth":
            refresh = getattr(agent, "refresh_ground_truth", None)
            if callable(refresh):
                n_gt = refresh()
                if n_gt == 0:
                    raise RuntimeError("ground-truth mode: no sim_object_placements in session")

        t_map0 = time.monotonic()
        n_steps = run_mapping_protocol(
            agent,
            explore_steps=episode.explore_steps,
            not_rotate=run_cfg.not_rotate,
        )
        mapping_wall_s = time.monotonic() - t_map0

        session = robot.get_emet_session()
        placements = read_sim_object_placements(session)
        memory = get_memory_backend_for_agent(agent, run_cfg.backend)
        vm = getattr(agent, "voxel_map", None)
        nav_world = sim_kind == "robocasa"

        object_query = resolve_object_query(episode, placements)

        prefer_voxel = run_cfg.prefer_voxel and run_cfg.backend != "ground_truth"
        t_query0 = time.monotonic()
        obj_xyz, obj_ok, obj_q_used, obj_source = query_find_phase_localization(
            memory,
            object_query,
            placements=placements,
            session=session,
            near_recep=episode.start_recep,
            voxel_map=vm,
            convert_nav_to_world=nav_world or run_cfg.backend == "dynamem",
            prefer_voxel=prefer_voxel,
        )
        recep_xyz, recep_ok, recep_q_used, recep_source = query_find_phase_localization(
            memory,
            episode.goal_recep,
            placements=placements,
            session=session,
            near_recep=episode.goal_recep,
            voxel_map=vm,
            convert_nav_to_world=nav_world or run_cfg.backend == "dynamem",
            prefer_voxel=prefer_voxel,
        )

        find_metrics = compute_find_phase_metrics(
            obj_pred_xyz=obj_xyz,
            recep_pred_xyz=recep_xyz,
            placements=placements,
            object_query=object_query,
            start_recep=episode.start_recep,
            goal_recep=episode.goal_recep,
            radius_m=episode.success_radius_m,
            object_gt_body=episode.object_gt_body,
        )
        query_wall_s = time.monotonic() - t_query0

        scaling = collect_scaling_diagnostics(
            agent,
            placements,
            episode_wall_s=time.monotonic() - t0,
            n_controller_steps=n_steps,
        )

        gt_metrics: dict[str, Any] = {}
        if agent.graph_memory is not None and placements:
            gt_metrics = {
                "gt_graph_completeness": gt_graph_completeness(agent.graph_memory, placements),
                "instance_gt_association_recall": instance_gt_association_recall(agent.graph_memory, placements),
            }

        metrics = {
            "episode_id": episode.id,
            "tier": episode.tier,
            "backend": run_cfg.backend,
            "sim": episode.sim,
            "object_query": object_query,
            "start_recep": episode.start_recep,
            "goal_recep": episode.goal_recep,
            "explore_steps": episode.explore_steps,
            "merge_xy_m": parameters.get("dynagraph_merge_xy_m"),
            "staleness_horizon": parameters.get("dynagraph_staleness_horizon"),
            "perfect_depth": bool(run_cfg.perfect_depth),
            "use_sensor_perception": bool(run_cfg.use_sensor_perception),
            "prefer_voxel": bool(prefer_voxel),
            "manip_mode": str(run_cfg.manip_mode),
            "init_wall_s": float(init_wall_s),
            "mapping_wall_s": float(mapping_wall_s),
            "query_wall_s": float(query_wall_s),
            "obj_localize_success": bool(obj_ok),
            "recep_localize_success": bool(recep_ok),
            "obj_query_used": obj_q_used,
            "recep_query_used": recep_q_used,
            "obj_localize_source": obj_source,
            "recep_localize_source": recep_source,
            "seed": run_cfg.seed,
            **localization_pred_fields(obj_xyz, recep_xyz),
            **find_metrics,
            **scaling,
            **gt_metrics,
        }
        if run_cfg.manip_mode != "skip":
            from emet.eval.ovmm_full import augment_find_metrics_with_manip

            metrics = augment_find_metrics_with_manip(
                agent,
                robot,
                episode,
                run_cfg,
                metrics,
                placements=placements,
                object_query=object_query,
            )
        return metrics
    finally:
        if agent is not None:
            try:
                agent.stop()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if server is not None:
            terminate_process_tree(server, grace_s=10.0)
        from emet.utils.port_utils import get_ports, kill_processes_on_port

        for p in get_ports(port_offset):
            kill_processes_on_port(p)
