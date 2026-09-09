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

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from emet.config.rerun_config import eval_rerun_enabled
from emet.eval.memory_backends import (
    DYNAGRAPH,
    GROUND_TRUTH,
    LAZY_GRAPH,
    OVMM_MEMORY_BACKEND,
    STATIC_GRAPH,
    normalize_benchmark_backend,
)
from emet.utils.config import resolve_config_yaml_path

MemoryBackendName = OVMM_MEMORY_BACKEND
ManipMode = Literal["skip", "oracle", "sim", "attempt", "mcts"]
MANIP_MODES = ("skip", "oracle", "sim", "attempt", "mcts")
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
    "gt_placement",
    "agentic_verify",
]

_HEX_TOKEN_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"^\d+$")

# ``explore_steps`` is a deprecated alias of ``mapping_max_nav_steps`` (mapping
# coverage / agentic explore max_nav_steps — not FindObj/FindRec). Warn once
# per source so sweep files that still use the old key are visible.
_EXPLORE_STEPS_ALIAS_WARNED: set[str] = set()


class MappingBudgetConflict(ValueError):
    """Both mapping budget keys were set to different integers."""


def _warn_explore_steps_alias(source: str) -> None:
    key = str(source or "explore_steps")
    if key in _EXPLORE_STEPS_ALIAS_WARNED:
        return
    _EXPLORE_STEPS_ALIAS_WARNED.add(key)
    from emet.utils.logger import Logger

    Logger(__name__).warning(
        f"{key}: explore_steps is a deprecated alias of mapping_max_nav_steps "
        "(mapping coverage max_nav_steps, not FindObj/FindRec). "
        "Prefer mapping_max_nav_steps / --mapping-max-nav-steps."
    )


def resolve_mapping_max_nav_steps(
    mapping_max_nav_steps: int | None = None,
    explore_steps: int | None = None,
    *,
    source: str = "",
    default: int | None = 0,
    warn: bool = True,
) -> int | None:
    """Canonical mapping coverage budget (agentic ``max_nav_steps``).

    ``mapping_max_nav_steps`` wins. ``explore_steps`` is a deprecated alias.
    If both are set and differ, raise :class:`MappingBudgetConflict`. If neither
    is set, return *default* (``0`` for YAML rows, ``None`` for a CLI override).
    """
    has_new = mapping_max_nav_steps is not None
    has_old = explore_steps is not None
    if has_new and has_old and int(mapping_max_nav_steps) != int(explore_steps):
        loc = f"{source}: " if source else ""
        raise MappingBudgetConflict(
            f"{loc}mapping_max_nav_steps={mapping_max_nav_steps} conflicts with "
            f"deprecated explore_steps={explore_steps}"
        )
    if has_old and not has_new and warn:
        _warn_explore_steps_alias(source or "explore_steps")
    if has_new:
        return int(mapping_max_nav_steps)
    if has_old:
        return int(explore_steps)
    return default


def mapping_budget_from_row(
    row: dict[str, Any],
    *,
    source: str,
    default: int = 0,
    warn: bool = True,
) -> int:
    """Resolve ``mapping_max_nav_steps`` / ``explore_steps`` from a YAML mapping."""
    new = row.get("mapping_max_nav_steps")
    old = row.get("explore_steps")
    resolved = resolve_mapping_max_nav_steps(
        None if new is None else int(new),
        None if old is None else int(old),
        source=source,
        default=default,
        warn=warn,
    )
    return int(resolved or 0)


def semantic_label_from_instance(name: str) -> str:
    """Strip Molmo/iTHOR instance hashes from a body or category string.

    Examples::

        bowl_6befd62f08fd322391939c2b44d3f839_1_1_0 -> bowl
        bowl 6befd62f08fd322391939c2b44d3f839 1 0 0 -> bowl
        kitchen cabinet door -> kitchen cabinet door
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    tokens = re.split(r"[\s_]+", raw)
    keep = [t for t in tokens if t and not _HEX_TOKEN_RE.match(t) and not _NUMERIC_TOKEN_RE.match(t)]
    if keep:
        return " ".join(keep)
    return tokens[0] if tokens else raw


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
    mapping_max_nav_steps: int | None = None
    explore_steps: int | None = None
    object_gt_body: str | None = None
    # TAMP floor pick/place: drop the object to the floor before the pick phase
    # (RoboCasa "pick something off the floor" / room-exploration tasks).
    floor_object: bool = False
    floor_z_m: float | None = None
    # Optional per-episode full-OVMM override; batch CLI options take precedence.
    manip_mode: ManipMode | None = None

    def __post_init__(self) -> None:
        n = resolve_mapping_max_nav_steps(
            self.mapping_max_nav_steps,
            self.explore_steps,
            source="FindPhaseEpisode",
            default=0,
            warn=False,
        )
        object.__setattr__(self, "mapping_max_nav_steps", int(n or 0))
        object.__setattr__(self, "explore_steps", int(n or 0))


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
    # None → on for dynagraph/static_graph (shared AgenticEQA loop); off for dynamem/GT.
    agentic_find: bool | None = None
    query_driven_memory: bool = False
    manip_mode: ManipMode = "skip"
    nav_step_timeout_s: float | None = None
    explore_steps_override: int | None = None
    use_scene_cache: bool = True
    agentic_max_rounds: int | None = None
    agentic_max_nav_steps: int | None = None
    # None → full 8-step rotate_in_place; 4 is a fast rby1 gate (~table sweep).
    mapping_rotate_steps: int | None = None
    # None → on for S0 default-table episodes (pytest-aligned sim + interactive profile).
    s0_parity: bool | None = None


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
    n_alias = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("explore_steps") is not None and row.get("mapping_max_nav_steps") is None:
            n_alias += 1
        budget = mapping_budget_from_row(row, source=str(full), default=0, warn=False)
        out.append(
            FindPhaseEpisode(
                id=str(row["id"]),
                tier=str(row.get("tier", "")),
                sim=str(row["sim"]),
                object=str(row["object"]),
                start_recep=str(row["start_recep"]),
                goal_recep=str(row["goal_recep"]),
                success_radius_m=float(row.get("success_radius_m", 0.75)),
                mapping_max_nav_steps=budget,
                explore_steps=budget,
                object_gt_body=(str(row["object_gt_body"]) if row.get("object_gt_body") else None),
                floor_object=bool(row.get("floor_object", False)),
                floor_z_m=(float(row["floor_z_m"]) if row.get("floor_z_m") is not None else None),
                manip_mode=(str(row["manip_mode"]).strip().lower() if row.get("manip_mode") is not None else None),
            )
        )
        if out[-1].manip_mode is not None and out[-1].manip_mode not in MANIP_MODES:
            raise ValueError(f"invalid manip_mode={out[-1].manip_mode!r} in {full}")
    if n_alias:
        _warn_explore_steps_alias(f"{full} ({n_alias} episode(s))")
    return out


def is_s0_default_table_episode(episode: FindPhaseEpisode) -> bool:
    """True for Stretch default-table S0 rows in ``find_phase_episodes.yaml``."""
    if str(episode.tier or "").strip().upper() != "S0":
        return False
    sim = str(episode.sim or "").lower()
    return "default_table" in sim


def resolve_s0_parity_flags(
    episode: FindPhaseEpisode,
    run_cfg: FindPhaseRunConfig,
    *,
    use_agentic: bool,
) -> tuple[bool, bool]:
    """Return ``(sim_and_profile_parity, phrase_only_localize)`` for S0 default table.

    *sim_and_profile_parity* aligns harness startup with pytest / ``DynamemTaskExecutor``
    (interactive dynagraph profile, no ZMQ throttle, etc.). *phrase_only_localize* skips
    ``_query_variants`` token expansion on the oneshot memory-query path.
    """
    import os

    env_raw = os.environ.get("EMET_OVMM_S0_PARITY", "").strip().lower()
    if env_raw in ("0", "false", "no"):
        return False, False
    if run_cfg.s0_parity is False:
        return False, False
    if run_cfg.s0_parity is True:
        return True, not use_agentic
    if run_cfg.backend == "ground_truth":
        return False, False
    if not is_s0_default_table_episode(episode):
        return False, False
    return True, not use_agentic


def resolve_object_query(
    episode: FindPhaseEpisode,
    placements: dict[str, dict[str, Any]] | None,
) -> str:
    """Resolve the **agent** object query from episode language (open-vocab).

    Uses ``episode.object`` (cleaned of instance hashes). Does **not** replace a
    usable task string with sim GT ``cat`` from ``object_gt_body`` — that body is for
    *scoring* only. GT cat is consulted only when the episode label is a useless stub
    (``obj`` / ``object``), so full-OVMM episodes that store the manipulable as
    ``object: obj`` + ``object_gt_body`` still get a searchable name.
    """
    raw = str(episode.object or "").strip()
    cleaned = semantic_label_from_instance(raw)
    if cleaned.lower() not in {"", "obj", "object", "body"}:
        return cleaned
    if raw.lower() not in {"", "obj", "object", "body"}:
        return raw
    # Stub episode label: fall back to cleaned GT cat for the designated body only.
    if episode.object_gt_body and placements and episode.object_gt_body in placements:
        gt_raw = str(placements[episode.object_gt_body].get("cat") or "")
        gt_clean = semantic_label_from_instance(gt_raw)
        if gt_clean.lower() not in {"", "obj", "object", "body"}:
            return gt_clean
        if gt_raw:
            return gt_raw
    return cleaned or raw or "object"


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


def take_voxel_localize_stats(voxel_map: Any) -> dict[str, Any]:
    """Copy last DynaMem localize diagnostics (max SigLIP cosine, YoloE hit)."""
    stats = getattr(voxel_map, "_last_localize_stats", None) if voxel_map is not None else None
    if not isinstance(stats, dict):
        return {"max_cosine": None, "yoloe_hit": False, "query": None}
    cos = stats.get("max_cosine")
    return {
        "max_cosine": float(cos) if cos is not None else None,
        "yoloe_hit": bool(stats.get("yoloe_hit")),
        "query": stats.get("query"),
    }


def _merge_voxel_localize_stats(acc: dict[str, Any], last: Any) -> dict[str, Any]:
    if not isinstance(last, dict):
        return acc
    cos = last.get("max_cosine")
    acc_cos = acc.get("max_cosine")
    if cos is not None and (acc_cos is None or float(cos) > float(acc_cos)):
        acc["max_cosine"] = float(cos)
        if last.get("query") is not None:
            acc["query"] = last.get("query")
    if last.get("yoloe_hit"):
        acc["yoloe_hit"] = True
    return acc


def localization_detect_fields(
    obj_stats: dict[str, Any] | None,
    recep_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """YoloE / SigLIP audit fields for find-phase JSON (oneshot voxel localize)."""
    obj_stats = obj_stats or {}
    recep_stats = recep_stats or {}
    return {
        "obj_max_cosine": obj_stats.get("max_cosine"),
        "obj_yoloe_hit": bool(obj_stats.get("yoloe_hit")) if obj_stats.get("yoloe_hit") is not None else False,
        "recep_max_cosine": recep_stats.get("max_cosine"),
        "recep_yoloe_hit": bool(recep_stats.get("yoloe_hit")) if recep_stats.get("yoloe_hit") is not None else False,
    }


def ovmm_find_query_row(query: Any) -> dict[str, Any]:
    """Shared FindObj/FindRec JSON keys (Habitat and sim).

    Includes localize success/source, agentic meta, pred XYZ, and oneshot
    voxel-detect audit fields (empty / false for agentic).
    """
    return {
        **query.localize_fields(),
        **localization_pred_fields(query.obj_xyz, query.recep_xyz),
        **localization_detect_fields(query.obj_detect_stats, query.recep_detect_stats),
    }


def score_ovmm_find_query(
    query: Any,
    *,
    placements: dict[str, dict[str, Any]] | None,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    radius_m: float,
    object_gt_body: str | None = None,
    frame: PlanarFrame = "mujoco_xy",
) -> dict[str, Any]:
    """Score a shared FindObj/FindRec outcome (same GT radius logic on both runners)."""
    return compute_find_phase_metrics(
        obj_pred_xyz=query.obj_xyz,
        recep_pred_xyz=query.recep_xyz,
        placements=placements,
        object_query=object_query,
        start_recep=start_recep,
        goal_recep=goal_recep,
        radius_m=radius_m,
        object_gt_body=object_gt_body,
        frame=frame,
    )


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


def _query_variants(query: str) -> list[str]:
    """Expand a text query with cleaned labels and substring tokens (language-side only).

    Sim GT category strings are never injected here: adding fixture paths (e.g.
    ``cab … door handle``) to open-vocab ``localize_text`` leaked long descriptive
    strings into the search query.
    """
    base = str(query or "").strip()
    variants: list[str] = []
    if base:
        variants.append(base)
    cleaned = semantic_label_from_instance(base)
    if cleaned and cleaned.lower() != base.lower():
        variants.append(cleaned)
    low = (cleaned or base).lower()
    for token in low.replace("_", " ").split():
        if len(token) >= 2 and token not in {v.lower() for v in variants}:
            if _HEX_TOKEN_RE.match(token) or _NUMERIC_TOKEN_RE.match(token):
                continue
            variants.append(token)
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        key = v.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _category_match_rank(query: str, cat: str) -> tuple[int, int, int]:
    """Rank GT category quality for scoring (lower is better).

    Prefer exact label matches and shorter category strings over long descriptive
    fixture paths that only substring-match the episode query.
    """
    q = str(query or "").strip().lower()
    c = str(cat or "").strip().lower()
    c_clean = semantic_label_from_instance(cat).strip().lower()
    exact = 0 if q and (c == q or c_clean == q) else 1
    label = c_clean or c
    words = len(label.replace("_", " ").split()) if label else 999
    return (exact, words, len(label))


def pick_find_recep_gt_body(
    placements: dict[str, dict[str, Any]],
    goal_recep: str,
) -> str | None:
    """Choose a single GT body for FindRec scoring (analogous to FindObj disambiguation).

    Among category matches, prefer exact / short labels so long articulated-part
    descriptions that merely contain the query token are not the scoring target.
    """
    bodies = bodies_matching_category(placements, goal_recep)
    if not bodies:
        return None

    def _key(body: str) -> tuple:
        cat = str(placements[body].get("cat") or body)
        return (*_category_match_rank(goal_recep, cat), body)

    return min(bodies, key=_key)


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
    session: dict[str, Any] | None,
    phrase_only: bool = False,
) -> tuple[np.ndarray | None, str]:
    """Voxel-map localization only (preferred for find-phase; avoids merged-graph centroid drift)."""
    acc: dict[str, Any] = {"query": query, "max_cosine": None, "yoloe_hit": False}
    found_xyz: np.ndarray | None = None
    found_q = query
    variants = [str(query or "").strip()] if phrase_only else _query_variants(query)
    for q in variants:
        if not q:
            continue
        result = voxel_map.localize_text(q, debug=False, return_debug=True)
        acc = _merge_voxel_localize_stats(acc, getattr(voxel_map, "_last_localize_stats", None))
        target = result[0] if isinstance(result, (list, tuple)) else result
        if target is not None:
            xyz = localize_point_to_world_xy(target, session)
            if xyz is not None and found_xyz is None:
                found_xyz = xyz
                found_q = q
    voxel_map._last_localize_stats = acc
    if found_xyz is not None:
        return found_xyz, found_q
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
    phrase_only: bool = False,
) -> tuple[np.ndarray | None, bool, str, LocalizeSource | None]:
    """
    Query memory for FindObj/FindRec localization with query variants and fallbacks.

    Returns:
        ``(world_xyz, success, query_used, localize_source)`` where ``world_xyz`` is suitable
        for GT distance checks and ``localize_source`` names the winning code path (``None`` on miss).
    """
    sess = session if convert_nav_to_world else None

    if prefer_voxel and voxel_map is not None and hasattr(voxel_map, "localize_text"):
        xyz, q_used = _voxel_localize(voxel_map, query, session=sess, phrase_only=phrase_only)
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
    fallback_variants = [str(query or "").strip()] if phrase_only else _query_variants(query)
    for q in fallback_variants:
        if not q:
            continue
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


def run_ovmm_oneshot_find_pair(
    memory: Any,
    *,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    placements: dict[str, dict[str, Any]] | None,
    voxel_map: Any | None,
    prefer_voxel: bool,
    session: dict[str, Any] | None = None,
    convert_nav_to_world: bool = False,
    planar_frame: PlanarFrame = "mujoco_xy",
    phrase_only: bool = False,
    capture_voxel_stats: bool = False,
) -> Any:
    """One-shot FindObj + FindRec memory localize (shared Habitat / sim ablation)."""
    from emet.eval.ovmm_agentic_find import OvmmFindQueryOutcome, empty_ovmm_agentic_meta

    obj_xyz, obj_ok, obj_q_used, obj_source = query_find_phase_localization(
        memory,
        object_query,
        placements=placements,
        session=session,
        near_recep=start_recep,
        voxel_map=voxel_map,
        convert_nav_to_world=convert_nav_to_world,
        prefer_voxel=prefer_voxel,
        planar_frame=planar_frame,
        phrase_only=phrase_only,
    )
    obj_stats = take_voxel_localize_stats(voxel_map) if capture_voxel_stats else {}
    recep_xyz, recep_ok, recep_q_used, recep_source = query_find_phase_localization(
        memory,
        goal_recep,
        placements=placements,
        session=session,
        near_recep=goal_recep,
        voxel_map=voxel_map,
        convert_nav_to_world=convert_nav_to_world,
        prefer_voxel=prefer_voxel,
        planar_frame=planar_frame,
        phrase_only=phrase_only,
    )
    recep_stats = take_voxel_localize_stats(voxel_map) if capture_voxel_stats else {}
    return OvmmFindQueryOutcome(
        obj_xyz=obj_xyz,
        obj_ok=obj_ok,
        obj_query_used=obj_q_used,
        obj_source=obj_source,
        recep_xyz=recep_xyz,
        recep_ok=recep_ok,
        recep_query_used=recep_q_used,
        recep_source=recep_source,
        meta=empty_ovmm_agentic_meta(use_agentic=False),
        obj_detect_stats=obj_stats,
        recep_detect_stats=recep_stats,
    )


def run_ovmm_gt_oracle_find_pair(
    memory: Any,
    *,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    object_gt_body: str | None,
    placements: dict[str, dict[str, Any]] | None,
    voxel_map: Any | None = None,
    session: dict[str, Any] | None = None,
    convert_nav_to_world: bool = False,
    planar_frame: PlanarFrame = "mujoco_xy",
) -> Any:
    """Sim-only FindObj/FindRec from MuJoCo placements (upper bound).

    Habitat ground_truth is **not** this path: it localizes from the GT graph
    after ``refresh_ground_truth``. Fallback to one-shot memory query when a
    placement key is missing.
    """
    from emet.eval.ovmm_agentic_find import OvmmFindQueryOutcome, empty_ovmm_agentic_meta

    place = placements or {}
    obj_xyz = None
    obj_ok = False
    obj_q_used = object_query
    obj_source: LocalizeSource | None = None
    if object_gt_body and object_gt_body in place:
        obj_xyz = np.asarray(place[object_gt_body]["pos"][:3], dtype=np.float64)
        obj_ok = True
        obj_source = "gt_placement"
        obj_q_used = str(place[object_gt_body].get("cat") or object_gt_body)
    else:
        obj_xyz, obj_ok, obj_q_used, obj_source = query_find_phase_localization(
            memory,
            object_query,
            placements=place,
            session=session,
            near_recep=start_recep,
            voxel_map=voxel_map,
            convert_nav_to_world=convert_nav_to_world,
            prefer_voxel=False,
            planar_frame=planar_frame,
        )
    recep_xyz = None
    recep_ok = False
    recep_q_used = goal_recep
    recep_source: LocalizeSource | None = None
    needle = goal_recep.lower()
    for bname, meta in place.items():
        cat = str(meta.get("cat") or meta.get("label") or bname).lower()
        if needle in cat or cat in needle:
            recep_xyz = np.asarray(meta["pos"][:3], dtype=np.float64)
            recep_ok = True
            recep_source = "gt_placement"
            recep_q_used = cat
            break
    if not recep_ok:
        recep_xyz, recep_ok, recep_q_used, recep_source = query_find_phase_localization(
            memory,
            goal_recep,
            placements=place,
            session=session,
            near_recep=goal_recep,
            voxel_map=voxel_map,
            convert_nav_to_world=convert_nav_to_world,
            prefer_voxel=False,
            planar_frame=planar_frame,
        )
    return OvmmFindQueryOutcome(
        obj_xyz=obj_xyz,
        obj_ok=obj_ok,
        obj_query_used=obj_q_used,
        obj_source=obj_source,
        recep_xyz=recep_xyz,
        recep_ok=recep_ok,
        recep_query_used=recep_q_used,
        recep_source=recep_source,
        meta=empty_ovmm_agentic_meta(use_agentic=False),
    )


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
    """Score FindObj: predicted XYZ within ``radius_m`` of chosen GT object body.

    When no GT body can be resolved, the phase is **unscored** (not a localization miss):
    ``find_object_scored=False`` and ``find_object_unscored_reason`` explain why.
    """
    if not placements:
        return {
            "find_object_success": False,
            "find_object_scored": False,
            "find_object_unscored_reason": "no_placements",
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
    if gt_body is None:
        return {
            "find_object_success": False,
            "find_object_scored": False,
            "find_object_unscored_reason": "no_gt_match",
            "localization_err_obj_m": None,
            "gt_object_body": None,
            "obj_pred_present": pred is not None,
        }
    if pred is None:
        return {
            "find_object_success": False,
            "find_object_scored": True,
            "find_object_unscored_reason": None,
            "localization_err_obj_m": None,
            "gt_object_body": gt_body,
        }
    err_xy = distance_to_placement_xy(pred, placements[gt_body], frame=frame)
    return {
        "find_object_success": err_xy <= float(radius_m),
        "find_object_scored": True,
        "find_object_unscored_reason": None,
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
    """Score FindRec against one disambiguated GT body matching ``goal_recep``.

    Candidate matches are retained in ``gt_recep_bodies`` for debugging; scoring uses
    :func:`pick_find_recep_gt_body` so long substring matches are not vacuous hits.
    """
    if not placements:
        return {
            "find_recep_success": False,
            "find_recep_scored": False,
            "find_recep_unscored_reason": "no_placements",
            "localization_err_recep_m": None,
            "gt_recep_body": None,
            "gt_recep_bodies": [],
        }
    recep_bodies = bodies_matching_category(placements, goal_recep)
    gt_body = pick_find_recep_gt_body(placements, goal_recep)
    pred = _pred_xyz_array(pred_xyz)
    if gt_body is None:
        return {
            "find_recep_success": False,
            "find_recep_scored": False,
            "find_recep_unscored_reason": "no_gt_match",
            "localization_err_recep_m": None,
            "gt_recep_body": None,
            "gt_recep_bodies": recep_bodies,
            "recep_pred_present": pred is not None,
        }
    if pred is None:
        return {
            "find_recep_success": False,
            "find_recep_scored": True,
            "find_recep_unscored_reason": None,
            "localization_err_recep_m": None,
            "gt_recep_body": gt_body,
            "gt_recep_bodies": recep_bodies,
        }
    err_xy = distance_to_placement_xy(pred, placements[gt_body], frame=frame)
    return {
        "find_recep_success": err_xy <= float(radius_m),
        "find_recep_scored": True,
        "find_recep_unscored_reason": None,
        "localization_err_recep_m": err_xy,
        "gt_recep_body": gt_body,
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
    """Combine FindObj / FindRec scores and OVMM-style partial success.

    Partial success averages **scored** phases only. Unscored phases (no GT match)
    do not count as localization failures in the mean.
    """
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
    scored_vals: list[float] = []
    if obj.get("find_object_scored"):
        scored_vals.append(float(obj["find_object_success"]))
    if rec.get("find_recep_scored"):
        scored_vals.append(float(rec["find_recep_success"]))
    partial = float(sum(scored_vals) / len(scored_vals)) if scored_vals else 0.0
    return {
        **obj,
        **rec,
        "find_partial_success": partial,
        "find_phases_scored": len(scored_vals),
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
    s0_parity: bool = False,
    use_agentic: bool = False,
) -> Any:
    """Configure dynagraph merge/staleness for backend comparison runs.

    One-shot and agentic runs share the evaluation profile. Legacy S0 flags
    remain accepted but no longer select a different memory configuration.
    """
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
    graph_memory_input_path: str | None = None,
    s0_parity: bool = False,
):
    """Instantiate the controller for a memory backend.

    When ``graph_memory_input_path`` is set, GraphEQA / Dynagraph reload the
    exported scene map (graph + voxel). Dynamem loads ``voxel_map.pkl`` only.
    """
    from emet.eval.benchmark_dynagraph import harness_controller_kwargs
    from emet.memory.format import VOXEL_PICKLE_FILENAME

    backend = normalize_benchmark_backend(backend)
    harness_name = "interactive" if s0_parity else "ovmm_find_phase"
    harness_kw = harness_controller_kwargs(parameters, harness=harness_name, method=str(backend))
    use_instance_graph = bool(
        harness_kw.get(
            "use_instance_graph",
            backend in (STATIC_GRAPH, DYNAGRAPH, LAZY_GRAPH, GROUND_TRUTH),
        )
    )
    manipulation_only = bool(harness_kw.get("manipulation_only", False))
    input_path = str(graph_memory_input_path) if graph_memory_input_path else None
    live_rerun = eval_rerun_enabled()
    if s0_parity and backend in (STATIC_GRAPH, DYNAGRAPH, LAZY_GRAPH, GROUND_TRUTH):
        from emet.eval.stack import build_memory_agent

        agent = build_memory_agent(
            robot=robot,
            parameters=parameters,
            backend=backend,
            harness="interactive",
            server_ip="127.0.0.1",
            cpu_only=cpu_only,
            manipulation_only=manipulation_only,
            use_instance_graph=use_instance_graph,
            use_sensor_perception=use_sensor_perception,
            eqa=False,
            defer_eqa_vllm=True,
            apply_harness_profile=False,
        )
        if input_path and backend != "dynamem":
            from emet.memory.format import VOXEL_PICKLE_FILENAME

            voxel_pickle = Path(input_path) / VOXEL_PICKLE_FILENAME
            vm = getattr(agent, "voxel_map", None)
            if voxel_pickle.is_file() and vm is not None and hasattr(vm, "read_from_pickle"):
                vm.read_from_pickle(str(voxel_pickle))
        if hasattr(agent, "_fast_explore_lookaround"):
            agent._fast_explore_lookaround = True
        agent.start()
        return agent
    if backend == "dynamem":
        from emet.controller.controller_dynamem import DynamemController

        agent = DynamemController(
            robot,
            parameters,
            save_rerun=False,
            enable_live_rerun=live_rerun,
            use_instance_memory=True,
            cpu_only=cpu_only,
            eqa=False,
            defer_eqa_vllm=True,
        )
        if input_path:
            voxel_pickle = Path(input_path) / VOXEL_PICKLE_FILENAME
            vm = getattr(agent, "voxel_map", None)
            if voxel_pickle.is_file() and vm is not None and hasattr(vm, "read_from_pickle"):
                vm.read_from_pickle(str(voxel_pickle))
    elif backend == STATIC_GRAPH:
        from emet.controller.controller_graph_eqa import GraphEQAController

        agent = GraphEQAController(
            robot,
            parameters,
            save_rerun=False,
            enable_live_rerun=live_rerun,
            use_instance_graph=use_instance_graph,
            cpu_only=cpu_only,
            use_sensor_perception=use_sensor_perception,
            manipulation_only=manipulation_only,
            graph_memory_input_path=input_path,
        )
    elif backend == DYNAGRAPH:
        from emet.controller.controller_dynagraph import DynagraphController

        agent = DynagraphController(
            robot,
            parameters,
            save_rerun=False,
            enable_live_rerun=live_rerun,
            cpu_only=cpu_only,
            use_instance_graph=use_instance_graph,
            use_sensor_perception=use_sensor_perception,
            manipulation_only=manipulation_only,
            visualize_ground_truth=compare_to_gt,
            graph_memory_input_path=input_path,
        )
    elif backend == LAZY_GRAPH:
        from emet.controller.controller_lazy_graph import LazyGraphController

        agent = LazyGraphController(
            robot,
            parameters,
            save_rerun=False,
            enable_live_rerun=live_rerun,
            cpu_only=cpu_only,
            manipulation_only=manipulation_only,
            graph_memory_input_path=input_path,
        )
    elif backend == GROUND_TRUTH:
        from emet.controller.controller_dynagraph import DynagraphController

        agent = DynagraphController(
            robot,
            parameters,
            save_rerun=False,
            enable_live_rerun=live_rerun,
            cpu_only=cpu_only,
            use_instance_graph=use_instance_graph,
            use_sensor_perception=False,
            manipulation_only=manipulation_only,
            ground_truth_mode=True,
            graph_memory_input_path=input_path,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")
    # OVMM find is exploration-heavy (frontier sweeps after every investigate);
    # the full 4-pan look-around dominated wall time (>=90% of an episode in
    # teleport mode). Halve the pan count like run_dynagraph's explore loop.
    if hasattr(agent, "_fast_explore_lookaround"):
        agent._fast_explore_lookaround = True
    agent.start()
    return agent


def get_memory_backend_for_agent(agent: Any, backend: MemoryBackendName):
    """Return unified memory backend for find-phase queries."""
    from emet.memory.backend import get_memory_backend

    if backend == "dynamem":
        return get_memory_backend("dynamem", voxel_map=agent.voxel_map)
    return get_memory_backend(
        STATIC_GRAPH,
        graph_memory=agent.graph_memory,
        voxel_map=getattr(agent, "voxel_map", None),
    )


def _is_default_table_rby1_agent(agent: Any) -> bool:
    """True when agent runs on Galaxea / rby1 in the default MuJoCo table scene."""
    robot = getattr(agent, "robot", None)
    if robot is None:
        return False
    get_sess = getattr(robot, "get_emet_session", None)
    if not callable(get_sess):
        return False
    from emet.simulation.sim_object_placements import is_default_table_environment

    sess = get_sess() or {}
    env = sess.get("environment") or {}
    env_kind = env.get("kind")
    if not is_default_table_environment(env_kind if isinstance(env_kind, str) else None):
        return False
    rid = str(sess.get("emet_robot_id") or getattr(robot, "name", "") or "").lower()
    return rid in ("rby1", "galaxea_r1")


def _prepare_default_table_rby1_mapping_view(agent: Any) -> bool:
    """Back up and face the default-table workspace so the horizontal ZED sees tabletop objects.

    Galaxea R1 / rby1 spawn near the origin with a level ZED; Stretch ``look_front`` pitches
    the head down ~30°. Without backing up (~2.5 m toward +Y) and pitching torso1, SigLIP
    voxel memory never gets points on object1/object2 (false floor/sky matches instead).
    """
    import os

    if os.environ.get("EMET_OVMM_SKIP_TABLE_MAPPING_POSE", "").strip().lower() in ("1", "true", "yes"):
        return False
    if not _is_default_table_rby1_agent(agent):
        return False
    robot = getattr(agent, "robot", None)
    assert robot is not None
    timeout_fn = getattr(agent, "_find_phase_nav_timeout", None)
    timeout = float(timeout_fn()) if callable(timeout_fn) else 30.0
    move = getattr(robot, "move_base_to", None)
    if callable(move):
        # Episode frame: face −world Y (table at y≈−1). y≈1.5 m keeps tabletop depth < max_depth (2.5 m).
        move(
            np.array([0.0, 1.5, np.pi], dtype=np.float64),
            relative=False,
            blocking=True,
            timeout=timeout,
        )
    look_front = getattr(robot, "look_front", None)
    if callable(look_front):
        look_front(blocking=True, timeout=timeout)
    wait_obs = getattr(robot, "wait_for_obs", None)
    if callable(wait_obs):
        wait_obs(timeout=timeout)
    from emet.controller.controller_dynamem import DYNAMEM_HEAD_SETTLE_S

    time.sleep(DYNAMEM_HEAD_SETTLE_S)
    return True


def run_mapping_protocol(
    agent: Any,
    mapping_max_nav_steps: int | None = None,
    not_rotate: bool = False,
    mapping_rotate_steps: int | None = None,
    trace_meta: dict[str, Any] | None = None,
    *,
    explore_steps: int | None = None,
) -> int:
    """Rotate in place and optionally run frontier explore steps.

    When ``mapping_max_nav_steps>0`` and the agent has a graph/voxel stack
    (dynagraph), mapping uses the shared :class:`AgenticEQAExecutor` in
    ``mode=explore`` (coverage, no object ``toward``) — same loop as HM-EQA.
    ``S0`` ``mapping_max_nav_steps=0`` stays rotate-only.

    ``explore_steps`` is a deprecated alias of ``mapping_max_nav_steps``.
    """
    from emet.utils.logger import Logger

    _logger = Logger(__name__)

    n_explore = int(
        resolve_mapping_max_nav_steps(
            mapping_max_nav_steps,
            explore_steps,
            source="run_mapping_protocol",
            default=0,
            warn=bool(explore_steps is not None and mapping_max_nav_steps is None),
        )
        or 0
    )
    steps = 0
    # Agentic coverage mapping when explore steps are requested and not GT.
    if n_explore > 0 and not backend_uses_ground_truth(agent):
        gm = getattr(agent, "graph_memory", None)
        vm = getattr(agent, "voxel_map", None)
        if vm is None and hasattr(agent, "get_voxel_map"):
            try:
                vm = agent.get_voxel_map()
            except Exception:
                vm = None
        has_stack = gm is not None
        # Keep S0 rotate-only: S0 episodes have mapping_max_nav_steps==0, so they
        # never reach here. Kitchen / robocasa mapping with mapping_max_nav_steps>0
        # is coverage-only and must not depend on an 8-way spawn spin — but it
        # does need a seeded disk: without any scan the explored area is just the
        # ``local_radius`` start disk (~0.25 m), every frontier is outside the reachable
        # map, and sample_navigation clamps every explore goal to ~0 m (nav
        # "happens" but never moves). A 4-step rotate seed fixes that.
        if has_stack:
            if not not_rotate:
                rotate = getattr(agent, "rotate_in_place", None)
                if callable(rotate):
                    rotate(n_steps=mapping_rotate_steps)
                    steps += 1
            seed_fn = getattr(agent, "_seed_local_radius_explored", None)
            if callable(seed_fn) and vm is not None:
                try:
                    seed_fn(vm)
                except Exception:
                    pass
            try:
                from emet.memory.graph_eqa.agentic_eqa import run_agentic_eqa_result

                meta = dict(trace_meta or {})
                meta.setdefault("ovmm_phase", "mapping")
                result = run_agentic_eqa_result(
                    agent,
                    None,
                    goal="explore and map the environment",
                    max_nav_steps=n_explore,
                    max_rounds=n_explore + 1,
                    # Coverage mapping must not require the VLM router — the VL
                    # worker only starts at find query time. Deterministic
                    # explore_frontier / finish fallback is exactly the coverage
                    # policy (uncovered-first frontier picks).
                    router=False,
                    trace_meta=meta,
                )
                # Stash for per-episode JSON (jobs report ``map=`` already exists).
                agent._ovmm_mapping_result = result  # type: ignore[attr-defined]
                n_agentic = int(getattr(result, "n_explore", 0) + getattr(result, "n_nav", 0))
                steps += n_agentic
                if n_agentic > 0:
                    if backend_uses_ground_truth(agent):
                        refresh = getattr(agent, "refresh_ground_truth", None)
                        if callable(refresh):
                            refresh()
                    return steps
                _logger.warning("agentic mapping produced 0 nav/explore steps; falling back to execute_action")
            except Exception as exc:
                _logger.warning(f"agentic mapping failed, falling back to execute_action: {exc}")

    if not not_rotate and steps == 0:
        # rotate_in_place backs up + look_front on default_table_rby1 via
        # _prepare_default_table_rby1_mapping_view, then scans with update() each step.
        rotate = getattr(agent, "rotate_in_place", None)
        if callable(rotate):
            rotate(n_steps=mapping_rotate_steps)
        steps += 1
    seed_fn = getattr(agent, "_seed_local_radius_explored", None)
    vm = getattr(agent, "voxel_map", None)
    if vm is None and hasattr(agent, "get_voxel_map"):
        vm = agent.get_voxel_map()
    if callable(seed_fn) and vm is not None:
        seed_fn(vm)
    for _ in range(max(0, n_explore)):
        agent.execute_action("")
        steps += 1
    if backend_uses_ground_truth(agent):
        refresh = getattr(agent, "refresh_ground_truth", None)
        if callable(refresh):
            refresh()
    return steps


def backend_uses_ground_truth(agent: Any) -> bool:
    return getattr(agent, "ground_truth_mode", False) is True


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
    vl_worker: Any | None = None,
) -> dict[str, Any]:
    """
    Run one find-phase episode: sim subprocess, mapping, memory queries, GT scoring.

    Caller should ensure sim dependencies are available for the episode tier.
    """
    import os
    import socket
    import subprocess
    import sys
    import tempfile
    from dataclasses import replace

    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.core.parameters import get_parameters
    from emet.eval.ovmm_agentic_find import (
        attach_ovmm_episode_debug_dir,
        run_ovmm_find_queries,
        should_use_agentic_find,
    )
    from emet.memory.graph_eqa.sim_ground_truth_graph import (
        gt_graph_completeness,
        instance_gt_association_recall,
        read_sim_object_placements,
    )
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
    from emet.utils.process_tree import popen_session, terminate_process_tree

    use_agentic = should_use_agentic_find(run_cfg.backend, agentic_find=run_cfg.agentic_find)
    if run_cfg.query_driven_memory and (run_cfg.backend != "lazy_graph" or not use_agentic):
        raise ValueError("query-driven memory requires lazy_graph and agentic find")
    s0_parity, s0_phrase_only = resolve_s0_parity_flags(episode, run_cfg, use_agentic=use_agentic)

    if run_cfg.seed is not None:
        set_find_phase_run_seed(int(run_cfg.seed))

    repo = repo_root or Path(__file__).resolve().parents[3]
    sim_cfg = load_sim_launch_config_from_path(episode.sim)
    robot_kind = str(getattr(sim_cfg, "robot", None) or "stretch")
    port_offset = int(run_cfg.port_offset)
    recv_port = 4401 + port_offset

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    if not s0_parity:
        # Image encoding/rendering must not run in unbounded busy loops while the
        # simulator is also executing navigation. These rates are ample for mapping.
        env.setdefault("EMET_ZMQ_FULL_HZ", "5")
        env.setdefault("EMET_ZMQ_STATE_HZ", "30")
        env.setdefault("EMET_ZMQ_SERVO_HZ", "10")
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

    def wait_port(port: int, timeout: float, *, proc: subprocess.Popen | None = None) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=2):
                    return True
            except OSError:
                time.sleep(0.5)
        return False

    robot = None
    agent = None
    server = None
    server_log: Path | None = None
    server_log_fh = None
    t0 = time.monotonic()
    init_wall_s = 0.0
    mapping_wall_s = 0.0
    query_wall_s = 0.0
    previous_vl_endpoint = os.environ.get("EMET_VL_ENDPOINT")
    vl_endpoint_used = previous_vl_endpoint
    worker_started = False
    try:
        parameters = apply_backend_parameters(
            get_parameters("dynav_config.yaml", robot=robot_kind),
            run_cfg.backend,
            merge_xy_m=run_cfg.merge_xy_m,
            staleness_horizon=run_cfg.staleness_horizon,
            s0_parity=s0_parity,
            use_agentic=use_agentic,
        )
        if run_cfg.query_driven_memory:
            from emet.eval.benchmark_dynagraph import enable_query_driven_memory

            enable_query_driven_memory(parameters, run_cfg.backend)
        # Keep the shared SigLIP encoder (get_shared_mask_siglip_encoder, load-once)
        # so the voxel semantic memory gets per-point features — the agentic find
        # seeds receptacle search from SigLIP text grounding (no label match needed).
        # Setting encoder=None here disabled semantic memory entirely.
        if run_cfg.perfect_depth:
            parameters["debug_perfect_sensor_depth"] = True
        parameters["find_phase_nav_step_timeout_s"] = nav_timeout
        parameters["enable_tts"] = False
        det_conf = float((parameters.get("detection", {}) or {}).get("confidence_threshold", 0.05))

        # Load SigLIP/YoloE before MuJoCo EGL is up. Concurrent Robocasa EGL + VL worker
        # + YoloE get_text_pe has wedged find-phase init (no progress after SigLIP weights,
        # mujoco cancelled_write_bytes thrash). Agent construction reuses shared instances.
        if not run_cfg.cpu_only:
            print("OVMM find: preloading SigLIP + YoloE before sim…", flush=True)
            from emet.perception.detection.yoloe import get_shared_yoloe_perception
            from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

            get_shared_mask_siglip_encoder(version="so400m", device="cuda", feature_matching_threshold=0.14)
            get_shared_yoloe_perception(confidence_threshold=det_conf, device="cuda", size="l")
            print("OVMM find: perception preload done", flush=True)

        # Capture server stderr so bind failures include the real crash reason
        # (DEVNULL hid layout/asset errors on Robocasa multi-env sweeps).
        log_dir = Path(tempfile.mkdtemp(prefix="emet_ovmm_sim_"))
        server_log = log_dir / "mujoco_server.stderr"
        server_log_fh = server_log.open("w", encoding="utf-8")
        server = popen_session(
            server_cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=server_log_fh,
        )
        bind_timeout = 180.0 if sim_kind in ("molmospaces", "robocasa") else 120.0
        if not wait_port(recv_port, bind_timeout, proc=server):
            try:
                server_log_fh.flush()
            except Exception:
                pass
            err_tail = ""
            try:
                err_tail = server_log.read_text(encoding="utf-8", errors="replace")[-2000:]
            except Exception:
                err_tail = ""
            rc = server.poll()
            raise RuntimeError(
                f"sim server did not bind port {recv_port} (port_offset={port_offset}, "
                f"exit={rc}, sim={episode.sim})" + (f":\n{err_tail}" if err_tail.strip() else "")
            )
        settle = 2.0 if s0_parity else (25.0 if sim_kind in ("molmospaces", "robocasa") else 15.0)
        if run_cfg.cpu_only and not s0_parity:
            settle += 15.0
        time.sleep(settle)

        # Defer ZMQ start until controller is ready unless S0 parity (pytest-style).
        robot = create_robot_client_from_cli(
            robot_kind,
            "127.0.0.1",
            port_offset=port_offset,
            enable_rerun_server=eval_rerun_enabled(),
            start_immediately=False,
            allow_missing_depth=True,
        )

        cache_dir = None
        map_source = "live"
        # S0 parity always maps live (pytest path); cached find_phase maps skew localize.
        use_scene_cache = bool(run_cfg.use_scene_cache) and not s0_parity
        if use_scene_cache and run_cfg.backend != "ground_truth" and not episode.floor_object:
            from emet.eval.scene_map_cache import resolve_scene_cache_for_sim

            cache_dir = resolve_scene_cache_for_sim(sim_cfg, enabled=True)
            if cache_dir is not None:
                map_source = "cache"

        t_init0 = time.monotonic()
        agent = create_find_phase_agent(
            robot,
            parameters,
            run_cfg.backend,
            cpu_only=run_cfg.cpu_only,
            compare_to_gt=run_cfg.compare_to_gt,
            use_sensor_perception=run_cfg.use_sensor_perception,
            graph_memory_input_path=str(cache_dir) if cache_dir is not None else None,
        )
        attach_ovmm_episode_debug_dir(agent)
        if not s0_parity:
            # Controller already started ZMQ + nav posture; apply eval velocity after.
            robot.set_velocity(v=30.0, w=15.0)
        init_wall_s = time.monotonic() - t_init0
        if episode.floor_object:
            placements_before_floor = read_sim_object_placements(robot.get_emet_session()) or {}
            floor_body = episode.object_gt_body
            if not floor_body or floor_body not in placements_before_floor:
                raise RuntimeError(f"floor setup requires object_gt_body present in sim placements: {floor_body!r}")
            from emet.eval.ovmm_full import drop_object_to_floor

            if not drop_object_to_floor(
                robot,
                floor_body,
                placements_before_floor,
                floor_z_m=episode.floor_z_m if episode.floor_z_m is not None else 0.02,
            ):
                raise RuntimeError(f"floor setup failed for body {floor_body!r}")
        if run_cfg.backend == "ground_truth":
            refresh = getattr(agent, "refresh_ground_truth", None)
            if callable(refresh):
                n_gt = refresh()
                if n_gt == 0:
                    raise RuntimeError("ground-truth mode: no sim_object_placements in session")

        t_map0 = time.monotonic()
        mapping_budget = (
            int(run_cfg.explore_steps_override)
            if run_cfg.explore_steps_override is not None
            else int(episode.mapping_max_nav_steps)
        )
        not_rotate = bool(run_cfg.not_rotate)
        if cache_dir is not None:
            # Baseline already mapped; skip rotate/explore.
            mapping_budget = 0
            not_rotate = True
        mapping_frames_before = len(getattr(getattr(agent, "voxel_map", None), "observations", ()) or ())
        n_steps = run_mapping_protocol(
            agent,
            mapping_max_nav_steps=mapping_budget,
            not_rotate=not_rotate,
            mapping_rotate_steps=run_cfg.mapping_rotate_steps,
            trace_meta={
                "ovmm_phase": "mapping",
                "episode_id": episode.id,
                "object": str(episode.object),
                "start_recep": episode.start_recep,
                "goal_recep": episode.goal_recep,
            },
        )
        mapping_frames_added = (
            len(getattr(getattr(agent, "voxel_map", None), "observations", ()) or ()) - mapping_frames_before
        )
        mapping_wall_s = time.monotonic() - t_map0
        mapping_result = getattr(agent, "_ovmm_mapping_result", None) if agent is not None else None

        session = robot.get_emet_session()
        placements = read_sim_object_placements(session)
        memory = get_memory_backend_for_agent(agent, run_cfg.backend)
        vm = getattr(agent, "voxel_map", None)
        nav_world = sim_kind == "robocasa"

        object_query = resolve_object_query(episode, placements)

        prefer_voxel = run_cfg.prefer_voxel and run_cfg.backend != "ground_truth"
        t_query0 = time.monotonic()
        if run_cfg.backend == "ground_truth":
            # Oracle: localize directly from sim placements (upper bound for FindObj/FindRec).
            query = run_ovmm_gt_oracle_find_pair(
                memory,
                object_query=object_query,
                start_recep=episode.start_recep,
                goal_recep=episode.goal_recep,
                object_gt_body=episode.object_gt_body,
                placements=placements,
                voxel_map=vm,
                session=session,
                convert_nav_to_world=nav_world,
            )
        else:
            if use_agentic and vl_worker is not None:
                # Keep the VLM unloaded while MuJoCo and the mapping stack initialize.
                vl_endpoint_used = vl_worker.start()
                os.environ["EMET_VL_ENDPOINT"] = vl_endpoint_used
                worker_started = True
                print(f"Managed OVMM VL worker ready for query: {vl_endpoint_used}", flush=True)
            query = run_ovmm_find_queries(
                agent=agent,
                memory=memory,
                use_agentic=use_agentic,
                object_query=object_query,
                start_recep=episode.start_recep,
                goal_recep=episode.goal_recep,
                episode_id=episode.id,
                object_gt_body=episode.object_gt_body,
                max_rounds=run_cfg.agentic_max_rounds,
                max_nav_steps=run_cfg.agentic_max_nav_steps,
                placements=placements,
                voxel_map=vm,
                prefer_voxel=prefer_voxel,
                session=session,
                convert_nav_to_world=nav_world,
                phrase_only=s0_phrase_only,
            )
        find_metrics = score_ovmm_find_query(
            query,
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

        # Mapping-phase agentic coverage diagnostics (mapping_max_nav_steps>0 uses
        # the agentic explore loop; S0 rotate-only has no mapping_result).
        mapping_diag: dict[str, Any] = {}
        if mapping_result is not None:
            mapping_diag = {
                "mapping_n_explore": int(getattr(mapping_result, "n_explore", 0)),
                "mapping_n_nav": int(getattr(mapping_result, "n_nav", 0)),
                "mapping_n_rounds": int(getattr(mapping_result, "n_rounds", 0)),
                "mapping_verified": bool(getattr(mapping_result, "verified", False)),
            }
        else:
            mapping_diag = {
                "mapping_n_explore": None,
                "mapping_n_nav": None,
                "mapping_n_rounds": None,
                "mapping_verified": None,
            }

        metrics = {
            "episode_id": episode.id,
            "tier": episode.tier,
            "backend": run_cfg.backend,
            "query_driven_memory": run_cfg.query_driven_memory,
            "sim": episode.sim,
            "object_query": object_query,
            "start_recep": episode.start_recep,
            "goal_recep": episode.goal_recep,
            "mapping_max_nav_steps": mapping_budget,
            "explore_steps": mapping_budget,
            **mapping_diag,
            "floor_object": bool(episode.floor_object),
            "floor_z_m": float(episode.floor_z_m) if episode.floor_z_m is not None else None,
            "episode_manip_mode": episode.manip_mode,
            "map_source": map_source,
            "scene_cache_dir": str(cache_dir) if cache_dir is not None else None,
            "merge_xy_m": parameters.get("dynagraph_merge_xy_m"),
            "staleness_horizon": parameters.get("dynagraph_staleness_horizon"),
            "perfect_depth": bool(run_cfg.perfect_depth),
            "use_sensor_perception": bool(run_cfg.use_sensor_perception),
            "prefer_voxel": bool(prefer_voxel),
            "s0_parity": bool(s0_parity),
            "s0_phrase_only": bool(s0_phrase_only),
            "s0_oneshot_pytest": False,  # historical result-schema field; special executor removed
            "mapping_rotate_steps_requested": run_cfg.mapping_rotate_steps,
            "mapping_frames_added": mapping_frames_added,
            **ovmm_find_query_row(query),
            "manip_mode": str(run_cfg.manip_mode),
            "init_wall_s": float(init_wall_s),
            "mapping_wall_s": float(mapping_wall_s),
            "query_wall_s": float(query_wall_s),
            "vl_endpoint": vl_endpoint_used,
            "seed": run_cfg.seed,
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
        if worker_started:
            vl_worker.stop()
            if previous_vl_endpoint is None:
                os.environ.pop("EMET_VL_ENDPOINT", None)
            else:
                os.environ["EMET_VL_ENDPOINT"] = previous_vl_endpoint
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
        if server_log_fh is not None:
            try:
                server_log_fh.close()
            except Exception:
                pass
        from emet.utils.port_utils import get_ports, kill_processes_on_port

        for p in get_ports(port_offset):
            kill_processes_on_port(p)
        # Brief settle so the next episode's bind is not racing a dying listener.
        time.sleep(0.5)
