# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""EQA view selection, visual find, scene-graph prompt text, and recall ranking."""

from __future__ import annotations

import math
import os
import re
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

from emet.memory.graph_eqa.eqa.eqa_views import (
    EQA_SAME_OBS_MAX_VISITS,
    EQA_SAME_OBS_PROGRESS_M,
    eqa_look_is_spent,
    rgb_uint8,
    spread_obs_ids_xy,
    tightest_node_crop,
)
from emet.memory.graph_eqa.graph_types import (
    _GRAPH_CANDIDATE_COUNT_DISCLAIMER,
    _RECALL_SOURCE_TIER,
    _WEAK_SIGLIP_FIND_TOKENS,
    SIGLIP_PRESENT_THRESHOLD,
    GraphNode,
    NavHypothesis,
    _object_match_tokens,
    consolidate_relevant_keywords,
    format_graph_node_candidates,
    heuristic_relevant_objects,
    heuristic_relevant_phrases,
    label_matches_relevant_object,
    location_mcq_landmark_phrases,
    node_display_name,
    parse_eqa_action,
)


def eqa_obs_look_spent(self, obs_id: int | None) -> bool:
    """True when this observation has already been inspected (do not re-nav there)."""
    if obs_id is None:
        return False
    oid = int(obs_id)
    attempts = 0
    for node in self._nodes:
        if int(node.obs_id) != oid:
            continue
        attempts = max(attempts, int(getattr(node, "nav_attempts", 0) or 0))
    return eqa_look_is_spent(self._obs_nav_dists.get(oid, ()), nav_attempts=attempts)

def next_unspent_eqa_obs_id(
    self,
    candidates: list[int] | None,
    *,
    skip: set[int] | None = None,
) -> int | None:
    """First usable observation in ``candidates`` that is not look-spent."""
    blocked = {int(x) for x in (skip or set())}
    for raw in candidates or []:
        oid = int(raw)
        if oid in blocked or not self._obs_usable_for_eqa_image(oid):
            continue
        if self.eqa_obs_look_spent(oid):
            continue
        return oid
    return None

def _obs_xy(self, obs_id: int) -> np.ndarray | None:
    obs = self._observation_by_id(int(obs_id))
    if obs is None or getattr(obs, "xyz", None) is None:
        return None
    return np.asarray(obs.xyz, dtype=float)[:2]

def _spread_obs_xy(self, obs_ids: list[int], *, max_n: int) -> list[int]:
    return spread_obs_ids_xy(obs_ids, self._obs_xy, max_n=max_n)

def _eqa_find_phrases(self) -> list[str]:
    phrases: list[str] = []
    for raw in (
        list(self._relevant_phrases or [])
        + list(self._relevant_objects or [])
        + list(self._confirmed_memory_phrases())
    ):
        text = str(raw or "").strip()
        if text and text not in phrases:
            phrases.append(text)
    return phrases

def _phrases_need_rgb_highlight(self, visual_pins: list[int], phrases: list[str]) -> bool:
    """True when SigLIP retrieve is empty or the question phrases are weak (e.g. ``time``)."""
    if len(visual_pins) >= 2:
        return False
    tokens: set[str] = set()
    for phrase in phrases:
        tokens |= _object_match_tokens(phrase)
    concrete = {tok for tok in tokens if tok not in _WEAK_SIGLIP_FIND_TOKENS}
    if visual_pins and concrete:
        return False
    return True

def _merge_highlight_phrases(self, extra: list[str]) -> None:
    objects = list(self._relevant_objects or [])
    phrases = list(self._relevant_phrases or [])
    for name in extra:
        key = str(name).strip().lower()
        if not key:
            continue
        if key not in objects:
            objects.append(key)
        if key not in phrases:
            phrases.append(key)
    self._relevant_objects = objects
    # Keep an empty phrase list empty so coverage still uses the full object list.
    if self._relevant_phrases:
        self._relevant_phrases = phrases

def _highlight_relevant_from_latest_rgb(self, question: str) -> list[str]:
    """One pass on the live/latest full frame: names that could answer the question.

    Not detector boxes. Skip when the client is missing, the latest view was already
    highlighted for this question, or the reply looks like an EQA field dump.
    """
    client = self.image_description_client
    if client is None:
        return []
    obs = None
    for candidate in reversed(self._observations):
        if self._obs_usable_for_eqa_image(int(candidate.obs_id)):
            obs = candidate
            break
    if obs is None:
        return []
    key = (str(question), int(obs.obs_id))
    if self._eqa_highlight_key == key:
        return []
    prompt = (
        f"Question: {question}\n"
        "Name 1-3 objects visible in this image that could help answer the question. "
        "Comma-separated short nouns only (for example: wall clock, stool). "
        "If nothing relevant is visible, reply none."
    )
    try:
        image = Image.fromarray(rgb_uint8(obs.rgb), mode="RGB")
        raw = client([prompt, image])
    except Exception:
        return []
    self._eqa_highlight_key = key
    text = raw if isinstance(raw, str) else str(raw or "")
    low = text.lower()
    if "reasoning:" in low or "answer:" in low or "confidence:" in low:
        return []
    out: list[str] = []
    for part in re.split(r"[,;\n]", text):
        name = part.strip(" .").lower()
        if not name or name in {"none", "n/a", "nothing", "no", "unknown"}:
            continue
        if len(name) < 3 or len(name) > 40:
            continue
        if len(name.split()) > 4:
            continue
        if name not in out:
            out.append(name)
        if len(out) >= 3:
            break
    return out

def resolve_voxel_frame_to_graph_obs_id(self, voxel_obs_count: int, voxel_map: Any | None = None) -> int | None:
    """Map a DynaMem voxel ``obs_count`` (1-based frame index) to a graph observation id.

    Voxel frames and graph obs ids diverge after instance merges. Prefer a graph view
    whose capture pose matches the voxel camera; fall back to the raw id when it is
    already a usable graph observation.
    """
    voc = int(voxel_obs_count)
    cam = None
    frame_rgb = None
    obs_list = getattr(voxel_map, "observations", None) if voxel_map is not None else None
    if obs_list:
        idx = voc - 1
        if 0 <= idx < len(obs_list):
            frame = obs_list[idx]
            rgb = getattr(frame, "rgb", None)
            if rgb is not None:
                frame_rgb = np.asarray(rgb)
            pose = getattr(frame, "camera_pose", None)
            if pose is not None:
                try:
                    cam = np.asarray(pose, dtype=float).reshape(4, 4)[:3, 3]
                except Exception:
                    cam = None
    best_oid: int | None = None
    best_score = -1.0
    for gobs in self._observations:
        oid = int(gobs.obs_id)
        if not self._obs_usable_for_eqa_image(oid):
            continue
        score = 0.0
        if cam is not None:
            view = gobs.viewer_xyz if gobs.viewer_xyz is not None else gobs.xyz
            if view is not None:
                dist = float(np.linalg.norm(np.asarray(view, dtype=float).reshape(-1)[:3] - cam[:3]))
                if dist < 0.75:
                    score += 10.0 - dist
        if frame_rgb is not None:
            graph_rgb = np.asarray(gobs.rgb)
            if graph_rgb.shape == frame_rgb.shape:
                score += 0.5
                if np.shares_memory(graph_rgb, frame_rgb) or graph_rgb is frame_rgb:
                    score += 5.0
        if score > best_score:
            best_score = score
            best_oid = oid
    if best_oid is not None and best_score > 0:
        return best_oid
    if self._obs_usable_for_eqa_image(voc):
        return voc
    return None

def nearest_graph_obs_to_xyz(self, xyz: Any, *, max_dist_m: float = 2.0) -> int | None:
    """Nearest usable graph observation to a SigLIP voxel point (planar XY)."""
    try:
        target = np.asarray(xyz, dtype=float).reshape(-1)[:2]
    except Exception:
        return None
    if target.size < 2:
        return None
    best_oid: int | None = None
    best_dist = float("inf")
    for gobs in self._observations:
        oid = int(gobs.obs_id)
        if not self._obs_usable_for_eqa_image(oid) or gobs.xyz is None:
            continue
        xy = np.asarray(gobs.xyz, dtype=float).reshape(-1)[:2]
        dist = float(np.linalg.norm(xy - target))
        if dist < best_dist:
            best_dist = dist
            best_oid = oid
    if best_oid is None or best_dist > float(max_dist_m):
        return None
    return best_oid

def _visual_find_obs_ids(self, phrases: list[str], *, max_n: int) -> list[int]:
    """Rank stored RGB by SigLIP retrieve vs question phrases (not YoloE names)."""
    if max_n <= 0:
        return []
    scored: list[tuple[float, int]] = []
    seen: set[int] = set()

    def _add(sim: float, oid: int | None) -> None:
        if oid is None:
            return
        oi = int(oid)
        if not self._obs_usable_for_eqa_image(oi):
            return
        if oi in seen:
            for i, (old_sim, old_oid) in enumerate(scored):
                if old_oid == oi and float(sim) > old_sim:
                    scored[i] = (float(sim), oi)
            return
        seen.add(oi)
        scored.append((float(sim), oi))

    visual_fn = getattr(self, "_visual_find_fn", None)
    if visual_fn is not None:
        for phrase in phrases:
            hits: Any = []
            try:
                hits = visual_fn(str(phrase), max(int(max_n) * 3, 8))
            except TypeError:
                try:
                    hits = visual_fn(str(phrase))
                except Exception:
                    hits = []
            except Exception:
                hits = []
            for item in hits or []:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    _add(float(item[0]), int(item[1]))
                else:
                    try:
                        _add(max(SIGLIP_PRESENT_THRESHOLD, 0.25), int(item))
                    except (TypeError, ValueError):
                        continue
    rank_cache = getattr(self, "_visual_find_rank_cache", None) or {}
    for phrase in phrases:
        key = str(phrase or "").strip().lower()
        if not key:
            continue
        for sim, oid in rank_cache.get(key, []):
            _add(float(sim), int(oid))
    enc = getattr(self, "_confirmed_memory_siglip_encoder", None)
    feats = getattr(self, "_obs_siglip_features", None) or {}
    if enc is not None and feats:
        from emet.memory.graph_eqa.eqa.graph_eqa_siglip import rank_observations_for_phrase

        for phrase in phrases:
            for sim, oid in rank_observations_for_phrase(str(phrase), enc, feats):
                if sim < SIGLIP_PRESENT_THRESHOLD:
                    break
                _add(sim, oid)
    for phrase in phrases:
        key = str(phrase or "").strip().lower()
        if not key:
            continue
        cached = self._siglip_phrase_cache.get(key)
        if cached is None or cached[2] is None:
            continue
        if float(cached[0]) >= SIGLIP_PRESENT_THRESHOLD:
            _add(float(cached[0]), int(cached[2]))
    grounder = getattr(self, "_obs_id_grounder", None)
    if grounder is not None:
        for phrase in phrases:
            try:
                oid = grounder(str(phrase))
            except Exception:
                oid = None
            if oid is not None:
                _add(max(SIGLIP_PRESENT_THRESHOLD, 0.25), int(oid))
    scored.sort(key=lambda t: -t[0])
    ordered = [oid for _sim, oid in scored]
    return self._spread_obs_xy(ordered, max_n=max_n)

def eqa_attached_target_obs_id(self) -> int | None:
    """Attached view to stay on: VLM ``read N`` / ``N``, else Image 1 FIND."""
    ids = list(self.last_eqa_obs_ids or [])
    if not ids:
        return None
    attached = {int(x) for x in ids}
    action = ""
    if self.last_eqa_parsed:
        action = str(self.last_eqa_parsed[3] or "")
    kind, display_index = parse_eqa_action(action)
    if display_index is not None:
        oid = self._resolve_eqa_action_image_ref(display_index, ids)
        if oid is not None and int(oid) in attached and self._obs_usable_for_eqa_image(int(oid)):
            if kind == "read":
                return int(oid)
            phrases = self._eqa_find_phrases()
            visual = set(self._visual_find_obs_ids(phrases, max_n=8))
            if not visual or int(oid) in visual:
                return int(oid)
    oid = int(ids[0])
    if not self._obs_usable_for_eqa_image(oid):
        return None
    phrases = self._eqa_find_phrases()
    visual = set(self._visual_find_obs_ids(phrases, max_n=8))
    if visual:
        return oid if oid in visual else None
    if self._target_visible_in_obs_ids([oid]):
        return oid
    return None

def eqa_stay_on_attached_view(self) -> bool:
    """True when an attached RGB is already the FIND or ``read N`` view — do not frontier-chase."""
    return self.eqa_attached_target_obs_id() is not None

def _obs_visit_count(self, obs_id: int) -> int:
    oid = int(obs_id)
    nav_dists = self._obs_nav_dists.get(oid, ())
    attempts = 0
    for node in self._nodes:
        if int(node.obs_id) == oid:
            attempts = max(attempts, int(getattr(node, "nav_attempts", 0) or 0))
    return max(len(nav_dists), attempts)

def _history_action_stats_for_images(self, obs_ids: list[int]) -> dict[int, dict[str, int]]:
    """Per attached Image index: look/read action picks and Unknown answers in HISTORY."""
    stats: dict[int, dict[str, int]] = {
        idx: {"look": 0, "read": 0, "unknown": 0} for idx in range(1, len(obs_ids) + 1)
    }
    for raw in self._history_outputs:
        line = str(raw or "")
        act_m = re.search(r"\baction=([^\s|]+)", line)
        ans_m = re.search(r"\banswer=([^\s|]+)", line)
        if not act_m:
            continue
        action_raw = act_m.group(1).strip().lower()
        kind, display_index = parse_eqa_action(action_raw)
        if display_index is None or display_index not in stats:
            continue
        if kind == "read":
            stats[display_index]["read"] += 1
        else:
            stats[display_index]["look"] += 1
        if ans_m:
            ans = ans_m.group(1).strip().lower()
            if ans in ("unknown", "none", "?"):
                stats[display_index]["unknown"] += 1
    return stats

def format_eqa_view_status(self, obs_ids: list[int]) -> str:
    """Investigation counters for attached views (debugging signal for the EQA VLM)."""
    if not obs_ids:
        return ""
    covered = True
    try:
        covered = self._graph_covers_relevant_objects()
    except Exception:
        pass
    action_stats = self._history_action_stats_for_images(obs_ids)
    lines = [
        "VIEW_STATUS (investigation counters — not the answer; "
        f">{EQA_SAME_OBS_MAX_VISITS} visits on one Image without progress is risky; "
        "pick another Image or leave action empty to explore when stuck):",
        f"- relevant_objects_in_graph: {'yes' if covered else 'no — explore other rooms'}",
    ]
    for idx, oid in enumerate(obs_ids, start=1):
        oid_i = int(oid)
        visits = self._obs_visit_count(oid_i)
        spent = self.eqa_obs_look_spent(oid_i)
        st = action_stats.get(idx, {})
        look_n = int(st.get("look", 0))
        read_n = int(st.get("read", 0))
        unk_n = int(st.get("unknown", 0))
        risky = spent or visits >= int(EQA_SAME_OBS_MAX_VISITS)
        parts = [
            f"visits={visits}",
            f"look={look_n}",
            f"read={read_n}",
            f"unknown={unk_n}",
        ]
        if spent:
            parts.append("spent=yes")
        if risky:
            parts.append("risky=yes")
        lines.append(f"  Image {idx} (obs {oid_i}): " + ", ".join(parts))
    return "\n".join(lines)

def eqa_should_stay_on_attached_view(self, *, answer: str, confidence: bool) -> bool:
    """Controller gate: investigate closer at most a few times, then release to explore."""
    if not self.eqa_stay_on_attached_view():
        return False
    oid = self.eqa_attached_target_obs_id()
    if oid is None:
        return False
    if self.eqa_obs_look_spent(int(oid)):
        return False
    try:
        if not self._graph_covers_relevant_objects():
            return False
    except Exception:
        pass
    visits = self._obs_visit_count(int(oid))
    if visits >= int(EQA_SAME_OBS_MAX_VISITS):
        return False
    ans = (answer or "").strip().lower()
    action = str(self.last_eqa_parsed[3] or "") if self.last_eqa_parsed else ""
    kind, _display_index = parse_eqa_action(action)
    if not confidence and ans in ("unknown", "none", "?", ""):
        # ``read N`` may need a closer pass; bare Unknown on a FIND view should explore.
        return kind == "read" and visits < int(EQA_SAME_OBS_MAX_VISITS)
    return True

def eqa_approach_attached_find(self, robot_xyt: Any | None = None) -> np.ndarray | None:
    """Waypoint toward Image-1 FIND when the robot is still far; None to stay put."""
    oid = self.eqa_attached_target_obs_id()
    if oid is None or self.eqa_obs_look_spent(oid):
        return None
    waypoint = self._navigation_waypoint_for_obs(int(oid), robot_xyt)
    if waypoint is None:
        return None
    robot_xy = self._robot_planar_xy(robot_xyt)
    if robot_xy is None:
        return waypoint
    dist = float(math.hypot(float(waypoint[0]) - robot_xy[0], float(waypoint[1]) - robot_xy[1]))
    if dist < EQA_SAME_OBS_PROGRESS_M:
        return None
    return waypoint

def _eqa_rgb_for_obs(self, obs_id: int) -> np.ndarray | None:
    """Full camera frame for an observation (scene context for counting)."""
    obs = self._observation_by_id(int(obs_id))
    if obs is None or not self._obs_usable_for_eqa_image(int(obs_id)):
        return None
    return rgb_uint8(obs.rgb)

def _eqa_crop_for_obs(self, obs_id: int) -> np.ndarray | None:
    """Tight detector crop, or None when the bbox is missing or covers the frame."""
    obs = self._observation_by_id(int(obs_id))
    if obs is None or not self._obs_usable_for_eqa_image(int(obs_id)):
        return None
    nodes = [n for n in self._nodes if int(n.obs_id) == int(obs_id) and not n.is_frontier and not n.is_viewpoint]
    return tightest_node_crop(nodes, rgb_uint8(obs.rgb))

def _eqa_pick_closeup_obs_id(
    self,
    obs_ids: list[int],
    look_obs_id: int | None,
    pin_obs: list[int],
) -> int | None:
    """Choose an already-attached scene whose detector crop is a useful extra view."""
    attached = {int(oid) for oid in obs_ids}
    ordered: list[int] = []
    if look_obs_id is not None and int(look_obs_id) in attached:
        ordered.append(int(look_obs_id))
    ordered.extend(int(oid) for oid in pin_obs if int(oid) in attached)
    ordered.extend(int(oid) for oid in obs_ids)
    seen: set[int] = set()
    for oid in ordered:
        if oid in seen:
            continue
        seen.add(oid)
        if self._eqa_crop_for_obs(oid) is not None:
            return oid
    return None

def _eqa_reserve_closeup_slot(
    self,
    obs_ids: list[int],
    pin_obs: list[int],
    look_obs_id: int | None,
) -> list[int]:
    """Drop one extra FIND view so a labeled close-up can occupy the last slot."""
    if len(obs_ids) < 2:
        return list(obs_ids)
    pin_set = {int(x) for x in pin_obs}
    look = int(look_obs_id) if look_obs_id is not None else None
    ids = list(obs_ids)
    for i in range(len(ids) - 1, 0, -1):
        oid = int(ids[i])
        if oid in pin_set and oid != int(ids[0]) and oid != look:
            return ids[:i] + ids[i + 1 :]
    return ids[:-1]

def alternate_nav_target_for_failed_action(
    self,
    question: str,
    blocked_obs_id: int,
    planner: Any,
    base_xyt: Any,
) -> np.ndarray | None:
    """Pick a different frontier/fluid goal when the VLM re-picks a failed image action."""
    frontier_nodes = [
        n for n in self._nodes if getattr(n, "is_frontier", False) and int(n.obs_id) != int(blocked_obs_id)
    ]
    if frontier_nodes:
        frontier_nodes.sort(key=lambda n: (int(getattr(n, "nav_failures", 0)), -int(n.last_seen)))
        pick = frontier_nodes[0]
        return np.array([float(pick.xyz[0]), float(pick.xyz[1]), 1.0], dtype=float)
    return None

def _rank_nodes_for_eqa_prompt(
    self,
    *,
    keywords: list[str] | None = None,
    prefer_obs_ids: list[int] | None = None,
) -> list[GraphNode]:
    """Rank object/frontier nodes for a bounded EQA SCENE_GRAPH block.

    Viewpoints are omitted from the ranked list (they bloat prompts); edges still
    reference kept object ids. Frontiers are included and ranked after objects.
    """
    from emet.memory.graph_eqa.spatial.frontier_nodes import keyword_overlap_score

    kws = list(keywords or self._relevant_objects or [])
    prefer = {int(x) for x in (prefer_obs_ids or self.last_eqa_obs_ids or [])}
    objects: list[tuple[float, GraphNode]] = []
    frontiers: list[tuple[float, GraphNode]] = []
    for n in self._nodes:
        if n.is_viewpoint:
            continue
        kw = keyword_overlap_score(list(n.labels or []), kws) if kws else 0.0
        support = float(getattr(n, "support_count", 1) or 1)
        prefer_bonus = 2.0 if int(n.obs_id) in prefer else 0.0
        score = 10.0 * kw + prefer_bonus + 0.1 * support
        if n.is_frontier:
            frontiers.append((score, n))
        else:
            objects.append((score, n))
    objects.sort(key=lambda t: (-t[0], int(t[1].node_id)))
    frontiers.sort(key=lambda t: (-t[0], int(t[1].node_id)))
    return [n for _, n in objects] + [n for _, n in frontiers]

def _eqa_cfg_value(self, key: str, default: Any = None) -> Any:
    """Read ``eqa.<key>`` from Parameters or a nested dict."""
    params = self.parameters
    if params is None:
        return default
    if isinstance(params, dict):
        eqa = params.get("eqa")
        if isinstance(eqa, dict) and key in eqa:
            return eqa.get(key, default)
        return params.get(f"eqa/{key}", params.get(key, default))
    if hasattr(params, "get"):
        return params.get(f"eqa/{key}", default)
    return default

def _eqa_override_gate(self, key: str, default: bool) -> bool:
    """Boolean eqa flag with an env escape hatch: ``EMET_EQA_<KEY>=0|1``.

    Lets the Habitat HM-EQA runner / overnight jobs toggle location-letter
    override behavior (equip/image gates) without editing a config file.
    """
    env_name = "EMET_EQA_" + key.upper()
    env = os.environ.get(env_name, "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return bool(self._eqa_cfg_value(key, default))

def _spatial_rag_enabled(self) -> bool:
    """True when eqa.spatial_rag or EMET_EQA_SPATIAL_RAG requests REGION prompts."""
    env = os.environ.get("EMET_EQA_SPATIAL_RAG", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    raw = self._eqa_cfg_value("spatial_rag", False)
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)

def _spatial_rag_float(self, key: str, default: float) -> float:
    try:
        return float(self._eqa_cfg_value(key, default))
    except Exception:
        return default

def _spatial_rag_int(self, key: str, default: int) -> int:
    try:
        return int(self._eqa_cfg_value(key, default))
    except Exception:
        return default

def _merged_memory_enabled(self) -> bool:
    """True when eqa.merged_memory / EMET_EQA_MERGED_MEMORY folds CONFIRMED_MEMORY into SCENE_GRAPH.

    Default on. The HM-EQA paper row pins ``merged_memory: false`` via
    ``configs/benchmarks/dynagraph.yaml`` (harness ``habitat_eqa.dynagraph``) so its
    numbers stay on the standalone summary block; every other path gets the folded
    format. When on, the main EQA prompt tags SCENE_GRAPH nodes with status
    (inspect / candidate) and room names, and emits a short CONFIRMED_MEMORY tail
    only for phrases with no tagged node, instead of a separate summary block
    (one fact, one line — no duplicate object mentions).
    """
    env = os.environ.get("EMET_EQA_MERGED_MEMORY", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    raw = self._eqa_cfg_value("merged_memory", True)
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)

def to_string(
    self,
    *,
    max_object_nodes: int | None = None,
    question_keywords: list[str] | None = None,
    prefer_obs_ids: list[int] | None = None,
    record_prompt_count: bool = False,
    merge_confirmed: bool = False,
) -> str:
    """Serialize the scene graph to a string for mLLM prompts.

    When ``max_object_nodes`` is set, keep the top-K ranked object/frontier nodes
    (keyword + support + Image-N preference) so blowups cannot starve the VLM.
    With ``eqa.spatial_rag`` / ``EMET_EQA_SPATIAL_RAG``, emit compact REGION blocks
    around keyword / preferred-obs neighborhoods instead of a flat node dump.
    Full untruncated serialization is the default for exports / debugging.

    When ``merge_confirmed`` is set (prompt path only), fold CONFIRMED_MEMORY status
    into the serialization instead of emitting a separate summary block: matching
    nodes get an ``inspect`` / ``candidate`` tag, object nodes are tagged with their
    room-cluster name, and a short CONFIRMED_MEMORY tail lists only phrases with no
    tagged node (SigLIP-only sightings, weak matches, unobserved objects) plus a
    compact ``Rooms:`` line. The spatial-RAG branch is left untouched.
    """
    lines = []

    def _prompt_labels(node: GraphNode, max_len: int = 120) -> str:
        return node_display_name(node, max_len=max_len)

    if max_object_nodes is not None and max_object_nodes > 0 and self._spatial_rag_enabled():
        from emet.memory.graph_eqa.spatial.spatial_rag import (
            format_regions_for_prompt,
            select_spatial_regions,
        )

        radius = self._spatial_rag_float("spatial_rag_radius_m", 2.5)
        max_regions = self._spatial_rag_int("spatial_rag_max_regions", 6)
        max_nodes = self._spatial_rag_int(
            "spatial_rag_max_nodes",
            int(max_object_nodes) if max_object_nodes else 48,
        )
        frontier_budget = max(4, int(max_object_nodes) // 4)
        rag = select_spatial_regions(
            list(self._nodes),
            keywords=question_keywords or list(self._relevant_objects or []),
            prefer_obs_ids=prefer_obs_ids or self.last_eqa_obs_ids,
            radius_m=radius,
            max_regions=max_regions,
            max_nodes=max_nodes,
            max_frontiers=frontier_budget,
        )
        if rag.regions:
            text = format_regions_for_prompt(rag)
            keep_ids = set(rag.kept_node_ids)
            for n in rag.frontier_nodes:
                keep_ids.add(int(n.node_id))
            edge_lines: list[str] = []
            for a, b, rel in self._edges:
                if int(a) not in keep_ids:
                    continue
                if b != -1 and int(b) not in keep_ids:
                    continue
                b_str = "floor" if b == -1 else str(b)
                edge_lines.append(f"  {rel}({a}, {b_str})")
            if edge_lines:
                text = text + "\n" + "\n".join(edge_lines)
            if record_prompt_count:
                self.last_eqa_prompt_node_count = len(keep_ids)
                self.last_eqa_prompt_regions = len(rag.regions)
                self.last_eqa_spatial_rag = {
                    "n_regions": len(rag.regions),
                    "n_nodes": len(keep_ids),
                    "seed_node_ids": list(rag.seed_node_ids),
                    "radius_m": radius,
                }
            return text

    if max_object_nodes is not None and max_object_nodes > 0:
        ranked = self._rank_nodes_for_eqa_prompt(
            keywords=question_keywords,
            prefer_obs_ids=prefer_obs_ids,
        )
        # Always keep at least a few frontiers if present.
        objects = [n for n in ranked if not n.is_frontier]
        frontiers = [n for n in ranked if n.is_frontier]
        keep_obj = objects[: max(0, int(max_object_nodes))]
        frontier_budget = max(0, min(len(frontiers), max(4, int(max_object_nodes) // 4)))
        keep = keep_obj + frontiers[:frontier_budget]
        keep_ids = {int(n.node_id) for n in keep}
        nodes_for_prompt = keep
    else:
        nodes_for_prompt = list(self._nodes)
        keep_ids = {int(n.node_id) for n in nodes_for_prompt}

    if record_prompt_count:
        self.last_eqa_prompt_node_count = len(nodes_for_prompt)
        self.last_eqa_prompt_regions = 0
        self.last_eqa_spatial_rag = None

    # Merged-memory mode: fold CONFIRMED_MEMORY status into node lines so each
    # confirmed object appears once (one fact, one line). Only in prompt path,
    # and only when confirmed-memory is enabled at all.
    merge_active = (
        merge_confirmed and self.memory_summary_enabled and max_object_nodes is not None and max_object_nodes > 0
    )
    statuses: dict[str, tuple[str, list[int], float | None, np.ndarray | None, int | None]] = {}
    node_rooms: dict[int, str] = {}
    tagged: set[int] = set()
    tail_lines: list[str] = []
    nearest_by_phrase: dict[str, tuple[int, str]] = {}
    if merge_active:
        statuses = self._confirmed_phrase_statuses()
        node_rooms = self._node_room_by_id()
        for phrase, (status, ids, sim, xyz, obs_id) in statuses.items():
            if status == "present" and ids:
                kept = [nid for nid in ids if nid in keep_ids]
                if kept:
                    tagged.update(kept)
                    # Preserve nearest-furniture context (old CONFIRMED_MEMORY).
                    anchor = next(
                        (n for n in self._nodes if int(n.node_id) == int(kept[0])),
                        None,
                    )
                    if anchor is not None:
                        neighbors = self._nearest_object_neighbors(
                            np.asarray(anchor.xyz, dtype=np.float64),
                            exclude_node_ids=set(kept),
                            max_neighbors=2,
                            max_dist_m=3.0,
                        )
                        if neighbors:
                            near_bits = []
                            for n, dist in neighbors:
                                lab = node_display_name(n)
                                near_bits.append(f"{lab} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f}) {dist:.1f}m")
                            nearest_by_phrase[phrase] = (
                                int(kept[0]),
                                "nearest: " + "; ".join(near_bits),
                            )
                else:
                    # All matches fell outside the shown node budget: keep the
                    # legacy-style facts (count, coordinates, nearest furniture)
                    # instead of dangling "Node 9" ids the model cannot see.
                    matches_nodes = sorted(
                        (n for n in self._nodes if int(n.node_id) in set(ids)),
                        key=lambda n: int(n.node_id),
                    )
                    parts = [
                        "candidate views: " + format_graph_node_candidates(matches_nodes, max_nodes=4),
                        _GRAPH_CANDIDATE_COUNT_DISCLAIMER,
                    ]
                    anchor = matches_nodes[0] if matches_nodes else None
                    if anchor is not None:
                        neighbors = self._nearest_object_neighbors(
                            np.asarray(anchor.xyz, dtype=np.float64),
                            exclude_node_ids=set(ids),
                            max_neighbors=2,
                            max_dist_m=3.0,
                        )
                        if neighbors:
                            near_bits = []
                            for n, dist in neighbors:
                                lab = node_display_name(n)
                                near_bits.append(f"{lab} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f}) {dist:.1f}m")
                            parts.append("nearest: " + "; ".join(near_bits))
                    tail_lines.append(
                        f"- {phrase}: LOOK — " + "; ".join(parts) + " (nodes not shown in graph above)"
                    )
            elif status == "candidate":
                pos = f" near ({xyz[0]:.1f}, {xyz[1]:.1f})" if xyz is not None else ""
                obs_note = f", obs_id={obs_id}" if obs_id is not None else ""
                sim_s = f"{sim:.2f}" if sim is not None else "?"
                tail_lines.append(
                    f"- {phrase}: CANDIDATE (SigLIP-only sim={sim_s}{pos}{obs_note}) "
                    "- verify in attached images before finalizing; "
                    "do not treat as confirmed present or absent"
                )
            elif status == "weak_siglip":
                # Do not assert absence — detector miss ≠ not in scene.
                sim_s = f"{sim:.2f}" if sim is not None else "?"
                tail_lines.append(
                    f"- {phrase}: weak SigLIP only (sim={sim_s}) — not evidence of absence; trust attached images"
                )
            elif status == "not_observed":
                tail_lines.append(f"- {phrase}: not observed during exploration")

    for n in nodes_for_prompt:
        lbl = _prompt_labels(n)
        sup = f" n={n.support_count}" if getattr(n, "support_count", 1) != 1 else ""
        if n.is_frontier:
            kind = "Frontier"
        elif n.is_viewpoint:
            kind = "View"
        else:
            kind = "Node"
        nid = int(n.node_id)
        room_tag = f" ({node_rooms[nid]})" if nid in node_rooms else ""
        status_tag = ""
        nearest_tag = ""
        if merge_active and not n.is_frontier and not n.is_viewpoint:
            if nid in tagged:
                status_tag = " inspect"
                # Attach nearest on the in-budget anchor node for each phrase.
                for _phrase, (anchor_nid, near_txt) in nearest_by_phrase.items():
                    if int(anchor_nid) == nid:
                        nearest_tag = f" ({near_txt})"
                        break
            elif any(
                status == "candidate" and obs_id is not None and int(n.obs_id) == obs_id
                for status, _ids, _sim, _xyz, obs_id in statuses.values()
            ):
                status_tag = " candidate"
        lines.append(
            f"{kind} {n.node_id}{room_tag}: {lbl} at ({n.xyz[0]:.2f}, {n.xyz[1]:.2f}, {n.xyz[2]:.2f}) "
            f"[Image {n.obs_id}]{sup}{self._node_nav_status_suffix(n)}{status_tag}{nearest_tag}"
        )
    for a, b, rel in self._edges:
        if int(a) not in keep_ids:
            continue
        if b != -1 and int(b) not in keep_ids:
            continue
        b_str = "floor" if b == -1 else str(b)
        lines.append(f"  {rel}({a}, {b_str})")
    if tail_lines:
        lines.append(
            "CONFIRMED_MEMORY (index of views to look at, not the answer; "
            "LOOK = candidate Image N to look at; CANDIDATE/weak SigLIP are "
            "navigation hints; if images contradict memory, trust the images):"
        )
        lines.extend(tail_lines)
    if merge_active and self._room_clusters:
        rooms_line = self.format_rooms_line(max_chars=200)
        if rooms_line.strip() and rooms_line.strip() != "Rooms:":
            lines.append(rooms_line)
    return (
        "SCENE_GRAPH (views to look at, not the answer; labels are proposals for WHERE to look):\n"
        + "\n".join(lines)
        if lines
        else "SCENE_GRAPH: (empty)"
    )

def to_tree_string(self, indent: str = "  ") -> str:
    """
    Format the 3D spatial scene graph as an indented tree (text).

    Root = Scene; Floor is a virtual node; objects on floor are children of Floor;
    objects on other objects are nested. "Near" relations are listed at the end.
    Includes object labels, (x,y,z), and optional descriptions.
    """
    edge_set = set(self._edges)
    node_by_id = {n.node_id: n for n in self._nodes}
    object_nodes = [n for n in self._nodes if not n.is_viewpoint]

    def on_floor(nid: int) -> bool:
        return (nid, -1, "on") in edge_set

    def has_on_parent(nid: int) -> int | None:
        """Return node_id that this node is 'on', or None if on floor or no 'on' edge."""
        for a, b, rel in edge_set:
            if rel == "on" and a == nid and b != -1:
                return b
        return None

    def children_of(nid: int | None) -> list[GraphNode]:
        if nid is None:
            # Floor children: explicitly on floor, or no "on" relation (in-scene)
            out = [
                node_by_id[n.node_id]
                for n in object_nodes
                if on_floor(n.node_id) or has_on_parent(n.node_id) is None
            ]
        else:
            out = [node_by_id[a] for a, b, rel in edge_set if rel == "on" and b == nid and a in node_by_id]
        return sorted(out, key=lambda n: n.node_id)

    near_pairs = [(a, b) for a, b, rel in self._edges if rel == "near" and a < b]

    lines: list[str] = []
    lines.append("Scene (3D spatial graph)")
    lines.append(f"{indent}Floor")

    def visit(node: GraphNode, depth: int) -> None:
        pref = indent * (depth + 1)
        x, y, z = float(node.xyz[0]), float(node.xyz[1]), float(node.xyz[2])
        lbl = node_display_name(node)
        line = f"{pref}[{node.node_id}] {lbl}  at ({x:.2f}, {y:.2f}, {z:.2f})"
        if node.description:
            d = node.description
            if len(d) > 160:
                d = d[:157] + "..."
            line += f"  — {d}"
        lines.append(line)
        for c in children_of(node.node_id):
            visit(c, depth + 1)

    for node in children_of(None):
        visit(node, 1)

    if near_pairs:
        lines.append("")
        lines.append("Near relations:")
        for a, b in near_pairs:
            na, nb = node_by_id.get(a), node_by_id.get(b)
            la = ", ".join(na.labels) if na and na.labels else str(a)
            lb = ", ".join(nb.labels) if nb and nb.labels else str(b)
            lines.append(f"{indent}{la} — {lb}")

    seen_from_edges = [(a, b) for a, b, rel in self._edges if rel == "seen_from"]
    if seen_from_edges:
        lines.append("")
        lines.append("Seen from (viewpoint node → object):")
        for a, b in seen_from_edges:
            na = node_by_id.get(a)
            nb = node_by_id.get(b)
            la = ", ".join(na.labels) if na and na.labels else str(a)
            if nb is not None:
                vx, vy, vz = (float(nb.xyz[i]) for i in range(3))
                lb = ", ".join(nb.labels) if nb.labels else str(b)
                lines.append(f"{indent}{la} ← {lb} [{b}] at ({vx:.2f}, {vy:.2f}, {vz:.2f})")
            else:
                lines.append(f"{indent}{la} ← node {b}")

    return "\n".join(lines) if lines else "Scene (3D spatial graph): (empty)"

def seed_object_hints(self, labels: str) -> None:
    """GraphEQA HM-EQA enrich labels (per-question object hints for planning)."""
    from emet.habitat.hmeqa_enrich_labels import parse_enrich_label_text

    self._enrich_object_hints = parse_enrich_label_text(labels)

def extract_relevant_objects(self, question: str) -> None:
    """Extract keywords from the question for image selection (same idea as DynaMem)."""
    if self._question == question:
        return
    self._obs_nav_dists.clear()
    self._question = question
    prompt = (
        "Assume there is an agent doing Question Answering in an environment. "
        "When it receives a question, tell the agent few objects (preferably 1-3) to pay attention to. "
        "Example: Where is the pen? -> pen. Is there grey cloth on cloth hanger? -> grey cloth, cloth hanger"
    )
    out = self.image_description_client([prompt, question])
    enrich_hints = getattr(self, "_enrich_object_hints", None) or []
    llm_parts = [s.strip() for s in out.split(",") if s.strip()]
    mcq_landmarks = location_mcq_landmark_phrases(question)
    phrase_seed = list(enrich_hints) + heuristic_relevant_phrases(question) + list(mcq_landmarks)
    if enrich_hints:
        for hint in enrich_hints:
            h = hint.strip().lower()
            if h and " " in h and h not in phrase_seed:
                phrase_seed.insert(0, h)
    extra_seed = llm_parts + heuristic_relevant_objects(question) + list(mcq_landmarks)
    # Location MCQs need stem object + option landmarks in the same recall set.
    max_items = 8 if mcq_landmarks else 4
    phrases, objects = consolidate_relevant_keywords(phrase_seed, extra_seed, max_items=max_items)
    self._relevant_phrases = phrases
    self._relevant_objects = objects

def set_confirmed_memory_siglip_encoder(self, encoder: Any | None) -> None:
    """Attach a SigLIP encoder used only for CONFIRMED_MEMORY (survives voxel encoder drop)."""
    self._confirmed_memory_siglip_encoder = encoder

def refresh_siglip_confirmed_memory(self) -> None:
    """Encode new graph observation RGBs and refresh phrase→best-view alignments."""
    if not self.memory_summary_enabled:
        return
    enc = self._confirmed_memory_siglip_encoder
    if enc is None:
        return
    from emet.memory.graph_eqa.eqa.graph_eqa_siglip import (
        align_phrase_to_observation_features,
        encode_observation_rgb,
    )

    for obs in self._observations:
        oid = int(obs.obs_id)
        if oid in self._obs_siglip_features:
            continue
        feat = encode_observation_rgb(enc, obs.rgb)
        if feat is not None:
            self._obs_siglip_features[oid] = feat
    phrases = self._confirmed_memory_phrases()
    for phrase in phrases:
        match = align_phrase_to_observation_features(
            phrase,
            enc,
            self._observations,
            self._obs_siglip_features,
        )
        if match is not None:
            self._siglip_phrase_cache[phrase.strip().lower()] = match

def _node_for_obs(self, obs_id: int) -> GraphNode | None:
    return next(
        (node for node in self._nodes if int(node.obs_id) == int(obs_id) and not node.is_viewpoint),
        None,
    )

def _answerability_gain_for_obs(self, question: str, obs_id: int, phrase: str) -> float:
    obs = self._observation_by_id(int(obs_id))
    labels = list(obs.labels or []) if obs is not None else []
    target_hit = any(label_matches_relevant_object(phrase, label) for label in labels)
    try:
        from emet.habitat.metrics import parse_mcq_choices_from_question

        choices = parse_mcq_choices_from_question(question)
    except Exception:
        choices = []
    if not choices:
        return 1.0 if target_hit else 0.25
    landmark_hit = any(label_matches_relevant_object(choice, label) for choice in choices for label in labels)
    if target_hit and landmark_hit:
        return 1.0
    if target_hit or landmark_hit:
        return 0.55
    return 0.15

def _recall_rank_score(
    self,
    hypothesis: NavHypothesis,
    question: str,
    robot_xyt: np.ndarray | None,
) -> NavHypothesis:
    """Cheap recall key for top-K packing (not a VLM decision policy)."""
    answerability = self._answerability_gain_for_obs(
        question,
        hypothesis.obs_id,
        hypothesis.phrase,
    )
    # Map answerability to a small recall boost (landmark/target label hits).
    if answerability >= 1.0:
        hit_boost = 20.0
    elif answerability >= 0.55:
        hit_boost = 10.0
    else:
        hit_boost = 0.0
    path_cost = 0.0
    if robot_xyt is not None and np.asarray(robot_xyt).size >= 2:
        path_cost = float(
            np.linalg.norm(
                np.asarray(hypothesis.xyz, dtype=float)[:2] - np.asarray(robot_xyt, dtype=float).reshape(-1)[:2]
            )
        )
    tier = float(_RECALL_SOURCE_TIER.get(str(hypothesis.source), 0.0))
    # Path is a weak tiebreak only (cm-scale in the key).
    total = tier + hit_boost - 0.01 * path_cost
    return replace(
        hypothesis,
        score=total,
        answerability_gain=answerability,
        belief_reduction=0.0,
        revisit_change_value=0.0,
        path_cost=path_cost,
        failure_risk=0.0,
    )

@staticmethod
def _pack_diversified_hypotheses(
    scored: list[NavHypothesis],
    max_k: int,
) -> list[NavHypothesis]:
    """Pack top-K with source diversity (graph + frontier when both exist)."""
    k = max(1, int(max_k))
    if not scored:
        return []
    picked: list[NavHypothesis] = []
    seen: set[int] = set()

    def _take_from(sources: tuple[str, ...]) -> None:
        for h in scored:
            oid = int(h.obs_id)
            if oid in seen:
                continue
            if str(h.source) in sources:
                picked.append(h)
                seen.add(oid)
                return

    # Seed diversity: one graph, one siglip/confirmed, one frontier when present.
    _take_from(("graph",))
    _take_from(("confirmed", "siglip"))
    _take_from(("frontier",))
    for h in scored:
        if len(picked) >= k:
            break
        oid = int(h.obs_id)
        if oid not in seen:
            picked.append(h)
            seen.add(oid)
    return picked[:k]
