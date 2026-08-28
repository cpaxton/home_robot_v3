# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Construction, settings, capture context, and observation refresh."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

from emet.core.parameters import Parameters
from emet.memory.graph_eqa.attempt_ledger import (
    summary_bits_for_obs,
)
from emet.memory.graph_eqa.graph_types import (
    GraphNode,
    is_ground_truth_node,
    label_matches_relevant_object,
)
from emet.memory.graph_eqa.world_evidence import (
    WorldEvidenceStore,
    resolve_world_evidence_mode,
)


def init_memory(
    self,
    parameters: Parameters | None = None,
    max_near_distance: float = 1.5,
    eqa_client: Callable[..., str] | None = None,
    image_description_client: Callable[..., str] | None = None,
    log_dir: str = "graph_eqa_log",
    defer_llm_clients: bool = False,
):
    self.parameters = parameters or {}
    self.max_near_distance = max_near_distance
    self.last_eqa_raw: str = ""
    self.last_eqa_parsed: tuple[str, str, bool, str, str] = ("", "", False, "", "")
    # Model-native output before salvage, memory-location, or equipment overrides.
    # Agentic grounded_v2 arbitration consumes these immutable fields.
    self.last_eqa_model_raw: str = ""
    self.last_eqa_model_parsed: tuple[str, str, bool, str, str] = ("", "", False, "", "")
    self.last_agentic_decision: dict[str, Any] | None = None
    self.last_eqa_obs_ids: list[int] = []
    self.last_eqa_action_obs_id: int | None = None
    # Next query_answer must attach this RGB as Image 1 (Action:N / post-nav look).
    self.last_eqa_look_obs_id: int | None = None
    self.last_eqa_prompt_node_count: int = 0
    self.last_eqa_prompt_regions: int = 0
    self.last_eqa_spatial_rag: dict[str, Any] | None = None
    self.last_eqa_prompt_text: str = ""
    self.last_router_state_text: str = ""
    self.last_room_clusters: list[Any] = []
    self.last_nav_result_note: str = ""
    self.last_eqa_nav_fallback_count: int = 0
    # Frames attached to the last EQA call, kept for salvage / counterfactual re-asks.
    self.last_relevant_images: list[Any] = []
    # Decode-budget health: did the generation reach ``answer:``, and did the terse
    # re-ask have to rescue it?
    self.last_eqa_answer_field_emitted: bool = False
    self.last_eqa_salvage_used: bool = False
    # Model's own confidence before the graph-coverage gate suppresses it (for early-stop).
    self.last_eqa_model_confident: bool = False
    self._question: str | None = None
    self._relevant_objects: list[str] | None = None
    # Dynagraph improvements (kept OFF here so GraphEQA stays a clean baseline; the
    # DynagraphController turns them on):
    #  * memory_summary_enabled: prepend the CONFIRMED_MEMORY block to the planner prompt.
    #  * _text_grounder: open-vocab visual grounder (text -> (similarity, xyz)) backed by
    #    the voxel map's SigLIP features, decoupling grounding from brittle caption labels
    #    (e.g. a "woven basket" captioned as "decorative plant").
    self.memory_summary_enabled: bool = False
    #  * mcq_debias_enabled: choice-rotation vote at episode end (see mcq_debias.py).
    self.mcq_debias_enabled: bool = False
    self.last_mcq_debias: dict[str, Any] = {}
    self._text_grounder: Callable[[str], tuple[float, np.ndarray] | None] | None = None
    self._obs_id_grounder: Callable[[str], int | None] | None = None
    self._visual_find_fn: Callable[..., list[Any]] | None = None
    # phrase → top-k (sim, obs_id) from find_all_images / whole-image rank, frozen
    # before prepare_dynagraph_vram_for_eqa drops GPU SigLIP.
    self._visual_find_rank_cache: dict[str, list[tuple[float, int]]] = {}
    self._eqa_highlight_key: tuple[str, int] | None = None
    self._enrich_object_hints: list[str] = []
    self._history_outputs: list[str] = []
    self._relevant_phrases: list[str] = []
    self._confirmed_memory_siglip_encoder: Any | None = None
    self._siglip_phrase_cache: dict[str, tuple[float, np.ndarray, int | None]] = {}

    self.log_dir = log_dir
    self.eqa_client = eqa_client
    self.image_description_client = image_description_client
    self._defer_llm_clients = defer_llm_clients
    eqa_cfg = self._parameters_dict().get("eqa")
    configured_world_mode = eqa_cfg.get("graph_evidence_mode", "off") if isinstance(eqa_cfg, dict) else "off"
    world_mode = resolve_world_evidence_mode(
        os.environ.get("EMET_EQA_GRAPH_EVIDENCE_MODE", "") or configured_world_mode
    )
    self.world_evidence = WorldEvidenceStore(
        mode=world_mode,
        session_id=os.environ.get("EMET_WORLD_SESSION_ID", ""),
    )
    self._capture_context: dict[str, Any] = {}
    self._load_navigation_settings()
    self._load_dynagraph_settings()
    self._load_frontier_settings()
    self._load_attempt_ledger_settings()

    if not defer_llm_clients and (self.eqa_client is None or self.image_description_client is None):
        self._init_clients()

def _parameters_dict(self) -> dict[str, Any]:
    p = self.parameters
    if isinstance(p, dict):
        return p
    if hasattr(p, "data") and isinstance(p.data, dict):
        return p.data
    return {}

def _load_navigation_settings(self) -> None:
    d = self._parameters_dict()
    if not d:
        return
    v = d.get("graph_eqa_record_navigation")
    if v is not None:
        self._record_navigation = bool(v)
    blk = d.get("graph_eqa_extract")
    if isinstance(blk, dict) and blk.get("navigation_samples_max") is not None:
        self._nav_max = max(1, int(blk["navigation_samples_max"]))
    eqa = d.get("eqa")
    if isinstance(eqa, dict) and eqa.get("image_nav_min_approach_m") is not None:
        self.image_nav_min_approach_m = max(0.05, float(eqa["image_nav_min_approach_m"]))

def _load_dynagraph_settings(self) -> None:
    d = self._parameters_dict()
    if not d:
        return
    if d.get("dynagraph_merge_xy_m") is not None:
        self.spatial_merge_m = float(d["dynagraph_merge_xy_m"])
    if d.get("dynagraph_staleness_horizon") is not None:
        self.staleness_horizon = max(0, int(d["dynagraph_staleness_horizon"]))
    if d.get("dynagraph_viewpoint_merge_m") is not None:
        self.viewpoint_merge_m = max(0.0, float(d["dynagraph_viewpoint_merge_m"]))

def _load_frontier_settings(self) -> None:
    d = self._parameters_dict()
    blk = d.get("graph_eqa_frontier_nodes")
    if not isinstance(blk, dict):
        eqa = d.get("graph_eqa")
        if isinstance(eqa, dict):
            blk = eqa.get("frontier_nodes")
    if not isinstance(blk, dict):
        return
    if blk.get("enabled") is not None:
        self.frontier_nodes_enabled = bool(blk["enabled"])
    if blk.get("max_nodes") is not None:
        self._frontier_max_nodes = max(1, int(blk["max_nodes"]))
    if blk.get("min_cluster_cells") is not None:
        self._frontier_min_cluster_cells = max(1, int(blk["min_cluster_cells"]))
    if blk.get("keyword_score_weight") is not None:
        self._frontier_keyword_score_weight = max(0.0, float(blk["keyword_score_weight"]))

def _load_attempt_ledger_settings(self) -> None:
    """Load ``eqa.attempt_ledger`` dict knobs (max_records, persist_absent_claims)."""
    d = self._parameters_dict()
    blk: dict[str, Any] = {}
    eqa = d.get("eqa")
    if isinstance(eqa, dict) and isinstance(eqa.get("attempt_ledger"), dict):
        blk = dict(eqa["attempt_ledger"])
    agent = d.get("agent")
    if isinstance(agent, dict) and isinstance(agent.get("attempt_ledger"), dict):
        blk = {**blk, **agent["attempt_ledger"]}
    if blk.get("max_records") is not None:
        self._attempt_ledger_max = max(32, int(blk["max_records"]))
    if blk.get("persist_absent_claims") is not None:
        self.persist_absent_claims = bool(blk["persist_absent_claims"])
    env_persist = os.environ.get("EMET_ATTEMPT_LEDGER_PERSIST_ABSENT", "").strip().lower()
    if env_persist in ("1", "true", "yes", "on"):
        self.persist_absent_claims = True
    elif env_persist in ("0", "false", "no", "off"):
        self.persist_absent_claims = False
    env_max = os.environ.get("EMET_ATTEMPT_LEDGER_MAX", "").strip()
    if env_max:
        try:
            self._attempt_ledger_max = max(32, int(env_max))
        except ValueError:
            pass

def attempt_summary_for_obs(self, obs_id: int, *, max_bits: int = 4) -> str:
    """Newest-first compact attempt tags for place cards / diagnostics."""
    return summary_bits_for_obs(self._attempt_records, int(obs_id), max_bits=max_bits)

def record_close_look_label(self, obs_id: int, label: str) -> None:
    """Store a Qwen close-look / vlm_assess name on graph nodes for this view.

    Tags object nodes whose candidate image is ``obs_id``, and objects linked
    ``seen_from`` a viewpoint with that observation id (fusion may have moved
    the node's own ``obs_id`` to an earlier frame).
    """
    text = str(label or "").strip()
    if not text:
        return
    oid = int(obs_id)
    clipped = text[:80]
    vp_ids = {int(node.node_id) for node in self._nodes if node.is_viewpoint and int(node.obs_id) == oid}
    seen_from_ids: set[int] = set()
    for src, dst, rel in self._edges:
        if rel == "seen_from" and int(dst) in vp_ids:
            seen_from_ids.add(int(src))
    for idx, node in enumerate(self._nodes):
        if node.is_frontier or node.is_viewpoint:
            continue
        if int(node.obs_id) != oid and int(node.node_id) not in seen_from_ids:
            continue
        self._nodes[idx] = replace(node, close_look_label=clipped)

def set_graph_timestep(self, step: int) -> None:
    """Set the discrete time index used for ``last_seen`` and staleness (e.g. controller ``obs_count``)."""
    self._graph_timestep = int(step)

def _position_update(
    self,
    node: GraphNode,
    measured_xyz: np.ndarray,
    *,
    step: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], float]:
    """Update a track without averaging a contradictory relocation into its centroid."""
    measured = np.asarray(measured_xyz, dtype=float).reshape(-1)[:3]
    current = np.asarray(node.xyz, dtype=float).reshape(-1)[:3]
    history = list(node.position_history)
    if not history:
        history.append(
            {
                "step": int(node.last_seen),
                "xyz": current.tolist(),
                "confidence": float(node.belief_confidence),
            }
        )
    distance = float(np.linalg.norm(measured - current))
    relocation_bar = max(0.45, float(self.spatial_merge_m) * 1.5)
    changes = list(node.change_events)
    confidence = min(0.99, float(node.belief_confidence) + 0.08)
    if distance > relocation_bar:
        event = {
            "type": "position_contradiction",
            "node_id": int(node.node_id),
            "step": int(step),
            "from_xyz": current.tolist(),
            "to_xyz": measured.tolist(),
            "displacement_m": distance,
            "confidence": min(0.99, distance / max(relocation_bar, 1e-6)),
        }
        changes.append(event)
        self._change_events.append(event)
        updated = measured.copy()
        confidence = max(0.2, float(node.belief_confidence) * 0.7)
    else:
        support = max(1, int(node.support_count))
        updated = (current * support + measured) / (support + 1)
    history.append(
        {
            "step": int(step),
            "xyz": measured.tolist(),
            "confidence": confidence,
        }
    )
    samples = np.asarray([entry["xyz"] for entry in history[-20:]], dtype=float)
    covariance = np.cov(samples.T) if samples.shape[0] >= 2 else np.zeros((3, 3), dtype=float)
    return updated, covariance, history[-64:], changes[-32:], confidence

def observe_visible_labels(
    self,
    labels: list[str],
    viewer_xyz: np.ndarray | None,
    *,
    step: int | None = None,
    viewpoint_tolerance_m: float = 0.75,
    absence_confirmations: int = 2,
    visibility_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Conservatively detect disappeared objects from repeated same-view contradictions.

    A node is expected only when the camera revisits approximately the viewpoint
    that originally saw it. This avoids treating out-of-FOV objects as absent.
    Ground-truth placements are never consulted.
    """
    if viewer_xyz is None:
        return []
    evidence_view_id = ""
    if self.world_evidence.enabled:
        context = dict(visibility_context or {})
        required = (
            bool(context.get("camera_frustum_ok")),
            bool(context.get("depth_coverage_ok")),
            bool(context.get("detector_ran")),
            bool(context.get("occlusion_free")),
        )
        evidence_view_id = str(context.get("evidence_view_id") or "")
        if not all(required) or not evidence_view_id:
            return []
    now = self._effective_timestep() if step is None else int(step)
    viewer = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3]
    visible = [str(label) for label in labels if str(label).strip()]
    events: list[dict[str, Any]] = []
    for index, node in enumerate(self._nodes):
        if node.is_viewpoint or node.is_frontier or is_ground_truth_node(node):
            continue
        original = self._observation_by_id(int(node.obs_id))
        expected_view = getattr(original, "viewer_xyz", None) if original is not None else None
        if expected_view is None:
            continue
        distance = float(np.linalg.norm(np.asarray(expected_view, dtype=float).reshape(-1)[:2] - viewer[:2]))
        if distance > float(viewpoint_tolerance_m):
            continue
        seen = any(label_matches_relevant_object(node.labels[0], label) for label in visible)
        if seen:
            self._nodes[index] = replace(
                node,
                expected_absence_count=0,
                belief_confidence=min(0.99, float(node.belief_confidence) + 0.1),
            )
            continue
        if self.world_evidence.enabled:
            entity = self.world_evidence.entity_for_node(int(node.node_id))
            if entity is not None:
                self.world_evidence.record_event(
                    subject_kind="entity",
                    subject_id=entity.entity_id,
                    predicate="visible_in_view",
                    polarity="negative",
                    source="detector",
                    confidence=0.6,
                    step=now,
                    view_id=evidence_view_id,
                    place_id=entity.place_id,
                    payload={
                        "claim_view_id": self.view_id_for_obs(int(node.obs_id)),
                        "viewpoint_distance_m": distance,
                        "visibility_qualified": True,
                        "frustum_checked": True,
                        "depth_coverage_checked": True,
                        "occlusion_checked": True,
                    },
                )
        consecutive = (
            int(node.expected_absence_count) + 1
            if int(node.last_absence_step) < 0 or now - int(node.last_absence_step) <= 2
            else 1
        )
        changes = list(node.change_events)
        if consecutive >= int(absence_confirmations):
            event = {
                "type": "expected_object_missing",
                "node_id": int(node.node_id),
                "obs_id": int(node.obs_id),
                "step": now,
                "last_xyz": np.asarray(node.xyz, dtype=float).tolist(),
                "viewpoint_distance_m": distance,
                "confirmations": consecutive,
                "confidence": min(0.95, 0.45 + 0.2 * consecutive),
            }
            if not changes or changes[-1].get("type") != event["type"]:
                changes.append(event)
                self._change_events.append(event)
                events.append(event)
        self._nodes[index] = replace(
            node,
            expected_absence_count=consecutive,
            last_absence_step=now,
            belief_confidence=max(0.05, float(node.belief_confidence) * 0.65),
            change_events=changes[-32:],
        )
    return events

def get_change_events(self) -> list[dict[str, Any]]:
    return [dict(event) for event in self._change_events]

def set_navigation_samples_max(self, n: int) -> None:
    """Raise or lower the cap on stored navigation viewpoint samples (default from config)."""
    self._nav_max = max(1, int(n))

@property
def navigation_samples_max(self) -> int:
    return int(self._nav_max)

def _effective_timestep(self) -> int:
    if self._graph_timestep > 0:
        return self._graph_timestep
    self._fallback_timestep += 1
    return self._fallback_timestep

def clear_eqa_working_memory(self) -> None:
    """Drop cached EQA / CONFIRMED_MEMORY state after a known world change.

    Forces the next planner call to re-ground from the live graph instead of
    reusing provisional memory summaries and Image-N selections from before
    objects moved.
    """
    self.last_eqa_raw = ""
    self.last_eqa_parsed = ("", "", False, "", "")
    self.last_eqa_model_raw = ""
    self.last_eqa_model_parsed = ("", "", False, "", "")
    self.last_agentic_decision = None
    self.last_eqa_obs_ids = []
    self.last_eqa_action_obs_id = None
    self.last_eqa_look_obs_id = None
    self.last_eqa_prompt_node_count = 0
    self.last_eqa_prompt_regions = 0
    self.last_eqa_spatial_rag = None
    self.last_eqa_prompt_text = ""
    self.last_router_state_text = ""
    self.last_eqa_nav_fallback_count = 0
    self.last_eqa_model_confident = False
    self.last_relevant_images = []
    self.last_eqa_answer_field_emitted = False
    self.last_eqa_salvage_used = False
    self._obs_nav_dists.clear()
    self.clear_room_events()

def invalidate_nodes_near(
    self,
    xyz: np.ndarray | list[float] | tuple[float, ...],
    *,
    radius_m: float = 0.75,
    current_step: int | None = None,
    prune: bool = True,
) -> tuple[int, int]:
    """Age object nodes near ``xyz`` so staleness pruning can drop them.

    Used after scripted body relocations (dynamic world-change / lifelong fuzz)
    when the old pose is known. Nodes keep their identity until ``maintain``
    runs (or immediately when ``prune=True``).

    Returns:
        ``(n_aged, n_pruned)``.
    """
    if not self._nodes:
        return 0, 0
    target = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if target.size < 2:
        return 0, 0
    cur = int(current_step if current_step is not None else self._effective_timestep())
    horizon = max(0, int(self.staleness_horizon))
    aged_last_seen = cur - horizon - 1 if horizon > 0 else cur - 10_000
    radius = float(radius_m)
    n_aged = 0
    for i, n in enumerate(self._nodes):
        if is_ground_truth_node(n) or n.is_frontier or n.is_viewpoint:
            continue
        node_xy = np.asarray(n.xyz, dtype=np.float64).reshape(-1)
        if node_xy.size < 2:
            continue
        if float(np.linalg.norm(node_xy[:2] - target[:2])) > radius:
            continue
        self._nodes[i] = replace(n, last_seen=int(aged_last_seen))
        n_aged += 1
    n_pruned = 0
    if prune and n_aged > 0:
        if horizon > 0:
            n_pruned = int(self.maintain(cur))
        else:
            # Staleness disabled: still drop explicitly invalidated object nodes.
            n_pruned = int(self._drop_nodes_near(target, radius_m=radius))
    return n_aged, n_pruned

def _drop_nodes_near(self, xyz: np.ndarray, *, radius_m: float) -> int:
    """Remove non-GT object nodes within ``radius_m`` of ``xyz`` (xy)."""
    target = np.asarray(xyz, dtype=np.float64).reshape(-1)
    to_drop = [
        n
        for n in self._nodes
        if not is_ground_truth_node(n)
        and not n.is_frontier
        and not n.is_viewpoint
        and float(np.linalg.norm(np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2] - target[:2]))
        <= float(radius_m)
    ]
    if not to_drop:
        return 0
    drop_obs = {n.obs_id for n in to_drop}
    drop_node_ids = {n.node_id for n in to_drop}
    drop_node_ids |= {n.node_id for n in self._nodes if n.is_viewpoint and int(n.obs_id) in drop_obs}
    self._nodes = [n for n in self._nodes if n.node_id not in drop_node_ids]
    self._observations = [o for o in self._observations if o.obs_id not in drop_obs]
    for i, n in enumerate(self._nodes, start=1):
        self._nodes[i - 1] = replace(n, node_id=i)
    self._reindex_world_entities()
    self._rebuild_viewpoint_index()
    self._update_edges()
    return len(to_drop)

def maintain(self, current_step: int) -> int:
    """
    Drop stale nodes (and their observations) when ``staleness_horizon`` > 0,
    then renumber ``node_id`` to 1..N and rebuild edges.

    Returns:
        Number of nodes removed.
    """
    if self.staleness_horizon <= 0 or not self._nodes:
        return 0
    cur = int(current_step)
    to_drop: list[GraphNode] = [
        n
        for n in self._nodes
        if not is_ground_truth_node(n) and not n.is_frontier and cur - int(n.last_seen) > self.staleness_horizon
    ]
    if not to_drop:
        return 0
    drop_obs = {n.obs_id for n in to_drop if not n.is_viewpoint}
    drop_node_ids = {n.node_id for n in to_drop}
    drop_node_ids |= {n.node_id for n in self._nodes if n.is_viewpoint and int(n.obs_id) in drop_obs}
    self._nodes = [n for n in self._nodes if n.node_id not in drop_node_ids]
    self._observations = [o for o in self._observations if o.obs_id not in drop_obs]
    for i, n in enumerate(self._nodes, start=1):
        self._nodes[i - 1] = replace(n, node_id=i)
    self._reindex_world_entities()
    self._rebuild_viewpoint_index()
    self._update_edges()
    return len(to_drop)

def _ensure_llm_clients(self) -> None:
    """Load shared Qwen3.5 multimodal on first use when defer_llm_clients=True."""
    if self.eqa_client is not None and self.image_description_client is not None:
        return
    self._init_clients()

def _init_clients(self) -> None:
    """Initialize EQA + keyword helper (one shared VLM: gemma4 / Qwen-VL / Qwen3.5)."""
    try:
        from emet.llms.eqa_vl_settings import get_eqa_vl_int
        from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

        kw = get_eqa_vl_int(self.parameters, "graph_keyword_max_tokens", 64)
        self.image_description_client, self.eqa_client = build_graph_eqa_vlm_clients(
            parameters=self.parameters,
            keyword_max_tokens=kw,
        )
    except ImportError as e:
        raise ImportError(
            "GraphEQA memory requires emet.llms for EQA. Install GPU extras (torch, transformers)."
        ) from e

def obs_revision(self, obs_id: int) -> int:
    """Content generation for *obs_id* (advances when candidate RGB is refreshed)."""
    return int(self._obs_revisions.get(int(obs_id), 0))

def set_capture_context(
    self,
    *,
    camera_pose_world: Any = None,
    base_pose_world: Any = None,
    session_id: str | None = None,
) -> None:
    """Attach immutable sensor-pose provenance to subsequent object views."""
    self._capture_context = {
        "camera_pose_world": (
            np.asarray(camera_pose_world, dtype=float).reshape(4, 4).copy()
            if camera_pose_world is not None
            else None
        ),
        "base_pose_world": (
            np.asarray(base_pose_world, dtype=float).reshape(-1)[:3].copy() if base_pose_world is not None else None
        ),
    }
    if session_id:
        self.world_evidence.session_id = str(session_id)

def clear_capture_context(self) -> None:
    self._capture_context = {}

def _record_world_view_for_obs(self, obs_id: int) -> str:
    if not self.world_evidence.enabled:
        return ""
    oid = int(obs_id)
    node = next(
        (
            item
            for item in self._nodes
            if int(item.obs_id) == oid and not item.is_frontier and not item.is_viewpoint
        ),
        None,
    )
    obs = self._observation_by_id(oid)
    if node is None or obs is None:
        return ""
    entity = self.world_evidence.ensure_entity(
        identity_key=str(node.identity_key or f"obs:{oid}"),
        node_id=int(node.node_id),
        labels=list(node.labels),
        xyz=node.xyz,
        step=self._effective_timestep(),
    )
    if entity is None:
        return ""
    context = dict(self._capture_context or {})
    view = self.world_evidence.append_view(
        obs_id=oid,
        revision=self.obs_revision(oid),
        rgb=obs.rgb,
        object_xyz=obs.xyz,
        labels=list(obs.labels),
        description=obs.description,
        entity_id=entity.entity_id,
        place_id=entity.place_id,
        captured_step=self._effective_timestep(),
        camera_pose_world=context.get("camera_pose_world"),
        base_pose_world=context.get("base_pose_world", obs.viewer_xyz),
    )
    if view is not None:
        self.world_evidence.record_event(
            subject_kind="entity",
            subject_id=entity.entity_id,
            predicate="observed",
            polarity="positive",
            source="graph_observation",
            confidence=float(getattr(node, "belief_confidence", 0.5) or 0.5),
            step=self._effective_timestep(),
            view_id=view.view_id,
            place_id=entity.place_id,
            payload={
                "labels": list(node.labels),
                "legacy_obs_id": oid,
                "revision": self.obs_revision(oid),
            },
        )
    return view.view_id if view is not None else ""

def view_id_for_obs(self, obs_id: int) -> str:
    return self.world_evidence.view_id_for_obs(int(obs_id))

def bind_episode_context(
    self,
    *,
    question_id: str | int | None,
    session_id: str | int | None,
) -> None:
    """Bind IDs before any episode-specific graph/evidence append."""
    self._attempt_ledger_question_id = str(question_id) if question_id is not None else None
    self.world_evidence.set_context(question_id=question_id, session_id=session_id)

def observation_room(self, obs_id: int) -> tuple[str | None, str]:
    """Return the room persisted on an observation's stable view/place."""
    view = self.world_evidence.view_for_obs(int(obs_id))
    if view is None or not view.place_id:
        return None, "unknown"
    place = self.world_evidence.places.get(view.place_id)
    if place is None or not place.room_id:
        return None, "unknown"
    room = self.world_evidence.rooms.get(place.room_id)
    return place.room_id, room.room_name if room is not None else "unknown"

def record_agentic_evidence(
    self,
    *,
    stage: str,
    outcome: str,
    obs_id: int,
    phrase: str,
    confidence: float,
    source: str,
    agent_round: int | None = None,
    score: float | None = None,
    threshold: float | None = None,
    supporting_event_ids: tuple[str, ...] = (),
    payload: dict[str, Any] | None = None,
) -> str:
    """Persist one agentic evidence stage and return its durable event ID."""
    oid = int(obs_id)
    room_id, room_name = self.observation_room(oid)
    event = self.world_evidence.record_agentic_evidence(
        stage=stage,
        outcome=outcome,
        confidence=float(confidence),
        step=self._effective_timestep(),
        obs_id=oid,
        phrase=phrase,
        source=source,
        view_id=self.view_id_for_obs(oid) or None,
        room_id=room_id,
        room_name=room_name,
        agent_round=agent_round,
        score=score,
        threshold=threshold,
        supporting_event_ids=supporting_event_ids,
        payload=payload,
    )
    return event.event_id if event is not None else ""

def durable_confirmation_event_ids(self, *, obs_id: int, phrase: str = "") -> tuple[str, ...]:
    return self.world_evidence.durable_confirmation_event_ids(
        obs_id=int(obs_id),
        phrase=phrase,
    )

def latest_world_view_id(self) -> str:
    if not self.world_evidence.views:
        return ""
    return max(
        self.world_evidence.views.values(),
        key=lambda view: (view.captured_step, view.obs_id, view.revision),
    ).view_id

def _reindex_world_entities(self) -> None:
    self.world_evidence.reindex_entities(
        self._nodes,
        step=self._effective_timestep(),
    )

def _bump_obs_revision(self, obs_id: int) -> int:
    oid = int(obs_id)
    nxt = int(self._obs_revisions.get(oid, 0)) + 1
    self._obs_revisions[oid] = nxt
    self._last_obs_content_update_id = oid
    # Stale SigLIP features would disagree with the refreshed RGB candidate.
    self._obs_siglip_features.pop(oid, None)
    return nxt

def refresh_observation_candidate(
    self,
    obs_id: int,
    rgb: np.ndarray | Image.Image,
    *,
    xyz: np.ndarray | None = None,
    labels: list[str] | None = None,
    description: str | None = None,
    viewer_xyz: np.ndarray | None = None,
) -> bool:
    """Update the stored RGB/candidate for an existing graph observation.

    Graph nodes keep a stable ``obs_id`` under spatial merge; revisits must still
    refresh the evidence image (and invalidate caches) so verify/EQA see the
    better view rather than the first frame forever.
    """
    if isinstance(rgb, Image.Image):
        rgb = np.array(rgb)
    rgb_a = np.asarray(rgb)
    oid = int(obs_id)
    viewer_a: np.ndarray | None = None
    if viewer_xyz is not None:
        viewer_a = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()
    xyz_a = None
    if xyz is not None:
        xyz_a = np.asarray(xyz, dtype=float).reshape(-1)[:3].copy()
    for o in self._observations:
        if int(o.obs_id) != oid:
            continue
        o.rgb = rgb_a.copy()
        if xyz_a is not None:
            o.xyz = xyz_a
        if labels is not None:
            o.labels = list(labels)
        if description:
            o.description = description
        if viewer_a is not None:
            o.viewer_xyz = viewer_a
        self._bump_obs_revision(oid)
        self._record_world_view_for_obs(oid)
        return True
    return False
