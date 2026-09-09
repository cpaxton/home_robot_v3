# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Place-approach sampling, inspect cards, and room-stamp for agentic investigate."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from emet.mapping.voxel_localize import is_proposal_handle, voxel_map_from_agent
from emet.memory.graph_eqa.agentic.config import (
    DEFAULT_INVESTIGATE_ANNULUS_OUTER_M,
    INVESTIGATE_ANNULUS_OUTER_M,
    INVESTIGATE_SOURCES,
    PLACE_APPROACH_SAMPLES,
    PLACE_INSPECT_RECENT_K,
)
from emet.memory.graph_eqa.agentic.types import PlaceInspectRecord, PlaceInspectVisit
from emet.memory.graph_eqa.graph_memory import NavHypothesis
from emet.memory.graph_eqa.spatial.frontier_nodes import _as_bool_numpy
from emet.memory.graph_eqa.spatial.place_approaches import (
    footprint_from_xyz,
    make_grid_converters,
    sample_annulus_approach_xy,
)
from emet.memory.graph_eqa.spatial.room_clusters import (
    merge_room_estimates,
    resolve_investigate_room_stamp,
)
from emet.memory.graph_eqa.spatial.room_labels import coerce_room_label, normalize_current_room
from emet.utils.logger import Logger

_logger = Logger(__name__)


def _investigate_hypotheses(self) -> list[NavHypothesis]:
    hypotheses = [h for h in self._hypotheses if str(h.source) in INVESTIGATE_SOURCES]
    if getattr(self.agent, "query_driven_memory", False) is True:
        rejected = {r.handle for r in self.agent.query_candidates.records.values() if r.rejected_revision is not None}
        hypotheses = [h for h in hypotheses if int(h.obs_id) not in rejected]
    return hypotheses


def _place_anchor_xy(self, obs_id: int, hyp: NavHypothesis | None) -> tuple[float, float] | None:
    if hyp is not None:
        xyz = np.asarray(hyp.xyz, dtype=float).reshape(-1)
        if xyz.size >= 2:
            return float(xyz[0]), float(xyz[1])
    gm = self.graph_memory
    if gm is not None and hasattr(gm, "_observation_by_id"):
        obs = gm._observation_by_id(int(obs_id))
        if obs is not None:
            xyz = np.asarray(obs.xyz, dtype=float).reshape(-1)
            if xyz.size >= 2:
                return float(xyz[0]), float(xyz[1])
    return None


def _dist_to_anchor_m(self, obs_id: int, hyp: NavHypothesis | None) -> float | None:
    anchor = self._place_anchor_xy(obs_id, hyp)
    robot = self._robot_xyt_world()
    if anchor is None or robot is None:
        return None
    return float(np.hypot(float(robot[0]) - anchor[0], float(robot[1]) - anchor[1]))


def _hypothesis_for_obs_id(self, obs_id: int) -> NavHypothesis | None:
    oid = int(obs_id)
    for h in self._investigate_hypotheses():
        if int(h.obs_id) == oid:
            return h
    for h in self._hypotheses:
        if int(h.obs_id) == oid:
            return h
    return None


def _hypothesis_nav_anchor_xyz(self, obs_id: int) -> np.ndarray | None:
    """Object anchor for investigate standoff + arrival yaw (graph obs or synthetic hyp)."""
    from emet.mapping.voxel_localize import is_proposal_handle

    if is_proposal_handle(obs_id):
        hyp = self._hypothesis_for_obs_id(obs_id)
        if hyp is not None:
            try:
                xyz = np.asarray(hyp.xyz, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                xyz = None
            if xyz is not None and xyz.size >= 2 and np.isfinite(xyz[:2]).all():
                return xyz[:3].copy() if xyz.size >= 3 else np.array([xyz[0], xyz[1], 0.0], dtype=float)
        return None
    gm = self.graph_memory
    if gm is not None and hasattr(gm, "_obs_nav_anchor"):
        try:
            anchor = gm._obs_nav_anchor(int(obs_id))
        except Exception:
            anchor = None
        if anchor is not None:
            try:
                arr = np.asarray(anchor, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                arr = None
            if arr is not None and arr.size >= 2 and np.isfinite(arr[:2]).all():
                return arr[:3].copy() if arr.size >= 3 else np.array([arr[0], arr[1], 0.0], dtype=float)
    hyp = self._hypothesis_for_obs_id(obs_id)
    if hyp is not None:
        try:
            xyz = np.asarray(hyp.xyz, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            xyz = None
        if xyz is not None and xyz.size >= 2 and np.isfinite(xyz[:2]).all():
            return xyz[:3].copy() if xyz.size >= 3 else np.array([xyz[0], xyz[1], 0.0], dtype=float)
    return None


def _investigate_annulus_outer_m(self) -> float:
    if self._prefers_nearby_investigate():
        return INVESTIGATE_ANNULUS_OUTER_M
    return DEFAULT_INVESTIGATE_ANNULUS_OUTER_M


def _investigate_arrival_look_at_xy(self, obs_id: int, target: np.ndarray) -> tuple[float, float]:
    anchor = self._hypothesis_nav_anchor_xyz(obs_id)
    if anchor is not None:
        return float(anchor[0]), float(anchor[1])
    t_arr = np.asarray(target, dtype=float).reshape(-1)
    return float(t_arr[0]), float(t_arr[1])


def _sample_investigate_waypoint_at_anchor(
    self,
    anchor: np.ndarray,
    *,
    approach_index: int,
    avoid: list[tuple[float, float]] | None,
    xyt: np.ndarray | None,
    _as_xy: Any,
) -> np.ndarray | None:
    gm = self.graph_memory
    if gm is None:
        return None
    voxel_map, planner = self._voxel_planner()
    robot_xy = gm._robot_planar_xy(xyt) if hasattr(gm, "_robot_planar_xy") else None
    outer_m = self._investigate_annulus_outer_m()
    r_in = max(0.35, float(getattr(gm, "image_nav_min_approach_m", 0.35) or 0.35))
    if voxel_map is not None and hasattr(voxel_map, "get_2d_map"):
        try:
            converters = make_grid_converters(voxel_map)
            if converters is not None:
                xy_to_ij, ij_to_xy, _res = converters
                obstacles, explored = voxel_map.get_2d_map()
                obstacles_b = _as_bool_numpy(obstacles)
                reachable = None
                frontier = None
                if xyt is not None and planner is not None:
                    if hasattr(voxel_map, "get_reachable_map"):
                        reachable = _as_bool_numpy(voxel_map.get_reachable_map(xyt, planner))
                    if hasattr(voxel_map, "get_outside_frontier"):
                        outside = voxel_map.get_outside_frontier(xyt, planner)
                        frontier = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
                xy = sample_annulus_approach_xy(
                    anchor_xy=(float(anchor[0]), float(anchor[1])),
                    robot_xy=robot_xy,
                    obstacles=obstacles_b,
                    reachable=reachable,
                    frontier=frontier,
                    footprint=footprint_from_xyz(anchor),
                    xy_to_ij=xy_to_ij,
                    ij_to_xy=ij_to_xy,
                    avoid_xy=avoid,
                    radius_inner_m=r_in,
                    radius_outer_m=outer_m,
                    approach_index=int(approach_index),
                )
                if xy is not None:
                    return _as_xy(xy)
        except Exception as e:
            _logger.debug(f"synthetic annulus approach unavailable: {e}")
    if hasattr(gm, "_orbit_approach_samples"):
        try:
            samples = gm._orbit_approach_samples(
                anchor,
                robot_xy,
                n=PLACE_APPROACH_SAMPLES,
                radius_m=outer_m,
            )
            idx = int(approach_index) % len(samples)
            got = _as_xy(samples[idx])
            if got is not None:
                return got
        except Exception as e:
            _logger.debug(f"synthetic orbit approach unavailable: {e}")
    if robot_xy is not None and hasattr(gm, "_standoff_waypoint_toward"):
        return _as_xy(gm._standoff_waypoint_toward(robot_xy, anchor))
    return _as_xy(anchor)


def _record_place_inspect(
    self,
    obs_id: int,
    *,
    closest_m: float | None,
    verify_out: dict[str, Any] | None,
    approach_index: int | None = None,
) -> PlaceInspectRecord:
    oid = int(obs_id)
    rec = self._place_inspect.get(oid) or PlaceInspectRecord()
    dist = float(closest_m) if closest_m is not None else (float(rec.closest_m) if rec.closest_m is not None else 99.0)
    if rec.closest_m is None or dist < float(rec.closest_m):
        rec.closest_m = dist
    verify_status = ""
    if isinstance(verify_out, dict):
        verify_status = str(verify_out.get("status") or verify_out.get("decision") or "")
    assess = self._last_vlm_assess if isinstance(self._last_vlm_assess, dict) else {}
    present = assess.get("present") if assess else None
    answerable = assess.get("answerable") if assess else None
    suggested = str(assess.get("suggested_answer") or "") if assess else ""
    if present is None and isinstance(verify_out, dict):
        present = verify_out.get("present")
        answerable = verify_out.get("answerable")
    if approach_index is not None:
        ap = int(approach_index) % PLACE_APPROACH_SAMPLES
        if ap not in rec.tried_approaches:
            rec.tried_approaches.append(ap)
    visit = PlaceInspectVisit(
        round=int(self._round),
        closest_m=dist,
        verify=verify_status,
        assess_present=bool(present) if present is not None else None,
        assess_answerable=bool(answerable) if answerable is not None else None,
        suggested=suggested,
        approach_index=int(approach_index) if approach_index is not None else None,
    )
    rec.investigate_count += 1
    rec.recent.append(visit)
    if len(rec.recent) > PLACE_INSPECT_RECENT_K:
        rec.recent = rec.recent[-PLACE_INSPECT_RECENT_K:]
    rec.last_verify = verify_status
    rec.last_assess_present = visit.assess_present
    rec.last_assess_answerable = visit.assess_answerable
    rec.last_suggested = suggested
    self._place_inspect[oid] = rec
    # Close look: only VLM assess_present=False nudges explore (not SigLIP ABSENT alone).
    # Absent at a non-target fixture must not pull explore away from the seek phrase.
    hyp = next((h for h in self._investigate_hypotheses() if int(h.obs_id) == oid), None)
    if (
        self.decision_policy != "grounded_v2"
        and dist <= 1.0
        and visit.assess_present is False
        and self._investigate_matches_target(hyp, oid)
        and not self._close_map_unresolved_stay(oid)
    ):
        self._prefer_explore = True
        self._prefer_explore_reason = "absent"
    return rec


def _place_approaches_exhausted(self, obs_id: int) -> bool:
    """True when the fixed approach sample budget is spent."""
    rec = self._place_inspect.get(int(obs_id))
    if rec is None:
        return False
    return int(rec.approaches_left) <= 0


def _next_approach_index(self, obs_id: int, *, prefer: int | None = None) -> int | None:
    """Next unused approach sample index, or None if count-exhausted."""
    rec = self._place_inspect.get(int(obs_id))
    tried = {int(i) for i in (rec.tried_approaches if rec is not None else [])}
    if prefer is not None:
        p = int(prefer) % PLACE_APPROACH_SAMPLES
        if p not in tried:
            return p
    for i in range(PLACE_APPROACH_SAMPLES):
        if i not in tried:
            return i
    return None


def _place_close_and_absent(self, obs_id: int) -> bool:
    rec = self._place_inspect.get(int(obs_id))
    if rec is None or not rec.approached_close or rec.investigate_count <= 0:
        return False
    if rec.last_assess_answerable:
        return False
    if rec.last_assess_present is False:
        return True
    return str(rec.last_verify).upper() in {"ABSENT", "SKIPPED_SAME_VIEW"}


def _labels_for_room_stamp(self, obs_id: int, hyp: NavHypothesis | None = None) -> list[str]:
    """Local labels for room stamping: hyp node labels + this obs only.

    Deliberately omits ``hyp.phrase`` (often question/MCQ text like
    ``wall clock kitchen``) and ``labels_near_obs`` / station merges, which
    leak nonlocal kitchen/bathroom words in open-plan scenes.
    """
    labels: list[str] = []
    if hyp is not None:
        for lab in list(getattr(hyp, "labels", None) or []):
            s = str(lab).strip()
            if s and s not in labels:
                labels.append(s)
    gm = self.graph_memory
    if gm is not None:
        for obs in list(getattr(gm, "_observations", None) or []):
            if int(getattr(obs, "obs_id", -1)) != int(obs_id):
                continue
            for lab in list(getattr(obs, "labels", None) or []):
                s = str(lab).strip()
                if s and s not in labels:
                    labels.append(s)
            break
    return labels[:48]


def _stamp_room_after_investigate(
    self,
    obs_id: int,
    *,
    hyp: NavHypothesis | None,
    station_oid: int | None,
) -> dict[str, Any]:
    """Refresh graph room cluster from close-look evidence (deferred room-stamp)."""
    if not bool(getattr(self, "_room_stamp_investigate", False)):
        return {"ok": False, "reason": "disabled"}
    gm = self.graph_memory
    if gm is None or not hasattr(gm, "stamp_vlm_room_at_robot"):
        return {"ok": False, "reason": "no_graph"}
    # Station labels stay out of the bag (trace-only); they bleed open-plan kitchens.
    labels = self._labels_for_room_stamp(int(obs_id), hyp)
    label_source = "obs_and_hyp_labels"
    proposed = resolve_investigate_room_stamp(
        labels=labels,
        current_room=self._last_room_estimate,
        room_policy=self.room_policy,
    )
    if proposed == "unknown":
        return {
            "ok": False,
            "reason": "no_room",
            "labels": labels[:12],
            "label_source": label_source,
            "station_obs_id": int(station_oid) if station_oid is not None else None,
        }
    stamp_xy = None
    if hyp is not None and getattr(hyp, "xyz", None) is not None:
        try:
            xyz = np.asarray(hyp.xyz, dtype=float).reshape(-1)
            stamp_xy = (float(xyz[0]), float(xyz[1]))
        except Exception:
            stamp_xy = None
    if stamp_xy is None:
        xyt = self._robot_xyt_world()
        if xyt is not None:
            stamp_xy = (float(xyt[0]), float(xyt[1]))
    if stamp_xy is None:
        return {"ok": False, "reason": "no_xy", "proposed": proposed}
    prev = "unknown"
    if hasattr(gm, "graph_room_at_robot"):
        try:
            prev = coerce_room_label(gm.graph_room_at_robot(stamp_xy), room_policy=self.room_policy)
        except Exception as e:
            _logger.warning(f"graph_room_at_robot before investigate stamp failed: {e}")
    try:
        stamp_kwargs = {
            "protect_indoor_from_outdoor": True,
            "corroborating_labels": labels,
            "source": "investigate_vlm",
            "source_view_id": (
                gm.view_id_for_obs(int(station_oid))
                if station_oid is not None and hasattr(gm, "view_id_for_obs")
                else None
            ),
        }
        stamped = gm.stamp_vlm_room_at_robot(
            stamp_xy,
            proposed,
            **stamp_kwargs,
            agent_round=int(self._round) + 1,
            pose_round=int(self._round) + 1,
        )
    except Exception as e:
        _logger.warning(f"stamp_vlm_room_at_robot after investigate failed: {e}")
        return {"ok": False, "reason": "stamp_failed", "error": str(e), "proposed": proposed}
    stamped_s = coerce_room_label(stamped, room_policy=self.room_policy)
    if stamped_s == "unknown":
        payload = {
            "ok": False,
            "reason": "blocked_or_noop",
            "proposed": proposed,
            "prev": prev,
            "labels": labels[:12],
            "label_source": label_source,
            "station_obs_id": int(station_oid) if station_oid is not None else None,
        }
        self._append_trace({"event": "room_stamp_investigate", **payload})
        return payload
    graph_room = stamped_s
    if hasattr(gm, "graph_room_at_robot"):
        try:
            graph_room = coerce_room_label(gm.graph_room_at_robot(stamp_xy), room_policy=self.room_policy)
        except Exception:
            graph_room = stamped_s
    self._graph_room_estimate = graph_room
    self._graph_room_stale = graph_room == "unknown"
    merged = merge_room_estimates(proposed, graph_room, room_policy=self.room_policy)
    # Prefer the stamp we just applied when merge would keep a stale VLM outdoor.
    if normalize_current_room(merged) == "outdoor" and not (normalize_current_room(proposed) == "outdoor"):
        merged = proposed
    self._last_room_estimate = merged
    self._last_router_room_estimate = proposed
    self._current_room_source = "investigate_vlm+graph"
    self._room_estimate_stale = False
    self._router_room_stale = False
    self._room_pose_round = int(self._round) + 1
    self._room_world_step = self._graph_world_step()
    self._room_estimates.append(merged)
    if len(self._room_estimates) > 8:
        self._room_estimates = self._room_estimates[-8:]
    payload = {
        "ok": True,
        "obs_id": int(obs_id),
        "station_obs_id": int(station_oid) if station_oid is not None else None,
        "proposed": proposed,
        "stamped": stamped_s,
        "prev": prev,
        "current_room": merged,
        "labels": labels[:12],
        "label_source": label_source,
        "xy": [float(stamp_xy[0]), float(stamp_xy[1])],
    }
    self._append_trace({"event": "room_stamp_investigate", **payload})
    self._record_room_timeline(
        kind="stamp",
        room=merged,
        obs_id=int(obs_id),
        note=f"investigate stamp prev={prev}",
    )
    if gm is not None and hasattr(gm, "record_attempt"):
        gm.record_attempt(
            action_kind="investigate",
            outcome="ok",
            status_code="room_stamp",
            note=f"stamp room={merged} at obs {int(obs_id)}",
            step=self._graph_world_step(),
            obs_id=int(obs_id),
            phrase="",
            source="eqa",
            room=normalize_current_room(merged),
        )
    return payload


def _voxel_planner(self) -> tuple[Any | None, Any | None]:
    agent = self.agent
    voxel_map = voxel_map_from_agent(agent)
    if voxel_map is None:
        voxel_map = getattr(agent, "voxel_map", None)
    planner = getattr(agent, "planner", None) or getattr(agent, "_planner", None)
    return voxel_map, planner


def _known_room_for_event(self) -> str:
    """Canonical room label for timeline writes; empty when unknown (never invent)."""
    if self._room_estimate_stale:
        return ""
    for raw in (self._last_room_estimate, self._graph_room_estimate):
        room = normalize_current_room(raw)
        if room != "unknown":
            return room
    return ""


def _record_room_timeline(
    self,
    *,
    kind: str,
    room: str | None = None,
    phrase: str = "",
    obs_id: int | None = None,
    note: str = "",
) -> dict[str, Any] | None:
    gm = self.graph_memory
    if gm is None or not hasattr(gm, "record_room_event"):
        return None
    observed_room = self._observation_room(obs_id)
    label = observed_room or (normalize_current_room(room) if room else self._known_room_for_event())
    if label == "unknown" or not label:
        label = self._known_room_for_event()
    if not label:
        return None
    try:
        return gm.record_room_event(
            room=label,
            kind=kind,
            step=self._graph_world_step(),
            phrase=phrase,
            obs_id=obs_id,
            note=note,
            agent_round=int(self._round) + 1,
        )
    except Exception as e:
        _logger.warning(f"record_room_event failed: {e}")
        return None


def _refresh_place_coverage(self, obs_id: int) -> PlaceInspectRecord:
    """Update Investigate-card coverage= from footprint ∩ unexplored frontier."""
    oid = int(obs_id)
    rec = self._place_inspect.get(oid) or PlaceInspectRecord()
    prev_cov = str(rec.coverage or "unknown")
    gm = self.graph_memory
    voxel_map, planner = self._voxel_planner()
    cov = None
    if gm is not None and hasattr(gm, "place_coverage_for_obs"):
        try:
            cov = gm.place_coverage_for_obs(
                oid,
                voxel_map=voxel_map,
                planner=planner,
                robot_xyt=self._robot_xyt_world(),
            )
        except Exception as e:
            _logger.warning(f"place coverage refresh failed: {e}")
            cov = None
    if cov is not None:
        rec.coverage = str(getattr(cov, "status", "unknown") or "unknown")
        rec.local_frontier_cells = int(getattr(cov, "local_frontier_cells", 0) or 0)
    self._place_inspect[oid] = rec
    if prev_cov != "closed" and rec.coverage == "closed":
        self._record_room_timeline(
            kind="coverage_closed",
            obs_id=oid,
            note=f"obs {oid} local frontier closed",
        )
    return rec


def _mark_approach_tried(
    self,
    obs_id: int,
    approach_index: int,
    *,
    target_xy: tuple[float, float] | None = None,
) -> None:
    oid = int(obs_id)
    rec = self._place_inspect.get(oid) or PlaceInspectRecord()
    ap = int(approach_index) % PLACE_APPROACH_SAMPLES
    if ap not in rec.tried_approaches:
        rec.tried_approaches.append(ap)
    if target_xy is not None:
        xy = (float(target_xy[0]), float(target_xy[1]))
        if all(math.hypot(xy[0] - p[0], xy[1] - p[1]) > 0.25 for p in rec.tried_xy):
            rec.tried_xy.append(xy)
    self._place_inspect[oid] = rec


def _investigate_target_xyz(self, obs_id: int, approach_index: int) -> np.ndarray | None:
    gm = self.graph_memory
    if gm is None:
        return None
    xyt = self._robot_xyt_world()
    voxel_map, planner = self._voxel_planner()
    rec = self._place_inspect.get(int(obs_id))
    avoid = list(rec.tried_xy) if rec is not None else None
    anchor_xyz = self._hypothesis_nav_anchor_xyz(obs_id)

    def _as_xy(raw: Any) -> np.ndarray | None:
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size < 2 or not np.isfinite(arr[:2]).all():
            return None
        return np.array([float(arr[0]), float(arr[1]), 1.0], dtype=float)

    # Habitat: prefer navmesh-reachable approaches (can sit through doorways).
    try:
        from emet.controller.habitat_nav import (
            habitat_perfect_nav_enabled,
            is_habitat_robot_client,
            sample_habitat_navmesh_approach_xy,
        )

        agent = self.agent
        params = getattr(agent, "parameters", None)
        robot = getattr(agent, "robot", None)
        if (
            params is not None
            and robot is not None
            and habitat_perfect_nav_enabled(params)
            and is_habitat_robot_client(robot)
        ):
            sim = getattr(robot, "_sim", None)
            if sim is not None and anchor_xyz is not None:
                robot_xy = None
                if xyt is not None:
                    robot_xy = (float(xyt[0]), float(xyt[1]))
                r_in = max(
                    0.35,
                    float(getattr(gm, "image_nav_min_approach_m", 0.35) or 0.35),
                )
                navmesh_xy = sample_habitat_navmesh_approach_xy(
                    sim,
                    anchor_xy=(float(anchor_xyz[0]), float(anchor_xyz[1])),
                    robot_xy=robot_xy,
                    approach_index=int(approach_index),
                    radius_inner_m=r_in,
                    radius_outer_m=self._investigate_annulus_outer_m(),
                    avoid_xy=avoid,
                )
                if navmesh_xy is not None:
                    return np.array([float(navmesh_xy[0]), float(navmesh_xy[1]), 1.0], dtype=float)
    except Exception as e:
        _logger.debug(f"habitat navmesh approach unavailable: {e}")

    fn = getattr(gm, "_navigation_approach_waypoint_for_obs", None)
    if callable(fn) and int(obs_id) >= 0:
        try:
            got = _as_xy(
                fn(
                    int(obs_id),
                    xyt,
                    approach_index=int(approach_index),
                    n_approaches=PLACE_APPROACH_SAMPLES,
                    avoid_xy=avoid,
                    voxel_map=voxel_map,
                    planner=planner,
                    radius_outer_m=self._investigate_annulus_outer_m(),
                )
            )
        except TypeError:
            try:
                got = _as_xy(fn(int(obs_id), xyt, approach_index=int(approach_index)))
            except TypeError:
                got = None
        if got is not None:
            return got
    if hasattr(gm, "_navigation_waypoint_for_obs") and int(obs_id) >= 0:
        got = _as_xy(gm._navigation_waypoint_for_obs(int(obs_id), xyt))
        if got is not None:
            return got
    if anchor_xyz is not None:
        try:
            got = self._sample_investigate_waypoint_at_anchor(
                anchor_xyz,
                approach_index=int(approach_index),
                avoid=avoid,
                xyt=xyt,
                _as_xy=_as_xy,
            )
            if got is not None:
                return got
        except Exception as e:
            _logger.debug(f"synthetic approach sample failed for obs_id={obs_id}: {e}")
        return _as_xy(anchor_xyz)
    return None


def _maybe_retract_claim_after_station(
    self,
    obs_id: int,
    *,
    closest_m: float | None,
    verify_out: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """If a close look says ABSENT for phrase P, stop advertising P at that obs."""
    gm = self.graph_memory
    if gm is None or not hasattr(gm, "retract_phrase_claim_at_obs"):
        return None
    if closest_m is None or float(closest_m) > 1.0:
        return None
    if not isinstance(verify_out, dict):
        return None
    status = str(verify_out.get("status") or verify_out.get("decision") or "").upper()
    if status != "ABSENT":
        return None
    evidence_obs_id = int(verify_out.get("obs_id") or obs_id)
    evidence_source = "siglip"
    if self.decision_policy == "grounded_v2":
        vlm = verify_out.get("vlm_assess")
        if not isinstance(vlm, dict) or not vlm.get("ok"):
            return None
        # Cheap ABSENT is proposal-only. Only an explicit VLM miss on this fresh
        # station view may become additive negative evidence.
        if vlm.get("present") is not False or bool(vlm.get("answerable")):
            return None
        evidence_obs_id = int(vlm.get("obs_id") or evidence_obs_id)
        evidence_source = "vlm"
    phrase = str(verify_out.get("phrase") or self._target_phrase or "").strip()
    if not phrase:
        return None
    # Only the voxel proposal that was looked at disproves the pin. A close
    # ABSENT on a nearby cabinet graph view must not drop the jar XYZ.
    if is_proposal_handle(obs_id):
        from emet.mapping.voxel_localize import unpin_localize_xyz

        voxel_map = voxel_map_from_agent(self.agent)
        if voxel_map is not None:
            try:
                if unpin_localize_xyz(voxel_map, phrase):
                    self._append_trace({"event": "unpin_localize", "phrase": phrase})
            except Exception as exc:  # noqa: BLE001
                _logger.warning(f"unpin_localize_xyz failed: {exc}")
        if str(self._voxel_score_phrase or "").strip().lower() == phrase.lower():
            self._voxel_score_xyz = None
            self._voxel_score_phrase = None
            self._voxel_score_from_pin = None
    out = gm.retract_phrase_claim_at_obs(
        int(obs_id),
        phrase,
        room=self._observation_room(evidence_obs_id) or self._known_room_for_event() or None,
        step=self._graph_world_step(),
        strip_matching_labels=self.decision_policy != "grounded_v2",
        apply_blacklist=self.decision_policy != "grounded_v2",
        evidence_obs_id=evidence_obs_id,
        evidence_source=evidence_source,
    )
    self._append_trace(
        {
            "event": "retract_claim",
            "obs_id": int(obs_id),
            "claim_obs_id": int(obs_id),
            "evidence_obs_id": evidence_obs_id,
            "evidence_source": evidence_source,
            "phrase": str(out.get("phrase") or phrase),
            "closest_m": float(closest_m),
            "room": out.get("room"),
            **{k: out.get(k) for k in ("stripped_obs", "stripped_nodes", "ok")},
        }
    )
    return out
