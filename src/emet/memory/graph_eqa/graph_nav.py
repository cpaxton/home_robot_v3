# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Navigation waypoints, place coverage, and EQA image-id composition."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_types import (
    _COUNT_PHRASE_ALIASES,
    _COUNT_WORD_ALIASES,
    _QUESTION_STOPWORDS,
    CountTarget,
    GraphNavigationSample,
    GraphNode,
    GraphObservation,
    _collapse_count_nodes_spatially,
    _count_phrase_matches,
    _count_target_from_stem,
    _count_tokens,
    _strip_count_wrappers,
    finder_label_texts,
    format_graph_node_candidates,
    is_ground_truth_node,
    label_matches_relevant_object,
    question_stem_for_keywords,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)


class GraphNavMixin:
    """Navigation waypoints, place coverage, and EQA image-id composition."""

    def _node_for_obs_id(self, obs_id: int) -> GraphNode | None:
        for n in self._nodes:
            if int(n.obs_id) == int(obs_id):
                return n
        return None

    def _robot_planar_xy(self, robot_xyt: Any | None) -> tuple[float, float] | None:
        if robot_xyt is None:
            return None
        r = np.asarray(robot_xyt, dtype=float).reshape(-1)
        if r.size < 2:
            return None
        return float(r[0]), float(r[1])

    def _viewpoint_xyz_for_obs(self, obs_id: int, obs: GraphObservation | None = None) -> np.ndarray | None:
        if obs is None:
            obs = self._observation_by_id(obs_id)
        if obs is not None and obs.viewer_xyz is not None:
            return np.asarray(obs.viewer_xyz, dtype=float).reshape(-1)[:3]
        vp_id = self._viewpoint_by_obs_id.get(int(obs_id))
        if vp_id is None:
            return None
        for n in self._nodes:
            if int(n.node_id) == int(vp_id) and n.is_viewpoint:
                return np.asarray(n.xyz, dtype=float).reshape(-1)[:3]
        return None

    def _standoff_waypoint_toward(
        self,
        robot_xy: tuple[float, float],
        anchor: np.ndarray,
        *,
        min_approach_m: float | None = None,
    ) -> np.ndarray:
        """Planar goal: move toward ``anchor``, stopping ``min_approach_m`` short of it.

        Habitat/navmesh snaps the goal to the nearest navigable cell; we only pick the
        geometric approach point (closest sensible XY to the object / frontier).
        """
        min_m = float(min_approach_m if min_approach_m is not None else self.image_nav_min_approach_m)
        rx, ry = robot_xy
        ax, ay = float(anchor[0]), float(anchor[1])
        dx, dy = ax - rx, ay - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return np.array([rx + min_m, ry, 1.0], dtype=float)
        travel = dist if dist <= min_m else max(min_m, dist - min_m)
        ux, uy = dx / dist, dy / dist
        return np.array([rx + ux * travel, ry + uy * travel, 1.0], dtype=float)

    def _obs_nav_anchor(self, obs_id: int) -> np.ndarray | None:
        obs = self._observation_by_id(obs_id)
        if obs is None:
            return None
        node = self._node_for_obs_id(obs_id)
        if node is not None:
            return np.asarray(node.xyz, dtype=float).reshape(-1)[:3]
        return np.asarray(obs.xyz, dtype=float).reshape(-1)[:3]

    def place_footprint_for_obs(self, obs_id: int) -> Any:
        """Planar footprint for coverage / annulus sampling (``PlaceFootprint`` or None)."""
        from emet.memory.graph_eqa.place_approaches import (
            footprint_from_node,
            footprint_from_xyz,
        )

        node = self._node_for_obs_id(int(obs_id))
        fp = footprint_from_node(node)
        if fp is not None:
            return fp
        anchor = self._obs_nav_anchor(int(obs_id))
        return footprint_from_xyz(anchor) if anchor is not None else None

    def place_coverage_for_obs(
        self,
        obs_id: int,
        *,
        voxel_map: Any | None = None,
        planner: Any | None = None,
        robot_xyt: Any | None = None,
    ) -> Any:
        """Local frontier completeness for a place card (``PlaceCoverage``)."""
        from emet.memory.graph_eqa.place_approaches import (
            count_frontier_in_footprint,
            coverage_from_frontier_count,
            make_grid_converters,
        )

        fp = self.place_footprint_for_obs(int(obs_id))
        if fp is None or voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
            return coverage_from_frontier_count(None)
        converters = make_grid_converters(voxel_map)
        if converters is None:
            return coverage_from_frontier_count(None)
        xy_to_ij, _ij_to_xy, res = converters
        try:
            from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

            xyt = robot_xyt
            if xyt is None:
                return coverage_from_frontier_count(None)
            if planner is not None and hasattr(voxel_map, "get_outside_frontier"):
                outside = voxel_map.get_outside_frontier(xyt, planner)
                _, explored = voxel_map.get_2d_map()
                frontier = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
            else:
                obstacles, explored = voxel_map.get_2d_map()
                exp = _as_bool_numpy(explored)
                obs = _as_bool_numpy(obstacles)
                from scipy.ndimage import binary_dilation

                frontier = binary_dilation(exp) & ~exp & ~obs
            n = count_frontier_in_footprint(fp, frontier, xy_to_ij=xy_to_ij, resolution_m=res)
            return coverage_from_frontier_count(n)
        except Exception as e:
            _logger.warning(f"place_coverage_for_obs({obs_id}) failed: {e}")
            return coverage_from_frontier_count(None)

    def _orbit_approach_samples(
        self,
        anchor: np.ndarray,
        robot_xy: tuple[float, float] | None,
        *,
        n: int = 4,
        radius_m: float | None = None,
    ) -> list[np.ndarray]:
        """Legacy evenly spaced bearings (fallback when voxel sampling is unavailable)."""
        n_ap = max(1, int(n))
        radius = float(radius_m if radius_m is not None else max(0.85, float(self.image_nav_min_approach_m) + 0.5))
        ax, ay = float(anchor[0]), float(anchor[1])
        if robot_xy is not None:
            rx, ry = robot_xy
            base = math.atan2(ry - ay, rx - ax)
            first = self._standoff_waypoint_toward(robot_xy, anchor)
        else:
            base = 0.0
            first = np.array([ax + radius, ay, 1.0], dtype=float)
        samples: list[np.ndarray] = [np.asarray(first, dtype=float).reshape(-1)[:3].copy()]
        for k in range(1, n_ap):
            bearing = base + (2.0 * math.pi * float(k) / float(n_ap))
            samples.append(
                np.array(
                    [ax + radius * math.cos(bearing), ay + radius * math.sin(bearing), 1.0],
                    dtype=float,
                )
            )
        return samples

    def _navigation_approach_waypoint_for_obs(
        self,
        obs_id: int,
        robot_xyt: Any | None = None,
        *,
        approach_index: int = 0,
        n_approaches: int = 4,
        avoid_xy: list[tuple[float, float]] | None = None,
        voxel_map: Any | None = None,
        planner: Any | None = None,
    ) -> np.ndarray | None:
        """Sample a planar approach around the observation (annulus when map available)."""
        from emet.memory.graph_eqa.place_approaches import (
            make_grid_converters,
            sample_annulus_approach_xy,
        )

        anchor = self._obs_nav_anchor(int(obs_id))
        if anchor is None:
            return None
        robot_xy = self._robot_planar_xy(robot_xyt)
        if voxel_map is not None and hasattr(voxel_map, "get_2d_map"):
            converters = make_grid_converters(voxel_map)
            if converters is not None:
                xy_to_ij, ij_to_xy, _res = converters
                try:
                    from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

                    obstacles, explored = voxel_map.get_2d_map()
                    obstacles_b = _as_bool_numpy(obstacles)
                    reachable = None
                    frontier = None
                    if robot_xyt is not None and planner is not None:
                        if hasattr(voxel_map, "get_reachable_map"):
                            reachable = _as_bool_numpy(voxel_map.get_reachable_map(robot_xyt, planner))
                        if hasattr(voxel_map, "get_outside_frontier"):
                            outside = voxel_map.get_outside_frontier(robot_xyt, planner)
                            frontier = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
                    xy = sample_annulus_approach_xy(
                        anchor_xy=(float(anchor[0]), float(anchor[1])),
                        robot_xy=robot_xy,
                        obstacles=obstacles_b,
                        reachable=reachable,
                        frontier=frontier,
                        footprint=self.place_footprint_for_obs(int(obs_id)),
                        xy_to_ij=xy_to_ij,
                        ij_to_xy=ij_to_xy,
                        avoid_xy=avoid_xy,
                        radius_inner_m=max(0.35, float(self.image_nav_min_approach_m)),
                        approach_index=int(approach_index),
                    )
                    if xy is not None:
                        return np.array([float(xy[0]), float(xy[1]), 1.0], dtype=float)
                except Exception as e:
                    _logger.warning(f"annulus approach sample for obs_id={obs_id} failed: {e}")
        samples = self._orbit_approach_samples(anchor, robot_xy, n=max(1, int(n_approaches)))
        idx = int(approach_index) % len(samples)
        return samples[idx]

    def _navigation_waypoint_for_obs(
        self,
        obs_id: int,
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Closest approachable planar goal for this observation.

        Always aim at the object/frontier/node anchor (node centroid when present).
        With a robot pose, stop a short standoff short of the anchor; navmesh snapping
        happens downstream. Capture ``viewer_xyz`` is evidence provenance, not a goal.
        """
        anchor = self._obs_nav_anchor(int(obs_id))
        if anchor is None:
            return None
        robot_xy = self._robot_planar_xy(robot_xyt)
        if robot_xy is not None:
            return self._standoff_waypoint_toward(robot_xy, anchor)
        return np.array([float(anchor[0]), float(anchor[1]), 1.0], dtype=float)

    def _target_point_from_image_id(
        self,
        image_id: int,
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Return ``(x, y, 1)`` nav waypoint for observation ``image_id``."""
        return self._navigation_waypoint_for_obs(int(image_id), robot_xyt)

    def _resolve_eqa_action_image_ref(
        self,
        display_index: int,
        obs_ids: list[int] | None,
        *,
        slots_only: bool = False,
    ) -> int | None:
        """Map ``Action: Image N`` to a graph observation id.

        Prompt-attached images are renumbered 1..K (``obs_ids`` order). For ``look``,
        also accept a raw graph obs id (FIND_QUEUE / GRAPH_COUNT, e.g. ``action: 37``).
        ``read N`` passes ``slots_only=True`` so graph obs ids cannot be treated as zoom slots.
        """
        idx = int(display_index)
        if idx < 1:
            return None
        ids = [int(x) for x in (obs_ids or [])]
        if ids and 1 <= idx <= len(ids):
            return ids[idx - 1]
        if slots_only:
            return None
        if self._observation_by_id(idx) is not None:
            return idx
        return None

    def _target_point_from_display_image_index(
        self,
        display_index: int,
        *,
        obs_ids: list[int],
        nav_fallback_tail: list[GraphNavigationSample],
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Map 1-based ``Image N`` from the EQA prompt (or graph obs_id) to a waypoint."""
        if display_index < 1:
            return None
        oid = self._resolve_eqa_action_image_ref(display_index, obs_ids)
        if oid is not None:
            pt = self._navigation_waypoint_for_obs(oid, robot_xyt)
            if pt is not None:
                return pt
            return self._target_point_from_image_id(oid, robot_xyt)
        if nav_fallback_tail and display_index <= len(nav_fallback_tail):
            nv = nav_fallback_tail[display_index - 1]
            anchor = np.asarray(nv.xyz, dtype=float).reshape(-1)[:3]
            robot_xy = self._robot_planar_xy(robot_xyt)
            if robot_xy is not None:
                return self._standoff_waypoint_toward(robot_xy, anchor)
            return np.array([float(anchor[0]), float(anchor[1]), 1.0], dtype=float)
        return None

    def _count_candidate_nodes(self, question: str) -> tuple[list[GraphNode], CountTarget | None]:
        """Instance nodes whose primary label matches a count-MCQ target (FIND pointers)."""
        try:
            from emet.habitat.metrics import choices_are_count_mcq, parse_mcq_choices_from_question

            choices = parse_mcq_choices_from_question(question)
            if not choices or not choices_are_count_mcq(choices):
                return [], None
        except Exception:
            return [], None

        target = _count_target_from_stem(question_stem_for_keywords(question or ""))
        if target is None:
            # Anaphoric questions such as "I saw several lamps. How many are there?"
            # need the already extracted phrase rather than a bare stem token.
            for phrase in list(self._relevant_phrases) + list(self._relevant_objects or []):
                tokens = _strip_count_wrappers(_count_tokens(phrase))
                tokens = [token for token in tokens if token not in _QUESTION_STOPWORDS]
                if tokens:
                    target = CountTarget(tokens=tuple(tokens))
                    break
        if target is None:
            return [], None

        matches: list[GraphNode] = []
        target_phrases = [target.tokens, *_COUNT_PHRASE_ALIASES.get(target.tokens, ())]
        # FIND recall, not a new alias table: "table lamps" must still point at
        # detector "lamp" views so the VLM can look (q86 without close-look).
        head = target.tokens[-1] if target.tokens else ""
        if len(target.tokens) > 1 and head and (head in _COUNT_WORD_ALIASES or head.rstrip("s") in _COUNT_WORD_ALIASES):
            head_phrase = (head,)
            if head_phrase not in target_phrases:
                target_phrases = [*target_phrases, head_phrase]
        for node in self._nodes:
            if node.is_frontier or node.is_viewpoint or is_ground_truth_node(node) or not node.countable_instance:
                continue
            texts = finder_label_texts(node)
            if not texts:
                continue
            if any(_count_phrase_matches(phrase, text) for phrase in target_phrases for text in texts):
                matches.append(node)
        if not matches:
            return [], target

        if target.scope_tokens:
            room_by_id = self._node_room_by_id()
            if room_by_id and all(int(node.node_id) in room_by_id for node in matches):
                scoped = [
                    node
                    for node in matches
                    if _count_phrase_matches(target.scope_tokens, room_by_id[int(node.node_id)])
                ]
                if scoped:
                    matches = scoped
                # Scope miss (q93: dining stools vs "kitchen counter"): keep
                # scene-wide FIND views. List length is still not a count.
        unique: dict[str, GraphNode] = {}
        for node in matches:
            key = str(node.identity_key).strip() if node.identity_key else f"node:{int(node.node_id)}"
            unique.setdefault(key, node)
        return _collapse_count_nodes_spatially(list(unique.values())), target

    def _graph_count_hint(self, question: str) -> str:
        """List views to inspect for count MCQs; never assert an exact integer."""
        matches, target = self._count_candidate_nodes(question)
        phrase = target.phrase if target is not None else "target"
        visual = self._visual_find_obs_ids(self._eqa_find_phrases(), max_n=6)
        if visual:
            labeled = "; ".join(f"obs{int(oid)}" for oid in visual)
            return (
                f"GRAPH_COUNT: views to look at for '{phrase}' "
                f"(navigate via action=<obs id>; not an exact count): {labeled}. "
                "Count from attached images after looking; "
                "do not use this list length as the answer."
            )
        if not matches:
            return ""
        scope_note = ""
        if target is not None and target.scope_tokens:
            room_by_id = self._node_room_by_id()
            if not (room_by_id and all(int(node.node_id) in room_by_id for node in matches)):
                scope_note = (
                    f" Scope '{target.scope_phrase}' is not grounded in the graph; "
                    "look at these scene-wide candidate views."
                )
        labeled = format_graph_node_candidates(matches, max_nodes=6)
        return (
            f"GRAPH_COUNT: views to look at for '{phrase}' "
            f"(navigate via action=<graph obs id>; not an exact count): {labeled}."
            f"{scope_note} Count from attached images after looking; "
            "do not use this list length as the answer."
        )

    def _location_finder_nodes(self) -> list[GraphNode]:
        """Object nodes matching the question phrase (close-look name or detector primary)."""
        phrases = [p for p in self._confirmed_memory_phrases() if str(p).strip()]
        for obj in list(self._relevant_objects or []):
            if obj and obj not in phrases:
                phrases.append(str(obj))
        if not phrases:
            return []
        matches: list[GraphNode] = []
        seen_obs: set[int] = set()
        for node in self._nodes:
            if node.is_frontier or node.is_viewpoint or is_ground_truth_node(node):
                continue
            oid = int(node.obs_id)
            if oid in seen_obs:
                continue
            texts = finder_label_texts(node)
            if not texts:
                continue
            hit = False
            for phrase in phrases:
                tokens = tuple(_strip_count_wrappers(_count_tokens(phrase)))
                for text in texts:
                    if label_matches_relevant_object(phrase, text):
                        hit = True
                        break
                    if tokens and _count_phrase_matches(tokens, text):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                seen_obs.add(oid)
                matches.append(node)
        return matches

    def _compose_eqa_answer_obs_ids(
        self,
        *,
        forced: list[int],
        pin_obs: list[int],
        selected: list[int],
        max_images: int,
        count_question: bool,
        look_obs_id: int | None,
    ) -> list[int]:
        """Build Image 1..K from stored views: question-relevant RGB first.

        Image 1 is a pin / FIND / keyword view, not a leftover hallway look.
        Remaining slots mix other recalled views so one FIND noun cannot occupy
        the whole ``max_images`` budget. ``count_question`` keeps FIND pins ahead
        of unverified force_obs when both exist.
        """
        if max_images <= 0:
            return []

        def _take(dst: list[int], src: list[int | None]) -> None:
            seen = set(dst)
            if len(dst) >= max_images:
                return
            for raw in src:
                if raw is None:
                    continue
                oid = int(raw)
                if oid in seen:
                    continue
                if not self._obs_usable_for_eqa_image(oid):
                    continue
                dst.append(oid)
                seen.add(oid)
                if len(dst) >= max_images:
                    return

        look: list[int] = []
        _take(look, [look_obs_id])
        pins: list[int] = []
        _take(pins, pin_obs)
        extra: list[int] = []
        _take(extra, selected)
        out: list[int] = []
        pin_set = set(pins)
        # Verified evidence on a non-count question is Image 1 (agentic verified submit):
        # a stale Action look or location-FIND pin must not displace the confirmed evidence.
        # Count MCQs keep FIND pins ahead of force_obs so a single verified frame cannot
        # occupy the whole budget.
        if not count_question:
            _take(out, forced)
        look_relevant = bool(look and look[0] in pin_set)
        # Image 1 is a question-relevant stored view, not the last nav frame.
        # Hallway look after frontier chase must not displace clock / FIND RGB.
        if look and look_relevant:
            _take(out, look)
        elif pins:
            _take(out, pins[:1])
        elif look:
            _take(out, look)
        elif count_question:
            _take(out, extra[:1])
        for oid in extra:
            if oid not in out and oid not in pin_set:
                _take(out, [oid])
                break
        _take(out, pins)
        _take(out, extra)
        _take(out, look)
        _take(out, forced)
        return out
