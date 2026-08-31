# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Planning-frame pose helpers, graph localize, and Rerun blueprint."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from emet.utils.geometry import nav_xyt_to_world_xyt
from emet.utils.logger import Logger

logger = Logger(__name__)


def _start_threads(self) -> None:
    """DynamemController does not use realtime update threads."""
    pass


def move_to_manip_posture(self) -> None:
    """Move the robot to manipulation posture (delegates to the robot client)."""
    self.robot.move_to_manip_posture()


def move_to_nav_posture(self) -> None:
    """Move the robot to navigation posture (delegates to the robot client)."""
    self.robot.move_to_nav_posture()


def _robot_emet_session(self) -> dict[str, Any] | None:
    get_sess = getattr(self.robot, "get_emet_session", None)
    if get_sess is None:
        return None
    return get_sess()


def _navigation_origin_xyt(self) -> np.ndarray | None:
    """World spawn pose from ZMQ ``emet_session`` (cached from first observation if needed)."""
    sess = self._robot_emet_session()
    if sess is not None:
        org = sess.get("navigation_origin_xyt")
        if org is not None:
            origin = np.asarray(org, dtype=np.float64).reshape(-1)[:3]
            self._cached_navigation_origin_xyt = origin.copy()
            return origin
    if self._cached_navigation_origin_xyt is not None:
        return self._cached_navigation_origin_xyt
    return None


def _planning_base_xyt(self, local_xyt: np.ndarray | list | tuple) -> np.ndarray:
    """Episode-relative ZMQ base pose → world frame for voxel-grid planning."""
    xyt = np.asarray(local_xyt, dtype=np.float64).reshape(-1)
    if xyt.size < 3:
        xyt = np.pad(xyt, (0, max(0, 3 - xyt.size)), mode="constant")
    sess = self._robot_emet_session()
    if sess is None and self._cached_navigation_origin_xyt is not None:
        sess = {"navigation_origin_xyt": self._cached_navigation_origin_xyt.tolist()}
    return nav_xyt_to_world_xyt(xyt[:3], sess)


def _current_planning_xyt(self) -> np.ndarray:
    """Current base ``(x, y, θ)`` in the voxel-map / world planning frame.

    ``get_base_pose`` is episode-relative (ZMQ gps/compass). A* and
    ``execute_trajectory(world_frame=True)`` plan in the world frame anchored at
    ``navigation_origin_xyt`` — on Robocasa that origin is several meters from
    ``(0, 0)``. Always convert before measuring progress or starting a hop.
    """
    return np.asarray(self._planning_base_xyt(self.robot.get_base_pose()), dtype=np.float64)


def world_base_xy(self) -> tuple[float, float] | None:
    """Robot base (x, y) in the voxel-map / world frame (not raw ZMQ gps)."""
    if self.robot is None or not hasattr(self.robot, "get_base_pose"):
        return None
    try:
        wxyt = self._planning_base_xyt(self.robot.get_base_pose())
        return float(wxyt[0]), float(wxyt[1])
    except Exception:
        return None


def _sync_graph_frontier_nodes(self) -> None:
    gm = self.graph_memory
    if gm is None or not getattr(gm, "frontier_nodes_enabled", False):
        return
    from emet.memory.graph_eqa.ingest.dynamem_graph_hooks import sync_graph_frontier_nodes

    question = getattr(self, "_eqa_question", None)
    sync_graph_frontier_nodes(
        graph_memory=gm,
        voxel_map=self.voxel_map,
        planner=self.planner,
        base_xyt=self._planning_base_xyt(self.robot.get_base_pose()),
        question=question,
    )


def _exploration_text(self, text: str | None) -> str | None:
    """Text used for question-guided frontier scoring (explicit query or active EQA question)."""
    if text is not None and str(text).strip():
        return str(text).strip()
    q = getattr(self, "_eqa_question", None)
    if q is not None and str(q).strip():
        return str(q).strip()
    return None


def _localize_point_from_graph_memory(self, text: str) -> np.ndarray | None:
    """Resolve a nav goal from graph nodes (GT or perception) when voxel localize misses."""
    gm = getattr(self, "graph_memory", None)
    if gm is None or not (text or "").strip():
        return None
    from emet.memory.graph_eqa.graph_memory import heuristic_relevant_objects

    query = text.lower().strip()
    tokens = heuristic_relevant_objects(text)
    best_node = None
    best_score = -1
    for node in gm.get_nodes():
        if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
            continue
        labels = [str(label).lower() for label in (node.labels or []) if str(label).strip()]
        if not labels:
            continue
        blob = " ".join(labels)
        score = 0
        if query in blob:
            score += 3
        for tok in tokens:
            if tok.lower() in blob:
                score += 1
        if score > best_score:
            best_score = score
            best_node = node
    if best_node is None or best_score <= 0:
        return None
    return np.array([float(best_node.xyz[0]), float(best_node.xyz[1]), 1.0], dtype=float)


def _best_frontier_point_from_graph(self, text: str | None) -> np.ndarray | None:
    """Pick the frontier graph node best matching *text* / the active EQA question."""
    gm = getattr(self, "graph_memory", None)
    if gm is None or not getattr(gm, "frontier_nodes_enabled", True):
        return None
    from emet.memory.graph_eqa.spatial.frontier_nodes import exploration_keywords_from_text, keyword_overlap_score

    frontier_nodes = [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)]
    if not frontier_nodes:
        return None
    keywords = exploration_keywords_from_text(text)
    robot = getattr(self, "robot", None)
    if robot is not None and hasattr(robot, "get_base_pose"):
        pose = self._planning_base_xyt(robot.get_base_pose())
        rx, ry = float(pose[0]), float(pose[1])
    else:
        rx, ry = 0.0, 0.0
    if not keywords:
        node = min(
            frontier_nodes,
            key=lambda n: math.hypot(float(n.xyz[0]) - rx, float(n.xyz[1]) - ry),
        )
        return np.array([float(node.xyz[0]), float(node.xyz[1]), 1.0], dtype=float)
    best_node = None
    best_score = -1.0
    best_dist = float("inf")
    for node in frontier_nodes:
        labels = [str(lbl).strip().lower() for lbl in (node.labels or []) if str(lbl).strip()]
        score = keyword_overlap_score(labels, keywords)
        dist = math.hypot(float(node.xyz[0]) - rx, float(node.xyz[1]) - ry)
        if score > best_score or (score == best_score and dist < best_dist):
            best_score = score
            best_dist = dist
            best_node = node
    if best_node is None or best_score <= 0:
        return None
    return np.array([float(best_node.xyz[0]), float(best_node.xyz[1]), 1.0], dtype=float)


def setup_custom_blueprint(self):
    """
    This function define rerun blueprint of DynaMem module.
    """
    if getattr(self.rerun_visualizer, "enabled", True) is False:
        return
    from emet.visualization.rerun import spatial3d_view_world

    main = rrb.Horizontal(
        spatial3d_view_world(),
        rrb.Vertical(
            rrb.TextDocumentView(name="text", origin="robot_monologue"),
            rrb.Spatial2DView(name="relevant image", origin="/observation_similar_to_text"),
        ),
        rrb.Vertical(
            rrb.Spatial2DView(name="head_rgb", origin="world/head_camera/rgb"),
            rrb.Spatial2DView(name="ee_rgb", origin="world/ee_camera/rgb"),
            rrb.Spatial2DView(name="map_topdown", origin="world/map_snapshot/topdown"),
        ),
        rrb.Vertical(
            rrb.TextDocumentView(name="Scene Graph", origin="world/scene_graph/summary"),
        ),
        column_shares=[3, 1, 1, 1],
    )
    collapse = getattr(self.rerun_visualizer, "collapse_panels", True)
    my_blueprint = rrb.Blueprint(
        rrb.Vertical(main, rrb.TimePanel(state=True)),
        collapse_panels=collapse,
    )
    rr.send_blueprint(my_blueprint)
