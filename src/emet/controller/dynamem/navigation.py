# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Text command routing, exploration, A* execute, and safety filters."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from emet.controller.dynamem.constants import (
    DYNAMEM_HEAD_SETTLE_S,
    DYNAMEM_NAV_CHUNK_WPS,
    DYNAMEM_NAV_MAX_HOPS,
    _finite_xyz_traj_target,
)
from emet.controller.habitat_nav import (
    goal_key_xy,
    pick_uncovered_explore_target,
)
from emet.motion.algo.a_star import AStar
from emet.utils.logger import Logger

logger = Logger(__name__)


def _maybe_emit_navgrid_ascii(self, *, context: str = "") -> None:
    from emet.mapping.debug_navgrid_ascii import (
        build_navgrid_from_voxel_map,
        maybe_print_navgrid_ascii,
        navgrid_context_allowed,
    )

    if not navgrid_context_allowed(context):
        return
    try:
        robot_xy = self.world_base_xy()
    except Exception:
        robot_xy = None
    try:
        text = build_navgrid_from_voxel_map(
            self.voxel_map,
            graph_memory=getattr(self, "graph_memory", None),
            robot_xy=robot_xy,
        )
        if context:
            text = f"[navgrid:{context}]\n{text}"
        maybe_print_navgrid_ascii(text)
    except Exception as exc:
        logger.warning(f"Navgrid ASCII render skipped: {exc}")


def _filter_unsafe_nav_traj(
    self,
    traj: list,
    *,
    start_xyt: np.ndarray | list[float] | None = None,
) -> tuple[list, str | None, float | None]:
    """Drop low-clearance / unexplored waypoints before confirm/exec.

    Returns:
        (filtered_traj, reject_reason, min_clearance_m). reject_reason is set when the
        executable chunk would be empty after filtering (same as planner failure).
    """
    if not traj:
        return [], "no_plan", None
    planner = getattr(self, "planner", None)
    if planner is None:
        return list(traj), None, None
    if getattr(planner, "_clearance_m", None) is None:
        try:
            planner.reset()
        except Exception:
            pass

    min_c = float(getattr(self, "_min_clearance_m", getattr(planner, "min_clearance_m", 0.0)) or 0.0)
    # Preserve trailing [nan, object_xyz] marker if present.
    object_tail: list = []
    body = list(traj)
    if len(body) >= 2:
        mid = np.asarray(body[-2], dtype=np.float64).reshape(-1)
        if mid.size >= 2 and np.isnan(mid[:2]).all():
            object_tail = body[-2:]
            body = body[:-2]

    start_xy = None
    if start_xyt is not None:
        s = np.asarray(start_xyt, dtype=np.float64).reshape(-1)
        if s.size >= 2 and np.isfinite(s[:2]).all():
            start_xy = (float(s[0]), float(s[1]))

    kept: list = []
    reject: str | None = None
    clearances: list[float] = []
    prev_xy: tuple[float, float] | None = None
    prev_is_start = False
    for raw in body:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size < 2 or not np.isfinite(arr[:2]).all():
            continue
        xy = (float(arr[0]), float(arr[1]))
        # Always keep the first waypoint when it matches start (robot may sit in tight clearance).
        is_start = start_xy is not None and abs(xy[0] - start_xy[0]) < 1e-3 and abs(xy[1] - start_xy[1]) < 1e-3
        if not is_start and not planner.is_explored_xy(xy):
            reject = "rejected_unexplored"
            break
        c = float(planner.clearance_at_xy(xy))
        clearances.append(c)
        if not is_start and min_c > 0 and c < min_c:
            reject = "rejected_low_clearance"
            break
        # Mid-segment samples: reject chords that scrape low-clearance cells
        # between two individually-safe waypoints (post-simplify hazard).
        # The first segment may leave a tight start cell for the planner's
        # nearest clearance-safe escape cell.
        if prev_xy is not None and not is_start and not prev_is_start and hasattr(planner, "to_pt"):
            try:
                if not planner.is_in_line_of_sight(planner.to_pt(prev_xy), planner.to_pt(xy)):
                    reject = "rejected_low_clearance_segment"
                    break
            except Exception:
                pass
        kept.append(raw if isinstance(raw, list) else arr.tolist())
        prev_xy = xy
        prev_is_start = is_start

    min_along = float(min(clearances)) if clearances else None
    if not kept:
        return [], reject or "rejected_low_clearance", min_along
    if reject is not None and len(kept) <= 1 and start_xy is not None:
        # Only start survived → nothing useful to execute.
        return [], reject, min_along
    if object_tail:
        kept.extend(object_tail)
    return kept, None, min_along


def _mark_nav_goal_blocked(self, *, reason: str = "aborted_waypoint_timeout") -> None:
    """Remember the last nav goal so explore multi-goal A* skips it next time."""
    blocked = getattr(self, "_habitat_blocked_goals", None)
    if blocked is None:
        self._habitat_blocked_goals = set()
        blocked = self._habitat_blocked_goals
    recent = getattr(self, "_habitat_recent_goals", None)
    if recent is None:
        self._habitat_recent_goals = []
        recent = self._habitat_recent_goals

    meta = dict(getattr(self, "_last_nav_plan", None) or {})
    candidates: list[tuple[float, float]] = []
    for key in ("goal_xyt", "object_xyz", "effective_goal_xy"):
        raw = meta.get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size >= 2 and np.isfinite(arr[:2]).all():
            candidates.append((float(arr[0]), float(arr[1])))
    traj = meta.get("traj") or []
    for p in reversed(list(traj)):
        arr = np.asarray(p, dtype=np.float64).reshape(-1)
        if arr.size >= 2 and np.isfinite(arr[:2]).all():
            candidates.append((float(arr[0]), float(arr[1])))
            break
    for xy in candidates:
        key = goal_key_xy(xy)
        blocked.add(key)
        recent.append(key)
    del recent[:-16]
    self._record_nav_plan_fields(outcome=reason, blocked_after_abort=True)
    logger.warning(f"Nav abort ({reason}): marked {len(candidates)} goal key(s) blocked for replan/explore skip")


def _record_nav_plan_fields(self, **fields: Any) -> None:
    meta = dict(getattr(self, "_last_nav_plan", None) or {})
    meta.update(fields)
    self._last_nav_plan = meta


def execute_action(
    self,
    text: str,
) -> tuple[bool | None, np.ndarray | None]:
    """
    This function is used to navigate the robot give text query.
    It will call the process_text function to get the trajectory for the robot to follow.
    It will then execute the trajectory using the execute_trajectory function.
    If text is empty, it will just explore the environment.

    Args:
        text: The text query for the robot to navigate to / explore.

    Returns:
        The first element is a boolean indicating whether the navigation is finished. If it is None, it means the navigation has some problem.
        The second element is the location of the target object, useful used to tell the robot how to orient itself and prepare pregrasp pose for manipulation.
            If it is None, it means the navigation has some problem.
    """
    if not self._realtime_updates:
        self.robot.look_front()
        self.look_around()
        self.robot.look_front()
        self.robot.switch_to_navigation_mode()

    self.robot.switch_to_navigation_mode()

    start = self._current_planning_xyt()
    # Chunked A* (8 wps) must resume the leftover goal. Mapping used to call
    # execute_action("") once per "explore step", drop leftover, and pick a new
    # frontier — so long kitchen paths never finished and look-at never ran.
    for hop in range(DYNAMEM_NAV_MAX_HOPS):
        res = self.process_text(text, start)
        if len(res) == 0 and text != "" and text is not None:
            res = self.process_text("", start)
        if len(res) == 0:
            if hop == 0:
                logger.warning("No plan from process_text; try again.")
                return None, None
            logger.info("execute_action: leftover explore exhausted after %d hop(s)", hop)
            return False, None
        plan_meta = getattr(self, "_last_nav_plan", None) or {}
        announce = plan_meta.get("announce") or "Navigating…"
        if not str(announce).lower().startswith("navigat"):
            announce = f"Navigating… {announce}"
        # Confirm before posture/exec so operators can reject wall-hugging plans.
        object_xyz = None
        if len(res) >= 2 and np.isnan(np.asarray(res[-2], dtype=np.float64)).all():
            object_xyz = res[-1]
        from emet.controller.nav_confirm import confirm_navigation_plan

        if not confirm_navigation_plan(self, res, meta=plan_meta, object_xyz=object_xyz):
            self._record_nav_plan_fields(outcome="user_cancelled", confirmed=False)
            return None, None
        self._record_nav_plan_fields(confirmed=True, outcome="executing")
        self.announce_action(announce)
        n_exec = sum(1 for p in res if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all())
        logger.info(
            "Navigation plan OK; executing %d waypoints (localize=%s mode=%s path≈%.2fm chunked=%s hop=%d)",
            n_exec,
            plan_meta.get("localize_source", "?"),
            plan_meta.get("mode", "?"),
            float(plan_meta.get("path_m") or 0.0),
            bool(plan_meta.get("chunked")),
            hop,
        )
        nav_timeout = self._find_phase_nav_timeout()
        wait_obs = getattr(self.robot, "wait_for_obs", None)
        if wait_obs is not None:
            wait_obs(timeout=nav_timeout)
        if self._navigation_origin_xyt() is None:
            logger.warning(
                "navigation_origin_xyt missing from emet_session; sim nav may use wrong frame "
                "(restart sim server and ensure first observation arrived)."
            )
        # process_text ends with robot.say(...); re-sync nav posture + forward gaze before base moves.
        self.robot.move_to_nav_posture()
        self.robot.look_front(blocking=True)
        time.sleep(DYNAMEM_HEAD_SETTLE_S)
        # This means that the robot has already finished all of its trajectories and should stop to manipulate the object.
        # We will append a nan and point coordinates of the target object on the trajectory to denote that the robot is reaching the target point
        if len(res) >= 2 and np.isnan(res[-2]).all():
            if len(res) > 2:
                exec_ok = self.robot.execute_trajectory(
                    res[:-2],
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    per_waypoint_timeout=nav_timeout,
                    final_timeout=max(nav_timeout, 30.0),
                    blocking=True,
                    world_frame=True,
                )
                if exec_ok is False:
                    self._record_nav_plan_fields(outcome="aborted_waypoint_timeout")
                    self._mark_nav_goal_blocked(reason="aborted_waypoint_timeout")
                    logger.warning("Navigation aborted: waypoint timeout during execute_trajectory")
                    return None, None

            self.robot.look_front()
            self.update()
            self._record_nav_plan_fields(outcome="ok")
            return True, res[-1]
        # Chunk: execute, grow the voxel/graph at this pose, resume leftover.
        exec_ok = self.robot.execute_trajectory(
            res,
            pos_err_threshold=self.pos_err_threshold,
            rot_err_threshold=self.rot_err_threshold,
            per_waypoint_timeout=nav_timeout,
            final_timeout=max(nav_timeout, 30.0),
            blocking=True,
            world_frame=True,
        )
        if exec_ok is False:
            self._record_nav_plan_fields(outcome="aborted_waypoint_timeout")
            self._mark_nav_goal_blocked(reason="aborted_waypoint_timeout")
            logger.warning("Navigation aborted: waypoint timeout during execute_trajectory")
            return None, None
        self.robot.look_front()
        self.update()
        self._record_nav_plan_fields(outcome="ok_chunk")
        start = self._current_planning_xyt()
    logger.info("execute_action: still chunked after %d hops", DYNAMEM_NAV_MAX_HOPS)
    return False, None


def run_exploration(self):
    """
    Go through exploration when the robot has not received any text query from the user.
    We use the voxel_grid map created by our collector to sample free space, and then use A* planner to get there.
    """

    self.announce_action("Exploring…")
    # "" means the robot has not received any text query from the user and should conduct exploration just to better know the environment
    status, _ = self.execute_action("")
    if status is None:
        self.announce_action("Exploring… no valid frontier right now")
        logger.warning("Exploration failed (no valid plan or frontier).")
        return False
    self._maybe_emit_navgrid_ascii(context="explore")
    return True


def process_text(self, text, start_pose):
    """
    Process the text query and return the trajectory for the robot to follow.
    """

    logger.debug("process_text: %r", text)

    clear_nav = getattr(self.rerun_visualizer, "clear_nav_plan", None)
    if callable(clear_nav):
        clear_nav()
    else:
        self.rerun_visualizer.clear_identity("world/object")
        self.rerun_visualizer.clear_identity("world/xyt_goal")
        self.rerun_visualizer.clear_identity("world/robot_start_pose")
        self.rerun_visualizer.clear_identity("world/direction")
    self.rerun_visualizer.clear_identity("robot_monologue")
    self.rerun_visualizer.clear_identity("/observation_similar_to_text")
    self._last_nav_plan = None

    debug_text = ""
    mode = "navigation"
    localize_source = ""
    obs = None
    localized_point = None
    waypoints = None

    if text is not None and text != "" and self.space.traj is not None:
        logger.debug("Reusing saved trajectory target: %s", self.space.traj)
        traj_target_point = self.space.traj[-1]
        if hasattr(self.encoder, "feature_matching_threshold") and self.voxel_map.verify_point(
            text,
            traj_target_point,
            similarity_threshold=self.encoder.feature_matching_threshold,
        ):
            localized_point = traj_target_point
            localize_source = "saved_traj+verify"
            debug_text += "## Reusing prior plan target (SigLIP neighborhood OK).\n"
        elif hasattr(self.encoder, "feature_matching_threshold") and _finite_xyz_traj_target(traj_target_point):
            # Short queries ("red object") often fail SigLIP neighborhood re-check; still navigate to last grounding.
            localized_point = traj_target_point
            localize_source = "saved_traj"
            debug_text += "## Reusing prior plan target; semantic re-check was not decisive.\n"

    # Mapping / empty-text explore: resume the leftover frontier instead of
    # sampling a new one (leftover was only reused for nonempty object queries).
    continue_saved = False
    if localized_point is None and (text is None or text == "") and self.space.traj is not None:
        traj_target_point = self.space.traj[-1]
        if _finite_xyz_traj_target(traj_target_point):
            localized_point = traj_target_point
            localize_source = "saved_traj"
            mode = "exploration"
            continue_saved = True
            debug_text += "## Continuing leftover explore chunk toward prior frontier.\n"

    logger.debug("Target verification done (localized_point=%s)", localized_point is not None)

    if text is not None and text != "" and localized_point is None:
        graph_point = self._localize_point_from_graph_memory(text)
        if graph_point is not None:
            localized_point = graph_point
            localize_source = "graph"
            debug_text += "## Localized target from graph memory.\n"
            mode = "navigation"
            logger.info("Localized %r from graph memory at %s", text, np.asarray(graph_point).reshape(-1)[:3])

    if text is not None and text != "" and localized_point is None:
        det = getattr(self.voxel_map, "detection_model", None)
        if det is not None or self.encoder is not None:
            try:
                (
                    localized_point,
                    loc_debug,
                    obs,
                    pointcloud,
                ) = self.voxel_map.localize_text(text, debug=True, return_debug=True)
                if localized_point is not None:
                    localize_source = "voxel"
                    debug_text += "## Localized target from voxel semantic memory.\n"
                if loc_debug:
                    debug_text += str(loc_debug)
                logger.info("Localized %r from voxel map: %s", text, localized_point is not None)
            except Exception as exc:
                logger.debug("voxel localize_text failed for %r: %s", text, exc)

    # Do Frontier based exploration (optionally biased by the active EQA question).
    if not continue_saved and (text is None or text == "" or localized_point is None):
        debug_text += "## No object localization; falling back to frontier exploration.\n"
        frontier_text = self._exploration_text(text)
        explore_pt = pick_uncovered_explore_target(
            self,
            question=frontier_text or None,
            blocked=getattr(self, "_habitat_blocked_goals", None),
            recent_goals=getattr(self, "_habitat_recent_goals", None),
        )
        if explore_pt is not None:
            localized_point = explore_pt
            localize_source = "frontier_uncovered"
            debug_text += "## Selected blocked-aware explore frontier.\n"
            mode = "exploration"
        else:
            graph_frontier = self._best_frontier_point_from_graph(frontier_text)
            if graph_frontier is not None:
                localized_point = graph_frontier
                localize_source = "frontier_graph"
                debug_text += "## Selected frontier target from graph memory.\n"
                mode = "exploration"
            else:
                localized_point = self.space.sample_frontier(self.planner, start_pose, frontier_text)
                localize_source = "frontier_space" if localized_point is not None else ""
                mode = "exploration"

    if obs is not None and mode == "navigation":
        obs = self.voxel_map.find_obs_id_for_text(text)
        if obs is not None:
            try:
                idx = int(obs.item()) if hasattr(obs, "item") else int(obs)
                if 0 < idx <= len(self.voxel_map.observations):
                    rgb = self.voxel_map.observations[idx - 1].rgb
                    self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb)
            except (TypeError, ValueError, IndexError):
                pass

    if localized_point is None:
        logger.warning("process_text: no localized point for query %r", text)
        return []

    # TODO: Do we really need this line?
    if len(localized_point) == 2:
        localized_point = np.array([localized_point[0], localized_point[1], 0])

    _lp = np.asarray(
        localized_point.detach().cpu().numpy() if isinstance(localized_point, torch.Tensor) else localized_point,
        dtype=np.float64,
    ).reshape(-1)
    ox, oy = float(_lp[0]), float(_lp[1])
    oz = float(_lp[2]) if _lp.size > 2 else 1.5
    if not np.isfinite(oz) or abs(oz) < 1e-9:
        oz = 1.5

    waypoints = None
    n_planned = 0
    res = None
    point = None

    # Exploration: top-K frontiers → one multi-goal A* (skip sealed / unreachable).
    # Object nav stays single-goal. Leftover chunks keep the same frontier.
    if mode == "exploration" and not continue_saved and isinstance(self.planner, AStar):
        from emet.motion.frontier_goals import collect_explore_frontier_candidates

        frontier_text = self._exploration_text(text)
        cands = collect_explore_frontier_candidates(
            self,
            question=frontier_text or None,
            k=8,
            blocked=getattr(self, "_habitat_blocked_goals", None),
            recent_goals=getattr(self, "_habitat_recent_goals", None),
            seeds=[localized_point],
        )
        object_xys: list[np.ndarray] = []
        nav_goals: list[np.ndarray] = []
        for cand in cands:
            g = self.space.sample_navigation(start_pose, self.planner, cand)
            if g is None:
                continue
            object_xys.append(np.asarray(cand, dtype=np.float64).reshape(-1))
            nav_goals.append(np.asarray(g, dtype=np.float64).reshape(-1))

        if len(nav_goals) >= 2:
            res = self.planner.plan(start_pose, nav_goals[0], goals=nav_goals)
            gi = getattr(res, "goal_index", None) if res is not None else None
            if res is not None and res.success and gi is not None and 0 <= int(gi) < len(nav_goals):
                gi_i = int(gi)
                point = nav_goals[gi_i]
                localized_point = object_xys[gi_i]
                _lp = np.asarray(localized_point, dtype=np.float64).reshape(-1)
                ox, oy = float(_lp[0]), float(_lp[1])
                oz = float(_lp[2]) if _lp.size > 2 else 1.5
                if not np.isfinite(oz) or abs(oz) < 1e-9:
                    oz = 1.5
                localize_source = f"{localize_source or 'frontier'}_multi_goal"
                logger.info(
                    "Multi-goal explore: %d candidates, chose index=%d xy=(%.2f, %.2f)",
                    len(nav_goals),
                    gi_i,
                    ox,
                    oy,
                )
            elif res is not None and not res.success:
                logger.warning("Multi-goal explore plan failed: %s", res.reason)
                res = None
        elif len(nav_goals) == 1:
            point = nav_goals[0]
            localized_point = object_xys[0]
            _lp = np.asarray(localized_point, dtype=np.float64).reshape(-1)
            ox, oy = float(_lp[0]), float(_lp[1])
            res = self.planner.plan(start_pose, point)

    if point is None and res is None:
        point = self.space.sample_navigation(start_pose, self.planner, localized_point)

    logger.info(
        "Nav endpoint sample: localize=%s target_xy=(%.2f, %.2f) base_goal=%s",
        localize_source or "?",
        ox,
        oy,
        None if point is None else np.asarray(point).reshape(-1)[:3],
    )

    if res is None:
        if point is None:
            logger.warning("No navigation endpoint sampled (planner may fail).")
        else:
            res = self.planner.plan(start_pose, point)

    if res is not None and res.success:
        waypoints = [pt.state for pt in res.trajectory]
        n_planned = len(waypoints)
    elif res is not None:
        waypoints = None
        logger.warning("Planner failure: %s", res.reason)

    # If we are navigating to some object of interest, send (x, y, z) of
    # the object so that we can make sure the robot looks at the object after navigation
    traj = []
    chunked = False
    full_traj_for_viz = None
    if waypoints is not None:
        finished = len(waypoints) <= DYNAMEM_NAV_CHUNK_WPS
        chunked = not finished
        full_traj_for_viz = self.planner.clean_path_for_xy(
            list(waypoints), start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0
        )
        if finished:
            self.space.traj = None
        else:
            self.space.traj = waypoints[DYNAMEM_NAV_CHUNK_WPS:] + [[np.nan, np.nan, np.nan], localized_point]
        if not finished:
            waypoints = waypoints[:DYNAMEM_NAV_CHUNK_WPS]
        traj = self.planner.clean_path_for_xy(waypoints, start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0)
        if finished:
            traj.append([np.nan, np.nan, np.nan])
            if isinstance(localized_point, torch.Tensor):
                localized_point = localized_point.tolist()
            traj.append(localized_point)
        traj, reject_reason, min_clr = self._filter_unsafe_nav_traj(traj, start_xyt=start_pose)
        if reject_reason is not None or not traj:
            logger.warning(
                "Nav plan rejected after safety filter: %s (min_clearance=%s)",
                reject_reason,
                min_clr,
            )
            self._last_nav_plan = {
                "mode": mode,
                "localize_source": localize_source,
                "n_planned": n_planned,
                "chunked": chunked,
                "path_m": 0.0,
                "min_clearance_m": min_clr,
                "outcome": reject_reason or "rejected_low_clearance",
                "announce": f"Plan rejected ({reject_reason or 'unsafe'})",
                "traj": [],
            }
            return []
        logger.info(
            "Planned trajectory: %d exec / %d planned waypoints (finished_chunk=%s min_clearance=%.3f)",
            len([p for p in traj if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all()]),
            n_planned,
            finished,
            float(min_clr) if min_clr is not None else float("nan"),
        )

    # Talk about what you are doing, as the robot.
    if self.robot is not None:
        if text is not None and text != "":
            self.robot.say("I am looking for a " + text + ".")
        else:
            self.robot.say("I am exploring the environment.")

    if text is not None and text != "":
        debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
    else:
        debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
    debug_text += (
        f"\n### Plan: mode=`{mode}` localize=`{localize_source or 'n/a'}` planned_wps={n_planned} chunked={chunked}\n"
    )
    debug_text = "# Robot's monologue: \n" + debug_text
    self._rerun_monologue_base = debug_text
    self._rerun_refresh_monologue_panel()

    log_plan = getattr(self.rerun_visualizer, "log_nav_plan", None)
    if callable(log_plan) and traj:
        self._last_nav_plan = log_plan(
            traj,
            full_traj=full_traj_for_viz,
            start_xyt=start_pose,
            goal_xyt=point,
            object_xyz=[ox, oy, oz],
            mode=mode,
            localize_source=localize_source,
            query=text or "",
            n_planned=n_planned or None,
            chunked=chunked,
        )
        # Attach clearance / safety fields for agent tools.
        try:
            clr = self.planner.clearance_at_xy(start_pose[:2])
            path_clrs = [
                self.planner.clearance_at_xy(np.asarray(p).reshape(-1)[:2])
                for p in traj
                if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all()
            ]
            self._record_nav_plan_fields(
                min_clearance_m=float(min(path_clrs)) if path_clrs else None,
                base_clearance_m=float(clr),
                min_clearance_required_m=float(getattr(self, "_min_clearance_m", 0.0)),
                traj=list(traj),
            )
        except Exception:
            pass
    elif traj:
        # NullVisualizer / older stubs: keep minimal legacy arrows.
        origins = []
        vectors = []
        for idx in range(len(traj) - 1):
            a = np.asarray(traj[idx], dtype=np.float64).reshape(-1)
            b = np.asarray(traj[idx + 1], dtype=np.float64).reshape(-1)
            if a.size < 2 or b.size < 2 or not np.isfinite(a[:2]).all() or not np.isfinite(b[:2]).all():
                continue
            origins.append([float(a[0]), float(a[1]), 1.5])
            vectors.append([float(b[0] - a[0]), float(b[1] - a[1]), 0.0])
        if origins:
            self.rerun_visualizer.log_arrow3D("world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1)
        path_clrs = [
            self.planner.clearance_at_xy(np.asarray(p).reshape(-1)[:2])
            for p in traj
            if np.isfinite(np.asarray(p, dtype=np.float64).reshape(-1)[:2]).all()
        ]
        self._last_nav_plan = {
            "mode": mode,
            "localize_source": localize_source,
            "n_planned": n_planned,
            "chunked": chunked,
            "path_m": 0.0,
            "min_clearance_m": float(min(path_clrs)) if path_clrs else None,
            "min_clearance_required_m": float(getattr(self, "_min_clearance_m", 0.0)),
            "announce": f"Navigating via {localize_source or mode}: {n_planned} wps",
            "traj": list(traj),
        }

    return traj


def navigate(self, text, max_step=10):
    """
    The robot calls this function to navigate to the object.
    It will call execute_action function until it is ready for manipulation
    """
    self.maybe_save_rerun_recording()
    finished = False
    step = 0
    end_point = None
    while not finished and step < max_step:
        logger.debug("navigate step %s/%s", step, max_step)
        step += 1
        finished, end_point = self.execute_action(text)
        if finished is None:
            logger.warning("Navigation failed (blocked or no progress).")
            return None
    return end_point
