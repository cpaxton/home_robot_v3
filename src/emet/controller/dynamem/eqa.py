# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Classic voxel EQA loop and navigate_to_target_pose."""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from emet.controller.dynamem.constants import DYNAMEM_NAV_CHUNK_WPS, DYNAMEM_NAV_MAX_HOPS
from emet.controller.habitat_nav import (
    NavAttemptResult,
    NavOutcome,
    goal_key_xy,
    habitat_navmesh_navigate,
    habitat_perfect_nav_enabled,
    is_habitat_robot_client,
)
from emet.utils.logger import Logger

logger = Logger(__name__)


def _patch_images(self, images: list[Image.Image], patch_size=(480, 640), gap=5):
    """
    Patch a list of PIL Images into a numpy array, used for dicrod bot
    """
    # Resize all images to the same patch size
    images = [img.resize(patch_size) for img in images]

    # Calculate total width and height
    n_images = len(images)
    total_width = patch_size[0] * n_images + gap * (n_images - 1)
    total_height = patch_size[1]

    # Create a blank canvas
    canvas = Image.new("RGB", (total_width, total_height))

    # Paste images side-by-side
    for idx, img in enumerate(images):
        x = idx * (patch_size[0] + gap)
        canvas.paste(img, (x, 0))

    # Convert to numpy array
    return np.array(canvas)


def run_eqa(self, question, max_planning_steps: int = 5):
    """
    API for calling EQA module
    """
    self.maybe_save_rerun_recording()

    self.robot.switch_to_navigation_mode()

    discord_text, relevant_images = "", []

    # Early-stop: when exploration stalls (the scene graph gains no new nodes) yet the
    # model keeps returning the same answer, further planning steps re-ask with
    # identical inputs and cannot change the result — common when a question keyword
    # never becomes a node label (abstract/action words) or the robot is physically
    # stuck. Stop after ``stall_patience`` such steps. Productive exploration (a growing
    # graph) always continues, so this never cuts a run that is still gathering evidence.
    stall_patience = int(self.parameters.get("eqa_stall_patience", 4) or 0)
    prev_node_count = -1
    prev_answer = None
    stall = 0

    for _cnt_step in range(max_planning_steps):
        logger.info(
            "EQA planning step %d/%d for %r",
            _cnt_step + 1,
            max_planning_steps,
            question if isinstance(question, str) else str(question)[:80],
        )
        answer, discord_text, relevant_images, confidence = self.run_eqa_one_iter(question)
        if confidence:
            self.robot.say("The answer to " + question + " is " + answer)
            break

        if stall_patience > 0 and self.graph_memory is not None:
            # Never early-stop on a repeated Yes/No while question objects are still
            # uncovered — absence is not evidence; keep exploring frontiers.
            covers = getattr(self.graph_memory, "_graph_covers_relevant_objects", None)
            uncovered = bool(callable(covers) and not covers())
            if uncovered:
                stall = 0
                prev_node_count = len(self.graph_memory.get_nodes())
                prev_answer = self.graph_memory.last_eqa_parsed[1]
            else:
                node_count = len(self.graph_memory.get_nodes())
                cur_answer = self.graph_memory.last_eqa_parsed[1]
                if node_count <= prev_node_count and cur_answer and cur_answer == prev_answer:
                    stall += 1
                else:
                    stall = 0
                prev_node_count = node_count
                prev_answer = cur_answer
                if stall >= stall_patience:
                    logger.info(
                        "EQA early stop after %d/%d planning steps: exploration stalled (no new graph "
                        "nodes, stable answer %r) for %d steps; accepting the answer.",
                        _cnt_step + 1,
                        max_planning_steps,
                        cur_answer,
                        stall + 1,
                    )
                    break

    relevant_image = self._patch_images(relevant_images, patch_size=(270, 360))
    self.rerun_iter += 1

    return discord_text, relevant_image


def run_eqa_one_iter(self, question, max_movement_step: int = 5):
    answer_output = None

    if not self._realtime_updates and not getattr(self, "_fast_explore_lookaround", False):
        self.robot.look_front()
        self.look_around()
        self.robot.look_front()
        self.robot.switch_to_navigation_mode()
    elif not self._realtime_updates:
        # Explore-loop (`_fast_explore_lookaround`) already swept / mapped; skip another ~60s look_around.
        self.robot.look_front()
        self.robot.switch_to_navigation_mode()

    try:
        logger.info("EQA query_answer start for %r", question if isinstance(question, str) else str(question)[:80])
        t_qa0 = time.monotonic()
        (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            target_point,
            relevant_images,
        ) = self.voxel_map.query_answer(question, self._planning_base_xyt(self.robot.get_base_pose()), self.planner)
        logger.info(
            "EQA query_answer done wall_s=%.1f confidence=%s answer=%r",
            time.monotonic() - t_qa0,
            confidence,
            answer,
        )
    except:
        reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images = (
            "Exception happens in LLM querying!",
            "Unknown",
            False,
            "Exception happens in LLM querying!",
            self.space.sample_frontier(self.planner, self._planning_base_xyt(self.robot.get_base_pose()), text=None),
            [],
        )

    # Log the texts to rerun visualizer
    confidence_text = "I am confident with the answer" if confidence else "I am NOT confident with the answer"

    reasoning_output = (
        "\n#### Reasoning for the answer: " + reasoning
        if confidence
        else "\n#### Reasoning for the confidence: " + confidence_reasoning
    )

    answer_output = (
        "#### **Question:** "
        + question
        + "\n#### **Answer:** "
        + answer
        + "\n#### **Confidence:** "
        + confidence_text
        + reasoning_output
    )

    self._rerun_monologue_base = answer_output
    self._rerun_refresh_monologue_panel()
    if len(relevant_images) != 0:
        self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", self._patch_images(relevant_images))

    # chat with user in the rerun
    if confidence:
        discord_text = answer + ". I believe this answer is correct because " + reasoning
    else:
        discord_text = "I am not confident to answer the question because " + confidence_reasoning

    discord_text += "\nI also provide relevant images here."

    if confidence:
        return answer, discord_text, relevant_images, confidence

    start_pose = self._planning_base_xyt(self.robot.get_base_pose())

    logger.debug("EQA navigate: target_point=%s", target_point)
    if target_point is None:
        # No usable navigation target (degenerate action parsed no image index): skip movement.
        return answer, discord_text, relevant_images, confidence

    # If we want to explore non obstacles (especially frontiers), remember where we currently want to face
    obstacles, _ = self.voxel_map.get_2d_map()
    target_grid = self.voxel_map.xy_to_grid_coords((target_point[0], target_point[1]))
    if not obstacles[int(target_grid[0]), int(target_grid[1])]:
        target_theta = self.space.sample_navigation(start_pose, self.planner, target_point)[-1]
        logger.debug("EQA navigate: target_theta=%s", target_theta)
    else:
        target_theta = None

    movement_step = 0
    while movement_step < max_movement_step:
        start_pose = self._planning_base_xyt(self.robot.get_base_pose())
        movement_step += 1
        self.update()
        finished = self.navigate_to_target_pose(target_point, start_pose, target_theta)
        if finished.finished:
            break

    return answer, discord_text, relevant_images, confidence


def _log_nav_attempt(
    self,
    nav_res: NavAttemptResult,
    *,
    target_obs_id: int | None,
    goal_xy: np.ndarray,
) -> None:
    if target_obs_id is not None:
        nav_res.target_obs_id = target_obs_id
    if getattr(nav_res, "goal_xy", None) is None:
        nav_res.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
    # Structured status_code → _last_nav_plan + optional attempt ledger.
    from emet.controller.nav_attempt import sync_nav_attempt_to_ledger

    sync_nav_attempt_to_ledger(self, nav_res, source="eqa")
    recorder = getattr(self, "_episode_diagnostics_recorder", None)
    if recorder is not None and hasattr(recorder, "append_nav_attempt"):
        row = {
            "target_obs_id": target_obs_id,
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "effective_goal_xy": (
                [float(nav_res.effective_goal_xy[0]), float(nav_res.effective_goal_xy[1])]
                if getattr(nav_res, "effective_goal_xy", None)
                else None
            ),
            "method": nav_res.method,
            "success": nav_res.success,
            "finished": nav_res.finished,
            "dist_m": nav_res.dist_m,
            "note": nav_res.note,
            "status_code": getattr(nav_res, "status_code", None),
        }
        if getattr(nav_res, "path_xy", None):
            row["path_xy"] = nav_res.path_xy
        recorder.append_nav_attempt(row)
    if nav_res.finished or nav_res.success:
        eff = getattr(nav_res, "effective_goal_xy", None) or (
            float(goal_xy[0]),
            float(goal_xy[1]),
        )
        key = goal_key_xy(eff)
        recent = getattr(self, "_habitat_recent_goals", None)
        if recent is not None:
            recent.append(key)
            del recent[:-8]
    elif (
        str(nav_res.note or "").startswith("already_at_goal")
        or (not nav_res.finished and float(nav_res.dist_m) < 0.08)
        or (not nav_res.success and float(nav_res.dist_m) < 0.12)
    ):
        # Stuck / noop / no-progress: remember so uncovered explore does not re-pick.
        eff = getattr(nav_res, "effective_goal_xy", None) or (
            float(goal_xy[0]),
            float(goal_xy[1]),
        )
        key = goal_key_xy(eff)
        recent = getattr(self, "_habitat_recent_goals", None)
        if recent is not None:
            recent.append(key)
            del recent[:-8]
        blocked = getattr(self, "_habitat_blocked_goals", None)
        if blocked is not None:
            blocked.add(key)
            blocked.add(goal_key_xy(goal_xy))


def navigate_to_target_pose(
    self,
    target_pose: torch.Tensor | np.ndarray | list | tuple | None,
    start_pose: torch.Tensor | np.ndarray | list | tuple | None,
    target_theta: float | None = None,
    *,
    target_obs_id: int | None = None,
    _hop: int = 0,
    explore_goal: bool = False,
):
    if target_pose is None:
        nav_res = NavAttemptResult(
            success=False,
            finished=False,
            dist_m=0.0,
            method="none",
            note="no_target",
            target_obs_id=target_obs_id,
        )
        self._last_nav_attempt = nav_res
        return NavOutcome.NO_TARGET

    res = None
    original_target_pose = target_pose
    tp = target_pose.detach().cpu().numpy() if hasattr(target_pose, "detach") else target_pose
    tp_arr = np.asarray(tp, dtype=np.float64).reshape(-1)
    goal_xy = np.array([float(tp_arr[0]), float(tp_arr[1])], dtype=np.float64)

    if habitat_perfect_nav_enabled(self.parameters) and is_habitat_robot_client(self.robot):
        nav_res = habitat_navmesh_navigate(
            self.robot,
            goal_xy,
            target_theta=target_theta,
        )
        nav_res.target_obs_id = target_obs_id
        self._last_nav_attempt = nav_res
        if nav_res.finished or nav_res.success:
            logger.info(f"EQA habitat navmesh: {nav_res.note} dist={nav_res.dist_m:.2f}m")
        else:
            logger.info(f"EQA habitat navmesh failed: {nav_res.note} (dist={nav_res.dist_m:.2f}m)")
        self._last_nav_plan = {
            "mode": "navigation",
            "localize_source": "eqa_target",
            "goal_xyt": [float(goal_xy[0]), float(goal_xy[1]), float(target_theta or 0.0)],
            "method": "habitat_navmesh",
            "note": nav_res.note,
        }
        self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
        # Stuck / noop / no-progress: same blocked-goal memory as voxel timeout abort
        # so uncover explore / multi-goal A* skip this frontier.
        stuck = (
            str(nav_res.note or "").startswith("already_at_goal")
            or (not nav_res.finished and float(nav_res.dist_m) < 0.08)
            or (not nav_res.success and float(nav_res.dist_m) < 0.12)
        )
        if stuck:
            self._mark_nav_goal_blocked(reason=f"habitat_navmesh_{nav_res.note or 'stuck'}")
        if nav_res.finished:
            return NavOutcome.REACHED
        if nav_res.success or float(getattr(nav_res, "dist_m", 0.0) or 0.0) >= 0.12:
            return NavOutcome.PROGRESS
        return NavOutcome.STUCK

    target_pose = self.space.sample_navigation(start_pose, self.planner, original_target_pose)

    # A* planning
    if target_pose is not None:
        res = self.planner.plan(start_pose, target_pose)

    # Parse A* results into traj
    if res is not None and res.success:
        waypoints = [pt.state for pt in res.trajectory]
    elif res is not None:
        waypoints = None
        logger.warning(f"navigate_to_target_pose planner failure: {res.reason}")
    else:
        waypoints = None

    if waypoints is not None:
        self.rerun_visualizer.log_custom_pointcloud(
            "world/target_pose",
            [original_target_pose[0], original_target_pose[1], 1.5],
            torch.Tensor([1, 0, 0]),
            0.1,
        )

    finished = False
    n_planned = 0
    truncated = False
    full_traj_for_viz = None
    if waypoints is not None:
        n_planned = len(waypoints)
        truncated = len(waypoints) > DYNAMEM_NAV_CHUNK_WPS
        full_traj_for_viz = self.planner.clean_path_for_xy(
            list(waypoints), start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0
        )
        if truncated:
            waypoints = waypoints[:DYNAMEM_NAV_CHUNK_WPS]
        traj = self.planner.clean_path_for_xy(waypoints, start_yaw=float(start_pose[2]) if len(start_pose) > 2 else 0.0)
        finished = not truncated
        if finished and target_theta is not None:
            traj[-1][2] = target_theta
        traj, reject_reason, min_clr = self._filter_unsafe_nav_traj(
            traj,
            start_xyt=start_pose,
            explore_goal=explore_goal,
        )
        if reject_reason is not None or not traj:
            logger.warning(f"navigate_to_target_pose rejected after safety filter: {reject_reason}")
            self._last_nav_plan = {
                "mode": "navigation",
                "localize_source": "eqa_target",
                "goal_xyt": list(np.asarray(target_pose, dtype=np.float64).reshape(-1)[:3])
                if target_pose is not None
                else [float(goal_xy[0]), float(goal_xy[1]), 0.0],
                "object_xyz": list(np.asarray(original_target_pose, dtype=np.float64).reshape(-1)[:3]),
                "n_planned": n_planned,
                "chunked": truncated,
                "min_clearance_m": min_clr,
                "outcome": reject_reason or "rejected_low_clearance",
            }
            reason = reject_reason or "rejected_low_clearance"
            self._mark_nav_goal_blocked(reason=reason)
            nav_res = NavAttemptResult(
                success=False,
                finished=False,
                dist_m=0.0,
                method="voxel_astar",
                note=reason,
                target_obs_id=target_obs_id,
            )
            self._last_nav_attempt = nav_res
            self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
            return NavOutcome.SAFETY_REJECTED
        logger.info(
            "navigate_to_target_pose: %d exec / %d planned waypoints (finished=%s)",
            len(traj),
            n_planned,
            finished,
        )
    else:
        traj = None

    before_xy = np.asarray(start_pose, dtype=np.float64).reshape(-1)[:2].copy()
    # draw traj on rerun and execute it
    if traj is not None:
        log_plan = getattr(self.rerun_visualizer, "log_nav_plan", None)
        if callable(log_plan):
            self._last_nav_plan = log_plan(
                traj,
                full_traj=full_traj_for_viz,
                start_xyt=start_pose,
                goal_xyt=target_pose,
                object_xyz=original_target_pose,
                mode="exploration" if explore_goal else "navigation",
                localize_source="eqa_target",
                n_planned=n_planned or None,
                chunked=truncated,
            )
            self._record_nav_plan_fields(traj=list(traj))
        else:
            origins = []
            vectors = []
            for idx in range(len(traj) - 1):
                origins.append([traj[idx][0], traj[idx][1], 1.5])
                vectors.append([traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0])
            self.rerun_visualizer.log_arrow3D("world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1)
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 1.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

        from emet.controller.nav_confirm import confirm_navigation_plan

        if not confirm_navigation_plan(
            self,
            traj,
            meta=getattr(self, "_last_nav_plan", None) or {},
            object_xyz=original_target_pose,
        ):
            self._record_nav_plan_fields(outcome="user_cancelled", confirmed=False)
            nav_res = NavAttemptResult(
                success=False,
                finished=False,
                dist_m=0.0,
                method="voxel_astar",
                note="user_rejected_plan",
                target_obs_id=target_obs_id,
            )
            self._last_nav_attempt = nav_res
            self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
            return NavOutcome.USER_CANCELLED

        nav_timeout = self._find_phase_nav_timeout()
        exec_ok = self.robot.execute_trajectory(
            traj,
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
            nav_res = NavAttemptResult(
                success=False,
                finished=False,
                dist_m=0.0,
                method="voxel_astar",
                note="aborted_waypoint_timeout",
                target_obs_id=target_obs_id,
            )
            self._last_nav_attempt = nav_res
            self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
            return NavOutcome.ABORTED_TIMEOUT
        after_xy = self._current_planning_xyt().reshape(-1)[:2]
        dist_m = float(np.hypot(after_xy[0] - before_xy[0], after_xy[1] - before_xy[1]))
        note = "ok" if finished else f"moved_{dist_m:.2f}m"
        nav_res = NavAttemptResult(
            success=dist_m >= 0.12 or finished,
            finished=finished,
            dist_m=dist_m,
            method="voxel_astar",
            note=note,
            target_obs_id=target_obs_id,
        )
    else:
        note = res.reason if res is not None else "sample_nav_failed"
        logger.info(f"EQA voxel nav failed: {note}")
        self._last_nav_plan = {
            "mode": "navigation",
            "localize_source": "eqa_target",
            "goal_xyt": [float(goal_xy[0]), float(goal_xy[1]), 0.0],
            "object_xyz": list(np.asarray(original_target_pose, dtype=np.float64).reshape(-1)[:3]),
            "outcome": str(note),
        }
        self._mark_nav_goal_blocked(reason=str(note))
        nav_res = NavAttemptResult(
            success=False,
            finished=False,
            dist_m=0.0,
            method="voxel_astar",
            note=note,
            target_obs_id=target_obs_id,
        )

    self._last_nav_attempt = nav_res
    self._log_nav_attempt(nav_res, target_obs_id=target_obs_id, goal_xy=goal_xy)
    if finished:
        return NavOutcome.REACHED
    progressed = bool(nav_res.success) or float(getattr(nav_res, "dist_m", 0.0) or 0.0) >= 0.12
    if progressed and _hop + 1 < DYNAMEM_NAV_MAX_HOPS:
        upd = getattr(self, "update", None)
        if callable(upd):
            try:
                upd()
            except Exception as exc:
                logger.warning(f"navigate_to_target_pose hop update failed: {exc}")
        nxt = self._current_planning_xyt()
        logger.info(
            "navigate_to_target_pose: chunk hop %d/%d, replanning from (%.2f, %.2f)",
            _hop + 1,
            DYNAMEM_NAV_MAX_HOPS,
            float(nxt[0]),
            float(nxt[1]),
        )
        return self.navigate_to_target_pose(
            original_target_pose,
            nxt,
            target_theta,
            target_obs_id=target_obs_id,
            _hop=_hop + 1,
            explore_goal=explore_goal,
        )
    if progressed:
        return NavOutcome.PROGRESS
    if nav_res.note == "sample_nav_failed" or str(nav_res.note or "").startswith("no_target"):
        return NavOutcome.NO_TARGET
    if nav_res.note and str(nav_res.note).startswith("plan"):
        return NavOutcome.PLAN_FAILED
    return NavOutcome.STUCK
