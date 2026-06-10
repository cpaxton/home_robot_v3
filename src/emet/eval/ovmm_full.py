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

"""Full OVMM task harness: FindObj, Pick, FindRec, Place (sim)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from emet.eval.ovmm_find_phase import (
    FindPhaseEpisode,
    FindPhaseRunConfig,
    ManipMode,
    bodies_matching_category,
    distance_to_placement_xy,
    pick_find_object_gt_body,
)


def _snapshot_placements(placements: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not placements:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for body, info in placements.items():
        row = dict(info)
        pos = row.get("pos")
        if pos is not None:
            row["pos"] = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        out[body] = row
    return out


def score_pick_success(
    placements_before: dict[str, dict[str, Any]] | None,
    placements_after: dict[str, dict[str, Any]] | None,
    *,
    object_gt_body: str | None,
    start_recep: str,
    radius_m: float,
    min_displacement_m: float = 0.05,
) -> dict[str, Any]:
    """
    Proxy pick success from sim GT: object left the start-recep neighborhood or moved enough.

    Official OVMM uses grasp/detachment checks; we use placement deltas from ``sim_object_placements``.
    """
    if not placements_before or not placements_after or not object_gt_body:
        return {"pick_success": False, "pick_displacement_m": None, "gt_object_body": object_gt_body}
    if object_gt_body not in placements_before or object_gt_body not in placements_after:
        return {"pick_success": False, "pick_displacement_m": None, "gt_object_body": object_gt_body}

    before = np.asarray(placements_before[object_gt_body]["pos"], dtype=np.float64).reshape(3)
    after = np.asarray(placements_after[object_gt_body]["pos"], dtype=np.float64).reshape(3)
    displacement = float(np.linalg.norm(after - before))

    start_bodies = bodies_matching_category(placements_before, start_recep)
    left_start = False
    if start_bodies:
        errs_before = [distance_to_placement_xy(before, placements_before[b]) for b in start_bodies]
        errs_after = [distance_to_placement_xy(after, placements_after[b]) for b in start_bodies]
        left_start = min(errs_after) > float(radius_m) and min(errs_after) > min(errs_before)

    pick_success = left_start or displacement >= float(min_displacement_m)
    return {
        "pick_success": bool(pick_success),
        "pick_displacement_m": displacement,
        "gt_object_body": object_gt_body,
    }


def score_place_success(
    placements_after: dict[str, dict[str, Any]] | None,
    *,
    object_gt_body: str | None,
    goal_recep: str,
    radius_m: float,
    placements_before: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Place success: object GT body within ``radius_m`` of any goal-recep GT body."""
    if not placements_after or not object_gt_body or object_gt_body not in placements_after:
        return {
            "place_success": False,
            "place_err_obj_to_recep_m": None,
            "gt_object_body": object_gt_body,
            "gt_recep_bodies": [],
        }
    recep_bodies = bodies_matching_category(placements_after, goal_recep)
    if not recep_bodies:
        return {
            "place_success": False,
            "place_err_obj_to_recep_m": None,
            "gt_object_body": object_gt_body,
            "gt_recep_bodies": [],
        }
    obj_pos = placements_after[object_gt_body]
    errors = [distance_to_placement_xy(obj_pos["pos"], placements_after[body]) for body in recep_bodies]
    best_err = min(errors)

    if placements_before and object_gt_body in placements_before:
        before_errors = [
            distance_to_placement_xy(placements_before[object_gt_body]["pos"], placements_before[body])
            for body in recep_bodies
            if body in placements_before
        ]
        if before_errors:
            before_best = min(before_errors)
            improved = best_err < before_best - 1e-4
            within = best_err <= float(radius_m)
            return {
                "place_success": bool(within and improved),
                "place_err_obj_to_recep_m": float(best_err),
                "gt_object_body": object_gt_body,
                "gt_recep_bodies": recep_bodies,
            }

    return {
        "place_success": best_err <= float(radius_m),
        "place_err_obj_to_recep_m": float(best_err),
        "gt_object_body": object_gt_body,
        "gt_recep_bodies": recep_bodies,
    }


def compute_ovmm_full_metrics(
    *,
    find_object_success: bool,
    find_recep_success: bool,
    pick_success: bool | None,
    place_success: bool | None,
) -> dict[str, Any]:
    """Aggregate four OVMM phases into partial and full success."""
    phases: list[float] = [float(find_object_success), float(find_recep_success)]
    if pick_success is not None:
        phases.append(float(pick_success))
    if place_success is not None:
        phases.append(float(place_success))
    ovmm_full_partial = float(sum(phases) / len(phases)) if phases else 0.0
    full_success = None
    if pick_success is not None and place_success is not None:
        full_success = bool(find_object_success and find_recep_success and pick_success and place_success)
    return {
        "ovmm_full_partial": ovmm_full_partial,
        "ovmm_full_success": full_success,
    }


def _read_placements(robot: Any) -> dict[str, dict[str, Any]] | None:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    session = robot.get_emet_session() if robot is not None else None
    return read_sim_object_placements(session)


def _sim_manip_supported(robot: Any) -> bool:
    session = robot.get_emet_session() if robot is not None else None
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return False
    caps = session.get("capabilities") or {}
    return bool(caps.get("sim_set_body_pose", False))


def _robot_set_body_pose(robot: Any, body: str, pos: np.ndarray) -> None:
    from emet.core.zmq_protocol import build_sim_set_body_pose_action

    step = int(getattr(robot, "_last_step", -1)) + 1
    if step < 1:
        step = 1
    action = build_sim_set_body_pose_action(step, body, pos.tolist())
    send_action = getattr(robot, "send_action", None)
    if callable(send_action):
        send_action(action, reliable=True)
    else:
        robot.send_message(action)
    wait_obs = getattr(robot, "wait_for_obs", None)
    if callable(wait_obs):
        wait_obs(timeout=5.0)
    time.sleep(0.25)


def _goal_place_xyz(placements: dict[str, dict[str, Any]], goal_recep: str) -> np.ndarray | None:
    recep_bodies = bodies_matching_category(placements, goal_recep)
    if not recep_bodies:
        return None
    anchor = np.asarray(placements[recep_bodies[0]]["pos"], dtype=np.float64).reshape(3).copy()
    anchor[2] += 0.02
    return anchor


def _run_sim_manip_phases(
    robot: Any,
    episode: FindPhaseEpisode,
    *,
    find_metrics: dict[str, Any],
    placements_before: dict[str, dict[str, Any]] | None,
    gt_body: str | None,
    mode: ManipMode,
) -> dict[str, Any]:
    radius_m = float(episode.success_radius_m)
    t_manip0 = time.monotonic()
    before_pick = _snapshot_placements(placements_before)
    if not gt_body or gt_body not in before_pick:
        return {
            "manip_mode": mode,
            "pick_attempted": True,
            "place_attempted": False,
            "pick_success": False,
            "place_success": False,
            "manip_error": "missing_gt_object_body",
            "manip_wall_s": float(time.monotonic() - t_manip0),
            "ovmm_full_partial": 0.0,
            "ovmm_full_success": False,
        }

    t_pick0 = time.monotonic()
    pick_pos = np.asarray(before_pick[gt_body]["pos"], dtype=np.float64).reshape(3).copy()
    pick_pos[2] += 0.12
    _robot_set_body_pose(robot, gt_body, pick_pos)
    after_pick = _read_placements(robot) or before_pick
    pick_scores = score_pick_success(
        before_pick,
        after_pick,
        object_gt_body=gt_body,
        start_recep=episode.start_recep,
        radius_m=radius_m,
    )
    pick_wall_s = time.monotonic() - t_pick0

    t_place0 = time.monotonic()
    place_pos = _goal_place_xyz(after_pick, episode.goal_recep)
    if place_pos is None:
        place_scores = {
            "place_success": False,
            "place_err_obj_to_recep_m": None,
            "gt_object_body": gt_body,
            "gt_recep_bodies": [],
        }
    else:
        _robot_set_body_pose(robot, gt_body, place_pos)
        after_place = _read_placements(robot) or after_pick
        place_scores = score_place_success(
            after_place,
            object_gt_body=gt_body,
            goal_recep=episode.goal_recep,
            radius_m=radius_m,
            placements_before=before_pick,
        )
    place_wall_s = time.monotonic() - t_place0
    manip_wall_s = time.monotonic() - t_manip0

    full = compute_ovmm_full_metrics(
        find_object_success=bool(find_metrics.get("find_object_success")),
        find_recep_success=bool(find_metrics.get("find_recep_success")),
        pick_success=bool(pick_scores["pick_success"]),
        place_success=bool(place_scores["place_success"]),
    )
    return {
        "manip_mode": mode,
        "pick_attempted": True,
        "place_attempted": place_pos is not None,
        "pick_controller_ok": None,
        "place_controller_ok": None,
        "pick_wall_s": float(pick_wall_s),
        "place_wall_s": float(place_wall_s),
        "manip_wall_s": float(manip_wall_s),
        **pick_scores,
        **place_scores,
        **full,
    }


def run_ovmm_manip_phases(
    agent: Any,
    robot: Any,
    episode: FindPhaseEpisode,
    run_cfg: FindPhaseRunConfig,
    *,
    find_metrics: dict[str, Any],
    placements_before: dict[str, dict[str, Any]] | None,
    object_query: str,
) -> dict[str, Any]:
    """Run Pick + Place after find-phase queries; score from sim GT."""
    mode: ManipMode = run_cfg.manip_mode
    if mode == "skip":
        return {
            "manip_mode": mode,
            "pick_attempted": False,
            "place_attempted": False,
            "pick_success": None,
            "place_success": None,
            "pick_wall_s": 0.0,
            "place_wall_s": 0.0,
            "manip_wall_s": 0.0,
        }

    gt_body = find_metrics.get("gt_object_body") or pick_find_object_gt_body(
        placements_before or {},
        object_query,
        episode.start_recep,
        object_gt_body=episode.object_gt_body,
    )
    radius_m = float(episode.success_radius_m)
    t_manip0 = time.monotonic()

    if mode == "oracle":
        pick_success = bool(find_metrics.get("find_object_success"))
        place_success = bool(find_metrics.get("find_recep_success"))
        manip_wall_s = time.monotonic() - t_manip0
        full = compute_ovmm_full_metrics(
            find_object_success=bool(find_metrics.get("find_object_success")),
            find_recep_success=bool(find_metrics.get("find_recep_success")),
            pick_success=pick_success,
            place_success=place_success,
        )
        return {
            "manip_mode": mode,
            "pick_attempted": False,
            "place_attempted": False,
            "pick_controller_ok": None,
            "place_controller_ok": None,
            "pick_success": pick_success,
            "place_success": place_success,
            "pick_wall_s": 0.0,
            "place_wall_s": 0.0,
            "manip_wall_s": float(manip_wall_s),
            **full,
        }

    use_sim = mode == "sim" or (mode == "attempt" and _sim_manip_supported(robot))
    if use_sim:
        effective = "sim" if mode == "sim" else "attempt_sim"
        return _run_sim_manip_phases(
            robot,
            episode,
            find_metrics=find_metrics,
            placements_before=placements_before,
            gt_body=gt_body,
            mode=effective,  # type: ignore[arg-type]
        )

    before_pick = _snapshot_placements(placements_before)
    pick_controller_ok = False
    t_pick0 = time.monotonic()
    manipulate = getattr(agent, "manipulate", None)
    if callable(manipulate):
        try:
            pick_controller_ok = bool(manipulate(object_query, skip_confirmation=True))
        except Exception:
            pick_controller_ok = False
    time.sleep(2.0)
    after_pick = _read_placements(robot) or before_pick
    pick_scores = score_pick_success(
        before_pick,
        after_pick,
        object_gt_body=gt_body,
        start_recep=episode.start_recep,
        radius_m=radius_m,
    )
    pick_wall_s = time.monotonic() - t_pick0

    place_controller_ok = False
    t_place0 = time.monotonic()
    place_fn = getattr(agent, "place", None)
    if callable(place_fn):
        try:
            place_controller_ok = bool(place_fn(episode.goal_recep))
        except Exception:
            place_controller_ok = False
    time.sleep(2.0)
    after_place = _read_placements(robot) or after_pick
    place_scores = score_place_success(
        after_place,
        object_gt_body=gt_body,
        goal_recep=episode.goal_recep,
        radius_m=radius_m,
        placements_before=after_pick,
    )
    place_wall_s = time.monotonic() - t_place0
    manip_wall_s = time.monotonic() - t_manip0

    full = compute_ovmm_full_metrics(
        find_object_success=bool(find_metrics.get("find_object_success")),
        find_recep_success=bool(find_metrics.get("find_recep_success")),
        pick_success=bool(pick_scores["pick_success"]),
        place_success=bool(place_scores["place_success"]),
    )
    return {
        "manip_mode": mode,
        "pick_attempted": True,
        "place_attempted": True,
        "pick_controller_ok": bool(pick_controller_ok),
        "place_controller_ok": bool(place_controller_ok),
        "pick_wall_s": float(pick_wall_s),
        "place_wall_s": float(place_wall_s),
        "manip_wall_s": float(manip_wall_s),
        **pick_scores,
        **place_scores,
        **full,
    }


def augment_find_metrics_with_manip(
    agent: Any,
    robot: Any,
    episode: FindPhaseEpisode,
    run_cfg: FindPhaseRunConfig,
    metrics: dict[str, Any],
    *,
    placements: dict[str, dict[str, Any]] | None,
    object_query: str | None = None,
) -> dict[str, Any]:
    """Append manip-phase fields to a find-phase metrics dict."""
    if run_cfg.manip_mode == "skip":
        return metrics
    query = object_query or str(metrics.get("object_query") or episode.object)
    manip = run_ovmm_manip_phases(
        agent,
        robot,
        episode,
        run_cfg,
        find_metrics=metrics,
        placements_before=placements,
        object_query=query,
    )
    return {**metrics, **manip}
