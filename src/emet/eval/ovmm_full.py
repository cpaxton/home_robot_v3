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
from typing import Any, Literal

import numpy as np

from emet.eval.ovmm_find_phase import (
    FindPhaseEpisode,
    FindPhaseRunConfig,
    bodies_matching_category,
    distance_to_placement_xy,
    pick_find_object_gt_body,
)

ManipMode = Literal["skip", "oracle", "attempt"]


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
    mode: ManipMode = getattr(run_cfg, "manip_mode", "skip")
    if mode == "skip":
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
