# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Human confirmation before executing a motion plan (terminal y/n or Discord)."""

from __future__ import annotations

import logging
import queue
import re
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DISCORD_PREFIX = re.compile(r"^\[discord\]\s*", re.IGNORECASE)


def normalize_confirm_line(text: str | None) -> str:
    """Strip Discord/agent prefixes and surrounding whitespace."""
    if text is None:
        return ""
    s = str(text).strip()
    s = _DISCORD_PREFIX.sub("", s).strip()
    # Drop leading @mentions so "@virgil yes" still parses.
    if s.startswith("@"):
        parts = s.split(None, 1)
        s = parts[1] if len(parts) > 1 else ""
    return s.strip()


def parse_nav_confirm_reply(text: str | None) -> bool | None:
    """Parse a confirm reply.

    Returns:
        ``True`` for yes, ``False`` for no, ``None`` if unclear (ask again).
    """
    s = normalize_confirm_line(text).lower()
    if not s:
        return None
    token = re.split(r"[\s,.;:!]+", s, maxsplit=1)[0]
    if token in {"y", "yes", "yeah", "yep", "ok", "okay", "sure", "go", "execute", "confirm"}:
        return True
    if token in {"n", "no", "nope", "nah", "cancel", "abort", "stop", "skip"}:
        return False
    if s in {"do it", "looks good", "lgtm", "ship it"}:
        return True
    if s in {"don't", "do not", "never mind", "nevermind"}:
        return False
    return None


def finite_traj_xyt(
    traj: list | np.ndarray | None,
) -> list[tuple[float, float, float]]:
    """Drop NaN finish markers; return ``(x, y, theta)`` rows for map overlay."""
    if traj is None:
        return []
    out: list[tuple[float, float, float]] = []
    for raw in traj:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size < 2 or not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
            continue
        yaw = float(arr[2]) if arr.size >= 3 and np.isfinite(arr[2]) else 0.0
        out.append((float(arr[0]), float(arr[1]), yaw))
    return out


def render_nav_plan_map_rgb(
    voxel_map: Any,
    robot_xy: np.ndarray | tuple[float, float] | None,
    traj: list | np.ndarray | None,
    *,
    max_side: int = 640,
    object_xy: np.ndarray | tuple[float, float] | None = None,
) -> np.ndarray | None:
    """Top-down explored map with planned trajectory overlay (Discord / terminal preview)."""
    if voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
        return None
    from emet.visualization.map_snapshot import (
        _grid_origin_xy,
        downsample_topdown_rgb_max_side,
        explored_crop_indices,
        overlay_trajectory_on_map_rgb,
        render_topdown_map_rgb,
        world_xy_to_grid_ij,
    )

    obstacles, explored = voxel_map.get_2d_map()
    go = _grid_origin_xy(getattr(voxel_map, "grid_origin", np.zeros(2)))
    res = float(getattr(voxel_map, "grid_resolution", 0.1) or 0.1)
    path = finite_traj_xyt(traj)
    rgb_full = render_topdown_map_rgb(
        obstacles,
        explored,
        go,
        res,
        robot_xy,
        max_side=None,
    )
    if path:
        rgb_full = overlay_trajectory_on_map_rgb(
            rgb_full,
            path,
            go,
            res,
            crop_offset_ij=(0, 0),
            full_shape_hw=tuple(rgb_full.shape[:2]),
        )
    if object_xy is not None:
        oxy = np.asarray(object_xy, dtype=np.float64).reshape(-1)[:2]
        if np.isfinite(oxy).all():
            h, w = rgb_full.shape[:2]
            ri, rj = world_xy_to_grid_ij(oxy, go, res, (h, w))
            r = 3
            i0, i1 = max(0, ri - r), min(h, ri + r + 1)
            j0, j1 = max(0, rj - r), min(w, rj + r + 1)
            rgb_full[i0:i1, j0:j1] = np.maximum(rgb_full[i0:i1, j0:j1], np.uint8([255, 80, 80]))
            if 0 <= ri < h and 0 <= rj < w:
                rgb_full[ri, rj] = (255, 40, 40)
    bbox = explored_crop_indices(
        explored,
        robot_xy,
        go,
        res,
        rgb_full.shape[:2],
        margin_cells=16,
        trajectory_xyt=path or None,
    )
    if bbox is None:
        cropped = np.ascontiguousarray(rgb_full)
    else:
        i0, i1, j0, j1 = bbox
        cropped = np.ascontiguousarray(rgb_full[i0:i1, j0:j1])
    return downsample_topdown_rgb_max_side(cropped, max_side)


def wait_for_nav_confirm(
    prompt: str,
    *,
    input_queue: queue.Queue[str] | None = None,
    timeout_s: float | None = None,
) -> bool:
    """Block until the user answers y/n on stdin or the shared agent input queue.

    ``timeout_s=None`` waits forever (preferred on the real robot). On timeout, returns False.
    """
    print(prompt, flush=True)
    deadline = None if timeout_s is None else (time.monotonic() + float(timeout_s))
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            logger.warning("Navigation confirm timed out — aborting plan")
            print("Timed out waiting for y/n — plan cancelled.", flush=True)
            return False
        line: str | None = None
        if input_queue is not None:
            wait = 0.5
            if deadline is not None:
                wait = min(wait, max(0.05, deadline - time.monotonic()))
            try:
                line = input_queue.get(timeout=wait)
            except queue.Empty:
                continue
        else:
            try:
                line = input()
            except EOFError:
                return False
            except KeyboardInterrupt:
                print("\nCancelled.", flush=True)
                return False
        decision = parse_nav_confirm_reply(line)
        if decision is None:
            print("Please reply y or n (yes/no).", flush=True)
            continue
        return bool(decision)


def confirm_navigation_plan(
    controller: Any,
    traj: list | np.ndarray | None,
    *,
    meta: dict[str, Any] | None = None,
    object_xyz: np.ndarray | list[float] | None = None,
) -> bool:
    """Show plan (Rerun + optional Discord map image) and require y/n before execute.

    Returns True if the plan may run. When ``controller.confirm_navigation`` is false,
    always returns True without prompting.
    """
    if not bool(getattr(controller, "confirm_navigation", False)):
        return True
    if getattr(controller, "_nav_confirm_auto_yes", False):
        logger.info("confirm-nav: scripted/auto-yes — executing without prompt")
        return True

    meta = dict(meta or {})
    path = finite_traj_xyt(traj)
    if not path:
        logger.warning("confirm-nav: empty trajectory — treating as reject")
        return False

    robot_xy = None
    robot = getattr(controller, "robot", None)
    if robot is not None and hasattr(robot, "get_base_pose"):
        try:
            pose = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
            if pose.size >= 2:
                robot_xy = pose[:2]
        except Exception:
            robot_xy = None

    vm = None
    if hasattr(controller, "get_voxel_map"):
        vm = controller.get_voxel_map()
    elif hasattr(controller, "voxel_map"):
        vm = controller.voxel_map

    obj_xy = None
    if object_xyz is not None:
        o = np.asarray(object_xyz, dtype=np.float64).reshape(-1)
        if o.size >= 2 and np.isfinite(o[:2]).all():
            obj_xy = o[:2]

    img = render_nav_plan_map_rgb(vm, robot_xy, path, object_xy=obj_xy)

    log_plan = getattr(getattr(controller, "rerun_visualizer", None), "log_nav_plan", None)
    if callable(log_plan):
        try:
            log_plan(
                traj,
                start_xyt=robot_xy,
                object_xyz=object_xyz,
                mode=str(meta.get("mode") or "navigation"),
                localize_source=str(meta.get("localize_source") or ""),
                query=str(meta.get("query") or ""),
                n_planned=meta.get("n_planned"),
                chunked=bool(meta.get("chunked")),
            )
        except Exception as exc:
            logger.debug("confirm-nav: Rerun log_nav_plan failed: %s", exc)

    announce = meta.get("announce") or f"{len(path)} waypoints"
    path_m = float(meta.get("full_path_m") or meta.get("path_m") or 0.0)
    summary = (
        f"Motion plan ready ({announce}"
        + (f", ~{path_m:.1f}m" if path_m > 0 else "")
        + "). Reply **y** to execute or **n** to cancel."
    )
    print(summary, flush=True)
    if img is not None:
        viz = getattr(controller, "rerun_visualizer", None)
        if viz is not None and hasattr(viz, "log_custom_2d_image"):
            try:
                viz.log_custom_2d_image("world/nav/plan_map", img)
            except Exception:
                pass

    discord_bot = getattr(controller, "discord_bot", None)
    if discord_bot is not None and hasattr(discord_bot, "push_task_to_all_channels"):
        try:
            discord_bot.push_task_to_all_channels(message=summary, content=img)
        except Exception as exc:
            logger.warning("confirm-nav: Discord map send failed: %s", exc)
            if hasattr(controller, "announce_action"):
                controller.announce_action(summary, discord=True)

    timeout = getattr(controller, "nav_confirm_timeout_s", None)
    input_q = getattr(controller, "_nav_confirm_input_queue", None)
    prompt = "Execute this motion plan? [y/n]: "
    ok = wait_for_nav_confirm(prompt, input_queue=input_q, timeout_s=timeout)
    if ok:
        logger.info("confirm-nav: accepted — executing")
        if hasattr(controller, "announce_action"):
            controller.announce_action("Plan confirmed — navigating…", discord=True)
    else:
        logger.info("confirm-nav: rejected — skipping execute")
        if hasattr(controller, "announce_action"):
            controller.announce_action("Plan cancelled.", discord=True)
    return ok
