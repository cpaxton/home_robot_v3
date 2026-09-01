# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Head sweep, look-around, rotate-in-place, and relative base motion."""

from __future__ import annotations

import os
import time
from uuid import uuid4

import numpy as np

from emet.controller.dynamem.constants import (
    DYNAMEM_HEAD_SETTLE_S,
    DYNAMEM_HEAD_SWEEP_FRAME_SETTLE_S,
    DYNAMEM_HEAD_SWEEP_MAX_WAIT_S,
    DYNAMEM_HEAD_SWEEP_MIN_MOVE_S,
    DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD,
    DYNAMEM_HEAD_SWEEP_POS_DELTA_TOL,
    DYNAMEM_HEAD_SWEEP_SPEED_TOL,
    DYNAMEM_HEAD_SWEEP_STOPPED_HOLD_S,
    default_table_mapping_relative_yaws,
)
from emet.controller.zmq_client import StretchZmqClient
from emet.motion import constants as motion_constants
from emet.utils.logger import Logger
from emet.visualization.null_visualizer import visualizer_is_enabled

logger = Logger(__name__)


def _env_flag_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def look_around_should_sweep(robot: object, parameters: object | None = None) -> bool:
    """Whether ``look_around`` should pan the head.

    Env wins: ``EMET_FORCE_HEAD_SWEEP=1`` / ``EMET_SKIP_HEAD_SWEEP=1``. Else the
    mapping key ``look_around_head_sweep`` from the unified robot overlay
    (``robots.<id>`` in ``configs/emet/default.yaml``). Default YAML sets this
    false for Stretch as well as rby1 (hardware 4-pan is opt-in). If the key is
    missing, Stretch pans (narrow Realsense) and other robots do not.
    """
    if _env_flag_on("EMET_FORCE_HEAD_SWEEP"):
        return True
    if _env_flag_on("EMET_SKIP_HEAD_SWEEP"):
        return False
    raw = None
    if parameters is not None:
        getter = getattr(parameters, "get", None)
        if callable(getter):
            raw = getter("look_around_head_sweep")
        elif isinstance(parameters, dict):
            raw = parameters.get("look_around_head_sweep")
    if raw is not None:
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    return isinstance(robot, StretchZmqClient)


def _head_to_sweep(self, pan: float, tilt: float) -> None:
    """Move head for a look-around pan; return once close enough or briefly settled.

    Real Stretch head Dynamixels are fast; soft-wait is only to avoid blocking on joint
    tolerance. Sim MJCF head gains were raised (was kp=10 crawl) so pans should be snappy.
    """
    head_to = getattr(self.robot, "head_to", None)
    if not callable(head_to):
        return
    # Non-blocking; reliable=False avoids extra resends while we soft-wait.
    head_to(float(pan), float(tilt), blocking=False, reliable=False)
    get_js = getattr(self.robot, "get_joint_state", None)
    if not callable(get_js):
        time.sleep(DYNAMEM_HEAD_SWEEP_MAX_WAIT_S * 0.5)
        return
    try:
        from emet.motion.kinematics import HelloStretchIdx
    except Exception:
        time.sleep(DYNAMEM_HEAD_SWEEP_MAX_WAIT_S * 0.5)
        return

    t0 = time.time()
    stopped_since: float | None = None
    last_pan: float | None = None
    last_tilt: float | None = None
    while time.time() - t0 < DYNAMEM_HEAD_SWEEP_MAX_WAIT_S:
        try:
            joints, vels, _ = get_js()
        except Exception:
            joints, vels = None, None
        now = time.time()
        elapsed = now - t0
        if joints is None or len(joints) <= HelloStretchIdx.HEAD_TILT:
            time.sleep(0.04)
            continue

        cur_pan = float(joints[HelloStretchIdx.HEAD_PAN])
        cur_tilt = float(joints[HelloStretchIdx.HEAD_TILT])
        pan_err = abs(cur_pan - float(pan))
        tilt_err = abs(cur_tilt - float(tilt))
        near_goal = pan_err < DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD and tilt_err < DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD
        # Good enough for a sweep frame — do not wait out residual crawl.
        if near_goal and elapsed >= DYNAMEM_HEAD_SWEEP_MIN_MOVE_S * 0.5:
            break

        speed = 0.0
        if vels is not None and len(vels) > HelloStretchIdx.HEAD_TILT:
            speed = abs(float(vels[HelloStretchIdx.HEAD_PAN])) + abs(float(vels[HelloStretchIdx.HEAD_TILT]))
        pos_delta = 0.0
        if last_pan is not None and last_tilt is not None:
            pos_delta = abs(cur_pan - last_pan) + abs(cur_tilt - last_tilt)
        last_pan, last_tilt = cur_pan, cur_tilt

        # Loose: slow creep counts as stopped so we do not burn max wait every pan.
        moving = speed > DYNAMEM_HEAD_SWEEP_SPEED_TOL or pos_delta > DYNAMEM_HEAD_SWEEP_POS_DELTA_TOL
        if not moving:
            if stopped_since is None:
                stopped_since = now
            if (now - stopped_since) >= DYNAMEM_HEAD_SWEEP_STOPPED_HOLD_S and (
                elapsed >= DYNAMEM_HEAD_SWEEP_MIN_MOVE_S
            ):
                break
        else:
            stopped_since = None
        time.sleep(0.04)


def look_around(self):
    """Look around for mapping / agentic capture.

    Policy: :func:`look_around_should_sweep` (robot overlay
    ``mapping.look_around_head_sweep``, then env). YAML default is off for
    Stretch as well as rby1 (single capture at look_front) — hardware 4-pan
    is opt-in. Paper coverage: ``EMET_FORCE_HEAD_SWEEP=1`` or
    ``--set mapping.look_around_head_sweep=true``.
    """
    skip_sweep = not look_around_should_sweep(self.robot, getattr(self, "parameters", None))
    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
        import traceback

        tb = " | ".join(f"{f.name}:{f.lineno}" for f in traceback.extract_stack(limit=6)[:-1])
        print(
            f"[sweep] skip={skip_sweep} fast_lookaround={getattr(self, '_fast_explore_lookaround', False)} caller={tb}",
            flush=True,
        )
    if skip_sweep:
        self.announce_action("Look around: single capture (no head sweep)")
        self.update()
        return

    self.announce_action("Look around: sweeping head")
    tilt = float(motion_constants.look_front[1])
    # Four pans for Realsense FOV coverage (left → right-ish). Soft-wait exits on settle.
    # Explore-loop / smoke: two extremes ~halves wall time (~100s → ~50s per excursion).
    # In fast-sim (teleport) eval mode the 4-pan sweep dominates wall time, so always
    # halve to two extremes — the teleport base already turns to face each frontier.
    fast = getattr(self, "_fast_explore_lookaround", False) or os.environ.get("EMET_SIM_NAV_TELEPORT") == "1"
    if fast:
        pans = [0.6, -1.8]
    else:
        pans = [0.6, -0.2, -1.0, -1.8]
    n = len(pans)
    t_sweep = time.time()
    for i, pan in enumerate(pans):
        self.announce_motion_progress(f"Look around: head pan {i + 1}/{n} (pan={pan:+.1f} rad, tilt={tilt:+.2f})")
        self._head_to_sweep(pan, tilt)
        time.sleep(DYNAMEM_HEAD_SWEEP_FRAME_SETTLE_S)
        self.update()
    self.announce_motion_progress(f"Look around: head sweep done ({time.time() - t_sweep:.1f}s)")
    # Return to look_front without a long blocking wait.
    self._head_to_sweep(float(motion_constants.look_front[0]), tilt)
    time.sleep(DYNAMEM_HEAD_SETTLE_S)


def _find_phase_nav_timeout(self, default: float = 10.0) -> float:
    raw = self.parameters.get("find_phase_nav_step_timeout_s")
    if raw is None:
        return default
    return float(raw)


def maybe_save_rerun_recording(self) -> None:
    """Write ``logs/…/data_N.rrd`` without resetting a live ``RerunVisualizer`` stream.

    ``rr.init`` during live ``rr.serve`` starts a new recording and empties the
    websocket view. Only init when there is no live visualizer (offline dump).
    """
    if not self.save_rerun:
        return
    # Deferred: rerun-sdk native extensions.
    import rerun as rr

    os.makedirs(self.log, exist_ok=True)
    dest = os.path.join(self.log, f"data_{self.rerun_iter}.rrd")
    live = visualizer_is_enabled(self.rerun_visualizer)
    if live:
        logger.info(f"save_rerun: writing {dest} (keeping live Rerun recording)")
        rr.save(dest)
        return
    spawn = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET"))
    rr.init("Stretch_robot", recording_id=uuid4(), spawn=spawn)
    rr.save(dest)


def rotate_in_place(self, *, n_steps: int | None = None):
    self.announce_action("Looking around: rotating in place")
    nav_timeout = self._find_phase_nav_timeout()
    self.maybe_save_rerun_recording()
    self.robot.move_to_nav_posture()
    from emet.eval.ovmm_find_phase import _prepare_default_table_rby1_mapping_view

    table_view = _prepare_default_table_rby1_mapping_view(self)
    if not table_view:
        self.announce_motion_progress("Looking around: nav posture + look_front")
        self.robot.look_front(blocking=True, timeout=nav_timeout)
    else:
        self.announce_motion_progress("Looking around: default-table mapping view + look_front")
    time.sleep(DYNAMEM_HEAD_SETTLE_S)
    wait_obs = getattr(self.robot, "wait_for_obs", None)
    if callable(wait_obs):
        wait_obs(timeout=nav_timeout)
    if n_steps is None:
        env_n = os.environ.get("EMET_ROTATE_SCAN_STEPS", "").strip()
        n_steps = int(env_n) if env_n.isdigit() and int(env_n) > 0 else 8
    else:
        n_steps = int(n_steps)
    n_steps = max(1, min(n_steps, 16))

    def _capture() -> None:
        # Mapping must store every scan pose, including when realtime threads exist.
        self.update()

    # Map the prepared heading first. The old loop yawed +45° before any
    # update(), so default-table rby1 never stored the table-facing frame.
    self.announce_motion_progress(f"Looking around: scan step 1/{n_steps} (current heading)")
    _capture()
    extra = n_steps - 1
    if table_view:
        yaws = default_table_mapping_relative_yaws(extra)
        logger.info(
            f"rotate_in_place: table scan {n_steps} views, extra yaw deg={[round(float(np.rad2deg(y))) for y in yaws]}"
        )
    else:
        yaws = [np.pi / 4.0] * extra
        logger.info(f"rotate_in_place: {n_steps} views, {extra}× relative +45° yaw (no XY translation)")
    for step_i, yaw in enumerate(yaws):
        self.announce_motion_progress(f"Looking around: scan step {step_i + 2}/{n_steps}")
        self.robot.move_base_to(
            [0.0, 0.0, float(yaw)],
            relative=True,
            blocking=True,
            timeout=nav_timeout,
        )
        _capture()
        if step_i in (2, 6):
            self.announce_action(f"Looking around: scan step {step_i + 2}/{n_steps}")
    self.announce_motion_progress("Looking around: rotate-in-place done")
    self.rerun_iter += 1
    self._maybe_emit_navgrid_ascii(context="rotate_in_place")


def rotate_base_degrees(self, degrees: float) -> float:
    """Relative in-place yaw (degrees). Positive = left/CCW. Returns commanded degrees."""
    deg = float(np.clip(float(degrees), -360.0, 360.0))
    if abs(deg) < 1e-3:
        return 0.0
    self.announce_action(f"Rotating {deg:+.0f}°")
    # Scale wait with angle (180° Spin ~5s); floor above find-phase default so large yaws finish.
    nav_timeout = max(float(self._find_phase_nav_timeout()), abs(deg) / 45.0 * 5.0 + 8.0)
    if hasattr(self.robot, "move_to_nav_posture"):
        self.robot.move_to_nav_posture()
    self.robot.move_base_to(
        [0.0, 0.0, float(np.deg2rad(deg))],
        relative=True,
        blocking=True,
        timeout=nav_timeout,
    )
    if not getattr(self, "_realtime_updates", False):
        try:
            self.update()
        except Exception:
            pass
    return deg


def _seed_local_radius_explored(self, vm) -> bool:
    """Stamp ``local_radius`` explored disk at the current base (start / fallback seed).

    Used when the map has no explored cells yet, and after OVMM mapping so A* can
    leave spawn. Does not fill camera-coverage holes — that is observed voxels only.
    Returns True if the map reports any explored cells afterward.
    """
    if vm is None or not hasattr(vm, "_update_visited"):
        return False
    try:
        # Use world-frame base (planning frame) so the visited disk lands on the
        # robot's actual map cell. Raw get_base_pose is episode-relative; for sims
        # with a non-zero nav origin (robocasa) that maps the disk to grid center
        # (0,0), leaving the real spawn cell unexplored → nav 'non navigable'.
        xyt = self._planning_base_xyt(self.robot.get_base_pose())
    except Exception:
        return False
    if xyt.size < 2:
        return False
    try:
        import torch

        pose = torch.as_tensor(xyt[:3], dtype=torch.float32)
        device = getattr(vm, "map_2d_device", None)
        if device is not None:
            pose = pose.to(device)
        vm._update_visited(pose)
        # Invalidate 2D cache so the next get_2d_map includes _visited.
        if hasattr(vm, "_map2d"):
            vm._map2d = None
    except Exception:
        return False
    try:
        obstacles, explored = vm.get_2d_map()
    except Exception:
        return False
    if explored is None:
        return False
    exp_np = explored.cpu().numpy() if hasattr(explored, "cpu") else np.asarray(explored)
    return int(np.count_nonzero(exp_np)) > 0


def clip_forward_distance_m(
    self,
    meters: float,
    *,
    step_m: float = 0.05,
    clearance_m: float = 0.05,
    require_map: bool = True,
) -> float:
    """Shorten a forward request using the 2D obstacle map.

    Always consults the voxel map before driving — including small nudges (0.1 m).
    When *require_map* is True (default), paths must stay on explored cells. If the map
    has no explored cells yet, stamps the configured ``local_radius`` disk at the base
    (same Stretch-style turn-around seed) and retries — never drives into unknown space
    beyond that disk. Stops *clearance_m* before the first occupied cell.
    """
    requested = float(np.clip(float(meters), 0.0, 1.5))
    if requested < 1e-3:
        return 0.0
    vm = self.get_voxel_map() if hasattr(self, "get_voxel_map") else None
    if vm is None:
        return 0.0 if require_map else requested

    def _load_maps():
        try:
            obstacles, explored = vm.get_2d_map()
        except Exception:
            return None, None
        if obstacles is None:
            return None, None
        obs_np = obstacles.cpu().numpy() if hasattr(obstacles, "cpu") else np.asarray(obstacles)
        exp_np = None
        if explored is not None:
            exp_np = explored.cpu().numpy() if hasattr(explored, "cpu") else np.asarray(explored)
        return obs_np, exp_np

    obs_np, exp_np = _load_maps()
    empty_cloud = bool(hasattr(vm, "is_empty") and vm.is_empty())
    n_obs = int(np.count_nonzero(obs_np)) if obs_np is not None else 0
    n_exp = int(np.count_nonzero(exp_np)) if exp_np is not None else 0
    if require_map and (obs_np is None or (empty_cloud and n_exp == 0) or (n_obs == 0 and n_exp == 0)):
        if self._seed_local_radius_explored(vm):
            obs_np, exp_np = _load_maps()
            n_obs = int(np.count_nonzero(obs_np)) if obs_np is not None else 0
            n_exp = int(np.count_nonzero(exp_np)) if exp_np is not None else 0
        if obs_np is None or (n_obs == 0 and n_exp == 0):
            return 0.0
    elif obs_np is None:
        return 0.0 if require_map else requested

    try:
        xyt = self._planning_base_xyt(self.robot.get_base_pose())
    except Exception:
        return 0.0 if require_map else requested
    if xyt.size < 3:
        return 0.0 if require_map else requested
    x0, y0, th = float(xyt[0]), float(xyt[1]), float(xyt[2])
    c, s = float(np.cos(th)), float(np.sin(th))
    traveled = 0.0
    step = max(0.02, float(step_m))
    clear = max(0.0, float(clearance_m))
    while traveled + step <= requested + 1e-9:
        probe = traveled + step
        xy = np.array([x0 + probe * c, y0 + probe * s], dtype=np.float64)
        try:
            grid = vm.xy_to_grid_coords(xy)
        except Exception:
            break
        if grid is None:
            break
        if hasattr(grid, "detach"):
            grid = grid.detach().cpu().numpy()
        gi, gj = int(grid[0]), int(grid[1])
        if gi < 0 or gj < 0 or gi >= obs_np.shape[0] or gj >= obs_np.shape[1]:
            break
        if bool(obs_np[gi, gj]):
            return max(0.0, traveled - clear)
        if require_map and exp_np is not None and not bool(exp_np[gi, gj]):
            # Do not leave the explored (incl. local_radius) disk into unknown space.
            return traveled
        traveled = probe
    return requested


def move_forward_meters(self, meters: float) -> float:
    """Drive forward along current heading; clips for obstacles. Returns distance commanded."""
    requested = float(np.clip(float(meters), 0.0, 1.5))
    dist = self.clip_forward_distance_m(requested)
    if dist < 0.02:
        self.announce_action("Cannot move forward — need explored free space (scan?) or obstacle too close")
        return 0.0
    if dist + 1e-3 < requested:
        self.announce_action(f"Moving forward {dist:.2f} m (map-clipped from {requested:.2f} m)")
    else:
        self.announce_action(f"Moving forward {dist:.2f} m (map clear)")
    nav_timeout = self._find_phase_nav_timeout()
    if hasattr(self.robot, "move_to_nav_posture"):
        self.robot.move_to_nav_posture()
    # Relative body-frame: +x forward.
    self.robot.move_base_to(
        [float(dist), 0.0, 0.0],
        relative=True,
        blocking=True,
        timeout=nav_timeout,
    )
    if not getattr(self, "_realtime_updates", False):
        try:
            self.update()
        except Exception:
            pass
    return dist
