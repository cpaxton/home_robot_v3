#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""TAMP clutter-clearance benchmark runner (MolmoSpaces iTHOR).

Each episode starts the robot surrounded by small floor objects that must be moved
(relocated to a drop receptacle / bin) as part of a plan:

  cleanup  — "clean up the room": relocate all scattered objects to the bin.
  nav_goal — "get to <landmark>": clear a path of objects, then navigate to a scene
             landmark (static furniture). ``goal_landmark: auto`` samples one live.

Per episode: launch a MolmoSpaces MuJoCo server, scatter N pickable objects in a ring
around the robot (``sim_set_body_pose``), record a GT validity probe (is the route
blocked by the clutter?), then run the multi-object TAMP chain
(:func:`emet.controller.task.tamp.clutter_chain.plan_clear_clutter`) to relocate each
object (``latch`` = kinematic IK + sim attach; ``sim`` = teleport oracle), and for
nav_goal navigate to the landmark.

Prefer running this as an ``emet jobs`` job (never block an agent turn on sim)::

  NEED_MIB=8000 uv run emet jobs run --name tamp-clutter --need-mib 8000 -- \\
    uv run python scripts/eval_tamp_clutter.py

Results: JSON + aggregate CSV under --output-dir (default ~/runs/emet/tamp_clutter).

Smoke (fast rby1 iTHOR gate, N=2, no Stretch head sweeps)::

  uv run python scripts/eval_tamp_clutter.py --smoke

Problem-set build time: ``--generate`` resolves scatter + validity probe on a live
scene and writes a deterministic episode registry (no execution)::

  uv run python scripts/eval_tamp_clutter.py --generate --output-dir ~/runs/emet/tamp_clutter/gen
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from emet.eval.tamp_clutter import BIN_FALLBACKS
from emet.utils.logger import Logger

logger = Logger(__name__)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = REPO / "configs" / "ovmm" / "clutter_episodes.yaml"
DEFAULT_OUTPUT = Path.home() / "runs" / "emet" / "tamp_clutter"
SMOKE_EPISODE_ID = "ithor_cleanup_s1_bin_n3"

# Static-furniture keywords for nav-goal landmarks (matched as substrings against the
# live placements cat, which carries instance hashes, e.g. "cabinet 9ec3f05...").
_FURNITURE_KEYWORDS = (
    "sofa",
    "couch",
    "fridge",
    "refrigerator",
    "table",
    "diningtable",
    "counter",
    "cabinet",
    "drawer",
    "shelf",
    "bed",
    "chair",
    "tv",
    "television",
    "countertop",
    "microwave",
    "oven",
    "stove",
    "sink",
    "desk",
    "toilet",
    "bathtub",
)

# Farthest navigable landmark: inside the GenericZmqClient ~12 m-from-origin guard.
_REACHABLE_LANDMARK_M = 10.0


def _matches_furniture(cat: str) -> bool:
    c = str(cat or "").lower().strip()
    if not c:
        return False
    return any(k in c for k in _FURNITURE_KEYWORDS)


# Per-robot MolmoSpaces base sim config for the GT+MCTS battery (scene_index overrides).
_BATTERY_SIM = {
    "nori": "configs/sim/molmospaces_ithor_train_nori_0.yaml",
    "innate_mars": "configs/sim/molmospaces_ithor_train_innate_mars_0.yaml",
    "rby1": "configs/sim/molmospaces_ithor_train_0.yaml",
    "stretch": "configs/sim/molmospaces_ithor_train_stretch_0.yaml",
}


def _battery_episodes(robots: list[str], scenes: list[int]) -> list[Any]:
    """Build the GT+MCTS battery episode matrix (no AI models — all sim GT + MCTS planner)."""
    from emet.eval.tamp_clutter import ClutterEpisode

    eps: list[Any] = []
    for robot in robots:
        sim = _BATTERY_SIM.get(robot)
        if sim is None:
            raise SystemExit(f"--battery-robots: unknown robot {robot!r}")
        for scene in scenes:
            base = {
                "tier": "TEST",
                "sim": sim,
                "robot": robot,
                "scene_index": int(scene),
                "seed": int(scene),
                # GT+MCTS battery runs the sim (teleport) oracle: validates the TAMP
                # chain (what to move / where / order) independent of per-robot arm
                # reach. Kinematic latch is a separate per-robot experiment
                # (--manip-mode latch); e.g. Nori's model arm bottoms out at z≈0.29.
                "manip_mode": "sim",
            }
            # 1) pick-and-place: relocate one object to the bin.
            eps.append(ClutterEpisode(id=f"test_pickplace_{robot}_s{scene}", mode="cleanup", n_objects=1, **base))
            # 2) declutter: relocate all three scattered objects to the bin.
            eps.append(ClutterEpisode(id=f"test_declutter_{robot}_s{scene}", mode="cleanup", n_objects=3, **base))
            # 3) blocked nav: a closed ring (8 objects @ 0.5 m) blocks the route; must
            #    clear then reach the landmark.
            eps.append(
                ClutterEpisode(
                    id=f"test_navblocked_{robot}_s{scene}",
                    mode="nav_goal",
                    n_objects=8,
                    goal_landmark="auto",
                    scatter_radius_m=0.5,
                    success_radius_m=0.7,
                    tight_ring=True,
                    **base,
                )
            )
            # 4) unblocked nav: no clutter; navigate straight to the landmark.
            eps.append(
                ClutterEpisode(
                    id=f"test_navclear_{robot}_s{scene}",
                    mode="nav_goal",
                    n_objects=0,
                    goal_landmark="auto",
                    success_radius_m=0.7,
                    **base,
                )
            )
    return eps


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=str, default=str(DEFAULT_EPISODES))
    p.add_argument("--episode-id", action="append", dest="episode_ids", default=None)
    p.add_argument("--backend", default="ground_truth", help="unused by deterministic chain; kept for parity")
    p.add_argument("--manip-mode", choices=("latch", "sim", "attempt"), default=None)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--port-offset", type=int, default=int(os.getpid() % 400 + 240))
    p.add_argument("--port-stride", type=int, default=2)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--generate", action="store_true", help="resolve scatter+validity probe and write YAML (no exec)")
    p.add_argument("--clearance-m", type=float, default=0.22, help="nav min-clearance for the GT validity probe")
    p.add_argument(
        "--test-battery",
        action="store_true",
        help="GT+MCTS battery: pick/place, declutter, blocked & unblocked nav per (robot x scene). No AI models.",
    )
    p.add_argument("--battery-robots", type=str, default="nori", help="comma list (default: nori)")
    p.add_argument("--battery-scenes", type=str, default="0,1", help="comma list of iTHOR scene indices")
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Stream the scene + scattered objects + robot to the Rerun viewer "
        "(http://localhost:9090?url=ws://localhost:9877) so operators can inspect "
        "clutter scenes across robots.",
    )
    return p.parse_args()


def _clamp2(xy: Any) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64).reshape(-1)[:2]


def _read_placements(robot: Any) -> dict[str, dict[str, Any]]:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    return read_sim_object_placements(robot.get_emet_session()) or {}


def _world_base_xy(robot: Any) -> np.ndarray:
    from emet.utils.geometry import nav_xyt_to_world_xyt

    pose = np.asarray(robot.get_base_pose(timeout=2.0), dtype=np.float64).reshape(-1)
    sess = robot.get_emet_session() if callable(getattr(robot, "get_emet_session", None)) else None
    world = nav_xyt_to_world_xyt(pose[:3], sess)
    state = getattr(robot, "_state", None)
    if isinstance(state, dict) and state.get("base_xyz") is not None:
        xyz = np.asarray(state["base_xyz"], dtype=np.float64).reshape(-1)
        if xyz.size >= 2:
            return np.array([float(xyz[0]), float(xyz[1])], dtype=np.float64)
    return _clamp2(world)


def _placement_category_map(placements: dict[str, dict[str, Any]], metadata: dict | None) -> dict[str, dict[str, Any]]:
    """Map placement bodies to clean category + static flag from scene metadata.

    Live placements carry hash-suffixed body labels (``'bottle 8382d4... 1 0 0'``);
    the metadata gives the clean category and staticness.
    """
    objects = (metadata or {}).get("objects") if isinstance(metadata, dict) else None
    if not isinstance(objects, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _root, info in objects.items():
        if not isinstance(info, dict):
            continue
        static = bool(info.get("is_static"))
        cat = str(info.get("category") or "").strip()
        for body in (info.get("name_map") or {}).get("bodies") or {}:
            if body in placements:
                out[str(body)] = {"cat": cat, "static": static}
    return out


def _pickable_bodies(
    placements: dict[str, dict[str, Any]],
    n: int,
    *,
    cat_map: dict[str, dict[str, Any]],
    exclude_bodies: set[str] | None = None,
) -> list[str]:
    """Choose up to ``n`` small movable floor bodies to scatter (non-static per scene
    metadata, not furniture, not the drop receptacle body)."""
    exclude_bodies = exclude_bodies or set()

    def is_clutter(body: str) -> bool:
        if body in exclude_bodies:
            return False
        meta = cat_map.get(body)
        if meta is None or meta.get("static"):
            return False
        cat = str(meta.get("cat") or "").lower().strip()
        if not cat or _matches_furniture(cat):
            return False
        return True

    cands = sorted(b for b in placements if is_clutter(b))
    return cands[:n]


def _resolve_bin(placements: dict[str, dict[str, Any]], bin_query: str | None) -> str | None:
    """Resolve a drop receptacle GT body (trash-only aliases, shared with the chain)."""
    from emet.eval.ovmm_find_phase import bodies_matching_category

    queries = [bin_query] if bin_query else []
    for q in BIN_FALLBACKS:
        if q not in queries:
            queries.append(q)
    for q in queries:
        bodies = bodies_matching_category(placements, q)
        if bodies:
            return sorted(bodies)[0]
    return None


def _goal_for_landmark(
    placements: dict[str, dict[str, Any]],
    robot_xy: np.ndarray,
    landmark: str | None,
    *,
    cat_map: dict[str, dict[str, Any]],
) -> str | None:
    """Pick a static-furniture landmark body (or ``landmark`` category) and return its body."""
    from emet.eval.ovmm_find_phase import bodies_matching_category

    if landmark and landmark.lower() not in ("auto", "any", ""):
        bodies = bodies_matching_category(placements, landmark)
        if bodies:
            return bodies[0]
        return None
    # Sample a furniture body, preferring the farthest **navigable** one: a straight-line
    # reachable landmark can still sit behind a furniture barrier (no 8-connected route),
    # which makes the nav_goal unwinnable. Prefer candidates with an actual route.
    from emet.eval.tamp_clutter import (
        nav_path_open_around_disks,
        placement_obstacle_disks,
    )

    furniture = [
        b for b, meta in cat_map.items() if meta.get("static") and _matches_furniture(str(meta.get("cat") or ""))
    ]
    if not furniture:
        furniture = [b for b, meta in cat_map.items() if meta.get("static")]
    if not furniture:
        return None

    def dist(b: str) -> float:
        return float(np.linalg.norm(_clamp2(placements[b]["pos"]) - robot_xy))

    reachable = [b for b in furniture if dist(b) <= _REACHABLE_LANDMARK_M]
    if not reachable:
        # Degenerate scene: everything beyond reach — use the nearest landmark anyway.
        return min(furniture, key=dist)
    # Navigable candidates only: 8-connected route from the start pose to the approach
    # point around all other furniture disks (skip the landmark's own bodies near it).
    from emet.eval.tamp_clutter import bodies_near_xy

    def navigable(b: str) -> bool:
        approach = _approach_xy_near(b, placements, robot_xy)
        skip = bodies_near_xy(placements, approach, keepout_m=0.75)
        disks = placement_obstacle_disks(placements, skip_bodies=skip)
        path_open, _probe = nav_path_open_around_disks(robot_xy, approach, disks, clearance_m=0.22)
        return path_open

    navigable_bodies = [b for b in reachable if navigable(b)]
    if not navigable_bodies:
        # No furniture is route-reachable from spawn: fall back to nearest (documented
        # degenerate case rather than erroring the whole episode).
        return max(reachable, key=dist)
    return max(navigable_bodies, key=dist)


def _approach_xy_near(landmark_body: str, placements: dict[str, dict[str, Any]], robot_xy: np.ndarray) -> np.ndarray:
    """A reachable base goal just in front of the landmark (away from the robot)."""
    lm = _clamp2(placements[landmark_body]["pos"])
    delta = lm - robot_xy
    d = float(np.linalg.norm(delta))
    if d < 1e-6:
        return lm
    offset = delta / d * 0.6
    return lm - offset


def _wait_port(port: int, timeout: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _launch_server(
    sim: str,
    port_offset: int,
    *,
    cpu_only: bool,
    scene_index: int | None = None,
    robot: str | None = None,
) -> tuple[Any, Path, Any]:
    """Launch the MuJoCo server subprocess; returns ``(proc, stderr_log_path, stderr_fh)``.

    ``robot`` (episode robot id) overrides the sim YAML so a mismatched
    ``sim:`` path cannot merge the wrong MJCF.
    """
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
    from emet.utils.process_tree import popen_session
    from emet.utils.pythonpath import sanitize_emet_subprocess_env

    sim_cfg = load_sim_launch_config_from_path(sim)
    if scene_index is not None:
        sim_cfg = replace(sim_cfg, index=int(scene_index))
    if robot:
        sim_cfg = replace(sim_cfg, robot=str(robot))
    sim_cfg = replace(sim_cfg, port_offset=port_offset, headless=True)
    server_argv = prepare_mujoco_server_argv(sim_cfg)
    server_cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *server_argv]
    env = sanitize_emet_subprocess_env(dict(os.environ))
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    log_dir = Path(tempfile.mkdtemp(prefix="emet_tamp_clutter_sim_"))
    server_log = log_dir / "mujoco_server.stderr"
    fh = server_log.open("w", encoding="utf-8")
    try:
        proc = popen_session(server_cmd, env=env, stdout=subprocess.DEVNULL, stderr=fh)
        recv_port = 4401 + port_offset
        if not _wait_port(recv_port, 180.0, proc=proc):
            try:
                fh.flush()
            except Exception:
                pass
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"sim server did not bind port {recv_port} (sim={sim}):\n{tail}")
        time.sleep(25.0)
        return proc, server_log, fh
    except Exception:
        try:
            fh.close()
        except Exception:
            pass
        raise


def run_one(ep: Any, args: argparse.Namespace, port_offset: int) -> dict[str, Any]:
    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.controller.task.tamp.clutter_chain import plan_clear_clutter
    from emet.eval.tamp_clutter import clutter_blocks_path, scatter_ring_targets
    from emet.simulation.sim_manipulation import robot_zmq_set_body_pose
    from emet.utils.process_tree import terminate_process_tree

    server = None
    robot = None
    server_log_fh = None
    t0 = time.monotonic()
    try:
        server, server_log, server_log_fh = _launch_server(
            ep.sim,
            port_offset,
            cpu_only=args.cpu_only,
            scene_index=ep.scene_index,
            robot=ep.robot,
        )
        robot = create_robot_client_from_cli(
            ep.robot,
            "127.0.0.1",
            port_offset=port_offset,
            enable_rerun_server=bool(args.rerun),
            start_immediately=True,
            allow_missing_depth=True,
        )
        robot.set_velocity(v=30.0, w=15.0)

        placements = _read_placements(robot)
        if not placements:
            raise RuntimeError("no sim_object_placements in session")
        robot_xy = _world_base_xy(robot)

        # Resolve goal landmark (nav_goal) + approach goal.
        goal_xy = None
        landmark_body = None
        cat_map: dict[str, dict[str, Any]] = {}
        if ep.mode == "nav_goal" or ep.n_objects > 0:
            from emet.eval.scene_task_extractor import load_scene_metadata, resolve_scene_metadata_for_session

            meta_path = resolve_scene_metadata_for_session(robot.get_emet_session())
            metadata = load_scene_metadata(meta_path) if meta_path is not None else None
            cat_map = _placement_category_map(placements, metadata)
        if ep.mode == "nav_goal":
            landmark_body = _goal_for_landmark(placements, robot_xy, ep.goal_landmark, cat_map=cat_map)
            if landmark_body is None:
                return {
                    "episode_id": ep.id,
                    "tier": ep.tier,
                    "mode": ep.mode,
                    "error": "missing_landmark",
                    "init_wall_s": time.monotonic() - t0,
                }
            goal_xy = _approach_xy_near(landmark_body, placements, robot_xy)

        # Scatter N pickable objects in a ring around the robot (between robot and goal).
        # n_objects == 0 → pure-nav episode (no clutter); used by the GT+MCTS battery.
        rng = np.random.default_rng(ep.seed if ep.seed is not None else args.seed)
        bodies: list[str] = []
        targets: list[np.ndarray] = []
        bin_query = ep.bin_query or "GarbageCan"
        bin_body = _resolve_bin(placements, bin_query)
        if ep.n_objects > 0:
            bodies = _pickable_bodies(
                placements,
                ep.n_objects,
                cat_map=cat_map,
                exclude_bodies={bin_body} if bin_body else None,
            )
            if len(bodies) < ep.n_objects:
                return {
                    "episode_id": ep.id,
                    "tier": ep.tier,
                    "mode": ep.mode,
                    "n_objects": ep.n_objects,
                    "error": f"not_enough_pickable_bodies:{len(bodies)}",
                    "init_wall_s": time.monotonic() - t0,
                }
            bodies = bodies[: ep.n_objects]
            targets = scatter_ring_targets(
                robot_xy,
                goal_xy,
                ep.n_objects,
                radius_m=ep.scatter_radius_m,
                rng=rng,
                radius_jitter=0.02 if ep.tight_ring else 0.15,
                angle_jitter_rad=0.02 if ep.tight_ring else 0.35,
            )
            for body, xy in zip(bodies, targets, strict=True):
                pos = [float(xy[0]), float(xy[1]), float(ep.floor_z_m)]
                robot_zmq_set_body_pose(robot, body, pos)

        # GT validity probe: is the route to the goal blocked by the clutter?
        # Re-read placements AFTER scatter so the probe sees the moved floor positions.
        placements = _read_placements(robot) or placements
        obj_xy = []
        for b in bodies:
            p = placements[b]["pos"]
            obj_xy.append(_clamp2(p) if p is not None else np.zeros(2))
        blocked, probe = clutter_blocks_path(
            robot_xy,
            goal_xy,
            obj_xy,
            clearance_m=args.clearance_m,
        )

        objects = []
        for b in bodies:
            meta = cat_map.get(b) or {}
            obj_cat = str(meta.get("cat") or "").strip() or str(placements[b].get("cat") or b).strip()
            objects.append({"object_query": obj_cat, "object_gt_body": b})
        manip = ep.resolved_manip_mode() if args.manip_mode is None else args.manip_mode
        # Cluttered nav_goal that is not actually blocked is reported, not scored.
        skipped_invalid = bool(ep.mode == "nav_goal" and ep.n_objects > 0 and not blocked)

        if ep.n_objects == 0:
            # Pure-nav: no clutter to clear; just reach the landmark (nav_goal).
            goal_reached = False
            nav_success = False
            nav_path_open = True
            nav_probe_after = None
            if ep.mode == "nav_goal" and goal_xy is not None:
                from emet.controller.task.tamp.clutter_chain import nav_to_landmark_if_clear

                goal_reached, nav_success, nav_path_open, nav_probe_after = nav_to_landmark_if_clear(
                    robot,
                    goal_xy=goal_xy,
                    objects=(),
                    goal_radius_m=ep.success_radius_m,
                    clearance_m=float(args.clearance_m),
                )
            elif ep.mode == "cleanup":
                goal_reached = True
                nav_success = True
            metrics0 = {
                "episode_id": ep.id,
                "tier": ep.tier,
                "mode": ep.mode,
                "robot": ep.robot,
                "sim": ep.sim,
                "scene_index": ep.scene_index,
                "n_objects": 0,
                "n_cleared": 0,
                "n_relocated": 0,
                "goal_reached": bool(goal_reached),
                "nav_success": bool(nav_success),
                "task_success": bool(goal_reached) and bool(nav_path_open),
                "manip_success_rate": 0.0,
                "manip_mode": manip,
                "episode_valid": False,
                "skipped_invalid": False,
                "nav_path_open": bool(nav_path_open),
                "validity_probe": probe,
                "landmark_body": landmark_body,
                "init_wall_s": time.monotonic() - t0,
            }
            if nav_probe_after is not None:
                metrics0["nav_probe_after"] = nav_probe_after
            return metrics0

        if args.generate:
            # Problem-set build: record the resolved deterministic episode, no execution.
            return {
                "episode_id": ep.id,
                "tier": ep.tier,
                "mode": ep.mode,
                "robot": ep.robot,
                "sim": ep.sim,
                "scene_index": ep.scene_index,
                "n_objects": ep.n_objects,
                "manip_mode": manip,
                "robot_start_xy": [float(x) for x in robot_xy],
                "goal_xy": [float(x) for x in goal_xy] if goal_xy is not None else None,
                "goal_landmark": ep.goal_landmark,
                "landmark_body": landmark_body,
                "bin_query": ep.bin_query,
                "scatter_radius_m": ep.scatter_radius_m,
                "episode_valid": bool(blocked),
                "skipped_invalid": skipped_invalid,
                "validity_probe": probe,
                "clutter": [
                    {"body": b, "cat": str(placements[b].get("cat") or b), "xy": [float(x) for x in xy]}
                    for b, xy in zip(bodies, targets, strict=True)
                ],
                "init_wall_s": time.monotonic() - t0,
            }

        executor = None
        if manip in ("latch", "attempt"):
            executor = KinematicPickPlaceExecutor(robot, manip_collision="none", traj_dt=0.05)
        metrics = plan_clear_clutter(
            robot,
            objects=objects,
            mode=ep.mode,
            bin_query=bin_query,
            goal_xy=goal_xy,
            goal_radius_m=ep.success_radius_m,
            drop_radius_m=ep.success_radius_m,
            manip_mode=manip,
            executor=executor,
            seed=ep.seed,
            clearance_m=float(args.clearance_m),
        )
        metrics.update(
            {
                "episode_id": ep.id,
                "tier": ep.tier,
                "mode": ep.mode,
                "robot": ep.robot,
                "sim": ep.sim,
                "scene_index": ep.scene_index,
                "n_objects": ep.n_objects,
                "manip_mode": manip,
                "episode_valid": bool(blocked),
                "skipped_invalid": skipped_invalid,
                "validity_probe": probe,
                "init_wall_s": time.monotonic() - t0,
            }
        )
        return metrics
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if server is not None:
            terminate_process_tree(server)
        if server_log_fh is not None:
            try:
                server_log_fh.close()
            except Exception:
                pass


def _run_battery(episodes: list[Any], args: argparse.Namespace) -> int:
    """Run the GT+MCTS battery; assert per-test pass/fail and write a summary JSON.

    Tests (all sim GT placements + MCTS planner, no VLM/LLM):
      pickplace   — relocate 1 scattered object to the bin (n_relocated >= 1)
      declutter   — relocate all 3 scattered objects (n_relocated == 3)
      navblocked  — clutter ring blocks the route (episode_valid) then cleared -> goal reached
      navclear    — no clutter -> navigate straight to the landmark (goal reached)
    """
    from emet.eval.tamp_clutter import clutter_success_flags

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        # Isolate battery artifacts from regular per-episode runs.
        base = Path(os.environ.get("EMET_TAMP_CLUTTER_OUTPUT") or DEFAULT_OUTPUT).expanduser().resolve()
        output_dir = base / "battery"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    stride = max(1, int(args.port_stride))
    for i, ep in enumerate(episodes):
        port_offset = int(args.port_offset) + i * stride
        print(f"Battery {ep.id} (port_offset={port_offset}) …", file=sys.stderr)
        try:
            metrics = run_one(ep, args, port_offset)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {ep.id}: {exc}", file=sys.stderr)
            metrics = {"episode_id": ep.id, "tier": ep.tier, "mode": ep.mode, "error": str(exc)}
        metrics.update(clutter_success_flags(metrics))
        rows.append(metrics)
        (output_dir / f"{ep.id}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        error = str(metrics.get("error") or "")
        test = str(ep.id).split("_")[1]
        ok = False
        reason = error or "ok"
        if not error:
            if test == "pickplace":
                ok = int(metrics.get("n_relocated", 0)) >= 1
                reason = "no object relocated" if not ok else "ok"
            elif test == "declutter":
                ok = int(metrics.get("n_relocated", 0)) == int(ep.n_objects)
                reason = f"relocated {metrics.get('n_relocated')}/{ep.n_objects}" if not ok else "ok"
            elif test == "navblocked":
                ok = (
                    bool(metrics.get("episode_valid"))
                    and int(metrics.get("n_relocated", 0)) == int(ep.n_objects)
                    and bool(metrics.get("goal_reached"))
                )
                reason = (
                    "ok"
                    if ok
                    else (
                        f"valid={metrics.get('episode_valid')} cleared="
                        f"{metrics.get('n_relocated')}/{ep.n_objects} goal={metrics.get('goal_reached')}"
                    )
                )
            elif test == "navclear":
                ok = bool(metrics.get("goal_reached"))
                reason = "goal not reached" if not ok else "ok"
            else:
                reason = "unknown test"
        n_pass += int(ok)
        n_fail += int(not ok)
        summary.append(
            {
                "id": ep.id,
                "robot": ep.robot,
                "scene_index": ep.scene_index,
                "test": test,
                "pass": bool(ok),
                "reason": reason,
                "n_relocated": int(metrics.get("n_relocated", 0)),
                "goal_reached": bool(metrics.get("goal_reached", False)),
                "episode_valid": bool(metrics.get("episode_valid", False)),
                "manip_wall_s": float(metrics.get("manip_wall_s", 0.0)),
            }
        )

    out_json = output_dir / "battery_summary.json"
    out_json.write_text(json.dumps({"passed": n_pass, "failed": n_fail, "tests": summary}, indent=2), encoding="utf-8")
    print(f"Battery: {n_pass} passed / {n_fail} failed  ->  {out_json}", file=sys.stderr)
    for row in summary:
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"  [{mark}] {row['robot']} scene{row['scene_index']} {row['test']:<10} {row['reason']}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    args = _parse_args()
    from emet.eval.tamp_clutter import clutter_success_flags, load_clutter_episodes

    episodes = load_clutter_episodes(args.episodes)
    if args.episode_ids:
        want = set(args.episode_ids)
        episodes = [e for e in episodes if e.id in want]
    if args.smoke:
        episodes = [e for e in episodes if e.id == SMOKE_EPISODE_ID]
        if not episodes:
            raise SystemExit(f"--smoke fixes episode to {SMOKE_EPISODE_ID!r} (not in {args.episodes})")

    if args.test_battery:
        robots = [r.strip() for r in args.battery_robots.split(",") if r.strip()]
        scenes = [int(s) for s in args.battery_scenes.split(",") if s.strip()]
        episodes = _battery_episodes(robots, scenes)
        if args.dry_run:
            for ep in episodes:
                print(f"{ep.id}\t{ep.robot}\tscene={ep.scene_index}\t{ep.mode}\tn={ep.n_objects}")
            return 0
        return _run_battery(episodes, args)

    if args.dry_run:
        for ep in episodes:
            print(
                f"{ep.id}\t{ep.tier}\t{ep.mode}\t{ep.robot}\t"
                f"idx={ep.scene_index or 0}\tn={ep.n_objects}\tmanip={ep.resolved_manip_mode()}"
            )
        return 0

    output_dir = (
        Path(args.output_dir or os.environ.get("EMET_TAMP_CLUTTER_OUTPUT") or DEFAULT_OUTPUT).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    stride = max(1, int(args.port_stride))
    for i, ep in enumerate(episodes):
        port_offset = int(args.port_offset) + i * stride
        print(f"Running {ep.id} (port_offset={port_offset}) …", file=sys.stderr)
        try:
            metrics = run_one(ep, args, port_offset)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {ep.id}: {exc}", file=sys.stderr)
            metrics = {"episode_id": ep.id, "tier": ep.tier, "mode": ep.mode, "error": str(exc)}
        if args.generate:
            resolved.append(metrics)
        else:
            metrics.update(clutter_success_flags(metrics))
        rows.append(metrics)
        (output_dir / f"{ep.id}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.generate:
        out_yaml = output_dir / "resolved_clutter_episodes.yaml"
        out_yaml.write_text(yaml.safe_dump({"episodes": resolved}, sort_keys=False), encoding="utf-8")
        print(f"Wrote resolved registry to {out_yaml}", file=sys.stderr)
        return 0

    # Aggregate CSV (union of keys).
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    csv_path = output_dir / "aggregate_tamp_clutter.csv"
    import csv as _csv

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
    n_skip = sum(1 for r in rows if r.get("skipped_invalid"))
    n_scored = len(rows) - n_skip
    n_ok = sum(1 for r in rows if not r.get("skipped_invalid") and r.get("task_success"))
    print(
        f"Wrote {len(rows)} runs to {output_dir} (CSV: {csv_path}); "
        f"scored={n_scored} skipped_invalid={n_skip} task_success={n_ok}/{n_scored}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
