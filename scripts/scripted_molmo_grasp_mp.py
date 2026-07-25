#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Non-agentic MolmoSpaces explore + grasp-oracle + motion-plan / teleport smoke.

Robot-agnostic grasp oracle (ZMQ); execution dispatches by server capabilities:
  - kinematic_manip → IK + RRT + attach (rby1 / galaxea_r1)
  - sim_set_body_pose only → teleport object to grasp XYZ (stretch)

Examples::

  # rby1 kinematic
  EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl \\
    uv run python scripts/scripted_molmo_grasp_mp.py --start-sim \\
    --sim configs/sim/molmospaces_ithor_train_0.yaml \\
    --port-offset 194 --object bowl --cpu-only

  # stretch teleport (same oracle)
  EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl \\
    uv run python scripts/scripted_molmo_grasp_mp.py --start-sim \\
    --sim configs/sim/molmospaces_ithor_train_stretch_0.yaml \\
    --port-offset 195 --object bowl --cpu-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _placement_pos(robot: Any, body: str) -> np.ndarray | None:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    pl = read_sim_object_placements(robot.get_emet_session())
    if not pl or body not in pl:
        return None
    return np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)


def _T_from_placement(info: dict[str, Any]) -> np.ndarray:
    from emet.perception.grasps.molmo_grasp_library import pose_matrix_from_pos_quat

    pos = info["pos"]
    quat = info.get("quat") or [1.0, 0.0, 0.0, 0.0]
    return pose_matrix_from_pos_quat(pos, quat)


def _find_graspable_body(
    robot: Any,
    *,
    object_query: str | None,
    asset_id: str | None,
    oracle_client: Any,
) -> tuple[str, dict[str, Any], list[Any]]:
    from emet.eval.ovmm_find_phase import bodies_matching_category
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
    from emet.perception.grasps.asset_id import resolve_asset_id_against_grasps_dir
    from emet.perception.grasps.molmo_grasp_library import default_grasps_dir

    pl = read_sim_object_placements(robot.get_emet_session()) or {}
    grasps_dir = default_grasps_dir()
    candidates: list[str]
    if object_query:
        candidates = bodies_matching_category(pl, object_query) or []
    else:
        candidates = list(pl.keys())
    for body in candidates:
        info = pl[body]
        cat = str(info.get("cat") or "")
        aid = asset_id or resolve_asset_id_against_grasps_dir(body, grasps_dir, category=cat)
        if not aid:
            continue
        T = _T_from_placement(info)
        try:
            poses = oracle_client.predict(object_pose_4x4=T, asset_id=aid, body_name=body, category=cat, top_k=24)
        except Exception as e:
            print(f"  skip body={body!r} asset={aid!r}: {e}")
            continue
        if poses:
            return body, info, poses
    raise RuntimeError(
        f"no graspable GT body (query={object_query!r} asset_id={asset_id!r}); placements={list(pl.keys())[:20]}"
    )


def _approach_pose(obj_xy: np.ndarray, *, standoff: float = 0.35) -> np.ndarray:
    # Face −Y (table / ithor convention used in default_table smoke).
    return np.array([float(obj_xy[0]), float(obj_xy[1]) + standoff, -np.pi / 2], dtype=np.float64)


def _rotate_in_place_light(robot: Any, n: int = 4) -> None:
    """Cheap non-agentic explore: relative yaw steps to fill a bit of the map."""
    pose = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
    for i in range(int(n)):
        goal = pose.copy()
        goal[2] = float(pose[2] + (i + 1) * (np.pi / 2))
        robot.move_base_to(goal, blocking=True, world_frame=True)
        time.sleep(0.2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sim", action="store_true")
    parser.add_argument("--sim", type=str, default="configs/sim/molmospaces_ithor_train_0.yaml")
    parser.add_argument("--robot", type=str, default=None, help="Override robot id from sim YAML")
    parser.add_argument("--port-offset", type=int, default=None)
    parser.add_argument("--object", type=str, default="bowl")
    parser.add_argument("--asset-id", type=str, default=None)
    parser.add_argument(
        "--manip-mode",
        type=str,
        default="auto",
        choices=["auto", "kinematic", "teleport"],
    )
    parser.add_argument("--oracle-bind", type=str, default="tcp://127.0.0.1:5558")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-explore", action="store_true")
    parser.add_argument("--verbose-sim", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))

    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.eval.sim_eval_session import (
        connect_benchmark_robot,
        launch_benchmark_sim_server,
        terminate_benchmark_sim_server,
    )
    from emet.motion.arm_manip_profile import resolve_manip_mode_for_robot
    from emet.perception.grasps.zmq_client import GraspOracleClient

    oracle_proc: subprocess.Popen | None = None
    sim_handle = None
    robot = None
    client: GraspOracleClient | None = None
    try:
        # Start grasp oracle (subprocess so it matches `emet grasp-oracle`).
        oracle_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emet.cli",
                "grasp-oracle",
                "--bind",
                args.oracle_bind,
            ],
            cwd=str(REPO),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.8)
        client = GraspOracleClient(args.oracle_bind, timeout_ms=8000)
        if not client.ping():
            raise RuntimeError("grasp-oracle ping failed")

        if not args.start_sim:
            raise SystemExit("this smoke requires --start-sim")

        sim_cfg = load_sim_launch_config_from_path(args.sim)
        if args.robot:
            sim_cfg = replace(sim_cfg, robot=str(args.robot))
        if args.port_offset is not None:
            sim_cfg = replace(sim_cfg, port_offset=int(args.port_offset))
        print(
            f"Starting sim {args.sim} robot={getattr(sim_cfg, 'robot', None)!r} …",
            flush=True,
        )
        sim_handle = launch_benchmark_sim_server(
            sim_cfg,
            repo=REPO,
            cpu_only=bool(args.cpu_only),
            cwd=REPO,
            server_stderr=sys.stderr if args.verbose_sim else None,
        )
        robot = connect_benchmark_robot(sim_cfg, sim_handle.port_offset)
        for _ in range(80):
            sess = robot.get_emet_session()
            if isinstance(sess, dict) and sess.get("is_simulation"):
                break
            time.sleep(0.25)

        sess = robot.get_emet_session() or {}
        caps = sess.get("capabilities") or {}
        mode = resolve_manip_mode_for_robot(robot, manip_mode=args.manip_mode)
        print(
            f"session robot={sess.get('robot_id') or getattr(sim_cfg, 'robot', None)!r} "
            f"kinematic_manip={caps.get('kinematic_manip')} "
            f"sim_set_body_pose={caps.get('sim_set_body_pose')} manip_mode={mode!r}",
            flush=True,
        )

        if not args.skip_explore:
            print("Explore: rotate_in_place (non-agentic) …", flush=True)
            _rotate_in_place_light(robot, n=4)

        body, info, poses = _find_graspable_body(
            robot, object_query=args.object, asset_id=args.asset_id, oracle_client=client
        )
        print(
            f"GT body={body!r} cat={info.get('cat')!r} n_grasps={len(poses)} asset={poses[0].asset_id!r}",
            flush=True,
        )
        before = _placement_pos(robot, body)
        obj_xy = np.asarray(info["pos"], dtype=np.float64).reshape(3)[:2]
        approach = _approach_pose(obj_xy)
        print(f"Approach base {approach.tolist()} …", flush=True)
        robot.move_base_to(approach, blocking=True, world_frame=True)
        time.sleep(0.5)

        if mode == "kinematic":
            from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

            exe = KinematicPickPlaceExecutor(robot, manip_collision="none", traj_dt=0.05)
            ok = False
            last = None
            for i, g in enumerate(poses[:8]):
                print(f"  try grasp[{i}] pos={np.round(g.position, 3).tolist()}", flush=True)
                last = exe.grasp_only(args.object, object_gt_body=body, grasp_T_world=g.T_world)
                print(f"    -> success={last.success} msg={last.message!r} err={last.grasp_err_m}")
                if last.success:
                    ok = True
                    break
            if not ok:
                print(f"FAIL kinematic grasp: {last}", file=sys.stderr)
                return 1
        else:
            from emet.simulation.sim_manipulation import sim_teleport_to_grasp_pose

            g0 = poses[0]
            ok = sim_teleport_to_grasp_pose(robot, body, g0.position, lift_m=0.12)
            print(f"teleport grasp success={ok} target={np.round(g0.position, 3).tolist()}")
            if not ok:
                print("FAIL teleport grasp", file=sys.stderr)
                return 1

        after = _placement_pos(robot, body)
        if before is not None and after is not None:
            disp = float(np.linalg.norm(after - before))
            print(f"pos_before={before.tolist()} pos_after={after.tolist()} displacement_m={disp:.4f}")
            if disp < 0.02:
                print("WARN: object barely moved (attach/teleport may be no-op)", file=sys.stderr)
        print("OK: molmo grasp-oracle + manip smoke")
        return 0
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if sim_handle is not None:
            terminate_benchmark_sim_server(sim_handle)
        if oracle_proc is not None and oracle_proc.poll() is None:
            oracle_proc.terminate()
            try:
                oracle_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                oracle_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
