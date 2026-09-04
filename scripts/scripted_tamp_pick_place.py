#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""No-LLM TAMP pick-place smoke: oracle grasps + deterministic search + figure export.

Does **not** load Qwen / chat agent. Writes PNG/PDF figures under ``--figures-dir``.

Example::

  EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl \\
    uv run python scripts/scripted_tamp_pick_place.py --start-sim \\
    --sim configs/sim/default_table_rby1.yaml \\
    --object \"red cylinder\" --receptacle \"blue cube\" --cpu-only

  # Sourccey table (z≈0.6 m). ``--rerun`` opens the web viewer (hold 30s after the plan).
  uv run python scripts/scripted_tamp_pick_place.py --start-sim \\
    --sim configs/sim/default_table_sourccey.yaml --manip-mode kinematic --skip-oracle --rerun
  uv run python scripts/scripted_tamp_pick_place.py --start-sim \\
    --sim configs/sim/robocasa_pick_place_sourccey.yaml --object obj --receptacle cab \\
    --object-gt-body obj_main --manip-mode kinematic --skip-oracle --rerun
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RERUN_URL = "http://127.0.0.1:9090?url=ws://127.0.0.1:9877"


def _hold_rerun(args: argparse.Namespace) -> None:
    if not args.rerun:
        return
    hold = float(args.rerun_hold_s)
    if hold <= 0:
        return
    print(f"Holding Rerun for {hold:.0f}s — {RERUN_URL}", flush=True)
    time.sleep(hold)


def _placement_pos(robot: Any, body: str) -> np.ndarray | None:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    pl = read_sim_object_placements(robot.get_emet_session())
    if not pl or body not in pl:
        return None
    return np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)


def _T_from_placement(info: dict[str, Any]) -> np.ndarray:
    from emet.perception.grasps.molmo_grasp_library import pose_matrix_from_pos_quat

    pos = info["pos"]
    quat = info.get("quat")
    if quat is None:
        quat = [1.0, 0.0, 0.0, 0.0]
    else:
        quat = np.asarray(quat, dtype=np.float64).reshape(-1)[:4].tolist()
    return pose_matrix_from_pos_quat(pos, quat)


def _find_graspable_body(
    robot: Any,
    *,
    object_query: str | None,
    asset_id: str | None,
    oracle_client: Any,
    object_gt_body: str | None = None,
) -> tuple[str, dict[str, Any], list[Any]]:
    from emet.eval.ovmm_find_phase import bodies_matching_category
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
    from emet.perception.grasps.asset_id import resolve_asset_id_against_grasps_dir
    from emet.perception.grasps.molmo_grasp_library import default_grasps_dir

    pl = read_sim_object_placements(robot.get_emet_session()) or {}
    grasps_dir = default_grasps_dir()
    if object_gt_body:
        if object_gt_body not in pl:
            raise RuntimeError(
                f"object_gt_body={object_gt_body!r} not in placements={list(pl.keys())[:20]}"
            )
        candidates = [object_gt_body]
    elif object_query:
        candidates = bodies_matching_category(pl, object_query) or []
        if object_query in pl and object_query not in candidates:
            candidates.insert(0, object_query)
    else:
        candidates = list(pl.keys())
    for body in candidates:
        info = pl[body]
        cat = str(info.get("cat") or "")
        aid = asset_id or resolve_asset_id_against_grasps_dir(body, grasps_dir, category=cat)
        if not aid:
            # Table scene objects often lack Molmo assets — synthesize a top-down grasp at COM.
            from emet.controller.task.tamp.grasp_frames import top_down_grasp_T
            from emet.perception.grasps.oracle import GraspPose

            T = top_down_grasp_T(info["pos"])
            synth = GraspPose(T_world=T, score=0.5, asset_id="synthetic_com", gripper="droid")
            return body, info, [synth]
        T = _T_from_placement(info)
        try:
            poses = oracle_client.predict(object_pose_4x4=T, asset_id=aid, body_name=body, category=cat, top_k=24)
        except Exception as e:
            print(f"  skip body={body!r} asset={aid!r}: {e}")
            continue
        if poses:
            return body, info, poses
    raise RuntimeError(f"no graspable GT body (query={object_query!r}); placements={list(pl.keys())[:20]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sim", action="store_true")
    parser.add_argument("--sim", type=str, default="configs/sim/default_table_rby1.yaml")
    parser.add_argument("--robot", type=str, default=None)
    parser.add_argument("--port-offset", type=int, default=None)
    parser.add_argument("--object", type=str, default="red cylinder")
    parser.add_argument(
        "--object-gt-body",
        default=None,
        help="Pin the GT body id (e.g. obj_main for RoboCasa PickPlace). Category query is skipped.",
    )
    parser.add_argument("--receptacle", type=str, default="blue cube")
    parser.add_argument("--asset-id", type=str, default=None)
    parser.add_argument(
        "--any-object",
        action="store_true",
        help="Pick the first freejoint GT body (ignore --object category filter when resolving).",
    )
    parser.add_argument(
        "--plant-infeasible-grasps",
        action="store_true",
        help="Prepend IK-unreachable decoy grasps so ranking must skip them (multi-option TAMP).",
    )
    parser.add_argument("--manip-mode", type=str, default="auto", choices=["auto", "kinematic", "teleport"])
    parser.add_argument("--oracle-bind", type=str, default="tcp://127.0.0.1:5558")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip grasp-oracle process (synthetic COM grasps)")
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Output dir for PNG/PDF (default ~/runs/emet/tamp_pick_place/<stamp>)",
    )
    parser.add_argument(
        "--record-mp4",
        action="store_true",
        help="Record third-person MuJoCo view to MP4 (sets EMET_SIM_THIRD_PERSON=1 on the sim).",
    )
    parser.add_argument("--video-fps", type=float, default=12.0, help="MP4 sample rate when --record-mp4.")
    parser.add_argument("--verbose-sim", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Live Rerun of the ZMQ client (MJCF robot + cameras). Web UI :9090.",
    )
    parser.add_argument(
        "--rerun-headless",
        action="store_true",
        help="With --rerun: do not auto-open a browser (still serves :9090 / :9877).",
    )
    parser.add_argument(
        "--rerun-hold-s",
        type=float,
        default=30.0,
        help="Seconds to keep Rerun up after the plan (default 30; only with --rerun).",
    )
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))

    if args.record_mp4:
        os.environ["EMET_SIM_THIRD_PERSON"] = "1"
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.controller.task.tamp.smoke_grasps import plant_mixed_grasp_poses
    from emet.controller.task.tamp.task_search import execute_task_plan, plan_pick_place
    from emet.eval.ovmm_find_phase import bodies_matching_category
    from emet.eval.sim_eval_session import (
        connect_benchmark_robot,
        launch_benchmark_sim_server,
        terminate_benchmark_sim_server,
    )
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
    from emet.motion.arm_manip_profile import resolve_manip_mode_for_robot
    from emet.visualization.manip_figures import write_tamp_figure_bundle

    oracle_proc: subprocess.Popen | None = None
    sim_handle = None
    robot = None
    client = None
    try:
        if not args.start_sim:
            raise SystemExit("this smoke requires --start-sim")
        os.environ.setdefault("EMET_SIM_NAV_TELEPORT", "1")

        if not args.skip_oracle:
            from emet.perception.grasps.zmq_client import GraspOracleClient

            oracle_proc = subprocess.Popen(
                [sys.executable, "-m", "emet.cli", "grasp-oracle", "--bind", args.oracle_bind],
                cwd=str(REPO),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.8)
            client = GraspOracleClient(args.oracle_bind, timeout_ms=8000)
            if not client.ping():
                print("WARN: grasp-oracle ping failed; falling back to synthetic COM grasps", flush=True)
                client = None

        sim_cfg = load_sim_launch_config_from_path(args.sim)
        if args.robot:
            sim_cfg = replace(sim_cfg, robot=str(args.robot))
        if args.port_offset is not None:
            sim_cfg = replace(sim_cfg, port_offset=int(args.port_offset))
        print(f"Starting sim {args.sim} robot={getattr(sim_cfg, 'robot', None)!r} …", flush=True)
        sim_handle = launch_benchmark_sim_server(
            sim_cfg,
            repo=REPO,
            cpu_only=bool(args.cpu_only),
            cwd=REPO,
            server_stderr=sys.stderr if args.verbose_sim else None,
        )
        robot = connect_benchmark_robot(
            sim_cfg,
            sim_handle.port_offset,
            enable_rerun_server=bool(args.rerun),
            rerun_headless=bool(args.rerun_headless),
        )
        for _ in range(80):
            sess = robot.get_emet_session()
            if isinstance(sess, dict) and sess.get("is_simulation"):
                break
            time.sleep(0.25)

        mode = resolve_manip_mode_for_robot(robot, manip_mode=args.manip_mode)
        print(f"manip_mode={mode!r}", flush=True)
        if args.rerun:
            print(
                "Rerun: http://127.0.0.1:9090?url=ws://127.0.0.1:9877  "
                "(MJCF robot + cameras; EE path under world/manip/ee_path)",
                flush=True,
            )

        class _SynthClient:
            def predict(self, **kwargs):
                return []

        object_query = None if args.any_object else args.object
        body, info, poses = _find_graspable_body(
            robot,
            object_query=object_query,
            asset_id=args.asset_id,
            oracle_client=client or _SynthClient(),
            object_gt_body=args.object_gt_body,
        )
        if args.plant_infeasible_grasps:
            # Decoys first — ranking must not pick index 0.
            com = np.asarray(info["pos"], dtype=np.float64).reshape(3)
            poses = plant_mixed_grasp_poses(com + np.array([0.0, 0.0, 0.02]), n_infeasible=2)
            print(
                f"planted mixed grasps: n={len(poses)} (decoys first, reachable asset={poses[-1].asset_id!r})",
                flush=True,
            )
        pl = read_sim_object_placements(robot.get_emet_session()) or {}
        receps = bodies_matching_category(pl, args.receptacle) or []
        recep_body = receps[0] if receps else None
        print(f"object body={body!r} n_grasps={len(poses)} recep={recep_body!r}", flush=True)

        before = _placement_pos(robot, body)
        base_path = [np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(3)]

        exe = None
        viz = getattr(robot, "_rerun", None) if args.rerun else None
        if mode == "kinematic":
            exe = KinematicPickPlaceExecutor(
                robot, manip_collision="none", traj_dt=0.05, visualizer=viz
            )

        plan = plan_pick_place(
            robot,
            object_query=args.object if not args.any_object else str(info.get("cat") or body),
            receptacle_query=args.receptacle,
            grasp_poses=poses,
            object_gt_body=body,
            receptacle_gt_body=recep_body,
            executor=exe,
            approach_standoff_m=0.55,
        )
        print(f"plan success={plan.success} chosen_grasp={plan.chosen_grasp_index} msg={plan.message!r}")
        for line in plan.expanded_nodes:
            print(f"  {line}")
        if plan.grasp_scores:
            print("grasp_scores:", flush=True)
            for idx, err, ok in plan.grasp_scores:
                print(f"  [{idx}] err={err:.3f} reachable={ok}", flush=True)
        if args.plant_infeasible_grasps:
            if not plan.grasp_scores:
                print("FAIL: expected IK grasp ranking scores", file=sys.stderr)
                return 1
            if not any(ok for _i, _e, ok in plan.grasp_scores):
                print("FAIL: no reachable grasp in scores", file=sys.stderr)
                return 1
            if not any(not ok for _i, _e, ok in plan.grasp_scores):
                print("FAIL: expected at least one infeasible decoy", file=sys.stderr)
                return 1
            if plan.chosen_grasp_index is None or plan.chosen_grasp_index < 2:
                # plant_mixed puts 2 decoys at 0,1 and reachable at 2
                print(
                    f"FAIL: chosen_grasp={plan.chosen_grasp_index} should be reachable index (>=2)",
                    file=sys.stderr,
                )
                return 1
        if not plan.success:
            return 1

        if exe is None and mode == "kinematic":
            exe = KinematicPickPlaceExecutor(
                robot, manip_collision="none", traj_dt=0.05, visualizer=viz
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_dir = Path(args.figures_dir) if args.figures_dir else Path.home() / "runs/emet/tamp_pick_place" / stamp
        fig_dir.mkdir(parents=True, exist_ok=True)

        video = None
        if args.record_mp4:
            from emet.visualization.manip_video import ManipVideoRecorder

            video = ManipVideoRecorder(
                robot,
                fig_dir / "third_person.mp4",
                fps=float(args.video_fps),
                title="tamp pick-place",
            )
            video.set_status(
                "plan",
                goal=f"{body} → {recep_body}",
                detail=f"chosen_grasp={plan.chosen_grasp_index}",
            )
            video.start()

        plan = execute_task_plan(
            robot,
            plan,
            executor=exe,
            grasp_poses=poses,
            manip_mode=mode,
            video_recorder=video,
        )
        if video is not None:
            mp4 = video.stop()
            if mp4 is not None:
                print(f"mp4 -> {mp4}", flush=True)
            else:
                print(
                    "WARN: --record-mp4 produced no frames (is EMET_SIM_THIRD_PERSON reaching the server?)",
                    file=sys.stderr,
                )

        base_path.append(np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(3))
        print(f"execute success={plan.success} msg={plan.message!r}", flush=True)

        after = _placement_pos(robot, body)
        disp = None
        if before is not None and after is not None:
            disp = float(np.linalg.norm(after - before))
            print(f"displacement_m={disp:.4f}")

        grasp_xy = None
        if plan.chosen_grasp_index is not None and plan.chosen_grasp_index < len(poses):
            grasp_xy = poses[plan.chosen_grasp_index].position[:2]
        recep_xy = None
        if recep_body and recep_body in pl:
            recep_xy = np.asarray(pl[recep_body]["pos"], dtype=np.float64).reshape(3)[:2]
        paths = write_tamp_figure_bundle(
            fig_dir,
            plan=plan,
            base_path_xyt=base_path,
            object_xy=np.asarray(info["pos"], dtype=np.float64).reshape(3)[:2],
            receptacle_xy=recep_xy,
            grasp_xy=grasp_xy,
            planned_ee_xyz=getattr(exe, "last_ee_path_world", None) if exe else None,
            joint_waypoints=getattr(exe, "last_plan_waypoints", None) if exe else None,
            joint_names=list(getattr(exe, "joint_names", ())) if exe else None,
            targets=getattr(exe, "last_targets", None) if exe else None,
        )
        print(f"figures -> {fig_dir}")
        for k, p in paths.items():
            print(f"  {k}: {p}")

        if not plan.success:
            return 1
        if disp is not None and disp < 0.02 and mode == "kinematic":
            print("WARN: object barely moved", file=sys.stderr)
        print("OK: tamp pick-place smoke")
        return 0
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if robot is not None:
            try:
                _hold_rerun(args)
            except KeyboardInterrupt:
                print("Rerun hold interrupted", flush=True)
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
