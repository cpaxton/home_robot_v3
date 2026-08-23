#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Debug robocasa find-phase nav failure: is the robot's A* start navigable?
#
# Boots the robocasa sim the find-phase uses, runs the mapping protocol
# (rotate-in-place), then prints:
#   * robot base pose (world frame)
#   * map explored bounds / cell counts
#   * the navigable flag + clearance at the base grid cell
#   * get_reachable_points count from the base cell
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _wait_port(port: int, timeout: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="configs/sim/robocasa_pick_place_stretch.yaml")
    parser.add_argument("--port-offset", type=int, default=97)
    parser.add_argument("--not-rotate", action="store_true")
    args = parser.parse_args()

    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.eval.ovmm_find_phase import run_mapping_protocol
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
    from emet.utils.process_tree import popen_session

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("EMET_ZMQ_FULL_HZ", "5")
    env.setdefault("EMET_ZMQ_STATE_HZ", "30")
    env.setdefault("EMET_ZMQ_SERVO_HZ", "10")
    env["EMET_SIM_NAV_TELEPORT"] = "1"

    sim_cfg = load_sim_launch_config_from_path(args.sim)
    sim_cfg.port_offset = args.port_offset
    sim_cfg.headless = True
    server_argv = prepare_mujoco_server_argv(sim_cfg)
    server_cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *server_argv]
    recv_port = 4401 + args.port_offset

    print("launching sim:", " ".join(server_cmd), file=sys.stderr)
    server_log = REPO / "debug_ovmm_robocasa_nav_sim.log"
    fh = server_log.open("w", encoding="utf-8")
    server = popen_session(server_cmd, env=env, stdout=subprocess.DEVNULL, stderr=fh)
    try:
        if not _wait_port(recv_port, timeout=180.0, proc=server):
            raise RuntimeError("sim server did not bind")
        time.sleep(20.0)

        from emet.app.robot_cli import create_robot_client_from_cli
        from emet.core.parameters import get_parameters
        from emet.eval.ovmm_find_phase import create_find_phase_agent

        params = get_parameters("dynav_config.yaml")
        params["debug_perfect_sensor_depth"] = True
        params["enable_tts"] = False
        robot_kind = str(getattr(sim_cfg, "robot", "stretch"))
        robot = create_robot_client_from_cli(
            robot_kind,
            "127.0.0.1",
            port_offset=args.port_offset,
            parameters=params,
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=True,
        )
        time.sleep(5.0)

        agent = create_find_phase_agent(robot, params, "dynagraph", cpu_only=False)
        if args.not_rotate:
            # Fast path: a few manual updates (avoids the slow 8-step rotate) then dump.
            for _ in range(3):
                agent.update()
        else:
            steps = run_mapping_protocol(agent, explore_steps=0, not_rotate=False)
            print(f"mapping protocol steps={steps}", file=sys.stderr)

        # --- inspect map + base pose ---
        vm = getattr(agent, "voxel_map", None)
        planner = getattr(agent, "planner", None)
        world_xy = agent.world_base_xy()
        print(f"base world xy={world_xy}", file=sys.stderr)

        if vm is not None:
            visited = getattr(vm, "_visited", None)
            if visited is not None:
                print(
                    f"visited cells={int((visited > 0).sum())} visited_norm={float((visited > 0).float().mean()):.5f}",
                    file=sys.stderr,
                )
            obs, exp = vm.get_2d_map()
            obs_np = np.asarray(obs)
            exp_np = np.asarray(exp)
            print(
                f"map shape={obs_np.shape} explored_cells={int(exp_np.sum())} "
                f"obstacle_cells={int(obs_np.sum())} explored_frac={float(exp_np.mean()):.4f}",
                file=sys.stderr,
            )
            if world_xy is not None:
                i, j = planner.to_pt((float(world_xy[0]), float(world_xy[1])))
                print(
                    f"base grid cell=({i},{j}) explored={bool(exp_np[i, j])} obstacle={bool(obs_np[i, j])}",
                    file=sys.stderr,
                )
                if visited is not None:
                    print(f"base cell visited={bool(visited[i, j] > 0)}", file=sys.stderr)
                reachable = planner.get_reachable_points((i, j))
                print(f"reachable_points from base cell={len(reachable)}", file=sys.stderr)
                clear = getattr(planner, "_clearance_m", None)
                if clear is not None:
                    print(
                        f"clearance at base cell={float(clear[i, j]):.3f} "
                        f"min_req={getattr(planner, 'min_clearance_m', None)}",
                        file=sys.stderr,
                    )
        else:
            print("no voxel_map on agent", file=sys.stderr)
        return 0
    finally:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except Exception:
            pass
        fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
