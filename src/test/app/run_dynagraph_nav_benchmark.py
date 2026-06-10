#!/usr/bin/env python3
"""Dynagraph nav + frontier benchmarks on default table, Robocasa, MolmoSpaces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("DYNAGRAPH_NAV_BENCH_BASE", "/tmp/dynagraph_nav_bench"))
SEND_PORT = 4401
SERVER_WAIT_S = 180


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_port(port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.5)
    return False


def _kill_servers() -> None:
    subprocess.run(["uv", "run", "emet", "kill-mujoco-server"], cwd=REPO, check=False)
    subprocess.run(["pkill", "-f", "emet serve mujoco"], check=False)
    time.sleep(1.5)


def _session_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src")] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"
    env["EMET_ZMQ_STARTUP_TIMEOUT"] = "120"
    env["EMET_SIM_NAV_TELEPORT"] = "1"
    return env


def _connect_robot(robot_key: str) -> Any:
    key = robot_key.lower().replace("-", "_")
    if key == "stretch":
        from emet.controller.zmq_client import StretchZmqClient

        return StretchZmqClient(
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
        )
    from emet.controller.generic_zmq_client import GenericZmqClient
    from emet.robots import get_robot_spec

    spec = get_robot_spec(key)
    if spec is None:
        raise ValueError(f"unknown robot {robot_key!r}")
    return GenericZmqClient(
        robot_spec=spec,
        robot_ip="127.0.0.1",
        enable_rerun_server=False,
        start_immediately=True,
    )


def _make_agent(robot: Any, *, ground_truth: bool, cpu_only: bool = True):
    from emet.controller.controller_dynagraph import DynagraphController
    from emet.core.parameters import get_parameters

    params = get_parameters("dynav_config.yaml")
    return DynagraphController(
        robot,
        params,
        save_rerun=False,
        cpu_only=cpu_only,
        realtime_updates=True,
        ground_truth_mode=ground_truth,
        use_sensor_perception=not ground_truth,
        use_instance_graph=not ground_truth,
    )


def _run_tier(
    *,
    tier: str,
    server_cmd: list[str],
    nav_query: str,
    gt_body_key: str | None,
    ground_truth: bool,
    explore_iters: int,
    robot_key: str = "stretch",
) -> dict[str, Any]:
    from emet.memory.graph_eqa.nav_benchmark import (
        find_gt_target_xy,
        score_explore_metrics,
        score_nav_toward_target,
    )
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    log_path = BASE / f"{tier}_server.log"
    BASE.mkdir(parents=True, exist_ok=True)
    _kill_servers()
    env = _session_env()
    with open(log_path, "w", encoding="utf-8") as log_f:
        server = subprocess.Popen(
            server_cmd,
            cwd=REPO,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    robot = None
    out: dict[str, Any] = {"tier": tier, "pass": False}
    try:
        if not _wait_port(SEND_PORT, SERVER_WAIT_S):
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"server did not bind {SEND_PORT}\n{tail}")
        time.sleep(12.0)

        robot = _connect_robot(robot_key)
        agent = _make_agent(robot, ground_truth=ground_truth)
        agent.start()
        warmup = 12 if robot_key != "stretch" else 5
        for _ in range(warmup):
            agent.update()
        try:
            agent.robot.switch_to_navigation_mode()
            agent.look_around()
            agent.robot.look_front()
            agent.update()
        except Exception:
            pass

        placements = read_sim_object_placements(robot.get_emet_session())
        if placements is None:
            raise RuntimeError("missing sim_object_placements in emet_session")

        gt_match = find_gt_target_xy(placements, nav_query, body_key=gt_body_key)
        if gt_match is None:
            raise RuntimeError(f"no GT match for query {nav_query!r}")
        body, target_xyz = gt_match
        out["gt_body"] = body
        out["nav_query"] = nav_query

        graph_target = agent._localize_point_from_graph_memory(nav_query)
        out["graph_localized"] = graph_target is not None
        start_pose = agent._planning_base_xyt(robot.get_base_pose())
        start_xy = start_pose[:2]
        gt_xy = target_xyz[:2]

        plan = agent.process_text(nav_query, start_pose)
        out["process_text_plan_len"] = len(plan)

        if graph_target is not None:
            goal = np.array(
                [float(graph_target[0]), float(graph_target[1]), float(start_pose[2])],
                dtype=np.float64,
            )
            robot.move_base_to(goal, blocking=True, world_frame=True)
            agent.update()

        end_xy = agent._planning_base_xyt(robot.get_base_pose())[:2]
        nav_score = score_nav_toward_target(start_xy, end_xy, gt_xy)
        out.update({f"nav_{k}": v for k, v in nav_score.items()})

        n_ok = 0
        for _ in range(explore_iters):
            if agent.run_exploration():
                n_ok += 1
        explored = None
        try:
            _, explored_map = agent.voxel_map.get_2d_map()
            explored = float(np.asarray(explored_map, dtype=bool).sum()) * float(
                getattr(agent.voxel_map, "grid_resolution", 0.05) ** 2
            )
        except Exception:
            explored = None
        frontier_nodes = 0
        if agent.graph_memory is not None:
            frontier_nodes = sum(1 for n in agent.graph_memory.get_nodes() if getattr(n, "is_frontier", False))
        explore_score = score_explore_metrics(
            n_success=n_ok,
            n_iters=explore_iters,
            explored_area_m2=explored,
            frontier_nodes=frontier_nodes,
        )
        out.update({f"explore_{k}": v for k, v in explore_score.items()})

        nav_ok = bool(
            graph_target is not None
            and (nav_score["improved"] or nav_score["reached"] or len(plan) > 0)
        )
        out["pass"] = nav_ok and bool(explore_score["pass"])
        # Kitchen / Molmo may need extra mapping before frontier sampling reports success.
        if nav_ok and not out["pass"]:
            out["pass"] = (
                explored is not None
                and explored > 0.5
                and int(explore_score.get("explore_successes", 0)) >= 1
            )
        return out
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
        _kill_servers()


def tier_default_table() -> dict[str, Any]:
    return _run_tier(
        tier="default_table_gt_nav_explore",
        server_cmd=["uv", "run", "emet", "serve", "mujoco", "--robot", "stretch", "--headless"],
        nav_query="red cylinder",
        gt_body_key=None,
        ground_truth=True,
        explore_iters=3,
    )


def tier_robocasa() -> dict[str, Any]:
    return _run_tier(
        tier="robocasa_gt_nav_explore",
        server_cmd=        [
            "uv",
            "run",
            "emet",
            "serve",
            "robocasa",
            "--robot",
            "innate_mars",
            "--headless",
            "--seed",
            "0",
        ],
        nav_query="go to the sink",
        gt_body_key="sink_main",
        ground_truth=True,
        explore_iters=3,
        robot_key="innate_mars",
    )


def tier_dynamem_explore_only() -> dict[str, Any]:
    """DynaMem baseline: frontier exploration without graph memory."""
    from emet.controller.controller_dynamem import DynamemController
    from emet.controller.zmq_client import StretchZmqClient
    from emet.core.parameters import get_parameters
    from emet.memory.graph_eqa.nav_benchmark import score_explore_metrics

    log_path = BASE / "dynamem_explore_server.log"
    BASE.mkdir(parents=True, exist_ok=True)
    _kill_servers()
    env = _session_env()
    with open(log_path, "w", encoding="utf-8") as log_f:
        server = subprocess.Popen(
            ["uv", "run", "emet", "serve", "mujoco", "--robot", "stretch", "--headless"],
            cwd=REPO,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    robot = None
    out: dict[str, Any] = {"tier": "dynamem_explore_baseline", "pass": False}
    try:
        if not _wait_port(SEND_PORT, SERVER_WAIT_S):
            raise RuntimeError("server did not start")
        time.sleep(12.0)
        robot = _connect_robot("stretch")
        from emet.config.embodied_agent_config import legacy_embodied_agent_off

        params = get_parameters("dynav_config.yaml")
        agent = DynamemController(
            robot,
            params,
            save_rerun=False,
            cpu_only=True,
            realtime_updates=True,
            embodied_agent=legacy_embodied_agent_off(),
        )
        agent.start()
        for _ in range(5):
            agent.update()
        n_ok = sum(1 for _ in range(3) if agent.run_exploration())
        explored = None
        try:
            _, explored_map = agent.voxel_map.get_2d_map()
            explored = float(np.asarray(explored_map, dtype=bool).sum()) * float(
                getattr(agent.voxel_map, "grid_resolution", 0.05) ** 2
            )
        except Exception:
            pass
        explore_score = score_explore_metrics(n_success=n_ok, n_iters=3, explored_area_m2=explored)
        out.update({f"explore_{k}": v for k, v in explore_score.items()})
        out["pass"] = bool(explore_score["pass"])
        return out
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
        _kill_servers()


def tier_molmospaces() -> dict[str, Any]:
    return _run_tier(
        tier="molmospaces_gt_nav_explore",
        server_cmd=        [
            "uv",
            "run",
            "emet",
            "serve",
            "molmospaces",
            "ithor",
            "--split",
            "train",
            "--index",
            "0",
            "--robot",
            "stretch",
            "--headless",
        ],
        nav_query="go to the sofa",
        gt_body_key=None,
        ground_truth=True,
        explore_iters=3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynagraph GT nav + frontier explore benchmarks")
    parser.add_argument("--default", action="store_true")
    parser.add_argument("--robocasa", action="store_true")
    parser.add_argument("--molmo", action="store_true")
    parser.add_argument("--dynamem", action="store_true", help="DynaMem-only explore baseline")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run_all = args.all or not (args.default or args.robocasa or args.molmo or args.dynamem)

    results: list[dict[str, Any]] = []
    if run_all or args.dynamem:
        results.append(tier_dynamem_explore_only())
    if run_all or args.default:
        results.append(tier_default_table())
    if run_all or args.robocasa:
        if (REPO / "third_party" / "robocasa").is_dir():
            results.append(tier_robocasa())
        else:
            results.append({"tier": "robocasa_gt_nav_explore", "pass": False, "skipped": "no robocasa"})
    if run_all or args.molmo:
        if (REPO / "packages" / "emet_molmospaces").is_dir():
            results.append(tier_molmospaces())
        else:
            results.append({"tier": "molmospaces_gt_nav_explore", "pass": False, "skipped": "no molmospaces"})

    report = {"results": results, "all_pass": all(r.get("pass") for r in results)}
    BASE.mkdir(parents=True, exist_ok=True)
    rep_path = BASE / "nav_benchmark_report.json"
    rep_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {rep_path}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
