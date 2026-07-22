#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Smoke: Dynagraph agent FIND before/after a Robocasa world change.

Flow (same memory stack as ``emet run agent --memory-backend dynagraph``):

1. Start Robocasa sim (or reuse ``--port-offset``).
2. Explore briefly, localize a known label via graph/voxel find.
3. Relocate ``obj_main`` over ZMQ.
4. Invalidate nodes near the old pose + clear EQA working memory.
5. FIND again — must not confidently navigate to the *old* pose.

Usage::

    uv run python scripts/smoke_dynagraph_agent_world_change_find.py --cpu-only
    uv run python scripts/smoke_dynagraph_agent_world_change_find.py --label apple --cpu-only

Exit 0 when post-move find does not reuse the old XY within 0.75 m (or clearly
fails / points elsewhere after invalidate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _localize(agent, text: str) -> np.ndarray | None:
    """Best-effort object localization from Dynagraph / DynaMem agent APIs."""
    vm = getattr(agent, "voxel_map", None)
    if vm is not None and hasattr(vm, "localize_text"):
        try:
            result = vm.localize_text(text, debug=False, return_debug=True)
            if isinstance(result, tuple) and result and result[0] is not None:
                xyz = np.asarray(result[0], dtype=np.float64).reshape(-1)
                if xyz.size >= 2:
                    return xyz[:3]
            if result is not None and not isinstance(result, tuple):
                xyz = np.asarray(result, dtype=np.float64).reshape(-1)
                if xyz.size >= 2:
                    return xyz[:3]
        except Exception:
            pass
    mem = getattr(agent, "graph_memory", None)
    if mem is None or not hasattr(mem, "get_nodes"):
        return None
    needle = text.strip().lower()
    best = None
    best_d = 1e9
    for n in mem.get_nodes():
        if getattr(n, "is_viewpoint", False) or getattr(n, "is_frontier", False):
            continue
        blob = " ".join(n.labels).lower()
        if needle not in blob and not any(needle in lb.lower() for lb in n.labels):
            continue
        xyz = np.asarray(n.xyz, dtype=np.float64)
        d = float(np.linalg.norm(xyz[:2]))
        if d < best_d:
            best_d = d
            best = xyz
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--port-offset", type=int, default=40)
    ap.add_argument("--explore-iters", type=int, default=3)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--relocate-body", type=str, default="obj_main")
    ap.add_argument("--match-radius-m", type=float, default=0.75)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "runs/emet/agent_world_change_find/smoke.json",
    )
    args = ap.parse_args()

    from emet.app.dynagraph_explore import dynagraph_explore_until_terminated
    from emet.controller.controller_dynagraph import DynagraphController
    from emet.controller.task.dynamem import EQAExecuter
    from emet.core.parameters import get_parameters
    from emet.eval.dynamic_exploration_config import load_dynamic_exploration_config
    from emet.eval.dynamic_exploration_runner import _resolve_sim_cfg
    from emet.eval.ovmm_find_phase import resolve_find_phase_nav_step_timeout
    from emet.eval.sim_eval_session import benchmark_sim_server, connect_benchmark_robot
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
    from emet.simulation.sim_manipulation import robot_zmq_set_body_pose

    cfg = load_dynamic_exploration_config()
    episode = next(e for e in cfg.episodes if e.env == "robocasa" and e.id == "robocasa_seed0")
    sim_cfg = replace(_resolve_sim_cfg(episode), port_offset=int(args.port_offset), headless=True)

    payload: dict = {
        "label": args.label or None,
        "relocate_body": args.relocate_body,
        "explore_iters": args.explore_iters,
        "pass": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sim_log = args.output.with_suffix(".sim.log")

    with open(sim_log, "w", encoding="utf-8") as sim_fh:
        with benchmark_sim_server(
            sim_cfg,
            repo=_REPO,
            cpu_only=args.cpu_only,
            cwd=_REPO,
            server_stderr=sim_fh,
        ) as sim:
            robot = connect_benchmark_robot(sim_cfg, int(args.port_offset))
            agent = None
            try:
                parameters = get_parameters("dynav_config.yaml")
                parameters["encoder"] = None
                parameters["debug_perfect_sensor_depth"] = True
                parameters["find_phase_nav_step_timeout_s"] = resolve_find_phase_nav_step_timeout(
                    cpu_only=args.cpu_only,
                    sim_kind=sim.sim_kind,
                )
                agent = DynagraphController(
                    robot,
                    parameters,
                    save_rerun=False,
                    cpu_only=args.cpu_only,
                    use_instance_graph=True,
                    use_sensor_perception=False,
                )
                agent.start()
                agent._fast_explore_lookaround = True
                executor = EQAExecuter(agent)
                executor.rotate_in_place()
                dynagraph_explore_until_terminated(agent, max_iterations=int(args.explore_iters))

                session = robot.get_emet_session()
                placements = read_sim_object_placements(session)
                body = args.relocate_body
                if body not in placements:
                    raise RuntimeError(f"{body!r} missing from sim_object_placements: {list(placements)}")
                old_pos = list(placements[body]["pos"])
                label = args.label.strip()
                if not label:
                    meta = placements[body]
                    label = str(meta.get("label") or meta.get("name") or body).replace("_", " ")

                pre_xyz = _localize(agent, label)
                payload["label"] = label
                payload["old_pos"] = old_pos
                payload["pre_find_xyz"] = pre_xyz.tolist() if pre_xyz is not None else None

                new_pos = [float(old_pos[0]) + 1.5, float(old_pos[1]) + 0.5, float(old_pos[2])]
                robot_zmq_set_body_pose(robot, body, new_pos)
                time.sleep(0.5)
                agent.update()

                mem = agent.graph_memory
                cur = int(getattr(agent, "obs_count", 0))
                n_aged, n_pruned = 0, 0
                if mem is not None and hasattr(mem, "invalidate_nodes_near"):
                    n_aged, n_pruned = mem.invalidate_nodes_near(
                        old_pos,
                        radius_m=float(args.match_radius_m),
                        current_step=cur,
                        prune=True,
                    )
                if mem is not None and hasattr(mem, "clear_eqa_working_memory"):
                    mem.clear_eqa_working_memory()
                payload["n_aged"] = int(n_aged)
                payload["n_pruned"] = int(n_pruned)
                payload["new_pos"] = new_pos

                post_xyz = _localize(agent, label)
                payload["post_find_xyz"] = post_xyz.tolist() if post_xyz is not None else None

                reused_old = False
                if post_xyz is not None:
                    reused_old = float(
                        np.linalg.norm(post_xyz[:2] - np.asarray(old_pos[:2]))
                    ) <= float(args.match_radius_m)
                payload["post_find_reused_old_pose"] = bool(reused_old)
                payload["pass"] = not reused_old
                payload["pre_find_ok"] = pre_xyz is not None
            finally:
                if agent is not None and callable(getattr(agent, "stop", None)):
                    agent.stop()
                if callable(getattr(robot, "stop", None)):
                    robot.stop()

    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload.get("pass"):
        print("FAIL: post-move find still points at old pose", file=sys.stderr)
        print(f"sim log: {sim_log}", file=sys.stderr)
        return 1
    if not payload.get("pre_find_ok"):
        print("WARN: pre-move find did not localize; invalidate path still exercised", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
