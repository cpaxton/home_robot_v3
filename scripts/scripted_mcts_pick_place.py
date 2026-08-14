#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Live MCTS TAMP pick-place on a running MuJoCo server.

Connects to an existing server (e.g. ``emet serve mujoco --scene ithor --robot rby1``),
builds candidate pick/place tasks from the MolmoSpaces scene metadata, runs the
distance-heuristic MCTS search (``plan_pick_place_mcts``), and executes the best
reachable plan via the kinematic executor.

No LLM/VLM: the search policy is ``PickPlaceDistancePolicy`` (distance-based +
sampling), grounding is IK-ranked grasp selection over DROID grasp assets.

Example (against a running rby1 iTHOR server at port-offset 70)::

  uv run python scripts/scripted_mcts_pick_place.py --robot rby1 --port-offset 70 \\
      --object-filter bowl --manip-mode kinematic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=str, default="rby1")
    parser.add_argument("--robot-ip", type=str, default="127.0.0.1")
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--object-filter", type=str, default="", help="Category substring (e.g. bowl).")
    parser.add_argument("--manip-mode", type=str, default="auto", choices=["auto", "kinematic", "teleport"])
    parser.add_argument("--mcts-iters", type=int, default=120)
    parser.add_argument("--mcts-breadth", type=int, default=4)
    parser.add_argument("--mcts-depth", type=int, default=5)
    parser.add_argument("--top-k-grasps", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=12, help="Cap on (object, receptacle) candidates.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))

    from emet.controller.generic_zmq_client import GenericZmqClient
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.controller.task.tamp.task_search import execute_task_plan, plan_pick_place_mcts
    from emet.eval.scene_task_extractor import (
        default_molmospaces_scenes_dir,
        load_scene_metadata,
        pickable_objects,
        receptacle_objects,
        scene_objects,
    )
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
    from emet.motion.arm_manip_profile import resolve_manip_mode_for_robot
    from emet.robots import get_robot_spec

    spec = get_robot_spec(args.robot)
    robot = GenericZmqClient(
        robot_spec=spec,
        robot_ip=args.robot_ip,
        port_offset=args.port_offset,
        start_immediately=True,
    )
    for _ in range(80):
        sess = robot.get_emet_session()
        if isinstance(sess, dict) and sess.get("is_simulation"):
            break
        import time

        time.sleep(0.25)
    mode = resolve_manip_mode_for_robot(robot, manip_mode=args.manip_mode)
    print(f"robot={args.robot} manip_mode={mode!r}", flush=True)

    # Scene candidates from MolmoSpaces metadata (pickables with grasp assets -> receptacles).
    md_dir = default_molmospaces_scenes_dir() / "ithor"
    md_files = sorted(md_dir.glob("*_physics_metadata.json")) if md_dir.is_dir() else []
    if not md_files:
        print("FAIL: no MolmoSpaces scene metadata", file=sys.stderr)
        return 1
    metadata = load_scene_metadata(md_files[0])
    objs = scene_objects(metadata)
    picks = pickable_objects(objs)
    recepts = receptacle_objects(objs)
    filt = (args.object_filter or "").strip().lower()

    pl = read_sim_object_placements(robot.get_emet_session()) or {}
    candidates: list[dict] = []
    for p in picks:
        if filt and filt not in p.category.lower():
            continue
        if p.body not in pl:
            continue
        for r in recepts:
            if r.category.lower() == p.category.lower():
                continue
            candidates.append(
                {
                    "object_query": p.category.lower(),
                    "receptacle_query": r.category.lower(),
                    "object_gt_body": p.body,
                    "receptacle_gt_body": r.body,
                    "asset_id": p.asset_id,
                }
            )
            if len(candidates) >= args.max_candidates:
                break
        if len(candidates) >= args.max_candidates:
            break
    if not candidates:
        print("FAIL: no candidates", file=sys.stderr)
        return 1
    print(f"candidates ({len(candidates)}):", flush=True)
    for c in candidates[:8]:
        print(f"  pick {c['object_query']} -> {c['receptacle_query']} ({c['object_gt_body']})", flush=True)

    exe = KinematicPickPlaceExecutor(robot, manip_collision="none", traj_dt=0.05) if mode == "kinematic" else None

    plan = plan_pick_place_mcts(
        robot,
        candidates=candidates,
        executor=exe,
        approach_standoff_m=0.55,
        top_k_grasps=args.top_k_grasps,
        mcts_iterations=args.mcts_iters,
        mcts_breadth=args.mcts_breadth,
        mcts_depth=args.mcts_depth,
        seed=args.seed,
    )
    print(f"\nMCTS plan success={plan.success} msg={plan.message!r} obj={plan.object_body}", flush=True)
    print(f"search steps: {plan.expanded_nodes[:12]}", flush=True)
    for s in plan.steps:
        print(f"  step: {s.op} {s.args}", flush=True)
    if plan.grasp_scores:
        print("grasp_scores:", flush=True)
        for idx, err, ok in plan.grasp_scores:
            print(f"  [{idx}] err={err:.3f} reachable={ok}", flush=True)
    if not plan.success:
        robot.stop() if hasattr(robot, "stop") else None
        return 1

    # Execute the winning plan using the grasps the planner grounded against.
    plan = execute_task_plan(robot, plan, executor=exe, grasp_poses=plan.grasp_poses, manip_mode=mode)
    print(f"\nexecute success={plan.success} msg={plan.message!r}", flush=True)
    if plan.success:
        after = read_sim_object_placements(robot.get_emet_session()) or {}
        if plan.object_body in after:
            pos = np.asarray(after[plan.object_body]["pos"], dtype=np.float64).reshape(3)
            print(f"object now at {pos.tolist()}", flush=True)

    try:
        robot.stop() if hasattr(robot, "stop") else None
    except Exception:
        pass
    return 0 if plan.success else 1


if __name__ == "__main__":
    sys.exit(main())
