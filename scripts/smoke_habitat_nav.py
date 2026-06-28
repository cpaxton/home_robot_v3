#!/usr/bin/env python3
"""Live Habitat nav smoke: compare navmesh vs voxel A* on one HM-EQA scene."""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.habitat_nav import habitat_navmesh_navigate
from emet.core.parameters import get_parameters
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.runner import _configure_habitat_nav
from emet_habitat.simulator import HabitatEQASimulator


def _pose_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _build_agent(robot, *, perfect_nav: bool) -> GraphEQAController:
    params = get_parameters("dynav_config.yaml")
    _configure_habitat_nav(params, habitat_perfect_nav=perfect_nav)
    agent = GraphEQAController(
        robot=robot,
        parameters=params,
        manipulation_only=True,
        cpu_only=True,
    )
    agent.start()
    for _ in range(5):
        agent.update()
    return agent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", type=int, default=14)
    args = parser.parse_args()

    questions = load_hmeqa_questions(None)
    q = get_question(questions, question_id=args.question_id)
    poses = load_scene_init_poses(None)
    init_pose = poses[(q.scene, q.floor)]
    hm3d = default_hm3d_scene_dir()

    sim = HabitatEQASimulator.from_scene_id(
        q.scene,
        hm3d_root=hm3d,
        use_hm3d_semantics=True,
    )
    sim.set_init_pose(init_pose)
    robot = HabitatRobotClient(sim)
    start = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
    print(f"scene={q.scene} start=({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")

    # Prefer a known reachable frontier-style goal over a blind forward offset (may be off navmesh).
    goal = np.array([-1.9, -1.0], dtype=np.float64)
    print(f"goal=({goal[0]:.2f}, {goal[1]:.2f})")

    nav_res = habitat_navmesh_navigate(robot, goal)
    after = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
    print(
        "navmesh_direct:",
        f"success={nav_res.success} finished={nav_res.finished} dist={nav_res.dist_m:.3f}m note={nav_res.note}",
        f"pose=({after[0]:.2f}, {after[1]:.2f})",
    )

    # Reset spawn for controller tests.
    sim.set_init_pose(init_pose)
    robot = HabitatRobotClient(sim)

    for label, perfect in [("perfect_nav", True), ("astar_only", False)]:
        sim.set_init_pose(init_pose)
        robot = HabitatRobotClient(sim)
        agent = _build_agent(robot, perfect_nav=perfect)
        pose0 = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
        target = np.array([float(goal[0]), float(goal[1]), 1.0], dtype=np.float64)
        finished = agent.navigate_to_target_pose(
            target,
            agent._planning_base_xyt(pose0),
            float(start[2]),
        )
        nav = getattr(agent, "_last_nav_attempt", None)
        pose1 = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
        moved = _pose_dist(pose0[:2], pose1[:2])
        method = nav.method if nav else "?"
        note = nav.note if nav else "?"
        print(
            f"{label}: finished={finished} moved={moved:.3f}m method={method} note={note} "
            f"pose=({pose1[0]:.2f}, {pose1[1]:.2f})",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
