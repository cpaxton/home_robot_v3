# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""VLM-free Habitat frontier exploration (mapping + nav coverage only)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.habitat_nav import (
    habitat_navmesh_navigate,
    habitat_random_walk_step,
    pick_habitat_exploration_target,
)
from emet.core.parameters import Parameters, get_parameters
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import SceneInitPose, get_question, load_hmeqa_questions, load_scene_init_poses
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.runner import _configure_frontier_parameters, _configure_habitat_nav
from emet_habitat.simulator import HabitatEQASimulator


@dataclass
class FrontierExploreStep:
    step: int
    pose_xyt: list[float]
    explored_cells: int
    free_cells: int
    explored_frac: float
    frontier_nodes: int
    graph_nodes: int
    nav_note: str
    nav_moved_m: float
    nav_finished: bool
    action: str


@dataclass
class FrontierExploreResult:
    scene: str
    question_id: int | None
    steps: int
    total_travel_m: float
    explored_frac: float
    frontier_nodes: int
    graph_nodes: int
    nav_attempts: int
    nav_finished: int
    nav_zero_move: int
    random_walk_steps: int
    step_log: list[FrontierExploreStep] = field(default_factory=list)
    output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["step_log"] = [asdict(s) for s in self.step_log]
        return d


def _voxel_coverage(agent: GraphEQAController) -> tuple[int, int, float]:
    vm = agent.voxel_map
    obstacles, explored = vm.get_2d_map()
    obs = np.asarray(obstacles, dtype=bool)
    exp = np.asarray(explored, dtype=bool)
    free = int(np.sum(~obs))
    explored_cells = int(np.sum(exp & ~obs))
    frac = float(explored_cells / free) if free > 0 else 0.0
    return explored_cells, free, frac


def _make_explore_agent(
    robot: HabitatRobotClient,
    parameters: Parameters,
) -> GraphEQAController:
    agent = GraphEQAController(
        robot=robot,
        parameters=parameters,
        manipulation_only=True,
        cpu_only=True,
        use_instance_graph=False,
        use_sensor_perception=False,
    )
    agent.start()
    return agent


def run_frontier_exploration(
    *,
    scene_id: str | None = None,
    question_id: int | None = 14,
    init_pose: SceneInitPose | None = None,
    hm3d_root: Path | None = None,
    max_steps: int = 40,
    warmup_updates: int = 5,
    rotate_in_place: bool = True,
    output_dir: Path | None = None,
    seed: int = 0,
    frontier_nodes_enabled: bool = True,
) -> FrontierExploreResult:
    """Explore by repeatedly navigating to voxel/graph frontiers (no EQA VLM)."""
    rng = random.Random(int(seed))
    hm3d = hm3d_root or default_hm3d_scene_dir()

    if scene_id is None and question_id is not None:
        q = get_question(load_hmeqa_questions(None), question_id=question_id)
        scene_id = q.scene
        question = q.question_formatted
        if init_pose is None:
            init_pose = load_scene_init_poses(None)[(q.scene, q.floor)]
    else:
        question = None
        if scene_id is None:
            raise ValueError("Provide scene_id or question_id")
        if init_pose is None:
            raise ValueError("init_pose required when scene_id is given without question_id")

    sim = HabitatEQASimulator.from_scene_id(
        scene_id,
        hm3d_root=hm3d,
        use_hm3d_semantics=True,
    )
    sim.set_init_pose(init_pose)
    robot = HabitatRobotClient(sim)

    parameters = get_parameters("dynav_config.yaml")
    _configure_habitat_nav(parameters, habitat_perfect_nav=True)
    _configure_frontier_parameters(parameters, frontier_nodes_enabled=frontier_nodes_enabled)
    parameters.set("force_eqa_siglip_encoder", False)

    agent = _make_explore_agent(robot, parameters)
    if rotate_in_place and hasattr(agent, "look_around"):
        agent.robot.look_front()
        agent.look_around()
        agent.robot.look_front()

    for _ in range(max(0, warmup_updates)):
        agent.update()
        if hasattr(agent, "_sync_graph_frontier_nodes"):
            agent._sync_graph_frontier_nodes()

    blocked: set[tuple[float, float]] = set()
    step_log: list[FrontierExploreStep] = []
    poses: list[np.ndarray] = []
    nav_attempts = nav_finished = nav_zero = random_walks = 0
    exp_cells = free_cells = 0
    exp_frac = 0.0
    frontier_n = graph_n = 0

    try:
        for step in range(max_steps):
            agent.update()
            if hasattr(agent, "_sync_graph_frontier_nodes"):
                agent._sync_graph_frontier_nodes()

            pose = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
            poses.append(pose.copy())
            exp_cells, free_cells, exp_frac = _voxel_coverage(agent)
            gm = agent.graph_memory
            frontier_n = sum(1 for n in gm.get_nodes() if getattr(n, "is_frontier", False)) if gm else 0
            graph_n = len(gm.get_nodes()) if gm else 0

            target = pick_habitat_exploration_target(
                agent,
                question=question,
                blocked=blocked,
            )
            nav_note = "no_target"
            nav_moved = 0.0
            finished = False
            action = "observe"

            if target is not None:
                nav_attempts += 1
                start = agent._planning_base_xyt(pose)
                nav_res = habitat_navmesh_navigate(
                    robot,
                    target[:2],
                    target_theta=float(pose[2]),
                )
                agent._last_nav_attempt = nav_res
                nav_note = nav_res.note
                nav_moved = float(nav_res.dist_m)
                finished = bool(nav_res.finished)
                if finished:
                    nav_finished += 1
                elif nav_moved < 0.05:
                    nav_zero += 1
                    from emet.controller.habitat_nav import goal_key_xy

                    blocked.add(goal_key_xy(target))
                    action = habitat_random_walk_step(robot, rng=rng)
                    random_walks += 1
                else:
                    action = "navmesh"
            else:
                action = habitat_random_walk_step(robot, rng=rng)
                random_walks += 1
                nav_note = "random_walk"

            pose_after = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
            step_log.append(
                FrontierExploreStep(
                    step=step,
                    pose_xyt=[float(pose_after[0]), float(pose_after[1]), float(pose_after[2])],
                    explored_cells=exp_cells,
                    free_cells=free_cells,
                    explored_frac=round(exp_frac, 4),
                    frontier_nodes=frontier_n,
                    graph_nodes=graph_n,
                    nav_note=nav_note,
                    nav_moved_m=round(nav_moved, 3),
                    nav_finished=finished,
                    action=action,
                )
            )
        exp_cells, free_cells, exp_frac = _voxel_coverage(agent)
        gm = agent.graph_memory
        frontier_n = sum(1 for n in gm.get_nodes() if getattr(n, "is_frontier", False)) if gm else 0
        graph_n = len(gm.get_nodes()) if gm else 0
    finally:
        if hasattr(agent, "stop"):
            agent.stop()
        sim.close()

    total_travel = 0.0
    if len(step_log) >= 2:
        for a, b in zip(step_log, step_log[1:]):
            total_travel += float(
                math.hypot(b.pose_xyt[0] - a.pose_xyt[0], b.pose_xyt[1] - a.pose_xyt[1])
            )
    elif len(step_log) == 1 and poses:
        total_travel = float(
            math.hypot(step_log[0].pose_xyt[0] - poses[0][0], step_log[0].pose_xyt[1] - poses[0][1])
        )

    result = FrontierExploreResult(
        scene=str(scene_id),
        question_id=question_id,
        steps=max_steps,
        total_travel_m=round(total_travel, 3),
        explored_frac=round(exp_frac, 4),
        frontier_nodes=frontier_n,
        graph_nodes=graph_n,
        nav_attempts=nav_attempts,
        nav_finished=nav_finished,
        nav_zero_move=nav_zero,
        random_walk_steps=random_walks,
        step_log=step_log,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frontier_explore.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        with (out / "trajectory.jsonl").open("w", encoding="utf-8") as fh:
            for row in step_log:
                fh.write(json.dumps({"step": row.step, "pose_xyt": row.pose_xyt, "action": row.action}) + "\n")
        result.output_dir = str(out.resolve())

    return result
