# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run HM-EQA episodes with emet GraphEQA / Dynagraph controllers."""

from __future__ import annotations

from pathlib import Path

from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import get_parameters
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
from emet.habitat.metrics import EpisodeMetrics, grade_mcq_answer

from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.simulator import HabitatEQASimulator


def _mock_eqa_response(gold_letter: str) -> str:
    return (
        "reasoning: mock habitat harness\n"
        f"answer: {gold_letter}\n"
        "confidence: true\n"
        "action:\n"
        "confidence_reasoning: mocked for smoke test\n"
    )


def _apply_method_parameters(parameters: dict, method: str) -> dict:
    params = dict(parameters)
    if method == "graph_eqa":
        params["dynagraph_merge_xy_m"] = 0.0
        params["dynagraph_staleness_horizon"] = 0
    elif method == "dynagraph":
        params.setdefault("dynagraph_merge_xy_m", 0.45)
        params.setdefault("dynagraph_staleness_horizon", 256)
    else:
        raise ValueError(f"Unknown method {method!r}; use graph_eqa or dynagraph")
    return params


def _make_controller(
    robot: HabitatRobotClient,
    parameters: dict,
    *,
    method: str,
    mock_llm: bool,
    gold_letter: str,
    no_rerun: bool,
):
    params = _apply_method_parameters(parameters, method)
    common = dict(
        robot=robot,
        parameters=params,
        save_rerun=False if no_rerun else False,
        cpu_only=True,
        use_sensor_perception=False,
        use_instance_graph=False,
    )
    if method == "dynagraph":
        agent = DynagraphController(**common)
    else:
        agent = GraphEQAController(**common)

    if mock_llm and agent.graph_memory is not None:
        agent.graph_memory.eqa_client = lambda _q: _mock_eqa_response(gold_letter)
        agent.graph_memory.image_description_client = lambda _x: "object"
    return agent


def run_hmeqa_episode(
    *,
    question_id: int,
    method: str = "dynagraph",
    mock_llm: bool = True,
    max_planning_steps: int = 3,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
    no_rerun: bool = True,
    rotate_in_place: bool = False,
) -> EpisodeMetrics:
    questions = load_hmeqa_questions(questions_path)
    q = get_question(questions, question_id=question_id)
    poses = load_scene_init_poses(init_poses_path)
    init_pose = poses.get((q.scene, q.floor))
    if init_pose is None:
        raise KeyError(f"No init pose for scene={q.scene!r} floor={q.floor}")

    hm3d = hm3d_root or default_hm3d_scene_dir()
    sim = HabitatEQASimulator.from_scene_id(q.scene, hm3d_root=hm3d)
    try:
        sim.set_init_pose(init_pose)
        robot = HabitatRobotClient(sim)
        parameters = get_parameters("dynav_config.yaml")
        agent = _make_controller(
            robot,
            parameters,
            method=method,
            mock_llm=mock_llm,
            gold_letter=q.answer_letter,
            no_rerun=no_rerun,
        )
        agent.start()
        executor = EQAExecuter(agent)
        if rotate_in_place:
            executor.rotate_in_place()
        # Warm-start mapping with a few perception updates
        for _ in range(3):
            agent.update()

        discord_text, _images = agent.run_eqa(q.question_formatted, max_planning_steps=max_planning_steps)
        predicted = discord_text.split("---")[-1].strip() if "---" in discord_text else discord_text
        correct = grade_mcq_answer(predicted, q.answer_letter)

        return EpisodeMetrics(
            dataset="hmeqa",
            method=method,
            question_id=question_id,
            scene=q.scene,
            floor=q.floor,
            question=q.question,
            gold_answer_letter=q.answer_letter,
            predicted_answer=predicted[:200],
            correct=correct,
            confident=correct,
            planning_steps=getattr(agent, "obs_count", 0),
            success=correct,
        )
    finally:
        sim.close()
