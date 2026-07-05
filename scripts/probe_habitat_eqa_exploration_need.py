#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Probe HM-EQA questions for spin-only vs needs-navigation behavior.

Runs a short Habitat episode per question: initial look-around only (no EQA nav),
then one mocked confident EQA response. Questions that become confident immediately
are poor exploration demos; prefer those marked ``needs_navigation``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.core.parameters import get_parameters
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.runner import _apply_method_parameters, _configure_habitat_nav, _make_controller
from emet_habitat.simulator import HabitatEQASimulator


def probe_question(question_id: int, *, method: str = "dynagraph") -> dict:
    questions = load_hmeqa_questions(None)
    q = get_question(questions, question_id=question_id)
    init_pose = load_scene_init_poses(None)[(q.scene, q.floor)]
    sim = HabitatEQASimulator.from_scene_id(
        q.scene,
        hm3d_root=default_hm3d_scene_dir(),
        use_hm3d_semantics=True,
    )
    sim.set_init_pose(init_pose)
    robot = HabitatRobotClient(sim)
    parameters = get_parameters("dynav_config.yaml")
    _configure_habitat_nav(parameters)
    agent = _make_controller(
        robot,
        _apply_method_parameters(parameters, method),
        method=method,
        mock_llm=True,
        mock_llm_explore=False,
        gold_letter=q.answer_letter,
        no_rerun=True,
        use_real_vlm=False,
        device="cpu",
        use_hm3d_semantics=True,
    )
    try:
        agent.start()
        if agent.graph_memory is not None:
            agent.graph_memory.extract_relevant_objects(q.question_formatted)
        agent.look_around()
        for _ in range(3):
            agent.update()
        gm = agent.graph_memory
        n_nodes = len(gm.get_nodes()) if gm else 0
        n_frontier = sum(1 for n in gm.get_nodes() if getattr(n, "is_frontier", False)) if gm else 0
        covered = bool(gm._graph_covers_relevant_objects()) if gm else False
        pose = robot.get_base_pose()
        return {
            "question_id": question_id,
            "scene": q.scene,
            "graph_covers_relevant_after_spin": covered,
            "graph_nodes": n_nodes,
            "frontier_nodes": n_frontier,
            "spawn_xy": [float(pose[0]), float(pose[1])],
            "needs_navigation": not covered,
            "question": q.question_formatted[:120],
        }
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-ids",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
        help="Comma-separated HM-EQA question ids",
    )
    parser.add_argument("--method", default="dynagraph", choices=["dynagraph", "graph_eqa"])
    parser.add_argument("--json", action="store_true", help="Print JSON array")
    args = parser.parse_args()
    ids = [int(x.strip()) for x in args.question_ids.split(",") if x.strip()]
    rows = [probe_question(qid, method=args.method) for qid in ids]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    needs = [r for r in rows if r["needs_navigation"]]
    immediate = [r for r in rows if not r["needs_navigation"]]
    print(f"Probed {len(rows)} questions ({args.method})")
    print(f"\nNeeds navigation ({len(needs)}) — good exploration demos:")
    for r in needs:
        print(
            f"  Q{r['question_id']:02d} scene={r['scene']} "
            f"nodes={r['graph_nodes']} frontiers={r['frontier_nodes']}"
        )
    print(f"\nConfident graph coverage after spin ({len(immediate)}) — poor movement demos:")
    for r in immediate:
        print(f"  Q{r['question_id']:02d} scene={r['scene']} nodes={r['graph_nodes']}")


if __name__ == "__main__":
    main()
