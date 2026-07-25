# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Figure exporter smoke (no sim / no GPU)."""

from __future__ import annotations

from pathlib import Path

from emet.controller.task.tamp.task_search import TaskPlan, TaskPlanStep
from emet.visualization.manip_figures import write_tamp_figure_bundle


def test_write_tamp_figure_bundle(tmp_path: Path):
    plan = TaskPlan(
        steps=[TaskPlanStep("approach"), TaskPlanStep("grasp"), TaskPlanStep("place")],
        object_body="obj",
        receptacle_body="recep",
        chosen_grasp_index=1,
        expanded_nodes=["goal:on(obj,recep)", "approach@[0,0,0]", "grasp[1] err=0.01 reachable=True", "chosen_grasp=1"],
        success=True,
        message="planned",
    )
    paths = write_tamp_figure_bundle(
        tmp_path,
        plan=plan,
        base_path_xyt=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.05, 0.1]],
        object_xy=[0.25, 0.0],
        receptacle_xy=[0.4, 0.1],
        grasp_xy=[0.26, 0.01],
        planned_ee_xyz=[[0.2, 0.0, 0.9], [0.25, 0.0, 0.85], [0.25, 0.0, 0.95]],
        joint_waypoints=[[0.0, 0.1], [0.1, 0.2], [0.2, 0.15]],
        joint_names=["j1", "j2"],
        targets={"grasp": [0.25, 0.0, 0.85], "lift": [0.25, 0.0, 0.95]},
    )
    assert "topdown" in paths and paths["topdown"].is_file() and paths["topdown"].stat().st_size > 0
    assert paths["ee_path"].is_file() and paths["ee_path"].stat().st_size > 0
    assert paths["joint_traj"].is_file() and paths["joint_traj"].stat().st_size > 0
    assert paths["plan_tree"].is_file() and paths["plan_tree"].stat().st_size > 0
    assert paths["topdown"].with_suffix(".pdf").is_file()
