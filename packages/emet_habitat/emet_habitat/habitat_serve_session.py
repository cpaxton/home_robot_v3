# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Resolve HM3D scene + spawn pose for ``emet-habitat serve``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emet.habitat.config import default_hm3d_scene_dir, hm3d_scene_glb_path
from emet.habitat.datasets import SceneInitPose, get_question, load_hmeqa_questions, load_scene_init_poses
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.simulator import HabitatEQASimulator


@dataclass(frozen=True)
class HabitatServeConfig:
    """Inputs for opening one interactive Habitat ZMQ session."""

    scene_id: str
    init_pose: SceneInitPose
    hm3d_root: Path
    use_hm3d_semantics: bool | None = None
    question_id: int | None = None


def _default_spawn_pose(scene_id: str, sim: HabitatEQASimulator) -> SceneInitPose:
    """Use the simulator default agent pose when no HM-EQA init CSV row exists."""
    state = sim._agent.get_state()
    pos = state.position
    return SceneInitPose(
        scene=scene_id,
        floor=0,
        x=float(pos[0]),
        y=float(pos[1]),
        z=float(pos[2]),
        heading=0.0,
    )


def resolve_habitat_serve_config(
    *,
    question_id: int | None = None,
    scene_id: str | None = None,
    floor: int = 0,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
    use_hm3d_semantics: bool | None = None,
) -> HabitatServeConfig:
    """Resolve scene + spawn for ``emet-habitat serve``.

    Either *question_id* (HM-EQA row + init pose CSV) or *scene_id* (free play) is required.
    """
    hm3d = hm3d_root or default_hm3d_scene_dir()
    if question_id is not None:
        questions = load_hmeqa_questions(questions_path)
        q = get_question(questions, question_id=question_id)
        poses = load_scene_init_poses(init_poses_path)
        init_pose = poses.get((q.scene, q.floor))
        if init_pose is None:
            raise KeyError(f"No init pose for scene={q.scene!r} floor={q.floor}")
        sid = q.scene
        qid = int(question_id)
    elif scene_id is not None and str(scene_id).strip():
        sid = str(scene_id).strip()
        glb = hm3d_scene_glb_path(sid, hm3d)
        if not glb.is_file():
            raise FileNotFoundError(f"HM3D scene not found: {glb}")
        poses = load_scene_init_poses(init_poses_path)
        init_pose = poses.get((sid, int(floor)))
        if init_pose is None:
            sim_probe = HabitatEQASimulator.from_scene_id(sid, hm3d_root=hm3d, use_hm3d_semantics=False)
            try:
                init_pose = _default_spawn_pose(sid, sim_probe)
            finally:
                sim_probe.close()
        qid = None
    else:
        raise ValueError("Provide --question-id or --scene-id for emet serve habitat.")

    return HabitatServeConfig(
        scene_id=sid,
        init_pose=init_pose,
        hm3d_root=hm3d,
        use_hm3d_semantics=use_hm3d_semantics,
        question_id=qid,
    )


def open_habitat_robot_for_serve(cfg: HabitatServeConfig) -> tuple[HabitatRobotClient, HabitatEQASimulator]:
    """Open Habitat-Sim, spawn the agent, and return (robot client, simulator)."""
    sim = HabitatEQASimulator.from_scene_id(
        cfg.scene_id,
        hm3d_root=cfg.hm3d_root,
        use_hm3d_semantics=cfg.use_hm3d_semantics,
    )
    sim.set_init_pose(cfg.init_pose)
    robot = HabitatRobotClient(sim)
    return robot, sim
