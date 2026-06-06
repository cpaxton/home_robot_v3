# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Habitat HM3D find-phase episodes (FindObj / FindRec) for OVMM-style memory benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from emet.core.parameters import get_parameters
from emet.eval.ovmm_find_phase import (
    FindPhaseEpisode,
    FindPhaseRunConfig,
    apply_backend_parameters,
    collect_scaling_diagnostics,
    compute_find_phase_metrics,
    create_find_phase_agent,
    get_memory_backend_for_agent,
    query_find_phase_localization,
    resolve_object_query,
    run_mapping_protocol,
)
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import load_scene_init_poses
from emet.habitat.hm3d_semantics import hm3d_placements_from_semantic_scene
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.runner import _release_gpu_memory
from emet_habitat.simulator import HabitatEQASimulator

MemoryBackendName = Literal["dynamem", "graph_eqa", "dynagraph", "ground_truth"]


@dataclass(frozen=True)
class HabitatFindPhaseEpisode:
    """One Habitat find-phase episode (HM3D scene + category queries)."""

    id: str
    scene: str
    floor: int
    object: str
    start_recep: str
    goal_recep: str
    success_radius_m: float = 0.75
    explore_steps: int = 0
    object_gt_body: str | None = None


def load_habitat_find_phase_episodes(path: str | Path) -> list[HabitatFindPhaseEpisode]:
    """Load episodes from ``configs/ovmm/habitat_find_phase_episodes.yaml``."""
    full = Path(path).expanduser().resolve()
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    rows = raw.get("episodes") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"expected list under 'episodes' in {full}")
    out: list[HabitatFindPhaseEpisode] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            HabitatFindPhaseEpisode(
                id=str(row["id"]),
                scene=str(row["scene"]),
                floor=int(row.get("floor", 0)),
                object=str(row["object"]),
                start_recep=str(row["start_recep"]),
                goal_recep=str(row["goal_recep"]),
                success_radius_m=float(row.get("success_radius_m", 0.75)),
                explore_steps=int(row.get("explore_steps", 0)),
                object_gt_body=(str(row["object_gt_body"]) if row.get("object_gt_body") else None),
            )
        )
    return out


def run_habitat_find_phase_episode(
    episode: HabitatFindPhaseEpisode,
    run_cfg: FindPhaseRunConfig,
    *,
    hm3d_root: Path | None = None,
    init_poses_path: Path | None = None,
    use_hm3d_semantics: bool | None = True,
    device: str | None = "cpu",
) -> dict[str, Any]:
    """Run one Habitat find-phase episode with emet memory backends."""
    poses = load_scene_init_poses(init_poses_path)
    init_pose = poses.get((episode.scene, episode.floor))
    if init_pose is None:
        raise KeyError(f"No init pose for scene={episode.scene!r} floor={episode.floor}")

    hm3d = hm3d_root or default_hm3d_scene_dir()
    sim = HabitatEQASimulator.from_scene_id(
        episode.scene,
        hm3d_root=hm3d,
        use_hm3d_semantics=use_hm3d_semantics,
    )
    if not sim.uses_hm3d_semantics:
        sim.close()
        raise RuntimeError(f"HM3D semantics required for find-phase GT: scene {episode.scene}")

    placements = hm3d_placements_from_semantic_scene(sim._sim.semantic_scene)
    if not placements:
        sim.close()
        raise RuntimeError(f"No semantic placements for scene {episode.scene}")

    robot = None
    agent = None
    t0 = time.monotonic()
    try:
        sim.set_init_pose(init_pose)
        robot = HabitatRobotClient(sim)
        from emet.memory.graph_eqa.sim_ground_truth_graph import placements_to_json_dict

        robot.set_emet_session(
            {
                "sim_object_placements": placements_to_json_dict(placements),
                "sim_object_placements_frame": "habitat_yup",
            }
        )

        parameters = apply_backend_parameters(
            get_parameters("dynav_config.yaml"),
            run_cfg.backend,
            merge_xy_m=run_cfg.merge_xy_m,
            staleness_horizon=run_cfg.staleness_horizon,
        )
        parameters["encoder"] = None

        agent = create_find_phase_agent(
            robot,
            parameters,
            run_cfg.backend,
            cpu_only=run_cfg.cpu_only or device == "cpu",
            compare_to_gt=run_cfg.compare_to_gt,
        )
        if run_cfg.backend == "ground_truth":
            refresh = getattr(agent, "refresh_ground_truth", None)
            if callable(refresh):
                n_gt = refresh()
                if n_gt == 0:
                    raise RuntimeError("ground-truth mode: no placements in habitat session")

        n_steps = run_mapping_protocol(
            agent,
            explore_steps=episode.explore_steps,
            not_rotate=run_cfg.not_rotate,
        )
        for _ in range(3):
            agent.update()

        memory = get_memory_backend_for_agent(agent, run_cfg.backend)
        vm = getattr(agent, "voxel_map", None)
        object_query = resolve_object_query(
            FindPhaseEpisode(
                id=episode.id,
                tier="habitat",
                sim=episode.scene,
                object=episode.object,
                start_recep=episode.start_recep,
                goal_recep=episode.goal_recep,
                success_radius_m=episode.success_radius_m,
                object_gt_body=episode.object_gt_body,
            ),
            placements,
        )

        prefer_voxel = run_cfg.backend not in ("ground_truth",)
        obj_xyz, obj_ok, obj_q_used = query_find_phase_localization(
            memory,
            object_query,
            placements=placements,
            near_recep=episode.start_recep,
            voxel_map=vm,
            convert_nav_to_world=False,
            prefer_voxel=prefer_voxel,
            planar_frame="habitat_xz",
        )
        recep_xyz, recep_ok, recep_q_used = query_find_phase_localization(
            memory,
            episode.goal_recep,
            placements=placements,
            near_recep=episode.goal_recep,
            voxel_map=vm,
            convert_nav_to_world=False,
            prefer_voxel=prefer_voxel,
            planar_frame="habitat_xz",
        )

        find_metrics = compute_find_phase_metrics(
            obj_pred_xyz=obj_xyz,
            recep_pred_xyz=recep_xyz,
            placements=placements,
            object_query=object_query,
            start_recep=episode.start_recep,
            goal_recep=episode.goal_recep,
            radius_m=episode.success_radius_m,
            object_gt_body=episode.object_gt_body,
            frame="habitat_xz",
        )
        scaling = collect_scaling_diagnostics(
            agent,
            placements,
            episode_wall_s=time.monotonic() - t0,
            n_controller_steps=n_steps,
        )
        return {
            "episode_id": episode.id,
            "scene": episode.scene,
            "floor": episode.floor,
            "backend": run_cfg.backend,
            "dataset": "habitat_hm3d",
            "object_query": object_query,
            "start_recep": episode.start_recep,
            "goal_recep": episode.goal_recep,
            "explore_steps": episode.explore_steps,
            "merge_xy_m": parameters.get("dynagraph_merge_xy_m"),
            "staleness_horizon": parameters.get("dynagraph_staleness_horizon"),
            "obj_localize_success": bool(obj_ok),
            "recep_localize_success": bool(recep_ok),
            "obj_query_used": obj_q_used,
            "recep_query_used": recep_q_used,
            **find_metrics,
            **scaling,
        }
    finally:
        if agent is not None:
            try:
                agent.stop()
            except Exception:
                pass
        sim.close()
        _release_gpu_memory()
