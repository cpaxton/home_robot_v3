# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Habitat HM3D find-phase episodes (FindObj / FindRec) for OVMM-style memory benchmarks."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from emet.core.parameters import get_parameters
from emet.eval.benchmark_dynagraph import apply_habitat_ovmm_find_parameters
from emet.eval.episode_diagnostics import (
    EpisodeDiagnosticsConfig,
    EpisodeDiagnosticsRecorder,
    bind_diagnostics_recorder,
    flush_episode_diagnostics,
    habitat_export_voxel_history_default,
    unbind_diagnostics_recorder,
)
from emet.eval.ovmm_agentic_find import should_use_agentic_find
from emet.eval.ovmm_find_phase import (
    FindPhaseEpisode,
    FindPhaseRunConfig,
    collect_scaling_diagnostics,
    compute_find_phase_metrics,
    create_find_phase_agent,
    get_memory_backend_for_agent,
    localization_pred_fields,
    mapping_budget_from_row,
    query_find_phase_localization,
    resolve_mapping_max_nav_steps,
    resolve_object_query,
    run_mapping_protocol,
    set_find_phase_run_seed,
)
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import load_scene_init_poses
from emet.habitat.episode_debug import default_episodes_root
from emet.habitat.hm3d_semantics import hm3d_placements_from_semantic_scene
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.runner import _release_gpu_memory
from emet_habitat.simulator import HabitatEQASimulator

MemoryBackendName = Literal["dynamem", "static_graph", "dynagraph", "ground_truth"]


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
    mapping_max_nav_steps: int | None = None
    explore_steps: int | None = None
    object_gt_body: str | None = None

    def __post_init__(self) -> None:
        n = resolve_mapping_max_nav_steps(
            self.mapping_max_nav_steps,
            self.explore_steps,
            source="HabitatFindPhaseEpisode",
            default=0,
            warn=False,
        )
        object.__setattr__(self, "mapping_max_nav_steps", int(n or 0))
        object.__setattr__(self, "explore_steps", int(n or 0))


def load_habitat_find_phase_episodes(path: str | Path) -> list[HabitatFindPhaseEpisode]:
    """Load episodes from ``configs/ovmm/habitat_find_phase_episodes.yaml``."""
    full = Path(path).expanduser().resolve()
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    rows = raw.get("episodes") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"expected list under 'episodes' in {full}")
    out: list[HabitatFindPhaseEpisode] = []
    n_alias = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("explore_steps") is not None and row.get("mapping_max_nav_steps") is None:
            n_alias += 1
        budget = mapping_budget_from_row(row, source=str(full), default=0, warn=False)
        out.append(
            HabitatFindPhaseEpisode(
                id=str(row["id"]),
                scene=str(row["scene"]),
                floor=int(row.get("floor", 0)),
                object=str(row["object"]),
                start_recep=str(row["start_recep"]),
                goal_recep=str(row["goal_recep"]),
                success_radius_m=float(row.get("success_radius_m", 0.75)),
                mapping_max_nav_steps=budget,
                explore_steps=budget,
                object_gt_body=(str(row["object_gt_body"]) if row.get("object_gt_body") else None),
            )
        )
    if n_alias:
        from emet.eval.ovmm_find_phase import _warn_explore_steps_alias

        _warn_explore_steps_alias(f"{full} ({n_alias} episode(s))")
    return out


def run_habitat_find_phase_episode(
    episode: HabitatFindPhaseEpisode,
    run_cfg: FindPhaseRunConfig,
    *,
    hm3d_root: Path | None = None,
    init_poses_path: Path | None = None,
    use_hm3d_semantics: bool | None = True,
    device: str | None = "cuda",
    debug_run_tag: str | None = None,
    export_map: bool | None = None,
    export_video: bool | None = None,
) -> dict[str, Any]:
    """Run one Habitat find-phase episode with emet memory backends."""
    if run_cfg.seed is not None:
        set_find_phase_run_seed(int(run_cfg.seed))

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
    diag_recorder: EpisodeDiagnosticsRecorder | None = None
    t0 = time.monotonic()
    init_wall_s = 0.0
    mapping_wall_s = 0.0
    query_wall_s = 0.0
    try:
        sim.set_init_pose(init_pose)
        spawn_record = sim.last_init_pose_record
        robot = HabitatRobotClient(sim)
        from emet.memory.graph_eqa.sim_ground_truth_graph import placements_to_json_dict

        robot.set_emet_session(
            {
                "sim_object_placements": placements_to_json_dict(placements),
                "sim_object_placements_frame": "habitat_yup",
            }
        )

        parameters = apply_habitat_ovmm_find_parameters(
            get_parameters("dynav_config.yaml"),
            run_cfg.backend,
            merge_xy_m=run_cfg.merge_xy_m,
            staleness_horizon=run_cfg.staleness_horizon,
        )
        parameters["encoder"] = None
        use_agentic = should_use_agentic_find(run_cfg.backend, agentic_find=run_cfg.agentic_find)
        if use_agentic:
            # The agentic loop explores via habitat navmesh frontiers (like HM-EQA).
            # Default dynav clamps (max_depth=2.5, pad_obstacles=2, smooth_kernel_size=3)
            # seal thin corridors and hug a ~2.5 m ring — wrong for whole-home OVMM finds.
            from emet_habitat.runner import _configure_habitat_mapping, _configure_habitat_nav

            _configure_habitat_mapping(parameters)
            _configure_habitat_nav(parameters)
            # Agentic find loads the shared Qwen3-VL for the tool router / keyword
            # extraction. The habitat torch (2.12/2.13+cu130) has no flash-attn wheel,
            # so permit PyTorch SDPA like hmeqa_launch / the OVMM VL worker do.
            os.environ.setdefault("EMET_ALLOW_SDPA_ATTN", "1")

        t_init0 = time.monotonic()
        agent = create_find_phase_agent(
            robot,
            parameters,
            run_cfg.backend,
            cpu_only=run_cfg.cpu_only or device == "cpu",
            compare_to_gt=run_cfg.compare_to_gt,
            use_sensor_perception=run_cfg.use_sensor_perception,
        )
        diag_cfg = EpisodeDiagnosticsConfig.from_env(
            parameters,
            export_map=export_map,
            export_video=export_video,
            export_voxel_history=habitat_export_voxel_history_default(),
        )
        diag_recorder = EpisodeDiagnosticsRecorder(cfg=diag_cfg)
        bind_diagnostics_recorder(
            agent,
            diag_recorder,
            spawn_record=spawn_record,
            habitat_pathfinder=sim.pathfinder,
            habitat_floor_y=sim.floor_y,
        )
        init_wall_s = time.monotonic() - t_init0
        if run_cfg.backend == "ground_truth":
            refresh = getattr(agent, "refresh_ground_truth", None)
            if callable(refresh):
                n_gt = refresh()
                if n_gt == 0:
                    raise RuntimeError("ground-truth mode: no placements in habitat session")

        t_map0 = time.monotonic()
        n_steps = run_mapping_protocol(
            agent,
            mapping_max_nav_steps=episode.mapping_max_nav_steps,
            not_rotate=run_cfg.not_rotate,
        )
        for _ in range(3):
            agent.update()
        mapping_wall_s = time.monotonic() - t_map0

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

        prefer_voxel = run_cfg.prefer_voxel and run_cfg.backend != "ground_truth"
        agentic_meta: dict[str, Any] = {
            "agentic_find": bool(use_agentic),
            "obj_agentic_question": None,
            "recep_agentic_question": None,
            "obj_n_retracted_claims": 0,
            "recep_n_retracted_claims": 0,
        }
        obj_xyz = None
        obj_ok = False
        obj_q_used = object_query
        obj_source: str | None = None
        recep_xyz = None
        recep_ok = False
        recep_q_used = episode.goal_recep
        recep_source: str | None = None

        t_query0 = time.monotonic()
        if use_agentic:
            # Same AgenticEQAExecutor loop as HM-EQA / sim OVMM find: phrase FindObj
            # and FindRec as open questions, navigate/verify/explore inside the shared
            # loop, then score the loop's object-phrase voxel XYZ (never camera pose).
            from emet.eval.ovmm_agentic_find import (
                ovmm_find_object_question,
                ovmm_find_recep_question,
                run_ovmm_agentic_localize,
            )

            obj_q = ovmm_find_object_question(object_query, episode.start_recep)
            recep_q = ovmm_find_recep_question(episode.goal_recep)
            agentic_meta["obj_agentic_question"] = obj_q
            agentic_meta["recep_agentic_question"] = recep_q
            obj_res = run_ovmm_agentic_localize(
                agent,
                obj_q,
                max_rounds=run_cfg.agentic_max_rounds,
                max_nav_steps=run_cfg.agentic_max_nav_steps,
                require_verified=True,
                trace_meta={
                    "ovmm_phase": "find_object",
                    "episode_id": episode.id,
                    "scene": episode.scene,
                    "object": object_query,
                    "start_recep": episode.start_recep,
                    "goal_recep": episode.goal_recep,
                    "gt_body_key": episode.object_gt_body,
                },
            )
            if obj_res.error:
                agentic_meta["agentic_find_error"] = obj_res.error
            obj_xyz = obj_res.xyz
            obj_ok = obj_xyz is not None
            obj_q_used = object_query
            obj_source = (obj_res.extra or {}).get("xyz_source")
            if not obj_source and obj_ok:
                obj_source = "agentic_verify" if obj_res.verified else None
            agentic_meta["obj_n_retracted_claims"] = obj_res.n_retracted_claims
            agentic_meta["obj_agentic_rounds"] = obj_res.n_rounds
            agentic_meta["obj_verified_obs_id"] = obj_res.verified_obs_id
            if obj_res.extra:
                agentic_meta["obj_xyz_source"] = obj_res.extra.get("xyz_source")
                agentic_meta["obj_from_pin"] = obj_res.extra.get("from_pin")
                if obj_res.extra.get("voxel_query_used"):
                    obj_q_used = str(obj_res.extra["voxel_query_used"])

            recep_res = run_ovmm_agentic_localize(
                agent,
                recep_q,
                max_rounds=run_cfg.agentic_max_rounds,
                max_nav_steps=run_cfg.agentic_max_nav_steps,
                require_verified=True,
                trace_meta={
                    "ovmm_phase": "find_recep",
                    "episode_id": episode.id,
                    "scene": episode.scene,
                    "object": object_query,
                    "start_recep": episode.start_recep,
                    "goal_recep": episode.goal_recep,
                },
            )
            if recep_res.error:
                agentic_meta["agentic_find_error_recep"] = recep_res.error
            recep_xyz = recep_res.xyz
            recep_ok = recep_xyz is not None
            recep_q_used = episode.goal_recep
            recep_source = (recep_res.extra or {}).get("xyz_source")
            if not recep_source and recep_ok:
                recep_source = "agentic_verify" if recep_res.verified else None
            agentic_meta["recep_n_retracted_claims"] = recep_res.n_retracted_claims
            agentic_meta["recep_agentic_rounds"] = recep_res.n_rounds
            agentic_meta["recep_verified_obs_id"] = recep_res.verified_obs_id
            if recep_res.extra:
                agentic_meta["recep_xyz_source"] = recep_res.extra.get("xyz_source")
                agentic_meta["recep_from_pin"] = recep_res.extra.get("from_pin")
                if recep_res.extra.get("voxel_query_used"):
                    recep_q_used = str(recep_res.extra["voxel_query_used"])
        else:
            # Ablation: one-shot memory localize (voxel-first when prefer_voxel).
            obj_xyz, obj_ok, obj_q_used, obj_source = query_find_phase_localization(
                memory,
                object_query,
                placements=placements,
                near_recep=episode.start_recep,
                voxel_map=vm,
                convert_nav_to_world=False,
                prefer_voxel=prefer_voxel,
                planar_frame="habitat_xz",
            )
            recep_xyz, recep_ok, recep_q_used, recep_source = query_find_phase_localization(
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
        query_wall_s = time.monotonic() - t_query0
        scaling = collect_scaling_diagnostics(
            agent,
            placements,
            episode_wall_s=time.monotonic() - t0,
            n_controller_steps=n_steps,
        )
        result = {
            "episode_id": episode.id,
            "scene": episode.scene,
            "floor": episode.floor,
            "backend": run_cfg.backend,
            "dataset": "habitat_hm3d",
            "object_query": object_query,
            "start_recep": episode.start_recep,
            "goal_recep": episode.goal_recep,
            "mapping_max_nav_steps": episode.mapping_max_nav_steps,
            "explore_steps": episode.explore_steps,
            "merge_xy_m": parameters.get("dynagraph_merge_xy_m"),
            "staleness_horizon": parameters.get("dynagraph_staleness_horizon"),
            "use_sensor_perception": bool(run_cfg.use_sensor_perception),
            "prefer_voxel": bool(prefer_voxel),
            "init_wall_s": float(init_wall_s),
            "mapping_wall_s": float(mapping_wall_s),
            "query_wall_s": float(query_wall_s),
            "obj_localize_success": bool(obj_ok),
            "recep_localize_success": bool(recep_ok),
            "obj_query_used": obj_q_used,
            "recep_query_used": recep_q_used,
            "obj_localize_source": obj_source,
            "recep_localize_source": recep_source,
            "seed": run_cfg.seed,
            **agentic_meta,
            **localization_pred_fields(obj_xyz, recep_xyz),
            **find_metrics,
            **scaling,
        }
        if debug_run_tag:
            bundle_dir = default_episodes_root() / debug_run_tag / f"ovmm_{episode.id}_{run_cfg.backend}"
            manifest = flush_episode_diagnostics(bundle_dir, agent, diag_recorder)
            result["debug_bundle_dir"] = str(bundle_dir)
            if manifest.get("topdown_map"):
                result["topdown_map_path"] = manifest["topdown_map"]
            if manifest.get("diagnostics_manifest"):
                result["diagnostics_manifest_path"] = manifest["diagnostics_manifest"]
            (bundle_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if agent is not None and diag_recorder is not None:
            unbind_diagnostics_recorder(agent, diag_recorder)
        if agent is not None:
            try:
                agent.stop()
            except Exception:
                pass
        sim.close()
        _release_gpu_memory()
