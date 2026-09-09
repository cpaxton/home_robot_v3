#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Generate a Habitat find-phase episode config across HM3D train scenes (CPU-only).

Loads each candidate scene's semantic mesh (no rendering / no GPU) and emits a
``habitat_find_phase_episodes`` YAML with one FindObj/FindRec episode per scene
that has a movable object category plus two receptacle categories present. Used
to build an overnight habitat-ovmm failure-case sweep (envs can grow big).

Usage:
  uv run python scripts/generate_habitat_ovmm_episodes.py \
      --max-scenes 120 --out configs/ovmm/habitat_ovmm_overnight.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "packages" / "emet_habitat"))

# HM3D categories we prefer as find targets (movable / instance-dense).
FIND_CATEGORIES = [
    "lamp",
    "chair",
    "table",
    "shelf",
    "cabinet",
    "tv",
    "sofa",
    "stool",
    "plant",
    "vase",
    "bowl",
    "book",
    "bottle",
    "cushion",
    "picture",
    "rack",
    "monitor",
    "clock",
    "toy",
    "bag",
]
# Receptacle categories for start_recep / goal_recep.
RECEPTACLE_CATEGORIES = [
    "table",
    "bed",
    "shelf",
    "cabinet",
    "counter",
    "sofa",
    "stool",
    "chair",
    "desk",
    "dresser",
    "chest",
    "box",
]


def _category_from_key(key: str) -> str:
    """Strip the ``hm3d_<cat>_<idx>`` placement key down to the category name."""
    parts = key.split("_")
    if parts and parts[0] == "hm3d":
        parts = parts[1:]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hm3d-root", default=None)
    parser.add_argument("--init-poses", default=None)
    parser.add_argument("--max-scenes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explore-steps", type=int, default=0, help="Mapping explore budget (env-growth stress)")
    parser.add_argument("--out", default=str(REPO / "configs" / "ovmm" / "habitat_ovmm_overnight.yaml"))
    args = parser.parse_args()

    import habitat_sim
    import habitat_sim.agent

    from emet.habitat.config import default_hm3d_scene_dir
    from emet.habitat.datasets import load_scene_init_poses
    from emet.habitat.hm3d_semantics import (
        hm3d_annotated_scene_dataset_config,
        hm3d_placements_from_semantic_scene,
    )

    hm3d_root = Path(args.hm3d_root) if args.hm3d_root else default_hm3d_scene_dir()
    # default_hm3d_scene_dir() IS the train dir; the data root is three parents up.
    train_root = hm3d_root
    hm3d_data_root = train_root.parent.parent.parent
    init_poses = load_scene_init_poses(args.init_poses)
    annotated = hm3d_annotated_scene_dataset_config(hm3d_data_root, split="train")
    if annotated is None:
        print("no HM3D annotated scene dataset config found", file=sys.stderr)
        return 1

    # Candidate scenes: floor 0, basis glb + semantic glb present (only ~145 train scenes have semantics).
    candidates: list[str] = []
    for scene, floor in init_poses:
        if floor != 0 or scene in candidates:
            continue
        glb = train_root / scene / f"{scene.split('-')[1]}.basis.glb"
        sem_glb = train_root / scene / f"{scene.split('-')[1]}.semantic.glb"
        if glb.is_file() and sem_glb.is_file():
            candidates.append(scene)
    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[: args.max_scenes]

    episodes = []
    skipped: dict[str, int] = {}
    sim = None
    try:
        for scene in candidates:
            glb = train_root / scene / f"{scene.split('-')[1]}.basis.glb"
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = str(glb)
            sim_cfg.enable_physics = False
            sim_cfg.scene_dataset_config_file = str(annotated)
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            if sim is None:
                sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
            else:
                sim.reconfigure(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
            ss = sim.semantic_scene
            if ss is None:
                skipped.setdefault("no_semantic", 0)
                skipped["no_semantic"] += 1
                continue
            placements = hm3d_placements_from_semantic_scene(ss)
            if not placements:
                skipped.setdefault("no_placements", 0)
                skipped["no_placements"] += 1
                continue
            cats = {_category_from_key(k) for k in placements}
            obj = next((c for c in FIND_CATEGORIES if c in cats), None)
            if obj is None:
                skipped.setdefault("no_find_category", 0)
                skipped["no_find_category"] += 1
                continue
            receps = [c for c in RECEPTACLE_CATEGORIES if c in cats and c != obj]
            if len(receps) < 2:
                skipped.setdefault("no_receps", 0)
                skipped["no_receps"] += 1
                continue
            start_recep = receps[0]
            goal_recep = receps[1]
            episodes.append(
                {
                    "id": f"hm3d_{obj}_{scene}",
                    "scene": scene,
                    "floor": 0,
                    "object": obj,
                    "start_recep": start_recep,
                    "goal_recep": goal_recep,
                    "success_radius_m": 0.75,
                    "explore_steps": args.explore_steps,
                }
            )
    finally:
        if sim is not None:
            sim.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Auto-generated Habitat find-phase episodes (overnight failure-case sweep).\n")
        fh.write("# Generated by scripts/generate_habitat_ovmm_episodes.py\n\n")
        fh.write("episodes:\n")
        for ep in episodes:
            fh.write(f"  - id: {ep['id']}\n")
            fh.write(f"    scene: {ep['scene']}\n")
            fh.write(f"    floor: {ep['floor']}\n")
            fh.write(f"    object: {ep['object']}\n")
            fh.write(f"    start_recep: {ep['start_recep']}\n")
            fh.write(f"    goal_recep: {ep['goal_recep']}\n")
            fh.write(f"    success_radius_m: {ep['success_radius_m']}\n")
            fh.write(f"    explore_steps: {ep['explore_steps']}\n")

    report = {
        "generated": len(episodes),
        "scenes_tried": len(candidates),
        "skipped": skipped,
        "out": str(out_path),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
