#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Generate 200 deterministic construction requests for scene-validated TAMP controls.

Writes ``configs/ovmm/clutter_candidates.yaml`` by default. These requests are
not scored tasks: actual furniture, receptacles, clutter layouts and executed
reference witnesses must be resolved by ``run_tamp_experiments.py
--validate-fixtures``. That stage emits a separate admitted registry for fresh
MCTS evaluation. The historical ``clutter_episodes_large.yaml`` is preserved.

The matrix covers RBY1 scenes 0..21 (latch) and Stretch/Mars/Nori scenes 0..5
(oracle teleport), with two cleanup and three navigation construction requests
per robot/scene. Navigation targets are selected from actual scene furniture.
Missing assets, missing receptacles, construction failures and reference failures
remain visible as coverage gaps, not TAMP planner failures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from emet.eval.tamp_clutter import NAV_GOAL_LANDMARKS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "configs" / "ovmm" / "clutter_candidates.yaml"

# (robot, base sim config, max scene index)
ROBOT_SCENES = {
    "rby1": ("configs/sim/molmospaces_ithor_train_0.yaml", 29),
    "stretch": ("configs/sim/molmospaces_ithor_train_stretch_0.yaml", 5),
    "innate_mars": ("configs/sim/molmospaces_ithor_train_innate_mars_0.yaml", 5),
    "nori": ("configs/sim/molmospaces_ithor_train_nori_0.yaml", 5),
}

# Floor clutter: nori/mars latch cannot reliably reach z≈0.02 m.
ROBOT_MANIP_MODE = {
    "rby1": "latch",
    "stretch": "sim",
    "innate_mars": "sim",
    "nori": "sim",
}

# Per scene: (mode, n_objects, landmark_or_None, scatter_radius_m, tight_ring)
EPISODE_TEMPLATES = (
    ("cleanup", 3, None, 0.8, False),
    ("cleanup", 6, None, 1.1, False),
    ("nav_goal", 8, None, 0.5, True),
    ("nav_goal", 8, None, 0.5, True),
    ("nav_goal", 8, "auto", 0.5, True),
)


def generate(robots: dict[str, tuple[str, int]], templates: tuple[tuple, ...]) -> list[dict]:
    episodes: list[dict] = []
    for robot, (sim, max_index) in robots.items():
        n_scenes = max_index + 1
        manip = ROBOT_MANIP_MODE[robot]
        for scene_index in range(n_scenes):
            for t_i, (mode, n_objects, landmark, radius, tight_ring) in enumerate(templates):
                # Rotate landmarks across the set; deterministic seed per scene/template.
                if mode == "nav_goal" and landmark is None:
                    landmark = NAV_GOAL_LANDMARKS[(scene_index + t_i) % len(NAV_GOAL_LANDMARKS)]
                ep_id = f"ithor_{mode}_{scene_index:02d}_{robot}_n{n_objects}"
                if landmark and landmark != "auto":
                    ep_id += f"_{landmark.lower()}"
                if t_i >= 2:
                    ep_id += f"_{t_i}"
                seed = (scene_index * 97 + t_i * 31) % (2**31)
                ep = {
                    "id": ep_id,
                    "tier": "S2" if scene_index >= 2 else "S1",
                    "sim": sim,
                    "robot": robot,
                    "scene_index": scene_index,
                    "mode": mode,
                    "n_objects": n_objects,
                    "scatter_radius_m": radius,
                    "success_radius_m": 0.6 if mode == "nav_goal" else 0.5,
                    "seed": seed,
                    "tight_ring": bool(tight_ring),
                    "manip_mode": manip,
                }
                # MolmoSpaces assigns FloorPlans 1..12 to train, 13..24 to
                # validation, and 25..30 to test. Preserve the exact house ID
                # while explicitly overriding the base YAML's training split.
                if scene_index >= 12:
                    ep["scene_split"] = "val" if scene_index < 24 else "test"
                if mode == "cleanup":
                    ep["bin_query"] = "GarbageCan"
                else:
                    ep["goal_landmark"] = landmark
                    ep["bin_query"] = "GarbageCan"
                episodes.append(ep)
    return episodes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--rby1-scenes", type=int, default=22, choices=range(1, 31), help="Number of iTHOR FloorPlans for rby1 (1..30)")
    args = p.parse_args()

    robots = dict(ROBOT_SCENES)
    base_sim, _ = robots["rby1"]
    robots["rby1"] = (base_sim, int(args.rby1_scenes) - 1)

    episodes = generate(robots, EPISODE_TEMPLATES)
    # These are construction requests, not scored tasks. Resolve actual furniture,
    # clutter poses and an executed reference witness before admission.
    for slot, episode in enumerate(episodes):
        episode["requires_fixture"] = True
        episode["id"] = f"tamp_v2_{episode['robot']}_scene{episode['scene_index']:02}_{episode['mode']}_{slot % 5}"
        if episode["mode"] == "nav_goal":
            episode["goal_landmark"] = "auto"
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# UNVALIDATED construction candidates. Run run_tamp_experiments.py --validate-fixtures before scoring.\n"
        + yaml.safe_dump({"episodes": episodes}, sort_keys=False),
        encoding="utf-8",
    )
    from emet.eval.tamp_clutter import load_clutter_episodes

    n = len(load_clutter_episodes(out))
    robots_seen = sorted({e["robot"] for e in episodes})
    modes = {e["mode"] for e in episodes}
    print(f"Wrote {n} episodes to {out} (robots={robots_seen}, modes={sorted(modes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
