#!/usr/bin/env python3
"""Generate independent nominal handoff cases; simulation admission is separate."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, load_suite


def generate(*, seed=2026111710, count_per_split=8, reset_seed_start=410400000):
    if (
        type(count_per_split) is not int
        or count_per_split < 4
        or type(reset_seed_start) is not int
        or not 0 <= reset_seed_start < 2**31 - 2 * count_per_split
    ):
        raise ValueError("four or more cases per split and disjoint valid reset seed range required")
    rng = np.random.default_rng(seed)
    suite = load_suite(COARSE_SUITE)
    cases, changes = [], {}
    for i in range(2 * count_per_split):
        split = "train" if i < count_per_split else "audit"
        mode = "hand_free_base" if i % 4 == 3 else "coupled"
        travel = (-1 if i % 2 else 1) * float(rng.uniform(0.12, 0.16)) if mode == "coupled" else 0.0
        turn = (-1 if (i // 2) % 2 else 1) * float(rng.uniform(0.15, 0.3)) if mode == "coupled" else 0.0
        delta = rng.uniform(-0.035, 0.035, 5)
        case = {
            "id": f"innate_mars:goal-change-v1:{split}:{i}",
            "seed": reset_seed_start + i,
            "split": split,
            "mode": mode,
            "travel_m": travel,
            "turn_rad": turn,
            "arm_delta": delta.tolist(),
            "base_mask": [False] * 3 if mode == "hand_free_base" else [True, True, i % 4 != 2],
            "purpose": "tracking",
        }
        cases.append(case)
        changes[case["id"]] = {
            "travel_m": -travel * float(rng.uniform(0.8, 1.05)),
            "turn_rad": -turn * float(rng.uniform(0.8, 1.05)),
            "arm_delta": (-delta * rng.uniform(0.75, 1.05) + rng.uniform(-0.005, 0.005, 5)).tolist(),
            "dwell_s": float(rng.uniform(0.05, 0.6)),
        }
    suite["robots"]["innate_mars"] = cases
    suite.update(
        curriculum_format="nominal-goal-change-v1",
        curriculum_robot="innate_mars",
        goal_changes=changes,
        training_mix={"general": 0.5, "tamp": 0.5, "within_general": {"hand": 0.6, "base": 0.2, "goal_changes": 0.2}},
        generator={"seed": seed, "count_per_split": count_per_split, "reset_seed_start": reset_seed_start},
        scope="Mars-only training curriculum. Stretch entries preserve base schema validation, not additional training. Both nominal stages require independent geometry/support admission.",
    )
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026111710)
    parser.add_argument("--count-per-split", type=int, default=8)
    parser.add_argument("--reset-seed-start", type=int, default=410400000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path; do not overwrite a frozen curriculum")
    suite = generate(seed=args.seed, count_per_split=args.count_per_split, reset_seed_start=args.reset_seed_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2) + "\n")
    load_suite(args.output)


if __name__ == "__main__":
    main()
