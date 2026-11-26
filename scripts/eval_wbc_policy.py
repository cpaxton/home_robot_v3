# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Gate one WBC checkpoint across evaluation seeds.

Writes one JSON report per seed plus a manifest with the profile fingerprint and
the mean aggregate. This is the reproducible held-out gate used for the
multi-robot artifacts; it never enables hard-case resets, so it measures the
clean task suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.controller.manipulation.wbc_robots import get_wbc_robot, wbc_profile_for
from emet.rl.manipulation.wbc_dagger import WbcTaskSuiteEnv, evaluate_suite
from emet.rl.manipulation.wbc_training import WbcPPO


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be a non-empty list of unique integers")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a WBC checkpoint across held-out seeds")
    parser.add_argument("--robot", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=_parse_seeds, default=(30000, 40000, 50000))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    robot = get_wbc_robot(args.robot)
    profile = wbc_profile_for(args.robot)
    trainer, metadata = WbcPPO.load(args.checkpoint, profile, device=args.device)
    env = WbcTaskSuiteEnv(robot=robot.name, seed=0)
    if trainer.policy.observation_size != env.observation_space.shape[0]:
        raise ValueError("checkpoint observation size does not match the evaluation environment")
    if trainer.policy.action_size != env.action_space.shape[0]:
        raise ValueError("checkpoint action size does not match the evaluation environment")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_results: dict[str, dict] = {}
    aggregates: list[float] = []
    try:
        for seed in args.seeds:
            result = evaluate_suite(env, trainer.policy, episodes_per_task=args.episodes, seed=seed)
            path = args.output / f"heldout_{args.episodes}_seed{seed}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            seed_results[str(seed)] = result
            aggregates.append(float(result["aggregate"]["success_rate"]))
            tasks = {name: value["success_rate"] for name, value in result.items() if name != "aggregate"}
            print(f"seed={seed} aggregate={result['aggregate']['success_rate']:.4f} tasks={tasks}", flush=True)
    finally:
        env.close()

    manifest = {
        "robot": robot.name,
        "checkpoint": str(args.checkpoint.expanduser()),
        "checkpoint_stage": (metadata or {}).get("stage"),
        "profile": profile.name,
        "profile_fingerprint": profile.fingerprint,
        "episodes_per_task": int(args.episodes),
        "seeds": list(args.seeds),
        "mean_aggregate": sum(aggregates) / len(aggregates),
        "min_aggregate": min(aggregates),
        "seed_results": seed_results,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"mean_aggregate={manifest['mean_aggregate']:.4f} min_aggregate={manifest['min_aggregate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
