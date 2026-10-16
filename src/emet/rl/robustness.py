"""Matched base-policy stress tests: python -m emet.rl.robustness --help."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.__main__ import evaluate
from emet.rl.env import EnvConfig
from emet.rl.policy import PPOPolicy
from emet.rl.training import evaluation_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", action="append", default=[], help="NAME=BUNDLE (repeatable); teacher is always included"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--no-teacher", action="store_true", help="Skip the conventional baseline")
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.episodes > 1000:
        parser.error("episodes must be between 1 and 1000")
    policies = {} if args.no_teacher else {"teacher": None}
    for value in args.policy:
        name, path = value.split("=", 1)
        if not name or name in policies or Path(name).name != name:
            parser.error("policy names must be unique simple filenames")
        policies[name] = PPOPolicy(path)
    if not policies:
        parser.error("at least one policy is required")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "episodes_per_scenario": args.episodes, "policies": {}}
    for name, policy in policies.items():
        summary["policies"][name] = {}
        for index, scenario in enumerate(("standard", "retarget", "push", "contact_retreat", "approach_retreat")):
            config = EnvConfig(
                task="base",
                base_scenario="contact_retreat" if scenario == "approach_retreat" else scenario,
                contact_training=scenario in {"contact_retreat", "approach_retreat"},
                horizon_seconds=10 if scenario == "standard" else 15,
            )
            # Matched physical contact history: the teacher approaches the box,
            # then the tested policy receives the retreat goal. Avoiding the
            # box in a free rollout is not evidence of recovery after contact.
            report = evaluate(
                policy,
                config,
                episodes=args.episodes,
                seed=args.seed + index * 1000,
                teacher_approach=scenario == "contact_retreat",
            )
            (directory / f"{name}-{scenario}.json").write_text(json.dumps(report, indent=2) + "\n")
            metrics = evaluation_metrics(report)
            contacted = [r for r in report["episodes"] if r["peak_contact_force"] > 1]
            metrics["contacted_episodes"] = len(contacted)
            metrics["contact_recovery_rate"] = (
                float(np.mean([r["is_success"] for r in contacted])) if contacted else None
            )
            handed_off = [r for r in contacted if r.get("event_count", 0) > 0]
            metrics["contact_handoff_count"] = len(handed_off)
            metrics["contact_handoff_success_rate"] = (
                float(np.mean([r["is_success"] for r in handed_off])) if handed_off else None
            )
            metrics["force_cutoffs_before_handoff"] = sum(
                r["unsafe_contact"] and r.get("event_count", 0) == 0 for r in report["episodes"]
            )
            metrics["worst_peak_contact_force"] = max(r["peak_contact_force"] for r in report["episodes"])
            metrics["mean_final_contact_force"] = float(np.mean([r["final_contact_force"] for r in report["episodes"]]))
            metrics["mean_post_event_peak_contact_force"] = float(
                np.mean([r["post_event_peak_contact_force"] for r in report["episodes"]])
            )
            metrics["mean_approach_distance"] = float(
                np.mean([r.get("approach_distance", 0) for r in report["episodes"]])
            )
            summary["policies"][name][scenario] = metrics
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(name, scenario, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
