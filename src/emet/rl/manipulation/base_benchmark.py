"""Fresh-seed, stratified base benchmark; thresholds match the saved contract.

Freeze the suite before scoring any checkpoint. Teacher and policy receive
identical reset seeds and configurations. This tests simulator base control,
not arm manipulation, frozen TAMP acceptance, or hardware readiness.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.evaluation import evaluate, paired_comparison, summarize
from emet.rl.manipulation.policy import ManipulationPolicy


def wilson(successes, total):
    """Two-sided 95% binomial interval, including all failures in denominator."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid success counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def make_suite(config, *, cases=20, seed=9300000):
    if cases < 20 or seed < 1000000:
        raise ValueError("use at least 20 cases per stratum and a fresh seed block >= 1000000")
    common = replace(
        config,
        scenario="reach",
        reach_target_mode="base",
        horizon=20,
        base_only_min_distance=0.08,
        base_only_max_distance=0.35,
        target_scale=0.15,
        randomization_scale=0.05,
        initial_heading_range=0.3,
        retarget_probability=0,
        reach_push_probability=0,
        disturbance_curriculum=False,
    )
    # Each stratum varies a stated axis; acceptance tolerances are unchanged.
    settings = {
        "near": {},
        "longer_translation": {"target_scale": 1.0},
        "all_headings": {"initial_heading_range": float(np.pi), "target_scale": 0.5},
        "goal_change": {"retarget_probability": 1.0, "target_scale": 0.5},
        "finite_push": {"reach_push_probability": 1.0, "randomization_scale": 0.3},
        "randomized_dynamics": {"randomization_scale": 1.0},
        "long_hold": {"horizon": 40.0},
    }
    if cases >= 10000:
        raise ValueError("seed blocks would overlap")
    return {
        "version": "base-stress-v1",
        "cases_per_stratum": cases,
        "scope": "simulation-only base control; finite pushes, no dedicated obstacle-impact or manipulation gate",
        "strata": [
            {"name": name, "seed": seed + index * 10000, "config": asdict(replace(common, **overrides))}
            for index, (name, overrides) in enumerate(settings.items())
        ],
    }


def suite_sha(suite):
    return hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()


def score_suite(suite, directory, policy=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, groups = [], {}
    for item in suite["strata"]:
        report = evaluate(
            policy, config=ManipulationConfig(**item["config"]), episodes=suite["cases_per_stratum"], seed=item["seed"]
        )
        report["suite_sha256"] = suite_sha(suite)
        group = report["summary"]["reach"]
        successes = sum(r["is_success"] for r in report["episodes"])
        group["success_95pct_interval"] = wilson(successes, len(report["episodes"]))
        # Rejected initializations remain failures in the reported denominator.
        for row in report["episodes"]:
            row["stratum"] = item["name"]
        (directory / f"{item['name']}.json").write_text(json.dumps(report, indent=2) + "\n")
        groups[item["name"]] = group
        rows.extend(report["episodes"])
        print(item["name"], "success", successes, "/", len(report["episodes"]), flush=True)
    aggregate = summarize(rows)["reach"]
    aggregate["success_95pct_interval"] = wilson(sum(r["is_success"] for r in rows), len(rows))
    result = {
        "suite_sha256": suite_sha(suite),
        "scope": suite["scope"],
        "controller": "teacher" if policy is None else policy.manifest["weights_sha256"],
        "steps": None if policy is None else policy.manifest["steps"],
        "strata": groups,
        "summary": {"reach": aggregate},
        "episodes": rows,
    }
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="bundle defining the unchanged contract")
    parser.add_argument("--policy", type=Path, help="optional frozen candidate bundle")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9300000)
    parser.add_argument("--wait-seconds", type=float, default=0, help="wait for candidate manifest before evaluation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = ManipulationConfig(**json.loads((args.template / "manifest.json").read_text())["environment"])
    suite = make_suite(config, cases=args.cases, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    suite_path = args.output / "suite.json"
    if suite_path.exists() and json.loads(suite_path.read_text()) != suite:
        raise ValueError("refusing to change frozen suite")
    suite_path.write_text(json.dumps(suite, indent=2) + "\n")
    teacher_path = args.output / "teacher/report.json"
    if teacher_path.exists():
        teacher = json.loads(teacher_path.read_text())
        if teacher["suite_sha256"] != suite_sha(suite):
            raise ValueError("teacher suite mismatch")
    else:
        teacher = score_suite(suite, args.output / "teacher")
    if args.policy:
        deadline = time.monotonic() + args.wait_seconds
        while not (args.policy / "manifest.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("candidate checkpoint not available; teacher results retained")
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        actor = ManipulationPolicy(args.policy)
        candidate = score_suite(suite, args.output / f"policy-step-{actor.manifest['steps']}", actor)
        comparison = paired_comparison(teacher, candidate)
        (args.output / "paired-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
