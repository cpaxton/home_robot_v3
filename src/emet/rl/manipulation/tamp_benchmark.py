"""Continuous GT TAMP transport with canonical benchmark objects at native mass.

Object geometry, inertia and contacts are transferred unchanged; initial support
and robot placement use the manipulation fixture. This is an object-transfer
acceptance gate, not a claim of full public-tool scene/task execution.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from emet.rl.manipulation.integrated import run_case
from emet.rl.manipulation.payload_assets import provenance
from emet.rl.manipulation.policy import ManipulationPolicy


def make_suite(*, robot="innate_mars", seed=1800006, episodes=3):
    if episodes < 1:
        raise ValueError("positive episode count required")
    return {
        "version": "native-payload-tamp-v1",
        "robot": robot,
        "assets": provenance("native_table"),
        "scope": "continuous object-transfer fixture with GT TAMP transport; not full public-tool oracle task",
        "cases": [{"payload_object": obj, "seed": seed + i} for obj in ("object1", "object2") for i in range(episodes)],
    }


def summarize(rows):
    groups = {}
    for obj in sorted({r["payload_object"] for r in rows}):
        group = [r for r in rows if r["payload_object"] == obj]
        failures = Counter()
        for row in group:
            if not row["success"]:
                stage = row["stages"][-1]["stage"] if row["stages"] else "initialization"
                failures[f"{stage}:{row['failure']}"] += 1
        groups[obj] = {
            "cases": len(group),
            "successes": sum(r["success"] for r in group),
            "failures_by_stage": dict(failures),
            "verified_transport_segments": sum(
                s.get("execution_verified", False) for r in group for s in r["stages"] if s["stage"] == "transport"
            ),
        }
    return groups


def run_suite(suite, output, *, policy=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()
    rows = []
    for case in suite["cases"]:
        row = run_case(
            robot=suite["robot"],
            seed=case["seed"],
            policy=policy,
            action_mode="direct_velocity",
            payload_asset_set="native_table",
            payload_object=case["payload_object"],
        )
        row["payload_object"] = case["payload_object"]
        row["case_id"] = f"{suite['robot']}:{case['payload_object']}:{case['seed']}"
        rows.append(row)
        report = {
            "suite_sha256": digest,
            "scope": suite["scope"],
            "complete": len(rows) == len(suite["cases"]),
            "summary": summarize(rows),
            "cases": rows,
        }
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")
        print(case, "success", row["success"], "failure", row["failure"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["innate_mars", "stretch"], default="innate_mars")
    parser.add_argument("--seed", type=int, default=1800006)
    parser.add_argument("--episodes", type=int, default=3, help="per object")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    suite = make_suite(robot=args.robot, seed=args.seed, episodes=args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = args.output / "suite.json"
    if fixture.exists() and json.loads(fixture.read_text()) != suite:
        raise ValueError("refusing to alter frozen native-payload suite")
    fixture.write_text(json.dumps(suite, indent=2) + "\n")
    teacher = run_suite(suite, args.output / "teacher")
    if args.policy:
        policy = ManipulationPolicy(args.policy, robot=args.robot)
        candidate = run_suite(suite, args.output / "policy", policy=policy)
        lost = [
            a["case_id"]
            for a, b in zip(teacher["cases"], candidate["cases"], strict=True)
            if a["success"] and not b["success"]
        ]
        # Per-stage force, retention and tracking traces remain in raw reports;
        # task completion alone does not promote a controller.
        (args.output / "paired.json").write_text(
            json.dumps(
                {
                    "lost_teacher_success_cases": lost,
                    "teacher": teacher["summary"],
                    "policy": candidate["summary"],
                    "promoted": False,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
