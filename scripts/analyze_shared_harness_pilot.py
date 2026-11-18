#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""CPU-only paired pilot summary from frozen artifacts; no model or simulation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-run", type=Path, required=True)
    parser.add_argument("--seed-zero-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hashes = {}

    def read(path, *, lines=False):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.decode().splitlines() if s.strip()] if lines else json.loads(raw)

    report = read(args.final_run / "report.json")
    manifest = read(args.final_run / "manifest.json")
    if report.get("infrastructure_error"):
        raise ValueError(report["infrastructure_error"])
    paired = []
    for qid in manifest["questions"]:
        answers = {}
        for arm in ("main", "candidate"):
            rows = read(args.final_run / f"eqa-{arm}/eqa/q{qid}.jsonl", lines=True)
            if len(rows) != 1 or type(rows[0].get("correct")) is not bool or rows[0].get("error"):
                raise ValueError(f"Invalid {arm} EQA artifact for {qid}")
            answers[arm] = rows[0]["correct"]
        paired.append({"question_id": qid, **answers})
    s0 = []
    for seed in manifest["s0_seeds"]:
        s0.extend({"seed": seed, **r} for r in read(args.final_run / f"s0-seed{seed}/results.jsonl", lines=True))
    if args.seed_zero_run:
        prior = read(args.seed_zero_run / "manifest.json")
        expected = {arm: value["sha"] for arm, value in manifest["sources"].items()}
        if prior["revisions"] != expected:
            raise ValueError("Seed-zero source revisions differ; do not pool these arms")
        s0.extend({"seed": 0, **r} for r in read(args.seed_zero_run / "results.jsonl", lines=True))
    tasks = []
    for seed in sorted({r["seed"] for r in s0}):
        rows = [r for r in s0 if r["seed"] == seed]
        if len(rows) != 2 or {r["arm"] for r in rows} != {"main", "candidate"}:
            raise ValueError(f"Missing/duplicate paired S0 arm for seed {seed}")
        for r in rows:
            if not r["complete"] or r.get("validation_error"):
                raise ValueError(f"Incomplete S0 arm: {seed}/{r['arm']}")
            m = r["metrics"]
            tasks.append(
                {
                    "seed": seed,
                    "arm": r["arm"],
                    "object_success": m["find_object_success"],
                    "receptacle_success": m["find_recep_success"],
                    "episode_wall_s": m["episode_wall_s"],
                    "receptacle_rounds": m["recep_agentic_rounds"],
                }
            )
    trace_coverage = {}
    for seed in manifest["s0_seeds"]:
        for path in sorted((args.final_run / f"s0-seed{seed}").rglob("*agentic_trace.jsonl")):
            arms = [a for a in ("main", "candidate") if any(p.endswith("-" + a) for p in path.parts)]
            if len(arms) != 1:
                raise ValueError(f"Ambiguous trace arm: {path}")
            arm = arms[0]
            count = trace_coverage.setdefault(arm, Counter())
            for row in read(path, lines=True):
                if row.get("event") == "tool_pick":
                    count["primary_pick:" + row.get("tool", "unknown")] += 1
                if row.get("tool") == "look_around" and row.get("event") != "tool_pick":
                    count["look_results"] += 1
                    if row.get("camera_motion"):
                        count["camera_motion_receipts"] += 1
    summary = {
        "scope": "Development pilot, complete-system comparison; not held-out generalization or a causal ablation.",
        "sources": manifest["sources"],
        "control": "Main agent plus shared evaluation-protocol patch; not unmodified main.",
        "contracts": report["contracts"],
        "eqa": {
            "n": len(paired),
            "correct": {arm: sum(r[arm] for r in paired) for arm in ("main", "candidate")},
            "candidate_only_wins": [r["question_id"] for r in paired if r["candidate"] and not r["main"]],
            "candidate_only_losses": [r["question_id"] for r in paired if r["main"] and not r["candidate"]],
            "paired": paired,
        },
        "s0": tasks,
        "s0_trace_coverage": trace_coverage,
        "limitations": [
            "Small tuned development population",
            "Wall time is not pure navigation or sim time",
            "No consistent per-run speedup",
            "Seed-zero success did not invoke camera-motion feedback",
            "No learned-perception pick/place or active EXPRESS/AbstainEQA qualification",
        ],
        "artifact_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"eqa": summary["eqa"]["correct"], "s0_pairs": len(tasks) // 2, "output": str(args.output)}))


if __name__ == "__main__":
    main()
