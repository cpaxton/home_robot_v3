#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Serial, paired development pilot: production emet vs adapted GraphEQA.

Both use the same sensor/navigation boundary. This is a whole-system pilot,
not a causal memory ablation or a published benchmark result. Each child owns
one episode so graph/perception allocations do not accumulate across arms.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_shared_room_pairs import assert_frozen_source, git

from emet.utils.job_registry import validated_gpu_lock_fd


def run(args):
    lock_fd = validated_gpu_lock_fd()
    if lock_fd is None:
        raise RuntimeError("Use emet jobs --gpu-exclusive")
    root = Path(__file__).resolve().parents[1]
    revision = git(root, "rev-parse", "HEAD")
    assert_frozen_source(root, revision)
    if not args.question_ids or len(set(args.question_ids)) != len(args.question_ids):
        raise ValueError("Explicit, unique development IDs required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": revision,
        "fork_commit": git(args.fork, "rev-parse", "HEAD"),
        "status": "running",
        "question_ids": args.question_ids,
        "seed": args.seed,
        "max_images": args.max_images,
        "max_decisions": args.max_decisions,
        "max_primitives": args.max_primitives,
        "max_seconds": args.max_seconds,
        "startup_scan_degrees": 360,
        "comparison": "whole agents; shared sensor/body/pose/motion; native memory and image selection differ",
        "generation": "same served Qwen; 2048-token per-call cap; total model calls/tokens measured, not equalized",
        "order": "alternating first arm by question index",
        "rows": [],
    }
    path = args.output / "paired.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for index, qid in enumerate(args.question_ids):
        for arm in ("emet", "grapheqa") if index % 2 == 0 else ("grapheqa", "emet"):
            assert_frozen_source(root, revision)
            output = args.output / f"q{qid}-{arm}"
            command = [
                "/home/cpaxton/src/home_robot_v3/.venv-habitat/bin/python",
                str(root / "scripts/run_grapheqa_active.py"),
                "--fork",
                str(args.fork),
                "--model-config",
                str(args.model_config),
                "--output",
                str(output),
                "--agent",
                arm,
                "--question-ids",
                str(qid),
                "--seed",
                str(args.seed),
                "--max-decisions",
                str(args.max_decisions),
                "--max-images",
                str(args.max_images),
                "--max-primitives",
                str(args.max_primitives),
                "--max-seconds",
                str(args.max_seconds),
                "--startup-scan-degrees",
                "360",
            ]
            started = time.monotonic()
            row = {"question_id": qid, "agent": arm, "complete": False, "artifact": str(output)}
            try:
                with (args.output / f"q{qid}-{arm}.log").open("w") as log:
                    result = subprocess.run(
                        command,
                        cwd=root,
                        env={**os.environ, "PYTHONPATH": str(root / "src")},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.max_seconds + 300,
                        pass_fds=(lock_fd,),
                        check=False,
                    )
                row["returncode"] = result.returncode
                scores = output / "scored_results.json"
                if scores.exists():
                    scored = json.loads(scores.read_text())
                    if len(scored) != 1 or scored[0]["question_id"] != qid:
                        raise ValueError("One matching episode required per child")
                    row.update(scored[0])
                    row["complete"] = bool(row["complete"] and result.returncode == 0)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["process_wall_s"] = time.monotonic() - started
            manifest["rows"].append(row)
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    manifest["status"] = "complete" if all(r["complete"] for r in manifest["rows"]) else "incomplete_or_failed"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-ids", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--max-decisions", type=int, default=4)
    parser.add_argument("--max-primitives", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=900)
    raise SystemExit(run(parser.parse_args()))
