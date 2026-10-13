# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Serial, frozen EQA attribution pilot. Invoke through CPU-safe emet jobs.

Each arm runs in a fresh process; original attempts and uniquely named debug
bundles survive retries. Wrong answers are never retried. No automatic promotion.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def run(args):
    roots = {"historical": args.historical.resolve(), "current": args.current.resolve()}
    revisions = {}
    for name, root in roots.items():
        if git(root, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"{name} checkout must be clean before freezing")
        revisions[name] = git(root, "rev-parse", "HEAD")
    if revisions["historical"] != "dc106273f4f87cc2cb3a911aa6f99dac7adb2c25":
        raise RuntimeError("historical baseline does not match the recorded #181 revision")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "revisions": revisions,
        "roots": {k: str(v) for k, v in roots.items()},
        "ids": args.ids,
        "seeds": args.seeds,
        "arms": args.arms,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "habitat_bin": str(args.habitat_bin),
        "planning_steps": 20,
        "movement_step": 10,
        "timeout_s": 600,
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "quantization": "int4",
        "config_hashes": {
            name: {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((root / "configs").rglob("*.yaml"))
            }
            for name, root in roots.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for seed in args.seeds:
        for index, qid in enumerate(args.ids):
            order = args.arms if (index + seed) % 2 == 0 else list(reversed(args.arms))
            for arm in order:
                name = "historical" if arm == "historical" else "current"
                root = roots[name]
                if git(root, "rev-parse", "HEAD") != revisions[name] or git(root, "diff", "HEAD", "--stat"):
                    raise RuntimeError(f"frozen source changed: {root}")
                env = os.environ.copy()
                # Do not inherit another experiment's evidence/model overrides.
                for key in list(env):
                    if key.startswith("EMET_EQA_") or key in {
                        "EMET_CONFIG",
                        "EMET_VL_ENDPOINT",
                        "EMET_SIM_EVAL_CONFIG",
                        "EMET_OPENAI_BASE_URL",
                        "EMET_LLM_HOST",
                        "EMET_CALIBAN_HOST",
                    }:
                        env.pop(key)
                env.update(
                    {
                        "PYTHONPATH": f"{root}/src:{root}/packages/emet_habitat",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "EMET_UV_RUN": "1",
                        "EMET_ALLOW_SDPA_ATTN": "1",
                        "MUJOCO_GL": "egl",
                        "LD_LIBRARY_PATH": str(args.habitat_bin.parent.parent / "lib")
                        + ":"
                        + env.get("LD_LIBRARY_PATH", ""),
                    }
                )
                if arm in {"pose", "quality_coverage"}:
                    env["EMET_EQA_VIEW_SELECTION_POLICY"] = arm
                for attempt in range(2):
                    case = out / f"s{seed}-q{qid}-{arm}-attempt{attempt}"
                    case.mkdir()
                    metrics = case / "metrics.jsonl"
                    command = [
                        "timeout",
                        "--signal=TERM",
                        "--kill-after=20s",
                        "600",
                        str(args.habitat_bin),
                        "run-episode",
                        "--question-id",
                        str(qid),
                        "--method",
                        "lazy_graph",
                        "--query-driven-memory",
                        "--seed",
                        str(seed),
                        "--max-planning-steps",
                        "20",
                        "--max-movement-step",
                        "10",
                        "--no-hm3d-semantics",
                        "--no-enrich-labels",
                        "--eqa-vl-family",
                        "qwen3_vl",
                        "--eqa-hf-model-id",
                        "Qwen/Qwen3-VL-8B-Instruct",
                        "--eqa-vl-quantization",
                        "int4",
                        "--debug-run-tag",
                        f"{out.name}/{case.name}",
                        "--output",
                        str(metrics),
                    ]
                    (case / "invocation.json").write_text(
                        json.dumps(
                            {
                                "command": command,
                                "revision": revisions[name],
                                "policy": env.get("EMET_EQA_VIEW_SELECTION_POLICY", "disabled"),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    started = time.monotonic()
                    print(f"START {case.name} revision={revisions[name]}", flush=True)
                    with (case / "episode.log").open("w") as log:
                        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                    rows = []
                    if metrics.exists():
                        try:
                            rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
                        except json.JSONDecodeError:
                            pass
                    valid = result.returncode == 0 and len(rows) == 1 and not rows[0].get("error")
                    record = {
                        "question_id": qid,
                        "seed": seed,
                        "arm": arm,
                        "attempt": attempt,
                        "exit_code": result.returncode,
                        "complete": valid,
                        "elapsed_s": time.monotonic() - started,
                        "artifact_dir": str(case),
                        "metrics": rows[0] if len(rows) == 1 else None,
                    }
                    records.append(record)
                    with (out / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    if valid and (rows[0].get("vl_hf_model_id") != manifest["model"] or rows[0].get("vl_endpoint")):
                        raise RuntimeError(f"model mismatch; inspect {case}")
                    print(
                        f"END {case.name} complete={valid} correct={rows[0].get('correct') if rows else None}",
                        flush=True,
                    )
                    if valid:
                        break
    return (
        0
        if all(
            any(r["complete"] for r in records if (r["question_id"], r["seed"], r["arm"]) == key)
            for key in {(r["question_id"], r["seed"], r["arm"]) for r in records}
        )
        else 1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--habitat-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", default=[6, 12, 14, 56])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["historical", "unchanged", "pose", "quality_coverage"],
        default=["historical", "unchanged", "pose", "quality_coverage"],
    )
    raise SystemExit(run(parser.parse_args()))
