#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Resume a frozen benchmark matrix, using existing runners and per-case managed jobs.

The coordinator owns no GPU. Each child releases the GPU between episodes;
other registered GPU work and unmanaged compute take priority. A pause file or
disk reserve stops admission, never changes a task's safety or scoring policy.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from run_shared_room_pairs import assert_frozen_source, git

from emet.utils.job_registry import active_gpu_job_pids

METHODS = ("lazy_graph", "dynamem", "static_graph", "dynagraph")


def build_matrix(source, output, python, habitat):
    """Existing gate populations, not a newly selected favorable question set."""
    registry = yaml.safe_load((source / "configs/benchmarks/dev_sets.yaml").read_text())
    questions = list(
        dict.fromkeys(
            registry["eqa"]["dev_ids"] + registry["eqa"]["gate_holdout8_ids"] + registry["eqa"]["gate_bal32_ids"]
        )
    )
    tasks = []

    def add(key, family, command, timeout, env=None, metric=None):
        tasks.append(
            {
                "id": key,
                "family": family,
                "command": list(map(str, command)),
                "timeout_s": timeout,
                "env": env or {},
                "metric": metric,
            }
        )

    # Broad primary harness coverage comes first; comparison rows follow. Every
    # assigned comparison remains in the manifest even if the night expires.
    for method in METHODS:
        for q in questions:
            key = f"eqa-{method}-q{q}-s0"
            metric = str(output / key / "metrics.jsonl")
            add(
                key,
                "eqa",
                [
                    habitat,
                    "run-episode",
                    "--question-id",
                    q,
                    "--method",
                    method,
                    *(["--query-driven-memory"] if method == "lazy_graph" else []),
                    "--seed",
                    0,
                    "--max-planning-steps",
                    20,
                    "--max-movement-step",
                    10,
                    "--no-hm3d-semantics",
                    "--no-enrich-labels",
                    "--debug-run-tag",
                    output.name + "-" + key,
                    "--output",
                    metric,
                ],
                600,
                metric=metric,
            )
        for ep in ("default_table_s0_distinct_recep", "molmo_rby1_ithor_s2_idx0"):
            key = f"room-{method}-{ep}-s0"
            add(
                key,
                "room_find",
                [
                    python,
                    "-m",
                    "emet.cli",
                    "ovmm",
                    "find",
                    "--episodes",
                    "configs/ovmm/find_phase_episodes.yaml",
                    "--backend",
                    method,
                    *(["--query-driven-memory"] if method == "lazy_graph" else []),
                    "--episode-id",
                    ep,
                    "--seed",
                    0,
                    "--sensor-perception",
                    "--mapping-rotate-steps",
                    4,
                    "--agentic-max-rounds",
                    6,
                    "--agentic-max-nav-steps",
                    3,
                    "--no-scene-cache",
                    "--port-offset",
                    2300,
                    "--output-dir",
                    output / key / "artifacts",
                ],
                1500,
                {"EMET_CONFIG": str(output / f"{method}-config.yaml")},
                str(output / key / "artifacts" / f"{ep}_{method}.json"),
            )
        for ep in registry["ovmm"]["gate_episodes"]:
            key = f"habitat-find-{method}-{ep}-s0"
            metric = str(output / key / "metrics.json")
            add(
                key,
                "habitat_find_proxy",
                [
                    habitat,
                    "run-ovmm-find-episode",
                    "--episodes",
                    "configs/ovmm/habitat_find_phase_nearest_v2.yaml",
                    "--episode-id",
                    ep,
                    "--backend",
                    method,
                    *(["--query-driven-memory"] if method == "lazy_graph" else []),
                    "--agentic-find",
                    "--agentic-max-rounds",
                    6,
                    "--agentic-max-nav-steps",
                    3,
                    "--seed",
                    0,
                    "--device",
                    "cuda",
                    "--output",
                    metric,
                ],
                1800,
                metric=metric,
            )
    # CPU privileged feasibility control. Do not pool it with learned success.
    key = "tamp-native-two-step"
    add(
        key,
        "privileged_tamp",
        [
            python,
            "scripts/eval_native_table_contact.py",
            "--output",
            output / key / "artifacts",
            "--sequence",
            "object2",
            "object1",
            "--separation-axis",
            "y",
            "--separation",
            ".15",
        ],
        1800,
        metric=str(output / key / "artifacts/summary.json"),
    )
    # ScanNet's registered first-30 slice, two backends, one isolated episode at
    # a time. No downloads or substitution of mesh replay for declared SENS.
    for q in range(30):
        for method in ("dynamem", "dynagraph"):
            key = f"sqa3d-{method}-index{q}"
            add(
                key,
                "sqa3d_recorded",
                [
                    python,
                    "-m",
                    "emet.cli",
                    "sqa3d",
                    "run-real-sweep",
                    "--split",
                    "val",
                    "--question-start",
                    q,
                    "--question-end",
                    q + 1,
                    "--method",
                    method,
                    "--replay-mode",
                    "sens",
                    "--no-download",
                    "--no-resume",
                    "--eqa-vl-family",
                    "qwen3_vl",
                    "--eqa-hf-model-id",
                    "Qwen/Qwen3-VL-8B-Instruct",
                    "--device",
                    "cuda",
                    "--output-dir",
                    output / key / "artifacts",
                ],
                1200,
            )
    return tasks


def select_tasks(tasks, requested):
    """Declare a retry population without changing the original matrix or ledger."""
    if not requested:
        return tasks
    selected = set(requested)
    unknown = selected - {task["id"] for task in tasks}
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    return [task for task in tasks if task["id"] in selected]


def read_outcome(task, rc):
    """Infrastructure completion and task success are different fields."""
    if rc:
        return {"complete": False, "success": None, "error": f"process_exit_{rc}"}
    if not task.get("metric"):
        return {"complete": None, "success": None, "error": "artifact_scoring_pending"}
    try:
        path = Path(task["metric"])
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != 1:
                raise ValueError("expected one episode")
            data = rows[0]
        else:
            data = json.loads(path.read_text())
        if data.get("error"):
            raise ValueError(str(data["error"]))
        if task["family"] == "privileged_tamp":
            if data.get("total") != 1 or type(data.get("passed")) is not int:
                raise ValueError("missing native control denominator")
            success = data["passed"] == data["total"]
        elif task["family"] == "eqa":
            if type(data.get("correct")) is not bool:
                raise ValueError("missing boolean correct")
            success = data["correct"]
        else:
            from check_acceptance_metrics import validate_record

            validate_record("room", data, path)
            success = bool(data["find_object_success"] and data["find_recep_success"])
        return {"complete": True, "success": success, "metrics": data, "error": None}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"complete": False, "success": None, "error": f"invalid_artifact: {exc}"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--habitat", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=7)
    p.add_argument("--reserve-gib", type=float, default=20)
    p.add_argument("--task-id", action="append", help="Run only these exact matrix IDs (repeatable; new manifest).")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    source, output = args.source.absolute(), args.output.absolute()
    revision = git(source, "rev-parse", "HEAD")
    assert_frozen_source(source, revision)
    tasks = select_tasks(
        build_matrix(source, output, str(args.python.absolute()), str(args.habitat.absolute())), args.task_id
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_sha": revision,
        "source": str(source),
        "tasks": tasks,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {
            f: hashlib.sha256((source / f).read_bytes()).hexdigest()
            for f in (
                "uv.lock",
                "configs/emet/default.yaml",
                "configs/emet/shared_find_acceptance.yaml",
                "configs/benchmarks/dev_sets.yaml",
            )
        },
        "seed": 0,
        "methods": METHODS,
        "selected_task_ids": args.task_id,
        "scope": "Registered gate/slice refresh, not full upstream datasets or a multi-seed superiority claim",
        "deferred": {
            "robocasa": "Visible larger-object replacement unqualified; tiny-object stress kept separate",
            "dynamic_exploration": "Legacy relocation/assisted protocol requires separate source/physics audit before pooling",
            "full_manipulation": "Find is localization only; native TAMP below is a privileged control",
        },
    }
    mf = output / "manifest.json"
    if mf.exists() and json.loads(mf.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError("Resume source, driver or population differs from frozen manifest")
    mf.write_text(json.dumps(manifest, indent=2) + "\n")
    for method in METHODS:
        config = yaml.safe_load((source / "configs/emet/shared_find_acceptance.yaml").read_text())
        config["extends"] = str(source / "configs/emet/default.yaml")
        config["mapping"]["query_driven_memory"] = method == "lazy_graph"
        config["mapping"]["eqa"]["agentic_decision_policy"] = "model_final"
        text = yaml.safe_dump(config, sort_keys=False)
        path = output / f"{method}-config.yaml"
        if path.exists() and path.read_text() != text:
            raise RuntimeError(f"Frozen config differs: {path}")
        path.write_text(text)
    print(f"Frozen {len(tasks)} cases at {revision}; execute={args.execute}", flush=True)
    if not args.execute:
        return
    deadline = time.monotonic() + args.max_hours * 3600
    ledger = output / "results.jsonl"
    done = (
        {json.loads(line)["id"] for line in ledger.read_text().splitlines() if line.strip()}
        if ledger.exists()
        else set()
    )
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("EMET_EQA_") or key in {
            "EMET_CONFIG",
            "EMET_VL_ENDPOINT",
            "EMET_SIM_EVAL_CONFIG",
            "EMET_OPENAI_BASE_URL",
            "EMET_LLM_HOST",
        }:
            env.pop(key)
    env.update(
        EMET_UV_RUN="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{source}/src:{source}/packages/emet_habitat",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MUJOCO_GL="egl",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        EMET_ALLOW_SDPA_ATTN="1",
        EMET_PHYSICAL_EXECUTION="1",
        EMET_SIM_NAV_TELEPORT="0",
        EMET_MOLMOSPACES_NAV_TELEPORT="0",
        EMET_EQA_AGENTIC_DECISION_POLICY="model_final",
        MOLMOSPACES_PYTHON=str(args.python.absolute().parents[2] / ".venv-molmospaces/bin/python"),
        LD_LIBRARY_PATH=str(args.habitat.absolute().parents[1] / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )
    # Insert native control early, then full primary harness, then baselines.
    tasks.sort(key=lambda t: 0 if t["family"] == "privileged_tamp" else 1)
    infrastructure_streak = 0
    for task in tasks:
        if task["id"] in done:
            continue
        while active_gpu_job_pids(exclude_job_id=os.environ.get("EMET_JOB_ID"), include_unmanaged=True):
            if time.monotonic() >= deadline or (output / "PAUSE").exists():
                return
            time.sleep(30)
        if time.monotonic() + task["timeout_s"] > deadline or (output / "PAUSE").exists():
            print("Admission window ended; remaining manifest cases stay pending", flush=True)
            return
        if shutil.disk_usage(output).free < args.reserve_gib * 1024**3:
            print("Disk reserve reached; remaining manifest cases stay pending", flush=True)
            return
        assert_frozen_source(source, revision)
        case = output / task["id"]
        case.mkdir(exist_ok=True)
        command = [
            str(args.python),
            "-m",
            "emet.cli",
            "jobs",
            "run",
            "--foreground",
            "--name",
            output.name + "-" + task["id"],
            "--out-dir",
            str(case / "job"),
            "--cpu-safe",
            "--gpu-exclusive",
            "--lock-timeout-sec",
            "1",
            "--",
            "taskset",
            "-c",
            "12",
            "timeout",
            "--signal=TERM",
            "--kill-after=20",
            str(task["timeout_s"]),
            *task["command"],
        ]
        started = time.monotonic()
        print("START " + task["id"], flush=True)
        with (case / "supervisor.log").open("a") as log:
            result = subprocess.run(
                command,
                cwd=source,
                env=env | task["env"] | {"EMET_PHYSICAL_AUDIT": str(case / "actuation.jsonl")},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 3:  # Lost lock race: no episode ran; retain log, retry on resume.
            print("GPU lock busy; leaving this and later cases pending", flush=True)
            return
        row = dict(
            id=task["id"],
            family=task["family"],
            source_sha=revision,
            wall_s=time.monotonic() - started,
            exit_code=result.returncode,
            **read_outcome(task, result.returncode),
        )
        with ledger.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        infrastructure_streak = infrastructure_streak + 1 if row["complete"] is False else 0
        print("END " + task["id"] + " " + json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
        if infrastructure_streak >= 3:
            print(
                "Three consecutive infrastructure failures; stop for diagnosis, keep remaining cases pending",
                flush=True,
            )
            return
        # Give any queued RL supervisor the first chance at the released lock.
        time.sleep(30)


if __name__ == "__main__":
    main()
