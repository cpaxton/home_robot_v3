"""Paired comparisons of existing immutable TAMP evaluation manifests/ledgers."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def search_protocol():
    """Record planner defaults and dependency versions used by fixed-run comparisons."""
    import inspect
    from importlib.metadata import version

    from emet.motion.placement import plan_placement_paths

    names = ("max_solutions", "ik_attempts", "max_candidates", "max_ik_calls",
             "max_ik_calls_per_base", "rrt_max_iter", "preplace_height_m", "margin_m", "seed")
    parameters = inspect.signature(plan_placement_paths).parameters
    return {"placement": {name: parameters[name].default for name in names},
            "runtime": {name: version(name) for name in ("mujoco", "numpy", "scipy")}}


def _load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    ledger = json.loads((directory / "ledger.json").read_text())
    if not manifest.get("search_protocol"):
        raise ValueError("Missing recorded search protocol; rerun rather than assume historical defaults")
    cases = {case["id"]: case for case in manifest["cases"]}
    if len(cases) != len(manifest["cases"]) or not cases or set(cases) != set(ledger):
        raise ValueError("Ledger must cover every scheduled case exactly once")
    for key, case in cases.items():
        if case.get("stage") != "evaluation" or not case.get("registry_sha256"):
            raise ValueError("Compare fixed-registry evaluation, not candidate-specific admission")
        if ledger[key].get("status") in {None, "pending", "running"}:
            raise ValueError("All scheduled trials must be terminal")
    return manifest, ledger, cases


def _case_signature(case):
    # Executable, checkout and registry paths vary across frozen worktrees. The
    # registry hash, arguments and semantic metadata must remain identical.
    command = list(case["command"][2:])
    for flag in ("--episodes",):
        if flag in command:
            command[command.index(flag) + 1] = case["registry_sha256"]
    return {**{k: v for k, v in case.items() if k != "command"}, "arguments": command}


def _success(row):
    return row.get("task_success") is True and row.get("status") == "completed" and row.get("exit_code") == 0


def _first_failure(row):
    if _success(row):
        return "success"
    result = row.get("result") or {}
    for event in result.get("execution_trace", []):
        if event.get("success") is not True:
            return event.get("failed_op") or "execution"
    return row.get("reason") or result.get("error") or row["status"]


def compare_runs(baseline, candidate):
    old_manifest, old, old_cases = _load(baseline)
    new_manifest, new, new_cases = _load(candidate)
    if set(old_cases) != set(new_cases):
        raise ValueError("Case sets differ; do not change the comparison denominator")
    for key in ("suite", "robot_filter", "episode_timeout_s", "search_protocol"):
        if old_manifest.get(key) != new_manifest.get(key):
            raise ValueError(f"Comparison protocol differs: {key}")
    for key in old_cases:
        if _case_signature(old_cases[key]) != _case_signature(new_cases[key]):
            raise ValueError(f"Case configuration/registry differs: {key}")
    strata = {(case.get("robot"), case["execution_mode"]) for case in old_cases.values()}
    if len(strata) != 1:
        raise ValueError("Compare each robot/execution mode separately; do not pool assistance")
    pairs = []
    clusters = {}
    for key, case in old_cases.items():
        a, b = _success(old[key]), _success(new[key])
        pairs.append({"case_id": key, "baseline_success": a, "candidate_success": b,
                      "baseline_first_failure": _first_failure(old[key]),
                      "candidate_first_failure": _first_failure(new[key])})
        cluster = (case.get("scene_split"), case.get("scene_index"))
        clusters.setdefault(cluster, []).append(int(b) - int(a))
    n = len(pairs)
    old_count = sum(p["baseline_success"] for p in pairs)
    new_count = sum(p["candidate_success"] for p in pairs)
    # Resample scenes, not individual repetitions. Too few scenes cannot support
    # useful generalization uncertainty; keep it explicitly unavailable.
    interval = None
    if len(clusters) >= 5:
        values = list(clusters.values())
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(2000):
            selected = [values[i] for i in rng.integers(len(values), size=len(values))]
            samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
        interval = np.quantile(samples, [.025, .975]).tolist()
    return {
        "schema_version": 1, "scheduled_pairs": n,
        "baseline_source": old_manifest["source_sha"], "candidate_source": new_manifest["source_sha"],
        "baseline_successes": old_count, "candidate_successes": new_count,
        "success_rate_delta": (new_count - old_count) / n,
        "gains": sum(not p["baseline_success"] and p["candidate_success"] for p in pairs),
        "losses": sum(p["baseline_success"] and not p["candidate_success"] for p in pairs),
        "scene_clusters": len(clusters), "scene_bootstrap_95_interval": interval,
        "baseline_status_counts": dict(Counter(row["status"] for row in old.values())),
        "candidate_status_counts": dict(Counter(row["status"] for row in new.values())),
        "pairs": pairs,
        "input_sha256": {f"{label}/{name}": hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
                         for label, root in (("baseline", baseline), ("candidate", candidate))
                         for name in ("manifest.json", "ledger.json")},
        "promotion": "requires_review",  # Improvement alone is not a safety/canary certificate.
    }
