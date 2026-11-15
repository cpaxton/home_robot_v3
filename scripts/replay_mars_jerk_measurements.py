# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Read-only native replay of the 24 frozen exploration-pair final cases."""

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.basic_learning import load_basic_learning
from emet.rl.manipulation.jerk_measurements import NativeJerkObserver
from emet.rl.manipulation.mastery import evaluate_suite, read_report
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_training import load_development

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_replay(expected, actual):
    old = {case["id"]: case for case in expected["cases"]}
    new = {case["id"]: case for case in actual["cases"]}
    if (
        len(old) != 12
        or len(new) != 12
        or len(old) != len(expected["cases"])
        or len(new) != len(actual["cases"])
        or old.keys() != new.keys()
    ):
        raise ValueError("all twelve unique frozen diagnostic cases must replay")
    for case_id, previous in old.items():
        replay = new[case_id]
        for key in ("initial_state_sha256", "reference_sha256"):
            if previous["admission"][key] != replay["admission"][key]:
                raise ValueError("diagnostic physical identity changed: " + case_id)
        for key in ("code", "mastered", "tracking_and_settling_pass", "smoothness_pass", "failure_reasons", "metrics"):
            if previous[key] != replay[key]:
                raise ValueError("diagnostic changed frozen outcome/metric: " + case_id + "/" + key)
        if len(previous["trace"]) != len(replay["trace"]):
            raise ValueError("diagnostic changed rollout duration: " + case_id)
        for before, after in zip(previous["trace"], replay["trace"], strict=True):
            for key in ("time", "base", "hand", "action", "reference_base", "reference_hand"):
                np.testing.assert_array_equal(before[key], after[key], err_msg=case_id + "/" + key)
    return {"cases": 12, "physical_identities_equal": True, "all_outcomes_metrics_actions_poses_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root", type=Path, default=Path("logs/manipulation/mars-reference-base-exploration-v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, parent = load_development()
    _, _, suite = load_basic_learning(Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json"), parent)
    teacher = read_report(args.reference_root / "preflight/general-teacher.json.gz")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {
        "format": "mars-aligned-base-jerk-measurements-v1",
        "producer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "producer_script_sha256": sha(Path(__file__)),
        "scope": "24 frozen final cases, new read-only instrumentation; no new qualification labels or teacher admission",
        "definitions": {
            "point": "base body origin, world XY; both native implementations share this point",
            "gate": "interval-average unsigned planar displacement speed and wrapped heading rate; zero derivative reset",
            "displacement_vector": "signed interval-average world XY/heading velocity; zero derivative reset",
            "jacobian": "exact v2 RewardMeter instantaneous world XY/angular-Z velocity; actual initial velocity, zero acceleration",
            "jacobian_zero_reset": "same instantaneous samples with zero initial velocity/acceleration, diagnostic only",
            "jacobian_trapezoid": "adjacent instantaneous-velocity average, approximating the displacement interval, zero derivative reset",
            "timestamps": "recorded absolute post-step MuJoCo time; instantaneous and interval measurements differ by half a control tick",
            "linear_jerk": "abs(scalar-speed second difference) for gate, norm(world-XY vector second difference) otherwise",
            "yaw_caution": "RewardMeter angular-Z approximates heading derivative for upright base; frozen heading channel is unchanged",
        },
        "hardware_used": False,
        "new_ppo_steps": 0,
        "policy_exported": False,
        "reward_changed": False,
        "acceptance_changed": False,
        "variants": {},
    }
    try:
        for name in ("control", "candidate"):
            directory = args.reference_root / name
            checkpoint = directory / "last/actor"
            expected_path = directory / "final-evaluation.json.gz"
            policy, observer = WholeBodyBundle(checkpoint), NativeJerkObserver()
            report = evaluate_suite(
                suite, stages=("base", "arm"), policy=policy, teacher_report=teacher, observer=observer
            )
            full = {"measurements": list(observer.cases.values()), "replay": report}
            raw_path = args.output / (name + "-native.json.gz")
            raw_path.write_bytes(gzip.compress((json.dumps(full, allow_nan=False) + "\n").encode(), mtime=0))
            summary["variants"][name] = {
                "actor_weights_sha256": policy.manifest["weights_sha256"],
                "actor_bundle_sha256": sha(checkpoint / "policy.zip"),
                "original_report_sha256": sha(expected_path),
                "original_source_hashes": read_report(expected_path)["source_hashes"],
                "instrumented_source_hashes": report["source_hashes"],
                "native_archive_sha256": sha(raw_path),
                "replay_validation": {"complete": False},
            }
            summary["variants"][name]["replay_validation"] = compare_replay(read_report(expected_path), report)
            summary["variants"][name]["cases"] = observer.summarize()
            print(name, "all12 frozen outcomes/actions/poses exact; channels recorded", flush=True)
        summary.update(complete=True, elapsed_wall_seconds=time.monotonic() - started)
    except BaseException as exc:
        summary.update(complete=False, error=repr(exc), elapsed_wall_seconds=time.monotonic() - started)
        if "observer" in locals():
            partial = args.output / "partial-measurements.json.gz"
            partial.write_bytes(
                gzip.compress(json.dumps(list(observer.cases.values()), allow_nan=False).encode(), mtime=0)
            )
            summary["partial_archive_sha256"] = sha(partial)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
