# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bounded inner-loop experiment: four variants on three existing teacher cases.

Interventions apply AFTER identical native reset/admission. They do not change
production defaults, references, rewards, limits, stop guards or old results.
"""

import argparse
import gzip
import json
import subprocess
import time
from pathlib import Path

import torch
from replay_mars_jerk_measurements import compare_replay, sha

from emet.rl.manipulation.basic_learning import load_basic_learning
from emet.rl.manipulation.jerk_measurements import NativeJerkObserver, current_feedback_velocity
from emet.rl.manipulation.mastery import evaluate_suite, read_report
from emet.rl.manipulation.whole_body_training import load_development

ROOT = Path(__file__).resolve().parents[1]
CASES = ("train:base:hold", "train:base:forward", "train:arm:pitch_up@amplitude:0.25")
VARIANTS = ("cached_com", "feedback_off", "current_com", "current_axle")


class FeedbackProbe(NativeJerkObserver):
    def __init__(self, variant):
        super().__init__(record_physics=True)
        self.variant = variant

    def reset(self, env, case):
        super().reset(env, case)
        # The default method is restored before every case, including after a
        # reset that keeps the same control instance. Reset conditioning is legacy.
        if self.variant == "feedback_off":
            env.control.base_velocity_feedback_gain = 0.0
        elif self.variant.startswith("current_"):
            point = self.variant.removeprefix("current_")
            env.control._base_feedback_velocity = lambda: current_feedback_velocity(env.control, point)

    def after_step(self, env, action, info):
        super().after_step(env, action, info)
        if info["code"] != "running":
            self.restore(env)

    def restore(self, env):
        env.control.__dict__.pop("_base_feedback_velocity", None)
        env.control.base_velocity_feedback_gain = env.config.wheel_feedback_gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-root", type=Path, default=Path("logs/manipulation/mars-reference-base-exploration-v1")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, parent = load_development()
    _, _, suite = load_basic_learning(Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json"), parent)
    suite["cases"] = [case for case in suite["cases"] if case["id"] in CASES]
    if {case["id"] for case in suite["cases"]} != set(CASES):
        raise ValueError("all three frozen feedback probe cases required")
    teacher = read_report(args.reference_root / "preflight/general-teacher.json.gz")
    expected = {**teacher, "cases": [case for case in teacher["cases"] if case["id"] in CASES]}
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "mars-wheel-feedback-probe-v1",
        "producer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "producer_script_sha256": sha(__file__),
        "scope": "12 teacher episodes, four predefined feedback interventions on identical admitted resets",
        "variants": {},
        "new_ppo_steps": 0,
        "hardware_used": False,
        "production_defaults_changed": False,
        "reward_or_acceptance_changed": False,
        "qualified": False,
    }
    started = time.monotonic()
    try:
        for name in VARIANTS:
            observer = FeedbackProbe(name)
            report = evaluate_suite(suite, stages=("base", "arm"), teacher_report=teacher, observer=observer)
            raw = args.output / (name + ".json.gz")
            raw.write_bytes(
                gzip.compress(
                    json.dumps({"measurements": list(observer.cases.values()), "replay": report}).encode(), mtime=0
                )
            )
            if name == "cached_com":
                compare_replay(expected, report, cases=len(CASES))
            for row in report["cases"]:
                previous = next(case for case in expected["cases"] if case["id"] == row["id"])
                for key in ("initial_state_sha256", "reference_sha256"):
                    if row["admission"][key] != previous["admission"][key]:
                        raise ValueError("intervention changed reset/reference identity: " + row["id"])
            summary["variants"][name] = {
                "raw_sha256": sha(raw),
                "all_reset_reference_identities_equal": True,
                "cases": observer.summarize(),
                "outcomes": [
                    {key: row[key] for key in ("id", "code", "failure_reasons", "metrics")} for row in report["cases"]
                ],
            }
            print(name, "complete, identical reset/reference identities", flush=True)
        summary["complete"] = True
    except BaseException as exc:
        summary.update(complete=False, error=repr(exc))
        if "observer" in locals():
            raw = args.output / "partial.json.gz"
            raw.write_bytes(gzip.compress(json.dumps(list(observer.cases.values())).encode(), mtime=0))
        raise
    finally:
        summary["elapsed_wall_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
