# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline audit of the four frozen TAMP traces; never steps a simulator."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from emet.rl.manipulation.mastery import read_report
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.policy import load_control_policy
from emet.rl.manipulation.profiles import MARS


def main():
    from validate_mars_current_feedback import LEGACY, sha, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text())
    if not summary["complete"] or not summary["recording_preserves_teacher_outcome"]:
        raise ValueError("complete four-episode recording-parity diagnostic required")
    bundle = load_control_policy(LEGACY)
    if bundle.manifest["weights_sha256"] != summary["weights_sha256"]:
        raise ValueError("offline audit actor differs from recorded actor")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {k: v for k, v in summary.items() if k != "groups"}
    result.update(summary_sha256=sha(args.input / "summary.json"), analysis_script_sha256=sha(__file__), groups={})
    result["native_log_sha256"] = sha(args.native_log)
    result["native_failure_details"] = [
        line for line in args.native_log.read_text().splitlines() if line.startswith("Native TAMP ")
    ]
    for name, group in summary["groups"].items():
        if "trace_sha256" in group and sha(args.input / name / "episode-0001.json.gz") != group["trace_sha256"]:
            raise ValueError("recorded trace hash changed: " + name)
        row = {k: v for k, v in group["row"].items() if k != "commands"}
        trace = {k: v for k, v in group.get("trace", {}).items() if k != "commands"}
        result["groups"][name] = {
            "row": row,
            "trace": trace,
            "parity_signature": group["parity_signature"],
            "trace_sha256": group.get("trace_sha256"),
        }
    legacy = read_report(args.input / "legacy_recorded/episode-0001.json.gz")
    decisions = legacy["decisions"]
    audit = {}
    for key, fields in (("history", history_layout(MARS, retention_context=True)[0]), ("preview", preview_layout()[0])):
        x = np.asarray([d["observation"][key] for d in decisions])
        mean, var = bundle.normalization[key]
        z = (x - mean) / np.sqrt(var + bundle.manifest["epsilon"])
        clipped = abs(z) > bundle.manifest["clip"]
        audit[key] = {}
        for field, columns in fields.items():
            hits = np.flatnonzero(np.any(clipped[:, :, columns], axis=(1, 2)))
            if len(hits):
                audit[key][field] = {
                    "fraction_of_field_elements_clipped": float(clipped[:, :, columns].mean()),
                    "first_decision_time_s": decisions[int(hits[0])]["time"],
                    "peak_absolute_standardized_value": float(abs(z[:, :, columns]).max()),
                }
    result["normalization_clipping"] = audit
    result["offline_grip_input_sensitivity"] = {
        "scope": "Replace only recorded grip position/error history and preview error with training means; these are hypothetical inputs, never executed actions or success evidence.",
        "samples": [],
    }
    replay_error = 0.0
    sample_times = {
        min(decisions, key=lambda d: abs(d["time"] - v))["time"] for v in (3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 12.0)
    }
    history, _ = history_layout(MARS, retention_context=True)
    preview, _ = preview_layout()
    for decision in decisions:
        observation = {k: np.asarray(decision["observation"][k], dtype=np.float32) for k in bundle.normalization}
        actual = bundle(observation)
        replay_error = max(replay_error, float(abs(actual - decision["action"]).max()))
        if decision["time"] not in sample_times:
            continue
        changed = {k: v.copy() for k, v in observation.items()}
        for name in ("proprio/grip_position", "proprio/grip_error"):
            columns = history[name]
            changed["history"][:, columns] = bundle.normalization["history"][0][:, columns]
        changed["preview"][:, preview["grip_error"]] = bundle.normalization["preview"][0][:, preview["grip_error"]]
        result["offline_grip_input_sensitivity"]["samples"].append(
            {
                "time_s": decision["time"],
                "recorded_action": actual.tolist(),
                "hypothetical_action": bundle(changed).tolist(),
            }
        )
    result["maximum_replayed_action_difference"] = replay_error
    if replay_error > 1e-7:
        raise ValueError("recorded actor inputs do not reproduce recorded actions")

    plt.rcParams["svg.hashsalt"] = "mars-tamp-input-diagnostic-v1"
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for name, label in (("teacher_recorded", "Teacher"), ("legacy_recorded", "Retained actor")):
        trace = read_report(args.input / name / "episode-0001.json.gz")
        ticks = [t for t in trace["ticks"] if t["phase"] == "pregrasp"]
        start = next(c["sim_time"] for c in trace["commands"] if c.get("phase") == "pregrasp")
        times = [t["time"] - start for t in ticks]
        axes[0].plot(
            times, [1000 * np.linalg.norm(np.asarray(t["hand"])[:3] - t["hand_goal"][:3]) for t in ticks], label=label
        )
        axes[1].plot(times, [max(abs(np.asarray(t["normalized_action"])[2:7])) for t in ticks], label=label)
        axes[2].plot(times, [1000 * (t["gripper_position"] - t["grip_reference"]) for t in ticks], label=label)
    axes[0].set_ylabel("Hand error (mm)")
    axes[1].set_ylabel("Peak arm action\n(normalized)")
    axes[2].set_ylabel("Jaw angle error\n(mrad)")
    axes[2].set_xlabel("Seconds since pregrasp command")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("One matched TAMP pregrasp: response and jaw deflection")
    figure_path = args.output / "tracking.svg"
    fig.savefig(figure_path, metadata={"Date": None})
    figure_path.write_text("\n".join(line.rstrip() for line in figure_path.read_text().splitlines()) + "\n")
    plt.close(fig)
    write(args.output / "results.json", result)


if __name__ == "__main__":
    main()
