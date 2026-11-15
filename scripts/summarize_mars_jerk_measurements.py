# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compact, reproducible readout; raw diagnostic archives stay outside git."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physics_summary(case):
    rows = case["physics"]
    layout = case["physics_layout"]
    base = next(j for j in layout["joints"] if j["name"] == "manipulation_base")
    if base["type"] != 0:
        raise ValueError("world translational qvel readout requires the Mars free base")
    offset = base["qvel_address"]
    left = layout["actuators"].index("left_wheel_vel")
    right = layout["actuators"].index("right_wheel_vel")
    times = np.array([r["time"] for r in rows])
    dt = case["physics_dt_seconds"]
    np.testing.assert_allclose(np.diff(times), dt, rtol=0, atol=1e-12)
    # Discard the first second for spectral readout, not for contract metrics.
    mask = times > case["initial_time"] + 1.0
    velocity = np.array([r["qvel"][offset : offset + 2] for r in rows])[mask]
    vx = velocity[:, 0] - velocity[:, 0].mean()
    frequencies = np.fft.rfftfreq(len(vx), dt)
    power = np.abs(np.fft.rfft(vx)) ** 2
    index = int(np.argmax(power[1:])) + 1
    commands = np.array([r["ctrl"] for r in rows])[mask][:, [left, right]]
    force = np.array([r["actuator_force"] for r in rows])[mask][:, [left, right]]
    return {
        "samples": len(rows),
        "spectral_samples_after_one_second": int(mask.sum()),
        "planar_velocity_rms_m_s": float(np.sqrt(np.mean(np.sum(velocity**2, axis=1)))),
        "vx_peak_to_peak_m_s": float(np.ptp(velocity[:, 0])),
        "vx_dominant_frequency_Hz": float(frequencies[index]),
        "vx_power_fraction_above_10Hz": float(power[frequencies > 10].sum() / power.sum()) if power.sum() else 0.0,
        "wheel_command_rms_rad_s": float(np.sqrt(np.mean(commands**2))),
        "wheel_actuator_force_rms_native": float(np.sqrt(np.mean(force**2))),
        "wheel_actuator_force_peak_native": float(np.max(abs(force))),
        "requested_base_command_peak": np.max(abs(np.array([r["base_command"] for r in rows])), axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/eval/mars-aligned-jerk-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "format": "mars-aligned-jerk-readout-v1",
        "scope": "24 final replays + 2 stationary anchors + 2 high-rate anchors + 12 corrected teacher feedback probes; first 12-probe attempt invalid",
        "new_ppo_steps": 0,
        "hardware_used": False,
        "qualified": False,
        "acceptance_or_reward_changed": False,
        "experiments": {},
        "readout_script_sha256": sha(Path(__file__)),
        "invalid_attempt": {
            "path": str(args.root / "feedback"),
            "reason": "observer restored feedback on nonterminal code=ok; excluded from comparison; raw archives retained",
            "summary_sha256": sha(args.root / "feedback/summary.json"),
        },
    }
    for name in ("data", "anchors", "substeps", "feedback"):
        directory = "feedback-v2" if name == "feedback" else name
        path = args.root / directory / "summary.json"
        summary = json.loads(path.read_text())
        if not summary["complete"]:
            raise ValueError("incomplete diagnostic: " + name)
        variants = {}
        for variant, record in summary["variants"].items():
            variants[variant] = {
                key: value
                for key, value in record.items()
                if key not in {"original_source_hashes", "instrumented_source_hashes"}
            }
            for key in ("original_source_hashes", "instrumented_source_hashes"):
                if key in record:
                    variants[variant][key + "_digest"] = hashlib.sha256(
                        json.dumps(record[key], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
            if name in {"substeps", "feedback"}:
                suffix = ".json.gz" if name == "feedback" else "-native.json.gz"
                raw = args.root / directory / (variant + suffix)
                content = json.loads(gzip.decompress(raw.read_bytes()))
                measurements = content["measurements"]
                variants[variant]["physics_readout"] = {case["id"]: physics_summary(case) for case in measurements}
        result["experiments"][name] = {
            "producer_revision": summary["producer_revision"],
            "original_summary_sha256": sha(path),
            "elapsed_wall_seconds": summary["elapsed_wall_seconds"],
            "variants": variants,
        }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = {
        "cached_com": "Legacy cached COM",
        "feedback_off": "Proportional feedback off",
        "current_com": "Current qvel at COM",
        "current_axle": "Current qvel at axle",
    }
    for variant, label in labels.items():
        raw = args.root / "feedback-v2" / (variant + ".json.gz")
        content = json.loads(gzip.decompress(raw.read_bytes()))
        case = next(c for c in content["measurements"] if c["id"] == "train:base:hold")
        rows = case["physics"]
        times = np.array([r["time"] for r in rows]) - case["initial_time"]
        mask = (times >= 1.0) & (times <= 1.2)
        qvel = np.array([r["qvel"] for r in rows])
        wheel = case["physics_layout"]["actuators"].index("left_wheel_vel")
        command = np.array([r["ctrl"][wheel] for r in rows])
        force = np.array([r["actuator_force"][wheel] for r in rows])
        axes[0].plot(times[mask], 1000 * qvel[mask, 0], label=label)
        axes[1].plot(times[mask], command[mask], label=label)
        axes[2].plot(times[mask], force[mask], label=label)
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Base vx (mm/s)")
    axes[0].set_title("Stationary Mars: zero requested base motion, same admitted reset")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].set_ylabel("Left wheel target (rad/s)")
    axes[2].set_ylabel("Left actuator force\n(native rotational units)")
    axes[2].set_xlabel("Seconds after diagnostic reset")
    fig.tight_layout()
    fig.savefig(args.output / "stationary-feedback.svg")
    plt.close(fig)
    print("Wrote compact results and stationary physics plot to", args.output)


if __name__ == "__main__":
    main()
