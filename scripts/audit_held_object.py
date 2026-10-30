"""Teacher-driven occupancy diagnostic; never a task or safety certificate."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.policy import ManipulationPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6, help="attempts per empty/loaded category")
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--horizon", type=float, default=8)
    parser.add_argument("--sample-seconds", type=float, default=2)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.sample_seconds <= args.horizon:
        parser.error("positive episode count and 0 < sample-seconds <= horizon required")
    if args.output.exists():
        parser.error("refusing to overwrite an existing diagnostic")
    torch.set_num_threads(1)
    policy = ManipulationPolicy(args.policy)
    if policy.held_estimator is None:
        parser.error("policy has no occupancy estimator")
    base = ManipulationConfig(**policy.manifest["environment"])
    rows, summary = [], {}
    for scenario in ("reach", "carry"):
        config = replace(
            base,
            scenario=scenario,
            horizon=args.horizon,
            empty_grasp_probability=1 if scenario == "reach" else 0,
            carry_hold_probability=1 if scenario == "carry" else 0,
        )
        env = ManipulationEnv(config)
        if env.fingerprint != policy.manifest["fingerprint"]:
            raise ValueError("occupancy diagnostic model mismatch")
        try:
            for index in range(args.episodes):
                seed = args.seed + index
                obs, info = env.reset(seed=seed)
                samples = []
                for _ in range(max(1, round(args.sample_seconds / env.profile.dt))):
                    obs, _, done, truncated, info = env.step(env.teacher_action())
                    label = info["held_object_supervision"]
                    estimate = policy.estimate_held_object(obs)
                    if label["valid"] and estimate["available"] and not info.get("invalid_physics"):
                        samples.append(
                            {
                                "target": label["target"],
                                "probability": estimate["probability"],
                                "time": info["time"],
                                "gripper_position": float(env.data.qpos[env.control.grip_q]),
                                "gripper_command_error": float(
                                    env.control.grip_target - env.data.qpos[env.control.grip_q]
                                ),
                                "gripper_velocity": float(env.data.qvel[env.control.grip_v]),
                                "retain_object_command": env.control.retain_object,
                            }
                        )
                    if done or truncated:
                        break
                rows.append({"scenario": scenario, "seed": seed, "info": info, "samples": samples})
                print(scenario, seed, info.get("code"), len(samples), flush=True)
        finally:
            env.close()
        group = [r for r in rows if r["scenario"] == scenario]
        samples = [s for r in group for s in r["samples"]]
        positive = [s for s in samples if s["target"] == 1]
        negative = [s for s in samples if s["target"] == 0]
        summary[scenario] = {
            "attempts": len(group),
            "initialization_failed": sum(r["info"]["initialization_failed"] for r in group),
            "positive_samples": len(positive),
            "negative_samples": len(negative),
            "recall": sum(s["probability"] >= 0.5 for s in positive) / len(positive) if positive else None,
            "empty_false_positive_rate": sum(s["probability"] >= 0.5 for s in negative) / len(negative)
            if negative
            else None,
            "brier": sum((s["probability"] - s["target"]) ** 2 for s in samples) / len(samples) if samples else None,
        }
    report = {
        "scope": "teacher-driven empty/loaded simulator occupancy diagnostic; not task qualification or calibration",
        "source_hashes": source_hashes(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_sha256": policy.manifest["weights_sha256"],
        "held_object_manifest_sha256": policy.manifest["held_object_manifest_sha256"],
        "config": policy.manifest["environment"],
        "horizon": args.horizon,
        "sample_seconds": args.sample_seconds,
        "summary": summary,
        "cases": rows,
        "calibrated": False,
        "safety_promoted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
