# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Evaluate a WBC checkpoint on recorded hard-case postures.

Hard cases replay a recorded failure posture and target through a backend that
supports ``load_hard_case``. This script reports per-case success, steps, and
tracking error; it is a robustness gate separate from the clean subtask suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emet.controller.manipulation.wbc_robots import get_wbc_robot, wbc_profile_for
from emet.rl.manipulation.wbc_backends import make_wbc_backend
from emet.rl.manipulation.wbc_env import WbcGymEnv
from emet.rl.manipulation.wbc_hard_cases import load_hard_cases
from emet.rl.manipulation.wbc_training import WbcPPO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a WBC checkpoint on hard-case postures")
    parser.add_argument("--robot", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-case", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    if args.episodes_per_case < 1:
        parser.error("--episodes-per-case must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    robot = get_wbc_robot(args.robot)
    profile = wbc_profile_for(args.robot)
    trainer, metadata = WbcPPO.load(args.checkpoint, profile, device=args.device)
    policy = trainer.policy
    device = trainer.device
    cases = load_hard_cases(args.cases)
    backend = make_wbc_backend(
        robot.name,
        sim="mujoco",
        seed=0,
        arm_goal_probability=0.0,
        base_goal_probability=0.0,
    )
    env = WbcGymEnv(backend, horizon_s=args.horizon, control_dt=0.05, success_hold_s=0.1)
    results: dict[str, dict] = {}
    try:
        for case in cases:
            episodes = []
            for _ in range(args.episodes_per_case):
                loaded = env.load_hard_case(case)
                if loaded is None:
                    raise ValueError(f"{robot.name} backend does not support hard cases")
                observation, _ = loaded
                done, success, steps = False, False, 0
                info: dict = {}
                while not done:
                    with torch.no_grad():
                        action = (
                            policy.deterministic_action(
                                torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                            )[0]
                            .cpu()
                            .numpy()
                        )
                    observation, _, terminated, truncated, info = env.step(action)
                    success |= bool(info.get("success", False))
                    done = bool(terminated or truncated)
                    steps += 1
                terms = info.get("reward_terms", {})
                episodes.append(
                    {
                        "success": bool(success),
                        "steps": steps,
                        "pos_err": float(terms.get("pos_err", float("nan"))),
                        "quat_err": float(terms.get("quat_err", float("nan"))),
                    }
                )
            results[case.name] = {
                "success_rate": float(np.mean([episode["success"] for episode in episodes])),
                "episodes": episodes,
            }
            print(
                f"{case.name}: success={results[case.name]['success_rate']:.2f} "
                f"mean_steps={np.mean([episode['steps'] for episode in episodes]):.1f}",
                flush=True,
            )
    finally:
        env.close()

    aggregate = float(np.mean([value["success_rate"] for value in results.values()]))
    manifest = {
        "robot": robot.name,
        "checkpoint": str(args.checkpoint.expanduser()),
        "checkpoint_stage": (metadata or {}).get("stage"),
        "profile": profile.name,
        "profile_fingerprint": profile.fingerprint,
        "cases": str(args.cases.expanduser()),
        "episodes_per_case": int(args.episodes_per_case),
        "horizon_s": float(args.horizon),
        "aggregate_success": aggregate,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"aggregate_success={aggregate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
