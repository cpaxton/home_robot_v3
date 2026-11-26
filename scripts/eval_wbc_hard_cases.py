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
from emet.rl.manipulation.wbc_dagger import TRACKING_WINDOW_STEPS
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--allow-missing-contract",
        action="store_true",
        help="allow a legacy checkpoint without saved task-contract metadata",
    )
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
    contract = (metadata or {}).get("contract")
    contract_source = "metadata"
    if contract is not None and "schema_version" not in contract:
        contract = None
    if contract is None:
        if not args.allow_missing_contract:
            raise ValueError(
                "checkpoint has no versioned task contract; re-save it with scripts/stamp_wbc_contract.py "
                "before hard-case evaluation, or pass --allow-missing-contract for a legacy artifact"
            )
        contract_source = "legacy_default_timing"
        contract = {
            "schema_version": 1,
            "robot": robot.name,
            "control_dt": 0.05,
            "success_hold_s": 0.25,
            "collision_force_n": 10.0,
            "subtasks": [],
        }
    elif contract.get("robot") != robot.name:
        raise ValueError(f"checkpoint contract robot {contract.get('robot')!r} != requested {robot.name!r}")
    cases = load_hard_cases(args.cases)
    if not cases:
        raise ValueError(f"no hard cases in {args.cases}")
    backend = make_wbc_backend(
        robot.name,
        sim="mujoco",
        seed=0,
        control_dt=float(contract["control_dt"]),
        arm_goal_probability=0.0,
        base_goal_probability=0.0,
    )
    env = WbcGymEnv(
        backend,
        horizon_s=args.horizon,
        control_dt=float(contract["control_dt"]),
        success_hold_s=float(contract["success_hold_s"]),
        collision_force_n=float(contract["collision_force_n"]),
        hold_base=True,
    )
    if policy.observation_size != env.observation_space.shape[0]:
        raise ValueError("checkpoint observation size does not match the hard-case environment")
    if policy.action_size != env.action_space.shape[0]:
        raise ValueError("checkpoint action size does not match the hard-case environment")
    results: dict[str, dict] = {}
    try:
        for case in cases:
            episodes = []
            for episode_index in range(args.episodes_per_case):
                # A hard case overrides an initialized reset; load_hard_case
                # alone leaves the torso/gripper at model defaults. Vary the
                # reset seed so repeated episodes are not bit-identical.
                env.reset(seed=args.seed + episode_index)
                loaded = env.load_hard_case(case)
                if loaded is None:
                    raise ValueError(f"{robot.name} backend does not support hard cases")
                observation, _ = loaded
                done, success, steps = False, False, 0
                collision = False
                collision_is_measured = True
                contact_available = 0
                pos_errors: list[float] = []
                first_success_index: int | None = None
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
                    collision |= bool(info.get("collision", False))
                    collision_is_measured &= bool(info.get("collision_measured", False))
                    contact_available += bool(info.get("contact_state_available", False))
                    pos_error = float(info.get("reward_terms", {}).get("pos_err", float("nan")))
                    if np.isfinite(pos_error):
                        pos_errors.append(pos_error)
                    if info.get("success", False) and first_success_index is None:
                        first_success_index = max(0, len(pos_errors) - 1)
                    done = bool(terminated or truncated)
                    steps += 1
                settled_errors = pos_errors if first_success_index is None else pos_errors[first_success_index:]
                settled = settled_errors[-TRACKING_WINDOW_STEPS:]
                episodes.append(
                    {
                        "success": bool(success),
                        "steps": steps,
                        "collisions": int(collision) if collision_is_measured else None,
                        "collision_measured": bool(collision_is_measured),
                        "duration_s": float(info.get("elapsed_s", steps * 0.05)),
                        "contact_state_rate": contact_available / max(steps, 1),
                        "pos_err": float(np.mean(settled)) if settled else float("inf"),
                        "pos_err_max": max(settled) if settled else float("inf"),
                        "tracking_error_m": float(np.mean(settled)) if settled else float("inf"),
                        "tracking_error_final_m": pos_errors[-1] if pos_errors else float("inf"),
                        "tracking_error_settled_max_m": max(settled) if settled else float("inf"),
                        "tracking_error_episode_max_m": max(pos_errors) if pos_errors else float("inf"),
                        "quat_err": float(info.get("reward_terms", {}).get("quat_err", float("nan"))),
                    }
                )
            measured = all(episode["collision_measured"] for episode in episodes)
            results[case.name] = {
                "success_rate": float(np.mean([episode["success"] for episode in episodes])),
                "collisions": int(sum(episode["collisions"] for episode in episodes)) if measured else None,
                "collision_measured": bool(measured),
                "tracking_error_m": float(np.mean([episode["tracking_error_m"] for episode in episodes])),
                "tracking_error_max_m": float(np.max([episode["pos_err_max"] for episode in episodes])),
                "tracking_error_final_m": float(np.mean([episode["tracking_error_final_m"] for episode in episodes])),
                "tracking_error_settled_max_m": float(
                    max(episode["tracking_error_settled_max_m"] for episode in episodes)
                ),
                "tracking_error_episode_max_m": float(
                    max(episode["tracking_error_episode_max_m"] for episode in episodes)
                ),
                "duration_s": float(np.mean([episode["duration_s"] for episode in episodes])),
                "contact_state_rate": float(np.mean([episode["contact_state_rate"] for episode in episodes])),
                "episodes": episodes,
            }
            print(
                f"{case.name}: success={results[case.name]['success_rate']:.2f} "
                f"mean_steps={np.mean([episode['steps'] for episode in episodes]):.1f}",
                flush=True,
            )
    finally:
        env.close()

    if not results:
        raise ValueError("hard-case evaluation produced no case results")
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
        "contract": contract,
        "contract_source": contract_source,
        "hold_base": True,
        "aggregate_success": aggregate,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"aggregate_success={aggregate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
