"""Compare teacher and idle discounted returns before spending more PPO steps."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv


def audit(*, seeds=(9100000, 9100001, 9100002, 9100003)):
    cfg = ManipulationConfig(
        robot="innate_mars",
        scenario="reach",
        horizon=20,
        action_mode="direct_velocity",
        trajectory_preview=True,
        tracking_reward=True,
        differential_base_reward=True,
        reach_target_mode="base",
        base_only_min_distance=0.08,
        base_only_max_distance=0.35,
        target_scale=0.15,
        randomization_scale=0,
        initial_heading_range=0.3,
        base_xy_tolerance=0.05,
        base_heading_tolerance=0.25,
        endpoint_motion_weight=0.25,
    )
    rows = []
    for gated in (False, True):
        env = ManipulationEnv(replace(cfg, goal_gated_settling=gated))
        try:
            for seed in seeds:
                for controller in ("teacher", "idle"):
                    env.reset(seed=seed)
                    rewards = []
                    for _ in range(round(cfg.horizon / env.profile.dt)):
                        action = env.teacher_action() if controller == "teacher" else np.zeros(env.action_space.shape)
                        _, reward, done, truncated, info = env.step(action)
                        rewards.append(float(reward))
                        if done or truncated:
                            break
                    row = {
                        "seed": seed,
                        "goal_gated_settling": gated,
                        "controller": controller,
                        "returns": {
                            str(gamma): float(sum(r * gamma**i for i, r in enumerate(rewards)))
                            for gamma in (0.99, 0.9975)
                        },
                        "rewards": rewards,
                        "success": info["is_success"],
                        "final_base_error_m": info["base_goal_error"],
                        "ever_settled": info["time_to_first_settle"] is not None,
                        "wheel_lift": info["wheel_lift"],
                        "unsafe_contact": info["unsafe_contact"],
                    }
                    rows.append(row)
                    print(seed, gated, controller, row["returns"], flush=True)
        finally:
            env.close()
    comparisons = []
    for gated in (False, True):
        for gamma in ("0.99", "0.9975"):
            deltas = []
            for seed in seeds:
                pair = {r["controller"]: r for r in rows if r["seed"] == seed and r["goal_gated_settling"] == gated}
                deltas.append(pair["teacher"]["returns"][gamma] - pair["idle"]["returns"][gamma])
            comparisons.append(
                {
                    "goal_gated_settling": gated,
                    "discount": float(gamma),
                    "teacher_minus_idle": deltas,
                    "teacher_preferred_count": sum(d > 0 for d in deltas),
                }
            )
    return {
        "scope": "fixed seeded reward incentive diagnostic, not held-out controller qualification",
        "comparisons": comparisons,
        "episodes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
