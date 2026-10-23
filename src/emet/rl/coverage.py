"""Report accepted arm curriculum coverage, including rejection bias."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.env import EnvConfig, MarsReachEnv


def audit(samples=500, seed=80000):
    if samples < 1:
        raise ValueError("samples must be positive")
    env = MarsReachEnv(EnvConfig(arm_only=True))
    configurations, positions, travel = [], [], []
    rejected = 0
    try:
        for index in range(samples):
            _, info = env.reset(seed=seed + index)
            rejected += info["sampling_rejected_starts"]
            start, target = env.control.arm_positions().copy(), env.control.goal.arm.copy()
            configurations.extend([start, target])
            positions.append(env.control.goal_pose[:3].copy())
            travel.append(float(np.max(np.abs(target - start))))
        q = np.asarray(configurations)
        return {
            "pairs": samples,
            "seed": seed,
            "rejected_starts": rejected,
            "joint_bins": [
                np.histogram(q[:, i], bins=np.linspace(*env.control.limits[i], 11))[0].tolist() for i in range(5)
            ],
            "ee_bounds": [np.min(positions, axis=0).tolist(), np.max(positions, axis=0).tolist()],
            "occupied_ee_5cm_voxels": len(np.unique(np.floor(np.asarray(positions) / 0.05), axis=0)),
            "max_joint_travel_quantiles": np.quantile(travel, [0, 0.5, 0.9, 1]).tolist(),
            "sampling": "uniform joint proposals, 50% local target proposals, collision/path rejection; not Cartesian-uniform",
            "scope": "sampled geometric clearance only, no continuous swept-volume or loaded dynamic guarantee",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=80000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args.samples, args.seed)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
