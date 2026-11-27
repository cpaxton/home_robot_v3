# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Evaluate carry checkpoints with and without measured contact inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_dagger import WbcTaskSuiteEnv, evaluate_suite, subtasks_from_contract
from emet.rl.manipulation.wbc_env import wbc_observation_layout
from emet.rl.manipulation.wbc_training import WbcPPO


class MaskedSuite:
    def __init__(self, suite: WbcTaskSuiteEnv, layout, *, include_joint_torque: bool = False) -> None:
        self.suite = suite
        self.slices = [
            layout.slices["arm.right.contact_flag"],
            layout.slices["arm.right.contact_wrench"],
            layout.slices["arm.right.gripper_force"],
        ]
        if include_joint_torque:
            self.slices.append(layout.slices["arm.right.joint_torque"])

    @property
    def names(self):
        return self.suite.names

    @property
    def envs(self):
        return self.suite.envs

    def _mask(self, observation: np.ndarray) -> np.ndarray:
        observation = np.array(observation, copy=True)
        for section in self.slices:
            observation[section] = 0.0
        return observation

    def reset(self, **kwargs):
        observation, info = self.suite.reset(**kwargs)
        return self._mask(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.suite.step(action)
        return self._mask(observation), reward, terminated, truncated, info

    def teacher_action(self):
        return self.suite.teacher_action()

    def close(self):
        self.suite.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-joint-torque", action="store_true")
    args = parser.parse_args()

    profile = pandaomron_sim_profile()
    layout = wbc_observation_layout(profile)
    payload = {}
    for checkpoint in args.checkpoint:
        trainer, metadata = WbcPPO.load(checkpoint, profile, device="cpu")
        contract = metadata.get("contract")
        tasks = subtasks_from_contract(contract)
        suite = WbcTaskSuiteEnv(tasks, robot="pandaomron", seed=args.seed)
        try:
            full = evaluate_suite(suite, trainer.policy, episodes_per_task=args.episodes, seed=args.seed)
            masked = evaluate_suite(
                MaskedSuite(suite, layout, include_joint_torque=args.include_joint_torque),
                trainer.policy,
                episodes_per_task=args.episodes,
                seed=args.seed,
            )
        finally:
            suite.close()
        payload[str(checkpoint)] = {"full": full, "masked": masked}
        print(
            f"{checkpoint.name}: full={full['aggregate']['success_rate']:.2f} "
            f"masked={masked['aggregate']['success_rate']:.2f} "
            f"full_track={full['aggregate']['tracking_error_m']:.4f} "
            f"masked_track={masked['aggregate']['tracking_error_m']:.4f}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
