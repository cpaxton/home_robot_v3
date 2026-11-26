# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Attach a task contract to a legacy DAgger checkpoint.

Checkpoints written before the contract field cannot be evaluated safely: the
held-out suite could use a different target range, base-goal probability, or
hold-base setting than training. This migration tool records the explicit
default (or requested) suite in the checkpoint metadata without changing the
actor weights. New DAgger checkpoints already carry the contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emet.controller.manipulation.wbc_robots import get_wbc_robot, wbc_profile_for
from emet.rl.manipulation.wbc_dagger import WbcTaskSuiteEnv, default_subtasks
from emet.rl.manipulation.wbc_training import WbcPPO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attach a task contract to a legacy WBC checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subtasks", default=None, help="comma-separated suite; defaults to the robot's suite")
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--success-hold", type=float, default=0.25)
    parser.add_argument("--collision-force-n", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    if args.control_dt <= 0 or args.success_hold < 0 or args.collision_force_n < 0:
        parser.error("control dt must be positive and success-hold/collision thresholds nonnegative")
    robot = get_wbc_robot(args.robot)
    profile = wbc_profile_for(args.robot)
    trainer, metadata = WbcPPO.load(args.checkpoint, profile, device=args.device)
    tasks = {task.name: task for task in default_subtasks(robot.name)}
    requested = (
        list(tasks) if args.subtasks is None else [name.strip() for name in args.subtasks.split(",") if name.strip()]
    )
    unknown = set(requested) - set(tasks)
    if unknown:
        parser.error(f"unknown subtasks: {sorted(unknown)}")
    subtasks = tuple(tasks[name] for name in requested)
    env = WbcTaskSuiteEnv(
        subtasks,
        robot=robot.name,
        control_dt=args.control_dt,
        success_hold_s=args.success_hold,
        collision_force_n=args.collision_force_n,
    )
    try:
        if trainer.policy.observation_size != env.observation_space.shape[0]:
            raise ValueError("checkpoint observation size does not match the declared suite")
        if trainer.policy.action_size != env.action_space.shape[0]:
            raise ValueError("checkpoint action size does not match the declared suite")
        contract = env.contract()
        trainer.save(
            args.output,
            env.envs[env.names[0]],
            metadata={**(metadata or {}), "contract": contract, "contract_migrated": True},
        )
    finally:
        env.close()
    print(
        f"stamped {args.output} robot={robot.name} subtasks={requested} "
        f"control_dt={args.control_dt} hold={args.success_hold}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
