# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Smoke-test a PandaOmron WBC checkpoint in a RoboCasa kitchen scene.

This is a reach-tracking smoke, not a task-success benchmark: the backend holds
the base and targets a small free-space offset from the current end effector.
The high-level skill/TAMP layer supplies real object/receptacle targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
from emet.rl.manipulation.wbc_training import WbcPPO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the PandaOmron WBC in RoboCasa")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/eval/wbc_multirobot_20261126/pandaomron/best.pt"),
    )
    parser.add_argument("--env-name", default="PickPlaceCounterToCabinet")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.08))
    parser.add_argument("--backoff", type=float, default=0.45)
    parser.add_argument("--teacher", action="store_true", help="use the privileged teacher instead of a checkpoint")
    parser.add_argument("--target-mode", choices=("offset", "joint"), default="offset")
    parser.add_argument("--target-range", type=float, default=0.15)
    args = parser.parse_args(argv)

    import robocasa  # noqa: F401
    import robosuite
    from robosuite import load_part_controller_config

    env = robosuite.make(
        env_name=args.env_name,
        robots="PandaOmron",
        controller_configs=load_part_controller_config(default_controller="OSC_POSE"),
        translucent_robot=False,
        layout_and_style_ids=[[args.layout, args.style]],
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    backend = RoboCasaPandaBackend(
        env,
        target_offset=tuple(args.offset),
        base_backoff=args.backoff,
        target_mode=args.target_mode,
        target_range=args.target_range,
    )
    wbc = WbcGymEnv(backend, horizon_s=args.horizon, control_dt=0.05, hold_base=True)
    policy = None
    device = None
    if not args.teacher:
        trainer, _ = WbcPPO.load(args.checkpoint, pandaomron_sim_profile())
        policy = trainer.policy
        device = trainer.device
    observation, _ = wbc.reset(seed=0)
    print(
        f"base={backend._base_pose()} torso={backend.data.qpos[backend._torso_q]:.3f} "
        f"ncon={backend.data.ncon} target={backend._target_ee}",
        flush=True,
    )
    for index in range(backend.data.ncon):
        contact = backend.data.contact[index]
        first = backend.model.body(int(backend.model.geom_bodyid[contact.geom1])).name
        second = backend.model.body(int(backend.model.geom_bodyid[contact.geom2])).name
        if first.startswith(("robot0", "gripper0")) or second.startswith(("robot0", "gripper0")):
            print(f"reset contact {first} | {second} dist={contact.dist:.3f}", flush=True)
    success = False
    info: dict = {}
    final_contacts: list[tuple[str, str, float]] = []
    final_arm_q = np.zeros(0)
    final_target = np.zeros(3)
    final_ee = np.zeros(3)
    try:
        for step in range(int(round(args.horizon / 0.05))):
            if args.teacher:
                action = backend.teacher_action()
            else:
                with torch.no_grad():
                    action = (
                        policy.deterministic_action(
                            torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        )[0]
                        .cpu()
                        .numpy()
                    )
            observation, _, terminated, truncated, info = wbc.step(action)
            success |= bool(info.get("success", False))
            if step % 10 == 0:
                terms = info["reward_terms"]
                print(
                    f"step={step} pos_err={terms['pos_err']:.3f} quat_err={terms['quat_err']:.3f} "
                    f"torso={backend.data.qpos[backend._torso_q]:.3f} base={backend._base_pose()} "
                    f"success={info['success']}",
                    flush=True,
                )
            if terminated or truncated:
                break
    finally:
        # Snapshot before close; the underlying Robosuite env owns this data.
        final_arm_q = np.asarray(backend.data.qpos[backend._arm_q], dtype=np.float64).copy()
        final_target = np.asarray(backend._target_ee[:3], dtype=np.float64).copy()
        final_ee = np.asarray(backend._observation().arms[0].ee_pose[:3], dtype=np.float64).copy()
        for index in range(backend.data.ncon):
            contact = backend.data.contact[index]
            first = backend.model.body(int(backend.model.geom_bodyid[contact.geom1])).name
            second = backend.model.body(int(backend.model.geom_bodyid[contact.geom2])).name
            if first.startswith(("robot0", "gripper0")) or second.startswith(("robot0", "gripper0")):
                final_contacts.append((first, second, float(contact.dist)))
        wbc.close()
    for first, second, distance in final_contacts:
        print(f"final contact {first} | {second} dist={distance:.3f}", flush=True)
    print(
        f"final arm_q={np.round(final_arm_q, 3)} target={np.round(final_target, 3)} ee={np.round(final_ee, 3)}",
        flush=True,
    )
    print(f"tracking_success={success} final={info.get('reward_terms', {})}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
