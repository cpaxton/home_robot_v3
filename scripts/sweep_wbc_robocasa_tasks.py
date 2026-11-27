# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run PandaOmron WBC through derived RoboCasa task targets.

This is the task-level bridge after the reach sweep: it loads one kitchen env,
derives object/cabinet/pregrasp/grasp/lift/place poses, drives the WBC policy or
the privileged teacher, and reports both tracking and the env's task success.
It is still a development scene check, not a RoboCasa benchmark claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
from emet.rl.manipulation.wbc_robocasa_task import RoboCasaPandaTask, TaskResult
from emet.rl.manipulation.wbc_training import WbcPPO


def _phase_metrics(result: TaskResult) -> dict[str, dict[str, float]]:
    metrics: dict[str, list[float]] = {}
    for step in result.steps:
        if not np.isfinite(step.pos_err):
            continue
        values = metrics.setdefault(step.phase, [])
        values.append(float(step.pos_err))
    return {
        phase: {"pos_err_mean": float(np.mean(values)), "pos_err_max": float(np.max(values))}
        for phase, values in metrics.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep WBC through derived RoboCasa task targets")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--env-name", default="PickPlaceCounterToCabinet")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--backoff", type=float, default=0.0)
    parser.add_argument("--max-move-steps", type=int, default=70)
    parser.add_argument("--teacher", action="store_true", help="run the privileged teacher instead of a policy")
    parser.add_argument("--save-data", action="store_true", help="save policy observations and teacher labels")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.teacher and args.checkpoint is None:
        parser.error("--checkpoint is required unless --teacher is set")

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
    backend = RoboCasaPandaBackend(env, base_backoff=args.backoff, target_mode="offset")
    task = RoboCasaPandaTask(env, backend, max_move_steps=args.max_move_steps)
    policy = None
    checkpoint_metadata: dict = {}
    if not args.teacher:
        trainer, checkpoint_metadata = WbcPPO.load(args.checkpoint, pandaomron_sim_profile(), device=args.device)
        policy = trainer.policy

    observations: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    phases: list[str] = []
    episode_ids: list[int] = []
    episodes: list[dict] = []
    successes = 0
    try:
        for episode in range(args.episodes):
            result = task.execute(teacher=args.teacher, policy=policy, seed=args.seed + episode)
            successes += int(result.success)
            record = {
                "episode": episode,
                "seed": args.seed + episode,
                "success": result.success,
                "failure_phase": result.failure_phase,
                "object_category": result.poses.object_category,
                "object_initial": list(result.poses.object_pose[:3]),
                "object_final": list(result.final_object_pose[:3]),
                "object_displacement_m": result.object_displacement_m,
                "max_object_lift_m": result.max_object_lift_m,
                "object_inside_cabinet": result.object_inside_cabinet,
                "gripper_far": result.gripper_far,
                "grasped_after_close": result.grasped_after_close,
                "grasped_after_lift": result.grasped_after_lift,
                "cabinet": result.poses.cabinet_name,
                "cabinet_open": result.poses.cabinet_open,
                "final_ee": list(result.final_ee_pose[:3]),
                "steps": len(result.steps),
                "collisions": sum(int(step.collision) for step in result.steps),
                "phase_metrics": _phase_metrics(result),
            }
            episodes.append(record)
            if args.save_data and not args.teacher:
                for step in result.steps:
                    if step.observation is None or step.teacher_action is None:
                        continue
                    observations.append(np.asarray(step.observation, dtype=np.float32))
                    teacher_actions.append(np.asarray(step.teacher_action, dtype=np.float32))
                    phases.append(step.phase)
                    episode_ids.append(episode)
            print(
                f"episode={episode} success={result.success} phase={result.failure_phase or 'ok'} "
                f"object={record['object_category']} displacement={record['object_displacement_m']:.3f} "
                f"steps={record['steps']}",
                flush=True,
            )
    finally:
        env.close()

    failure_counts = Counter(record["failure_phase"] for record in episodes if not record["success"])
    summary = {
        "robot": "pandaomron",
        "env": args.env_name,
        "layout": args.layout,
        "style": args.style,
        "episodes": args.episodes,
        "seed": args.seed,
        "teacher": bool(args.teacher),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.expanduser()),
        "checkpoint_stage": checkpoint_metadata.get("stage"),
        "success_rate": successes / args.episodes,
        "failure_counts": dict(failure_counts),
        "episodes_detail": episodes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task_sweep.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.save_data and observations:
        np.savez_compressed(
            args.output / "loaded_task_data.npz",
            observations=np.asarray(observations, dtype=np.float32),
            teacher_actions=np.asarray(teacher_actions, dtype=np.float32),
            phases=np.asarray(phases),
            episode_ids=np.asarray(episode_ids, dtype=np.int32),
        )
        (args.output / "loaded_task_data_meta.json").write_text(
            json.dumps(
                {
                    "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                    "observations": len(observations),
                    "action_size": int(teacher_actions[0].size) if teacher_actions else 0,
                    "scene": {
                        "env": args.env_name,
                        "layout": args.layout,
                        "style": args.style,
                        "base_backoff": args.backoff,
                    },
                    "phases": sorted(set(phases)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(
        f"success={summary['success_rate']:.2f} failures={dict(failure_counts)} episodes={args.episodes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
