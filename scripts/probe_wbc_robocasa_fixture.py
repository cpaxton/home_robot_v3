# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Probe a closed RoboCasa cabinet door with a closed-gripper push servo.

This is a development probe, not a learned open/close skill. It closes the
fixture, moves the PandaOmron to the handle, and follows the handle arc about
the true hinge with the gripper closed. The task bridge itself does not use
this path; the probe exists to measure how far a fixed-base push can get and
to leave a reproducible trajectory for a future base-following skill.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
from emet.rl.manipulation.wbc_robocasa_task import (
    _GRIPPER_CLOSE,
    _GRIPPER_OPEN,
    RoboCasaPandaTask,
    _pose7,
)


def _rotation_from_axes(approach: np.ndarray, closing: np.ndarray) -> Rotation:
    z = np.asarray(approach, dtype=np.float64)
    z /= np.linalg.norm(z)
    y = np.asarray(closing, dtype=np.float64)
    y = y - z * float(np.dot(y, z))
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return Rotation.from_matrix(np.column_stack((x, y, z)))


def _hinge_world(env, joint_name: str) -> np.ndarray:
    model = env.sim.model
    joint_id = int(model.joint_name2id(joint_name))
    body = int(model.jnt_bodyid[joint_id])
    return np.asarray(env.sim.data.xpos[body], dtype=np.float64) + np.asarray(
        env.sim.data.xmat[body], dtype=np.float64
    ).reshape(3, 3) @ np.asarray(model.jnt_pos[joint_id], dtype=np.float64)


def _handle_body(env, fixture, side: str) -> str:
    candidates = [
        str(env.sim.model.body_id2name(body))
        for body in range(env.sim.model.nbody)
        if str(env.sim.model.body_id2name(body)).startswith(fixture.name)
        and "handle_main" in str(env.sim.model.body_id2name(body))
    ]
    if not candidates:
        raise RuntimeError(f"{fixture.name} has no handle body")
    if side == "auto":
        return candidates[0]
    for name in candidates:
        if side in name:
            return name
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe a closed RoboCasa cabinet door")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--side", choices=("auto", "left", "right"), default="left")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("/tmp/opencode/wbc_fixture_probe"))
    args = parser.parse_args(argv)

    import robocasa  # noqa: F401
    import robosuite
    from robosuite import load_part_controller_config

    env = robosuite.make(
        env_name="PickPlaceCounterToCabinet",
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
    backend = RoboCasaPandaBackend(env, base_backoff=0.0, target_mode="offset")
    task = RoboCasaPandaTask(env, backend, max_move_steps=90)
    task.configure(teacher=True)
    try:
        task.build_poses(seed=args.seed)
        fixture = task.cabinet_fixture()
        fixture.set_door_state(0.0, 0.0, env)
        env.sim.forward()
        initial = fixture.get_joint_state(env, fixture.door_joint_names)
        handle_name = _handle_body(env, fixture, args.side)
        joint_name = (
            fixture.door_joint_names[0]
            if len(fixture.door_joint_names) == 1
            else (fixture.door_joint_names[0] if "left" in handle_name else fixture.door_joint_names[1])
        )
        handle_body = int(env.sim.model.body_name2id(handle_name))
        handle = np.asarray(env.sim.data.body_xpos[handle_body], dtype=np.float64).copy()
        hinge = _hinge_world(env, joint_name)
        approach = np.asarray((0.0, 1.0, 0.0))
        closing = np.asarray((0.0, 0.0, 1.0))
        rotation = _rotation_from_axes(approach, closing)

        grasp_pose = handle - 0.108 * approach
        pregrasp_pose = handle - 0.20 * approach
        q0 = np.asarray(backend.data.qpos[backend._arm_q], dtype=np.float64).copy()
        grasp_q = task._solve_position(grasp_pose, q0)
        pregrasp_q = task._solve_position(pregrasp_pose, q0)
        task._move("pregrasp", pregrasp_q, _pose7(pregrasp_pose, rotation), _GRIPPER_OPEN)
        task._move("grasp", grasp_q, _pose7(grasp_pose, rotation), _GRIPPER_OPEN)
        for _ in range(15):
            task._step_action(None, _GRIPPER_CLOSE)

        # The left door opens with +z rotation; the right door mirrors it.
        sign = 1.0 if "left" in handle_name or len(fixture.door_joint_names) == 1 else -1.0
        history: list[dict[str, object]] = []
        for iteration in range(args.iterations):
            handle = np.asarray(env.sim.data.body_xpos[handle_body], dtype=np.float64).copy()
            delta = Rotation.from_euler("z", sign * np.deg2rad(4.0))
            target_handle = hinge + delta.apply(handle - hinge)
            target_pose = target_handle - 0.108 * delta.apply(approach)
            target_rotation = delta * rotation
            target_q = task._solve_position(
                target_pose,
                np.asarray(backend.data.qpos[backend._arm_q], dtype=np.float64).copy(),
            )
            target_q[6] = grasp_q[6]
            task.curr_phase = "push_fixture"
            backend._target_q = target_q
            backend.set_ee_target(_pose7(target_pose, target_rotation))
            for _ in range(3):
                task._step_action(None, _GRIPPER_CLOSE)
            states = fixture.get_joint_state(env, fixture.door_joint_names)
            history.append({"iteration": iteration, "doors": {key: float(value) for key, value in states.items()}})
            if iteration % 5 == 0:
                print(
                    f"iteration={iteration} doors="
                    + ",".join(f"{key.split('_')[-1]}={value:.3f}" for key, value in states.items()),
                    flush=True,
                )

        final = fixture.get_joint_state(env, fixture.door_joint_names)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "fixture_probe.json").write_text(
            __import__("json").dumps(
                {
                    "seed": args.seed,
                    "layout": args.layout,
                    "style": args.style,
                    "handle": handle_name,
                    "joint": joint_name,
                    "initial": initial,
                    "final": final,
                    "history": history,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            "final=" + ",".join(f"{key.split('_')[-1]}={value:.3f}" for key, value in final.items()),
            flush=True,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
