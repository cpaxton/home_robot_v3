#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Probe grasp retention over depth/orientation candidates.

This is a privileged diagnostic: it uses the target object body to split finger
contact force into object vs environment and measures object rise while lifting.
It is the sub-problem behind "why is the object not captured?"; it does not
train or gate the actor.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import TorqueAdapter
from emet.controller.manipulation.wbc_contract import action_from_normalized_vector
from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.controller.task.tamp.grasp_frames import top_down_grasp_T
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend


def _pose(position: np.ndarray, quat: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.r_[position, quat])


def _body_pos(env, body: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()


def _finger_target_contacts(backend: RoboCasaPandaBackend, target_body: str) -> list[np.ndarray]:
    target = int(backend.model.body(target_body).id)
    finger_bodies = set().union(*backend._contact_sensor.finger_bodies_set)
    points = []
    for index in range(int(backend.data.ncon)):
        contact = backend.data.contact[index]
        first = int(backend.model.geom_bodyid[contact.geom1])
        second = int(backend.model.geom_bodyid[contact.geom2])
        if (first in finger_bodies and second == target) or (second in finger_bodies and first == target):
            points.append(np.asarray(contact.pos, dtype=np.float64).copy())
    return points


def _target_error(backend: RoboCasaPandaBackend) -> tuple[float, float]:
    target = np.asarray(backend._target_ee[:3], dtype=np.float64)
    pose = np.asarray(backend._observation().arms[0].ee_pose[:3], dtype=np.float64)
    return float(np.linalg.norm(target - pose)), 0.0


def _step(backend: RoboCasaPandaBackend, adapter: TorqueAdapter, steps: int) -> None:
    dt = backend.control_dt
    for _ in range(steps):
        normalized = np.asarray(backend.teacher_action(), dtype=np.float32)
        command = adapter.command(action_from_normalized_vector(normalized, backend.profile), backend._observation())
        backend.step(command, dt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="PickPlaceCounterToSink")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--object-body", default="obj_main")
    parser.add_argument("--receptacle-body", default="sink_main")
    parser.add_argument("--base-standoffs", default="0.55,0.70,0.85")
    parser.add_argument("--lift-m", type=float, default=0.03)
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
    backend = RoboCasaPandaBackend(env, base_backoff=0.45)
    adapter = TorqueAdapter(pandaomron_sim_profile())
    backend.reset(seed=0)
    object_pos = _body_pos(env, args.object_body)
    current_pose = backend._observation().arms[0].ee_pose
    _, current_rot = backend._world_pose(current_pose)
    ready_quat = current_rot.as_quat()
    topdown_quat = Rotation.from_matrix(top_down_grasp_T(object_pos)[:3, :3]).as_quat()

    print(f"object={args.object_body} start={np.round(object_pos, 3).tolist()}")
    results = []
    for standoff in (float(value) for value in args.base_standoffs.split(",") if value.strip()):
        base_world = backend._world_base_pose()
        direction = np.asarray(base_world[:2]) - object_pos[:2]
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        target_xy = object_pos[:2] + direction * standoff
        backend.set_world_base_target(
            (
                float(target_xy[0]),
                float(target_xy[1]),
                float(np.arctan2(object_pos[1] - target_xy[1], object_pos[0] - target_xy[0])),
            )
        )
        if not backend.teleport_base_target():
            print(f"base standoff={standoff:.2f}: teleport failed", flush=True)
            continue
        state_qpos = backend.data.qpos.copy()
        state_qvel = backend.data.qvel.copy()
        for orientation, quat in (("ready", ready_quat), ("topdown", topdown_quat)):
            for z_offset in (0.04, 0.02, 0.0, -0.02):
                backend.data.qpos[:] = state_qpos
                backend.data.qvel[:] = state_qvel
                env.sim.forward()
                backend.set_gripper(True)
                start = _body_pos(env, args.object_body)
                pregrasp = start + np.array([0.0, 0.0, 0.12 + z_offset])
                grasp = start + np.array([0.0, 0.0, z_offset])
                backend.set_ee_target(_pose(pregrasp, quat))
                _step(backend, adapter, 40)
                backend.set_ee_target(_pose(grasp, quat))
                _step(backend, adapter, 30)
                backend.set_gripper(False)
                _step(backend, adapter, 10)
                breakdown = backend.contact_breakdown(args.object_body)
                contacts = _finger_target_contacts(backend, args.object_body)
                target_error, _ = _target_error(backend)
                finger_q = np.asarray(backend.data.qpos[backend._grip_q], dtype=np.float64).copy()
                before_lift = _body_pos(env, args.object_body)
                backend.set_ee_target(_pose(grasp + np.array([0.0, 0.0, args.lift_m]), quat))
                _step(backend, adapter, 20)
                after_lift = _body_pos(env, args.object_body)
                rise = float(after_lift[2] - before_lift[2])
                row = {
                    "base_standoff": standoff,
                    "orientation": orientation,
                    "z_offset": z_offset,
                    "finger_target_N": float(breakdown["finger_target_N"]),
                    "finger_environment_N": float(breakdown["finger_environment_N"]),
                    "object_rise_m": rise,
                    "finger_q": finger_q.tolist(),
                    "target_error_m": target_error,
                    "contact_z": [float(point[2]) for point in contacts],
                    "retained": bool(breakdown["finger_target_N"] > 1.0 and rise > 0.01),
                }
                results.append(row)
                print(
                    f"base={standoff:.2f} {orientation:7s} z={z_offset:+.2f} "
                    f"target_N={row['finger_target_N']:7.2f} env_N={row['finger_environment_N']:7.2f} "
                    f"rise={rise:+.4f} err={target_error:.4f} "
                    f"contact_z={np.round(row['contact_z'], 3).tolist() if row['contact_z'] else []} "
                    f"object_z={start[2]:.3f} retained={row['retained']}",
                    flush=True,
                )
    retained = [row for row in results if row["retained"]]
    print("BEST", retained[0] if retained else None)
    backend.close()
    return 0 if retained else 1
    print(f"object={args.object_body} start={np.round(object_pos, 3).tolist()} base={np.round(backend._base_pose(), 3).tolist()}")
    results = []
    for orientation, quat in (("ready", ready_quat), ("topdown", topdown_quat)):
        for z_offset in (0.04, 0.02, 0.0, -0.02, -0.04):
            backend.data.qpos[:] = state_qpos
            backend.data.qvel[:] = state_qvel
            env.sim.forward()
            backend.set_gripper(True)
            start = _body_pos(env, args.object_body)
            pregrasp = start + np.array([0.0, 0.0, 0.12 + z_offset])
            grasp = start + np.array([0.0, 0.0, z_offset])
            backend.set_ee_target(_pose(pregrasp, quat))
            _step(backend, adapter, 40)
            backend.set_ee_target(_pose(grasp, quat))
            _step(backend, adapter, 30)
            backend.set_gripper(False)
            _step(backend, adapter, 10)
            breakdown = backend.contact_breakdown(args.object_body)
            contacts = _finger_target_contacts(backend, args.object_body)
            target_error, _ = _target_error(backend)
            finger_q = np.asarray(backend.data.qpos[backend._grip_q], dtype=np.float64).copy()
            before_lift = _body_pos(env, args.object_body)
            backend.set_ee_target(_pose(grasp + np.array([0.0, 0.0, args.lift_m]), quat))
            _step(backend, adapter, 20)
            after_lift = _body_pos(env, args.object_body)
            rise = float(after_lift[2] - before_lift[2])
            row = {
                "orientation": orientation,
                "z_offset": z_offset,
                "finger_target_N": float(breakdown["finger_target_N"]),
                "finger_environment_N": float(breakdown["finger_environment_N"]),
                "object_rise_m": rise,
                "finger_q": finger_q.tolist(),
                "target_error_m": target_error,
                "contact_z": [float(point[2]) for point in contacts],
                "retained": bool(breakdown["finger_target_N"] > 1.0 and rise > 0.01),
            }
            results.append(row)
            print(
                f"{orientation:7s} z={z_offset:+.2f} target_N={row['finger_target_N']:7.2f} "
                f"env_N={row['finger_environment_N']:7.2f} rise={rise:+.4f} "
                f"err={target_error:.4f} contact_z={np.round(row['contact_z'], 3).tolist() if row['contact_z'] else []} "
                f"object_z={start[2]:.3f} fingers={np.round(finger_q, 3).tolist()} retained={row['retained']}",
                flush=True,
            )
    retained = [row for row in results if row["retained"]]
    print("BEST", retained[0] if retained else None)
    backend.close()
    return 0 if retained else 1


if __name__ == "__main__":
    raise SystemExit(main())
