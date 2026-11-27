#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Inspect why a PandaOmron top-down grasp does not retain the object.

Privileged diagnostic. Prints object category/mass/geometry, finger contact
normals, EE motion, and object motion for one grasp attempt.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import TorqueAdapter
from emet.controller.manipulation.wbc_contract import action_from_normalized_vector
from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.controller.task.tamp.grasp_frames import top_down_grasp_T
from emet.rl.contact import solved_contact_force
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend


def _pose(position: np.ndarray, quat: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.r_[position, quat])


def _body_pos(env, body: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()


def _step(backend: RoboCasaPandaBackend, adapter: TorqueAdapter, steps: int) -> None:
    dt = backend.control_dt
    for _ in range(steps):
        normalized = np.asarray(backend.teacher_action(), dtype=np.float32)
        command = adapter.command(action_from_normalized_vector(normalized, backend.profile), backend._observation())
        backend.step(command, dt)


def _describe_object(env, name: str) -> None:
    model = env.sim.model
    data = env.sim.data
    body = int(model.body_name2id(name))
    joint_ids = [int(j) for j in range(model.njnt) if int(model.jnt_bodyid[j]) == body]
    print(f"body={name} id={body} mass={model.body_mass[body]:.4f} joints={joint_ids}", flush=True)
    for j in joint_ids:
        print(
            f"  joint={model.joint_id2name(j)} type={int(model.jnt_type[j])} "
            f"axis={np.round(model.jnt_axis[j], 3).tolist()} range={np.round(model.jnt_range[j], 3).tolist()}",
            flush=True,
        )
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body:
            continue
        print(
            f"  geom={model.geom_id2name(g)} type={int(model.geom_type[g])} "
            f"size={np.round(model.geom_size[g], 4).tolist()} "
            f"pos={np.round(model.geom_pos[g], 4).tolist()} "
            f"friction={np.round(model.geom_friction[g], 3).tolist()} "
            f"contype={int(model.geom_contype[g])} conaffinity={int(model.geom_conaffinity[g])}",
            flush=True,
        )
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body:
            continue
        center = np.asarray(data.geom_xpos[g], dtype=np.float64)
        size = np.asarray(model.geom_size[g], dtype=np.float64)
        radius = float(np.linalg.norm(size))
        low = np.minimum(low, center - radius)
        high = np.maximum(high, center + radius)
    print(f"  aabb(approx) low={np.round(low, 4).tolist()} high={np.round(high, 4).tolist()}", flush=True)


def _print_contacts(backend: RoboCasaPandaBackend, target: str) -> None:
    model = backend.model
    data = backend.data
    target_id = int(model.body(target).id)
    fingers = {
        "left": int(model.body("gripper0_right_leftfinger").id),
        "right": int(model.body("gripper0_right_rightfinger").id),
    }
    print(f"contacts ncon={int(data.ncon)}", flush=True)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if target_id not in (first, second):
            continue
        other = first if second == target_id else second
        finger = next((label for label, body in fingers.items() if body == other), None)
        normal, force = solved_contact_force(model, data, index)
        frame_normal = np.asarray(contact.frame, dtype=np.float64)[:3]
        print(
            f"  geom={model.geom_id2name(contact.geom1)}/{model.geom_id2name(contact.geom2)} "
            f"finger={finger} pos={np.round(contact.pos, 4).tolist()} "
            f"dir={np.round(frame_normal, 3).tolist()} normal={normal:.3f} force={force:.3f}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="PickPlaceCounterToSink")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--object-body", default="obj_main")
    parser.add_argument("--base-standoff", type=float, default=0.55)
    parser.add_argument("--z-offset", type=float, default=0.02)
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
    meta = env.get_ep_meta()
    print(f"lang={meta.get('lang')!r}", flush=True)
    _describe_object(env, args.object_body)

    object_pos = _body_pos(env, args.object_body)
    base_world = backend._world_base_pose()
    direction = np.asarray(base_world[:2]) - object_pos[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    target_xy = object_pos[:2] + direction * args.base_standoff
    backend.set_world_base_target(
        (
            float(target_xy[0]),
            float(target_xy[1]),
            float(np.arctan2(object_pos[1] - target_xy[1], object_pos[0] - target_xy[0])),
        )
    )
    if not backend.teleport_base_target():
        raise RuntimeError("base teleport failed")
    print(f"base={np.round(backend._world_base_pose(), 3).tolist()}", flush=True)
    print(f"object_start={np.round(object_pos, 4).tolist()}", flush=True)

    quat = Rotation.from_matrix(top_down_grasp_T(object_pos)[:3, :3]).as_quat()
    grasp = object_pos + np.array([0.0, 0.0, args.z_offset])
    backend.set_gripper(True)
    backend.set_ee_target(_pose(grasp + np.array([0.0, 0.0, 0.12]), quat))
    _step(backend, adapter, 40)
    backend.set_ee_target(_pose(grasp, quat))
    _step(backend, adapter, 30)
    ee = np.asarray(backend._observation().arms[0].ee_pose[:3], dtype=np.float64)
    ee_world, _ = backend._world_pose(tuple(np.r_[ee, [0, 0, 0, 1]]))
    print(f"before_close ee_world={np.round(ee_world, 4).tolist()} object={np.round(_body_pos(env, args.object_body), 4).tolist()}", flush=True)
    for finger in ("gripper0_right_leftfinger", "gripper0_right_rightfinger"):
        print(f"  {finger}={np.round(_body_pos(env, finger), 4).tolist()}", flush=True)

    backend.set_gripper(False)
    _step(backend, adapter, 10)
    print(f"after_close finger_q={np.round(backend.data.qpos[backend._grip_q], 4).tolist()}", flush=True)
    _print_contacts(backend, args.object_body)
    before = _body_pos(env, args.object_body)
    backend.set_ee_target(_pose(grasp + np.array([0.0, 0.0, args.lift_m]), quat))
    _step(backend, adapter, 20)
    ee = np.asarray(backend._observation().arms[0].ee_pose[:3], dtype=np.float64)
    ee_world, _ = backend._world_pose(tuple(np.r_[ee, [0, 0, 0, 1]]))
    after = _body_pos(env, args.object_body)
    print(
        f"after_lift ee_world={np.round(ee_world, 4).tolist()} object={np.round(after, 4).tolist()} "
        f"object_rise={after[2] - before[2]:+.4f}",
        flush=True,
    )
    _print_contacts(backend, args.object_body)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
