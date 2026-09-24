#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Serial, privileged MuJoCo carry diagnostic; NOT a learned-agent benchmark.

Restore a sampled physical trace checkpoint, keep arm/gripper references fixed,
and compare bounded wheel profiles. ``recorded`` replays interpolated controls
as a fidelity check, not an exact command replay. Solver overrides and release
negative controls are explicit diagnostic options, never production settings.
No teleport, attachment, or policy access to ground truth is used.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def drive_envelope(t, *, ramp_s=2.0, cruise_s=4.0, brake_s=2.0):
    if t < ramp_s:
        return max(0.0, t / ramp_s)
    if t < ramp_s + cruise_s:
        return 1.0
    return max(0.0, 1.0 - (t - ramp_s - cruise_s) / brake_s)


def wheel_controls(mode, t, *, linear_speed, angular_speed, radius, separation, gears):
    """Velocity actuator references include transmission gearing."""
    envelope = drive_envelope(t, brake_s=0.25 if mode == "brake" else 2.0)
    v = linear_speed * envelope if mode in {"straight", "brake"} else 0.0
    w = angular_speed * envelope if mode == "turn" else 0.0
    return np.array([v - separation * w / 2, v + separation * w / 2]) / radius * gears


def base_yaw(qpos):
    """Yaw of the fixture's leading free joint (MuJoCo wxyz quaternion)."""
    w, x, y, z = qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def bounded_turn_rate(error, previous, dt, speed, acceleration=0.25):
    """Same feedback/acceleration for both speed arms; no pose teleport."""
    target = np.sign(error) * min(speed, 1.5 * abs(error), np.sqrt(2 * acceleration * abs(error)))
    return float(previous + np.clip(target - previous, -acceleration * dt, acceleration * dt))


def local_point(world_point, origin, rotation):
    """Convert a world point to a body/geom frame (MuJoCo row-major xmat)."""
    return np.asarray(rotation).reshape(3, 3).T @ (np.asarray(world_point) - origin)


def object_contacts(model, data, targets, fingers):
    """Read-only contact evidence, including contacts outside the gripper."""
    import mujoco

    result = []
    for index, contact in enumerate(data.contact):
        geoms = [int(g) for g in contact.geom]
        bodies = [int(model.geom_bodyid[g]) for g in geoms]
        if not any(body in targets for body in bodies):
            continue
        other = 1 if bodies[0] in targets else 0
        geom = geoms[other]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        result.append(
            {
                "geoms": [model.geom(g).name for g in geoms],
                "other_geom_id": geom,
                "gripper_contact": bodies[other] in fingers,
                "distance_m": float(contact.dist),
                "dimension": int(contact.dim),
                "friction": contact.friction.tolist(),
                "contact_frame": contact.frame.tolist(),
                "wrench_contact_frame": wrench.tolist(),
                "position_other_geom": local_point(contact.pos, data.geom_xpos[geom], data.geom_xmat[geom]).tolist(),
                "other_geom_size": model.geom_size[geom].tolist(),
                "object_center_other_geom": local_point(
                    data.body(bodies[1 - other]).xpos, data.geom_xpos[geom], data.geom_xmat[geom]
                ).tolist(),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--duration", type=float, default=35.0)
    parser.add_argument("--sample-period", type=float, default=0.1)
    parser.add_argument(
        "--contact-details", action="store_true", help="Log object contacts and actuator state; read-only."
    )
    parser.add_argument(
        "--motion-delay",
        type=float,
        default=0.0,
        help="Stationary seconds before synthetic wheel profiles; recorded controls are unaffected.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["hold", "straight", "turn", "brake", "recorded", "recorded_wheels", "release"],
        default=["hold", "straight", "turn", "brake", "recorded"],
    )
    parser.add_argument("--linear-speed", type=float, default=0.05)
    parser.add_argument("--angular-speed", type=float, default=0.2)
    parser.add_argument("--wheel-radius", type=float, default=0.0508)
    parser.add_argument("--wheel-separation", type=float, default=0.3153)
    parser.add_argument("--left-actuator", default="left_wheel_vel")
    parser.add_argument("--right-actuator", default="right_wheel_vel")
    parser.add_argument("--turn-angle", type=float, help="Closed-loop measured yaw target, radians; turn mode only.")
    parser.add_argument(
        "--noslip-iterations", type=int, help="Explicit solver-only diagnostic override; omit for original physics."
    )
    parser.add_argument("--gripper-actuator", default="gripper")
    parser.add_argument(
        "--open-control", type=float, help="Explicit open actuator target for release negative control."
    )
    args = parser.parse_args()
    for value in (
        args.duration,
        args.sample_period,
        args.wheel_radius,
        args.wheel_separation,
        args.linear_speed,
        args.angular_speed,
    ):
        if not np.isfinite(value) or value <= 0:
            parser.error("duration, sample period, geometry and speeds must be finite and positive")
    if not np.isfinite(args.time):
        parser.error("time must be finite")
    if not np.isfinite(args.motion_delay) or not 0 <= args.motion_delay < args.duration:
        parser.error("motion delay must be finite, nonnegative and shorter than duration")
    if args.turn_angle is not None and (not np.isfinite(args.turn_angle) or not 0 < abs(args.turn_angle) < np.pi):
        parser.error("turn angle must be finite, nonzero and smaller than pi radians")
    if args.noslip_iterations is not None and args.noslip_iterations < 0:
        parser.error("noslip iterations must be nonnegative")
    if "release" in args.modes and (args.open_control is None or not np.isfinite(args.open_control)):
        parser.error("release requires an explicit finite open-control actuator value")
    with args.trace.open() as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    times = np.asarray([row["sim_time"] for row in rows])
    if not len(times) or not np.all(np.diff(times) > 0) or not times[0] <= args.time <= times[-1]:
        parser.error("checkpoint must lie within a strictly increasing trace")
    index = int(np.argmin(abs(times - args.time)))
    initial = rows[index]
    if not initial["gripper_contact"]:
        parser.error("checkpoint must have recorded gripper contact")
    if any(mode.startswith("recorded") for mode in args.modes) and initial["sim_time"] + args.duration > times[-1]:
        parser.error("recorded profile extends beyond trace; reduce duration")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        parser.error("diagnostic requires the fixture base to be the leading free joint")
    original_noslip = int(model.opt.noslip_iterations)
    if args.noslip_iterations is not None:
        model.opt.noslip_iterations = args.noslip_iterations
    gripper_id = model.actuator(args.gripper_actuator).id if "release" in args.modes else None
    if gripper_id is not None and model.actuator_ctrllimited[gripper_id]:
        if not model.actuator_ctrlrange[gripper_id, 0] <= args.open_control <= model.actuator_ctrlrange[gripper_id, 1]:
            parser.error("open-control exceeds actuator limits")
    ids = [model.actuator(args.left_actuator).id, model.actuator(args.right_actuator).id]
    gears = model.actuator_gear[ids, 0]
    if np.any(gears == 0):
        parser.error("wheel transmission gear must be nonzero")
    for mode in args.modes:
        peak = wheel_controls(
            mode,
            3,
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
            radius=args.wheel_radius,
            separation=args.wheel_separation,
            gears=gears,
        )
        for actuator, target in zip(ids, peak, strict=True):
            if (
                model.actuator_ctrllimited[actuator]
                and not model.actuator_ctrlrange[actuator, 0] <= target <= model.actuator_ctrlrange[actuator, 1]
            ):
                parser.error("requested wheel profile exceeds actuator limits")
    target_id = model.body(header["config"]["object_body"]).id
    ee_id = model.body(header["config"]["ee_body"]).id
    gripper_roots = {model.body(name).id for name in header["config"]["gripper_bodies"]}

    def under(body, roots):
        while body:
            if body in roots:
                return True
            body = int(model.body_parentid[body])
        return False

    targets = {i for i in range(model.nbody) if under(i, {target_id})}
    fingers = {i for i in range(model.nbody) if under(i, gripper_roots)}
    controls = np.asarray([row["ctrl"] for row in rows])
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(
        kind="privileged_checkpoint_diagnostic",
        checkpoint_time=initial["sim_time"],
        mujoco_version=mujoco.__version__,
        original_noslip_iterations=original_noslip,
        effective_noslip_iterations=int(model.opt.noslip_iterations),
        actuator_names=[model.actuator(i).name for i in range(model.nu)],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        caveat="Sampled state/controls, not exact replay. Wheel dimensions are explicit fixture assumptions.",
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "checkpoint.json").write_text(json.dumps(initial) + "\n")
    summaries = []
    for mode in args.modes:
        data = mujoco.MjData(model)
        for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
            values = np.asarray(initial[name])
            if values.shape != getattr(data, name).shape or not np.isfinite(values).all():
                raise ValueError(f"Checkpoint incompatible with model: {name}")
            getattr(data, name)[:] = values
        data.time = initial["sim_time"]
        mujoco.mj_forward(model, data)
        reference = data.body(ee_id).xmat.reshape(3, 3).T @ (data.body(target_id).xpos - data.body(ee_id).xpos)
        initial_z = float(data.body(target_id).xpos[2])
        previous_yaw = base_yaw(data.qpos)
        yaw_travel = 0.0
        turn_rate = 0.0
        next_sample = 0.0
        lost_since = None
        drop_time = None
        max_slip = 0.0
        with (args.out / f"{mode}.jsonl").open("x") as stream:
            while data.time - initial["sim_time"] < args.duration:
                elapsed = data.time - initial["sim_time"]
                if mode == "recorded":
                    data.ctrl[:] = [np.interp(data.time, times, controls[:, i]) for i in range(model.nu)]
                elif mode == "recorded_wheels":
                    data.ctrl[ids] = [np.interp(data.time, times, controls[:, i]) for i in ids]
                else:
                    data.ctrl[ids] = wheel_controls(
                        mode,
                        elapsed - args.motion_delay,
                        linear_speed=args.linear_speed,
                        angular_speed=args.angular_speed,
                        radius=args.wheel_radius,
                        separation=args.wheel_separation,
                        gears=gears,
                    )
                    if mode == "turn" and args.turn_angle is not None and elapsed >= args.motion_delay:
                        turn_rate = bounded_turn_rate(
                            args.turn_angle - yaw_travel, turn_rate, model.opt.timestep, args.angular_speed
                        )
                        data.ctrl[ids] = (
                            np.array([-1.0, 1.0]) * args.wheel_separation * turn_rate / (2 * args.wheel_radius) * gears
                        )
                    if mode == "release":
                        # Open after 2 s, ramping over 1 s. No attachment removal,
                        # object repositioning, or extra force is applied.
                        blend = np.clip(elapsed - 2.0, 0.0, 1.0)
                        data.ctrl[gripper_id] = (1 - blend) * initial["ctrl"][gripper_id] + blend * args.open_control
                mujoco.mj_step(model, data)
                yaw = base_yaw(data.qpos)
                yaw_travel += float(np.arctan2(np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw)))
                previous_yaw = yaw
                elapsed = data.time - initial["sim_time"]
                if elapsed < next_sample:
                    continue
                next_sample = elapsed + args.sample_period
                ee = data.body(ee_id)
                obj = data.body(target_id)
                relative = ee.xmat.reshape(3, 3).T @ (obj.xpos - ee.xpos)
                slip = float(np.linalg.norm(relative - reference))
                max_slip = max(max_slip, slip)
                forces = []
                for j, contact in enumerate(data.contact):
                    a, b = [int(model.geom_bodyid[g]) for g in contact.geom]
                    if (a in targets and b in fingers) or (b in targets and a in fingers):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, j, force)
                        if force[0] > 0:
                            forces.append(float(force[0]))
                # Sustained displacement, not momentary missing contact, defines
                # this diagnostic's loss event. It is not the benchmark scorer.
                if np.linalg.norm(relative) > 0.12:
                    lost_since = elapsed if lost_since is None else lost_since
                    if elapsed - lost_since >= 0.2 and drop_time is None:
                        drop_time = lost_since
                else:
                    lost_since = None
                row = {
                    "elapsed_s": elapsed,
                    "object_xyz": obj.xpos.tolist(),
                    "ee_xyz": ee.xpos.tolist(),
                    "relative_xyz": relative.tolist(),
                    "slip_m": slip,
                    "normal_forces_n": forces,
                    "wheel_controls": data.ctrl[ids].tolist(),
                    "base_qpos": data.qpos[:7].tolist(),
                    "yaw_travel_rad": yaw_travel,
                }
                if args.contact_details:
                    row.update(
                        object_contacts=object_contacts(model, data, targets, fingers),
                        actuator_ctrl=data.ctrl.tolist(),
                        actuator_length=data.actuator_length.tolist(),
                        actuator_velocity=data.actuator_velocity.tolist(),
                        actuator_force=data.actuator_force.tolist(),
                        object_rotation=obj.xmat.tolist(),
                        ee_rotation=ee.xmat.tolist(),
                    )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        summary = {
            "mode": mode,
            "loss_time_s": drop_time,
            "max_relative_slip_m": max_slip,
            "final_relative_distance_m": float(np.linalg.norm(relative)),
            "final_height_change_m": float(data.body(target_id).xpos[2]) - initial_z,
            "final_contact": bool(forces),
            "measured_yaw_rad": yaw_travel,
            "turn_target_reached": (
                abs(yaw_travel - args.turn_angle) <= 0.01 if mode == "turn" and args.turn_angle is not None else None
            ),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
