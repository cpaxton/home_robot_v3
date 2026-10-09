#!/usr/bin/env python3
"""Native-actuator Stretch table-transfer media; scripted fixture, not TAMP acceptance."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from emet.motion.mujoco_arm_ik import solve_pose_ik
from emet.simulation.mujoco_server import _load_default_scene_with_robot
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", choices=["object1", "object2"], default="object1")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    model = _load_default_scene_with_robot("stretch")
    data = mujoco.MjData(model)
    data.ctrl[:] = model.key_ctrl[0]
    data.ctrl[model.actuator("gripper").id] = 0.04
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    # Select a reachable fixture before simulation; never relocate objects during execution.
    if args.object == "object2":
        a = model.jnt_qposadr[model.body("object1").jntadr[0]]
        b = model.jnt_qposadr[model.body("object2").jntadr[0]]
        data.qpos[b : b + 2] = data.qpos[a : a + 2].copy()
        data.qpos[a] = 0.3
    mujoco.mj_forward(model, data)
    joints = (
        "joint_lift",
        *(f"joint_arm_l{i}" for i in range(4)),
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    )
    events, frames, trace = [], [], []
    renderer = mujoco.Renderer(model, height=480, width=640) if args.render else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, -0.38, 0.65]
    camera.distance, camera.azimuth, camera.elevation = 2.1, 145, -25
    if renderer:
        renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        renderer._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    def advance(stage, seconds=2.0):
        for _ in range(round(seconds * 12)):
            for _ in range(round(1 / (12 * model.opt.timestep))):
                mujoco.mj_step(model, data)
            row = {
                "stage": stage,
                "time": float(data.time),
                "object": data.body(args.object).xpos.tolist(),
                "ee": data.body("link_grasp_center").xpos.tolist(),
                "object_in_ee": (
                    data.body("link_grasp_center").xmat.reshape(3, 3).T
                    @ (data.body(args.object).xpos - data.body("link_grasp_center").xpos)
                ).tolist(),
                "object_speed_m_s": float(
                    np.linalg.norm(data.qvel[model.jnt_dofadr[model.body(args.object).jntadr[0]] :][:3])
                ),
            }
            row["contacts"] = [
                [model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name]
                for c in data.contact
                if model.body(args.object).id in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
            ]
            trace.append(row)
            if renderer:
                from emet.visualization.manip_video import overlay_manip_frame

                renderer.update_scene(data, camera=camera)
                frames.append(
                    overlay_manip_frame(
                        renderer.render(),
                        title="Stretch | table transfer",
                        action=stage,
                        goal=f"{args.object}: pick, transfer, release",
                        flags="NATIVE ACTUATORS | scripted GT fixture | no attachment",
                        overlay_style="border",
                    )
                )
        if renderer:
            from PIL import Image

            Image.fromarray(renderer.render()).save(args.output / f"{len(events):02d}.png")
        print(stage, row, flush=True)
        events.append(row)

    def move(stage, position):
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        result = solve_pose_ik(
            model,
            scratch,
            ee_body="link_grasp_center",
            joint_names=joints,
            target_pos=position,
            target_rotation=rotation,
            coupled_groups=(joints[1:5],),
            tol_m=0.003,
        )
        if not result.success:
            raise RuntimeError(f"{stage}: IK error {result.pos_error_m}")
        values = {n: result.qpos[model.joint(n).qposadr[0]] for n in joints}
        commands = {
            "lift": values["joint_lift"],
            "arm": sum(values[n] for n in joints[1:5]),
            **{n: values["joint_" + n] for n in ("wrist_yaw", "wrist_pitch", "wrist_roll")},
        }
        for name, value in commands.items():
            data.ctrl[model.actuator(name).id] = value
        advance(stage, 3.0)

    def verify_hold(stage):
        samples = [r for r in trace if r["stage"] == stage][-12:]
        for row in samples:
            contacts = {n for pair in row["contacts"] for n in pair}
            if not {"rubber_tip_left", "rubber_tip_right"}.issubset(contacts) or "table" in contacts:
                raise RuntimeError(f"{stage}: missing sustained bilateral grasp")
            if np.linalg.norm(np.asarray(row["object_in_ee"]) - hold_reference) > 0.025:
                raise RuntimeError(f"{stage}: payload slipped")

    status = "failed"
    error = None
    try:
        advance("settle / open")
        initial = data.body(args.object).xpos.copy()
        rotation = data.body("link_grasp_center").xmat.reshape(3, 3).copy()
        goal = initial + [0, -0.16, 0]
        move("pregrasp", initial + [0, 0.08, 0.07])
        move("grasp approach", initial + [0, 0, 0.03])
        data.ctrl[model.actuator("gripper").id] = -0.02
        advance("close gripper")
        move("lift", initial + [0, 0, 0.20])
        if data.body(args.object).xpos[2] < initial[2] + 0.08:
            raise RuntimeError("object_not_lifted")
        hold_reference = np.asarray(trace[-1]["object_in_ee"])
        verify_hold("lift")
        move("transfer", goal + [0, 0, 0.20])
        verify_hold("transfer")
        move("lower onto table", goal + [0, 0, 0.035])
        data.ctrl[model.actuator("gripper").id] = 0.04
        advance("release")
        move("retract", goal + [0, 0.10, 0.12])
        advance("settle / verify", 3.0)
        final = data.body(args.object).xpos.copy()
        if np.linalg.norm(final[:2] - goal[:2]) > 0.04 or abs(final[2] - initial[2]) > 0.015:
            raise RuntimeError("placement_outside_tolerance")
        for row in trace[-24:]:
            contacts = {n for pair in row["contacts"] for n in pair}
            if "table" not in contacts or any("tip_" in n for n in contacts) or row["object_speed_m_s"] > 0.01:
                raise RuntimeError("release_not_stable_on_table")
        status = "success"
    except RuntimeError as exc:
        error = str(exc)
        print(error, flush=True)
    finally:
        if renderer:
            renderer.close()
            from emet.eval.episode_video import write_rgb_sequence_mp4

            write_rgb_sequence_mp4(frames, args.output / "stretch-placement.mp4", fps=12)
        (args.output / "result.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "error": error,
                    "scope": "scripted native-actuator table transfer; not full TAMP acceptance",
                    "object": args.object,
                    "events": events,
                    "criteria": {
                        "lift_min_m": 0.08,
                        "sustained_bilateral_contact_s": 1.0,
                        "retention_translation_m": 0.025,
                        "placement_xy_m": 0.04,
                        "placement_z_m": 0.015,
                        "release_speed_m_s": 0.01,
                        "release_stable_s": 2.0,
                    },
                    "assistance": "ground-truth scripted goals; no latch, weld, or runtime pose writes",
                    "planning_scope": "coupled-joint pose IK; no collision-certified path or agent planning",
                    "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "script_sha256": script_sha256,
                },
                indent=2,
            )
            + "\n"
        )
        (args.output / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
