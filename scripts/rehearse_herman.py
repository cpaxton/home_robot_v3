"""Build and run an offline, captured-room controller rehearsal. Never connects to a robot."""

import argparse
import dataclasses
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.coupled_tracking import CoupledReference, admit_reference
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.goal_change_tracking import nominal_successor
from emet.rl.manipulation.herman_rehearsal import (
    capture_audit,
    joint_values,
    load_capture,
    pose_matrix,
    project_depth,
    scene_voxels,
    seal_artifacts,
    sha,
    transform_error,
)
from emet.rl.manipulation.herman_rehearsal_env import (
    HermanRehearsalEnv,
    geometry_diagnostic,
    scene_factory,
    select_start_proposal,
)
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_observation import actor_observation


# Physical hold envelope for the base during a rehearsal: the independent
# hold-XY / bounded-yaw region the running robot permits before its hold
# actuators give way. This is a safety stop, distinct from the reward's
# acceptance tolerance (`base_xy_tolerance`); the reward dead-zone must stay
# well inside it so the policy holds the base with margin rather than riding
# the boundary. See `whole_body_reward.py` for the reward side.
HOLD_XY_ENVELOPE_M = 0.05
HOLD_YAW_ENVELOPE_RAD = np.deg2rad(47)


def save(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def world_pose(data, body):
    t = np.eye(4)
    t[:3, 3] = data.body(body).xpos.copy()
    t[:3, :3] = data.body(body).xmat.copy().reshape(3, 3)
    return t


def install_posture(model, data, values, head):
    mujoco.mj_resetData(model, data)
    for name, value in {**values, "joint6M": -values["joint6"], "joint_head": head}.items():
        data.qpos[model.joint(name).qposadr[0]] = value
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator >= 0:
            data.ctrl[actuator] = value
    # Static diagnostic FK only, including when outside limits. Never mj_step.
    mujoco.mj_forward(model, data)


def cases(kind="tracking"):
    if kind == "wave":
        from emet.rl.manipulation.herman_gestures import wave_case

        return [wave_case()]
    if kind == "wrist-only":
        from emet.rl.manipulation.herman_gestures import wave_case

        return [wave_case("wrist_wave")]
    if kind != "tracking":
        raise ValueError("unknown rehearsal case set")
    rows = []
    specs = [
        ("hold", [0] * 5, 0, True),
        ("arm_shoulder", [0, 0.012, -0.006, 0, 0], 0, True),
        ("arm_wrist", [0, 0, 0, 0.01, -0.01], 0, True),
        ("yaw_left", [0] * 5, 0.05, False),
        ("yaw_right", [0] * 5, -0.05, False),
        ("handoff", [0, 0.012, -0.006, 0, 0], 0.03, True),
    ]
    for index, (name, delta, turn, hand) in enumerate(specs):
        rows.append(
            {
                "id": "herman-proposed-v1:" + name,
                "seed": 430_000_000 + index,
                "split": "audit",
                "mode": "coupled" if hand else "base_only",
                "travel_m": 0.0,
                "turn_rad": turn,
                "arm_delta": delta,
                "base_mask": [False, False, True] if turn else [True] * 3,
                "initial_kind": "synthetic_geometry_admitted_staging_proposal",
                "hardware_executable": False,
            }
        )
    return rows


def evaluate(env, case, policy, output, name):
    control = env.control
    trace = []
    qpos = []
    times = []
    targets = []
    raw, initial = env.reset(seed=case["seed"])
    if initial.get("initialization_failed"):
        control.stop()
        return {
            "id": case["id"],
            "seed": case["seed"],
            "admitted": False,
            "reason": "initialization_failed",
            "stop_confirmed": bool(whole_body_stopped(control)),
            "success": False,
            "simulation_only": True,
        }
    initial_hash = digest(
        {"qpos": env.data.qpos.tolist(), "qvel": env.data.qvel.tolist(), "ctrl": env.data.ctrl.tolist()}
    )
    if case.get("gesture") in {"wrist_wave", "elbow_wrist_wave"}:
        from emet.rl.manipulation.herman_gestures import HermanWaveReference

        reference = HermanWaveReference(control, gesture=case["gesture"])
    else:
        reference = CoupledReference(control, case)
    admission = admit_reference(env, reference)
    second = None
    second_admission = None
    if case["id"].endswith("handoff"):
        second = nominal_successor(
            control,
            reference,
            case,
            {
                "travel_m": 0,
                "turn_rad": -0.03,
                "arm_delta": (-np.asarray(case["arm_delta"]) * 0.9).tolist(),
                "dwell_s": 0.25,
            },
        )
        second_admission = admit_reference(env, second)
    if not admission["accepted"] or (second_admission is not None and not second_admission["accepted"]):
        control.stop()
        return {
            "id": case["id"],
            "seed": case["seed"],
            "admitted": False,
            "admission": admission,
            "second_admission": second_admission,
            "stop_confirmed": bool(whole_body_stopped(control)),
            "success": False,
            "simulation_only": True,
        }
    env.set_task_trajectory(reference)
    origin = float(env.data.time)
    delivered = False
    for _tick in range(int(env.config.horizon / env.profile.dt)):
        # Record active-loop feedback BEFORE any native terminal stop.
        control.update_task_reference()
        info = env._info()
        obs = actor_observation(control)
        action = task_pose_teacher_action(control) if policy is None else policy(obs)
        trace.append(
            {
                "time_s": float(env.data.time - origin),
                "position_error_m": info["ee_position_error"],
                "rotation_error_rad": info["ee_orientation_error"],
                "action": action.tolist(),
                "measured_joint_positions_rad": control.positions().tolist(),
                "reference_joint_witness_rad": reference.coordinates(control.trajectory_elapsed())[1].tolist(),
                "actuator_position_targets_rad": env.data.ctrl[control.act].copy().tolist(),
                "reference_time_s": float(control.trajectory_elapsed()),
                "measured_arm_velocity": control.velocities().tolist(),
                "commanded_arm_velocity": control.velocity.tolist(),
                "measured_base_velocity": [info["base_linear_velocity_m_s"], info["base_yaw_velocity_rad_s"]],
                "commanded_base_velocity": control.base_command.tolist(),
                "second_delivered": delivered,
            }
        )
        qpos.append(env.data.qpos.copy())
        times.append(float(env.data.time - origin))
        targets.append(np.r_[control.base_goal.copy(), control.hand_goal.copy()])
        _, _, done, truncated, terminal = env.step(action)
        if done or truncated:
            break
        if second is not None and not delivered and env.data.time - origin >= reference.end + 0.25 - 1e-8:
            env.set_task_trajectory(second)
            delivered = True
        # Match the independent physical hold-XY / bounded-yaw envelope.
        base = control.base_pose()
        if (
            np.linalg.norm(base[:2]) > HOLD_XY_ENVELOPE_M
            or abs(np.arctan2(np.sin(base[2]), np.cos(base[2]))) > HOLD_YAW_ENVELOPE_RAD
        ):
            control.stop()
            terminal = {**terminal, "is_success": False, "code": "rehearsal_boundary"}
            break
    metrics = env._info()
    wave_tracking = None
    if case.get("gesture") in {"wrist_wave", "elbow_wrist_wave"}:
        from emet.rl.manipulation.herman_gestures import wave_tracking_gate

        wave_tracking = wave_tracking_gate(trace, reference.end)
        if terminal.get("is_success") and not wave_tracking["passed"]:
            terminal = {**terminal, "is_success": False, "code": "wave_tracking_failed"}
    control.stop()
    stop_start = float(env.data.time)
    while not whole_body_stopped(control) and not env._physics_invalid and env.data.time - stop_start < 0.75 - 1e-9:
        env._tick(np.zeros(env.profile.action_size), score=False)
    stem = case["id"].rsplit(":", 1)[-1] + "-" + name
    (output / (stem + "-trace.json")).write_text(json.dumps(trace, allow_nan=False) + "\n")
    np.savez_compressed(
        output / (stem + "-replay.npz"),
        qpos=np.array(qpos),
        time_s=np.array(times),
        targets=np.array(targets),
        reference_joints=np.array([row["reference_joint_witness_rad"] for row in trace]),
        measured_joints=np.array([row["measured_joint_positions_rad"] for row in trace]),
        actuator_targets=np.array([row["actuator_position_targets_rad"] for row in trace]),
    )
    reference_joints = np.array([row["reference_joint_witness_rad"] for row in trace])
    measured_joints = np.array([row["measured_joint_positions_rad"] for row in trace])
    return {
        "id": case["id"],
        "seed": case["seed"],
        "admitted": True,
        "admission": admission,
        "second_admission": second_admission,
        "success": bool(terminal.get("is_success")),
        "wave_tracking": wave_tracking,
        "reference_certificate": getattr(reference, "certificate", None),
        "joint_motion": {
            "names": list(control.profile.actuators),
            "reference_span_deg": np.rad2deg(np.ptp(reference_joints, axis=0)).tolist(),
            "measured_span_deg": np.rad2deg(np.ptp(measured_joints, axis=0)).tolist(),
            "peak_measured_speed_rad_s": np.max(
                np.abs([row["measured_arm_velocity"] for row in trace]), axis=0
            ).tolist(),
        },
        "code": terminal.get("code"),
        "second_delivered": delivered,
        "stop_confirmed": bool(not env._physics_invalid and whole_body_stopped(control)),
        "initial_state_sha256": initial_hash,
        "reference_sha256": digest(reference.record()),
        "second_reference_sha256": None if second is None else digest(second.record()),
        "model_fingerprint": env.fingerprint,
        "replay": stem + "-replay.npz",
        "trace": stem + "-trace.json",
        "metrics": {
            k: metrics[k]
            for k in (
                "ee_position_error",
                "ee_orientation_error",
                "base_goal_error",
                "base_heading_error_rad",
                "peak_contact_force",
                "peak_base_linear_jerk_m_s3",
                "peak_base_yaw_jerk_rad_s3",
                "wheel_lift",
                "unsafe_contact",
                "settled_seconds",
            )
        },
        "simulation_only": True,
    }


def code_identity():
    return {**source_hashes(), "@rehearsal_runner": sha(__file__)}


def finish(output, report, sources):
    if code_identity() != sources:
        raise RuntimeError("rehearsal source changed during evaluation")
    report["sources"] = sources
    report["complete"] = True
    save(output / "report.json", report)
    seal_artifacts(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", action="append", type=Path, default=[])
    parser.add_argument("--case-set", choices=("tracking", "wave", "wrist-only"), default="tracking")
    parser.add_argument(
        "--include-proposals",
        action="store_true",
        help="also test explicitly synthetic valid starts, never the recorded blocked posture",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output directory required")
    sources = code_identity()
    capture, description = load_capture(args.capture)
    audit = capture_audit(capture, description)
    points, pixels, coverage = project_depth(capture)
    color = cv2.cvtColor(
        cv2.imread(str(args.capture / capture["samples"]["head_left"]["image_asset"]["path"])), cv2.COLOR_BGR2RGB
    )
    if color.shape[:2] != (capture["samples"]["depth"]["height"], capture["samples"]["depth"]["width"]):
        raise ValueError("rectified RGB/depth size mismatch")
    rgb = color[pixels[:, 1], pixels[:, 0]] / 255.0
    # Lidar is an observed thin horizontal band, not invented full-height walls.
    from emet.rl.manipulation.herman_rehearsal import decode_array

    scan = capture["samples"].get("scan")
    laser_points = np.empty((0, 3))
    laser_skew = None if not scan else abs(scan["stamp_ns"] - capture["samples"]["arm"]["stamp_ns"]) / 1e9
    if scan and laser_skew <= 0.05:
        ranges = decode_array(scan["ranges_zlib_base64"], "<f4", (scan["count"],))
        angles = scan["angle_min"] + np.arange(len(ranges)) * scan["angle_increment"]
        valid = np.isfinite(ranges) & (ranges >= scan["range_min"]) & (ranges <= min(scan["range_max"], 2.5))
        laser = np.column_stack(
            [ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid]), np.zeros(valid.sum())]
        )
        t = pose_matrix(capture["transforms"][scan["frame_id"]])
        laser_points = laser @ t[:3, :3].T + t[:3, 3]
    scene = scene_voxels(
        np.vstack([points, laser_points]), np.vstack([rgb, np.tile([0.2, 0.7, 0.9], (len(laser_points), 1))])
    )
    args.output.mkdir(parents=True)
    save(args.output / "capture.json", capture)
    (args.output / "robot_description.urdf").write_text(description)
    for name in ("head_left", "head_right"):
        import shutil

        shutil.copyfile(args.capture / capture["samples"][name]["image_asset"]["path"], args.output / (name + ".jpg"))
    save(
        args.output / "scene.json",
        {
            "geoms": scene,
            "coverage": coverage,
            "laser_band_only": True,
            "laser_pair_skew_s": laser_skew,
            "laser_used": bool(len(laser_points)),
            "laser_omission": None if len(laser_points) else "no scan within 50 ms of arm capture",
            "voxel_m": 0.06,
            "floor": "canonical model plane; physical floor height/softness uncalibrated",
            "hidden_space": "unknown, never certified free",
        },
    )
    baseline = None
    policies = {}
    for index, path in enumerate(args.policy):
        bundle = WholeBodyBundle(path, robot="innate_mars")
        cfg = ManipulationConfig(**bundle.manifest["environment"])
        canonical, canonical_fp = build_manipulation_model(
            "innate_mars",
            payload_asset_set=cfg.payload_asset_set,
            contact_cone=cfg.contact_cone,
            contact_impratio=cfg.contact_impratio,
            two_supports=cfg.two_supports,
            mesh_storage="binary",
        )
        if canonical_fp != bundle.manifest["fingerprint"]:
            raise ValueError("source actor/model mismatch")
        del canonical
        if baseline is None:
            baseline = cfg
        elif dataclasses.asdict(cfg) != dataclasses.asdict(baseline):
            raise ValueError("matched policy environment settings required")
        policies["policy_" + str(index)] = {
            "bundle": bundle,
            "path": str(path),
            "weights_sha256": bundle.manifest["weights_sha256"],
        }
    baseline = baseline or ManipulationConfig(
        robot="innate_mars",
        scenario="reach",
        action_mode="direct_velocity",
        trajectory_preview=True,
        tracking_reward=True,
        retention_context=True,
        whole_body_settling=True,
        base_xy_tolerance=0.05,
        base_heading_tolerance=float(np.deg2rad(5)),
        hand_position_tolerance=0.003,
        hand_orientation_tolerance=0.02,
        terminal_hold_seconds=2,
        horizon=25,
    )
    if args.case_set == "wave":
        baseline = dataclasses.replace(baseline, horizon=max(35.0, baseline.horizon))
    factory = scene_factory(capture, scene)
    model, fingerprint = factory(
        "innate_mars",
        mesh_storage="binary",
        diagnostics=baseline.diagnostics,
        payload_asset_set=baseline.payload_asset_set,
        contact_cone=baseline.contact_cone,
        contact_impratio=baseline.contact_impratio,
        two_supports=baseline.two_supports,
    )
    data = mujoco.MjData(model)
    values = joint_values(capture)
    install_posture(model, data, values, audit["head_rad"])
    base = world_pose(data, "base_link")
    mj_errors = {}
    for name in ("ee_link", "head", "camera_optical_frame", "right_camera_optical_frame"):
        mj_errors[name] = transform_error(
            np.linalg.inv(base) @ world_pose(data, name), pose_matrix(capture["transforms"][name])
        )
        if mj_errors[name]["position_m"] > 0.005 or mj_errors[name]["rotation_rad"] > 0.02:
            audit["physics_blockers"].append("mujoco_kinematic_mismatch:" + name)
    audit["geometry_diagnostic"] = geometry_diagnostic(model, data)
    if audit["geometry_diagnostic"]["proxy_signed_distance_m"] < -0.001:
        audit["physics_blockers"].append("captured_link3_chassis_proxy_penetration")
    audit["mujoco_vs_live_tf"] = mj_errors
    audit["captured_physics_admitted"] = not audit["physics_blockers"]
    mujoco.mj_saveModel(model, str(args.output / "scene.mjb"))
    np.savez_compressed(
        args.output / "captured-replay.npz", qpos=data.qpos.copy()[None], time_s=[0.0], targets=np.zeros((1, 12))
    )
    report = {
        "format": "herman-rehearsal-v1",
        "capture_sha256": sha(args.capture / "capture.json"),
        "audit": audit,
        "model_fingerprint": fingerprint,
        "scene_geoms": len(scene),
        "depth": coverage,
        "laser": {"used": bool(len(laser_points)), "pair_skew_s": laser_skew},
        "hardware_ready": False,
        "captured": {
            "admitted": audit["captured_physics_admitted"],
            "physics_steps": 0,
            "reason": audit["physics_blockers"],
        },
        "policy_sources": {k: {a: v[a] for a in ("path", "weights_sha256")} for k, v in policies.items()},
        "proposals": [],
        "results": {},
        "complete": False,
        "hardware_motion_commands_sent": 0,
        "environment": dataclasses.asdict(baseline),
    }
    save(args.output / "report.json", report)
    if args.include_proposals:
        start, proposal_attempts = select_start_proposal(model, data, values, audit["head_rad"])
        report["proposal_attempts"] = proposal_attempts
        if start is None:
            finish(args.output, report, sources)
            print("NO_ADMITTED_SYNTHETIC_STAGING_PROPOSAL", flush=True)
            return
        report["proposed_start"] = {
            "joint_positions_rad": start.tolist(),
            "delta_from_capture_rad": (start - np.array([values[f"joint{i}"] for i in range(1, 6)])).tolist(),
            "kind": "synthetic_geometry_admitted_staging_proposal",
            "executed_on_robot": False,
            "requires_reposition_validation": True,
        }
        report["proposals"] = cases(args.case_set)
        for name, policy in [("teacher", None)] + [(key, value["bundle"]) for key, value in policies.items()]:
            env = HermanRehearsalEnv(
                baseline,
                initial_arm=start,
                initial_grip=values["joint6"],
                initial_head=audit["head_rad"],
                model_factory=factory,
            )
            rows = []
            try:
                if env.fingerprint != fingerprint:
                    raise RuntimeError("replay and evaluation scene identity mismatch")
                for case in cases(args.case_set):
                    row = evaluate(env, case, policy, args.output, name)
                    rows.append(row)
                    report["results"][name] = rows
                    save(args.output / "report.json", report)
                    print(
                        "CASE",
                        name,
                        case["id"],
                        row.get("admitted"),
                        row.get("success"),
                        row.get("code", row.get("reason")),
                        flush=True,
                    )
            finally:
                env.close()
    # Every proposed case remains in the denominator, including admission fails.
    report["summary"] = {
        name: {
            "attempts": len(rows),
            "admitted": sum(r["admitted"] for r in rows),
            "successes": sum(r["success"] for r in rows),
            "confirmed_stops": sum(r["stop_confirmed"] for r in rows),
        }
        for name, rows in report["results"].items()
    }
    reference = None
    for _name, rows in report["results"].items():
        if reference is None:
            reference = rows
            continue
        for a, b in zip(reference, rows, strict=True):
            for key in (
                "id",
                "seed",
                "initial_state_sha256",
                "reference_sha256",
                "second_reference_sha256",
                "model_fingerprint",
            ):
                if a.get(key) != b.get(key):
                    raise RuntimeError("rehearsal pairing mismatch " + key)
    finish(args.output, report, sources)
    print(
        json.dumps(
            {
                "captured_admitted": report["captured"]["admitted"],
                "blockers": audit["physics_blockers"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
