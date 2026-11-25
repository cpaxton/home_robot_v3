"""Replay a parameterized Mars pick against a captured scene in MuJoCo."""

import argparse
import hashlib
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.core.command_client import cancel_reactive_goal, command_receipt
from emet.core.command_tracker import TERMINAL
from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.command_runtime import TaskPoseCommandRuntime, send_task_pose_trajectory
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.herman_rehearsal import capture_audit, load_capture, project_depth, scene_voxels
from emet.rl.manipulation.herman_rehearsal_env import HermanRehearsalEnv, geometry_diagnostic, scene_factory
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.tracking import TaskPoseTrajectory
from emet.rl.manipulation.whole_body_observation import ACTOR_KEYS, actor_observation


class FixedBaseApproach(HermanRehearsalEnv):
    frames = None

    def __init__(self, *args, payload_center=None, correction=None, stall_elbow=False, **kwargs):
        self.payload_center = payload_center
        self.correction = correction
        self.stall_elbow = stall_elbow
        self.stall_demand_s = 0.0
        super().__init__(*args, **kwargs)

    def _make_reference(self):
        super()._make_reference()
        if self.payload_center is not None:
            joint = self.model.body(self.payload).jntadr[0]
            address = self.model.jnt_qposadr[joint]
            self.data.qpos[address : address + 3] = self.payload_center
            mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        # Disable wheel requests while retaining physical support and measured drift.
        arm_action = np.r_[0.0, 0.0, action[2:]]
        if self.correction is not None:
            learned = self.correction(actor_observation(self.control))
            arm_action[2:-1] += 0.10 * np.clip(learned[2:-1] - arm_action[2:-1], -0.2, 0.2)
        if self.stall_elbow:
            demanding = abs(arm_action[4]) > 0.07 and abs(self.control.velocities()[2]) < 0.015
            self.stall_demand_s = self.stall_demand_s + self.profile.dt if demanding else 0.0
        if self.frames is not None:
            obs = actor_observation(self.control)
            if np.any(abs(obs["motion"]) > 1e-6):
                self.frames.append({**obs, "action": arm_action.copy()})
        return super().step(arm_action)


def targets(point, pivot, *, stand_off, clearance, lift, pitch):
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all() or point[0] <= 0:
        raise ValueError("grasp point must be finite and in front of the robot")
    relative = point[:2] - pivot
    radius = np.linalg.norm(relative)
    if radius <= max(0.0, stand_off):
        raise ValueError("pinch point is inside the approach stand-off from the arm pivot")
    direction = relative / radius
    rotation = Rotation.from_euler("ZY", [np.arctan2(direction[1], direction[0]), np.deg2rad(pitch)]).as_matrix()
    grip = np.r_[point, rotation[:, :2].ravel()]
    approach = grip.copy()
    approach[:2] -= stand_off * direction
    approach[2] += clearance
    raised = grip.copy()
    raised[2] += lift
    return approach, grip, raised


def approach_schedule(control, target, duration, swing_radius, entry_arm_delta=None):
    base, hand = control.base_pose(), control.ee_pose()
    entry = hand
    if entry_arm_delta is not None:
        start = control.positions()
        posture = start + entry_arm_delta
        if np.any(posture < control.limits[:, 0]) or np.any(posture > control.limits[:, 1]):
            raise ValueError("pickup opening posture exceeds the model joint limits")
        scratch = mujoco.MjData(control.model)
        scratch.qpos[:] = control.data.qpos
        for group, delta in zip(control.q, entry_arm_delta, strict=True):
            scratch.qpos[group] += delta / len(group)
        mujoco.mj_forward(control.model, scratch)
        entry = np.r_[scratch.xpos[control.ee], scratch.xmat[control.ee].reshape(3, 3)[:, :2].ravel()]
    # First lift in place, then cross above the support before descending.
    height = max(hand[2], entry[2], target[2]) + 0.03
    lift = entry.copy()
    lift[2] = height
    cross = target.copy()
    cross[2] = height
    hands = [hand, lift, cross, target]
    times = [0.0, 0.25, 0.75, 1.0]
    if swing_radius is not None:
        pivot = control.data.xanchor[control.model.joint("joint1").id, :2]
        start_x, start_y = lift[:2] - pivot
        end_x, end_y = cross[:2] - pivot
        start_yaw = np.arctan2(start_y, start_x)
        end_yaw = np.arctan2(end_y, end_x)
        turn = np.arctan2(np.sin(end_yaw - start_yaw), np.cos(end_yaw - start_yaw))
        columns = cross[3:].reshape(3, 2)
        arc = []
        for angle in np.linspace(start_yaw, start_yaw + turn, 9):
            orientation = Rotation.from_euler("Z", angle - end_yaw).as_matrix() @ columns
            xy = pivot + swing_radius * np.array([np.cos(angle), np.sin(angle)])
            arc.append(np.r_[xy, height, orientation.ravel()])
        hands = [hand, lift, *arc, cross, target]
        times = [0.0, 0.15, *np.linspace(0.35, 0.75, len(arc)), 0.9, 1.0]
    if entry_arm_delta is not None:
        hands.insert(1, entry)
        times = [0.0, 0.16, *(0.16 + 0.84 * t for t in times[1:])]
    return TaskPoseTrajectory(
        np.asarray(times) * duration,
        [base] * len(hands),
        hands,
        [control.grip_reference] * len(hands),
        base_mask=[True] * 3,
    )


def run(
    pose,
    head,
    grip,
    approach,
    duration,
    swing_radius,
    *,
    factory,
    collect=False,
    pickup=None,
    entry_arm_delta=None,
    correction=None,
    stall_elbow=False,
    two_waypoints=False,
):
    cfg = ManipulationConfig(
        robot="innate_mars",
        action_mode="direct_velocity",
        trajectory_preview=True,
        retention_context=True,
        tracking_reward=pickup is None,
        whole_body_settling=True,
        hand_position_tolerance=0.008,
        hand_orientation_tolerance=0.035,
        terminal_hold_seconds=0.6,
        horizon=duration + (14 + pickup["lift_seconds"] if pickup else 5),
        payload_object="object1" if pickup else None,
    )
    env = FixedBaseApproach(
        cfg,
        initial_arm=pose,
        initial_grip=grip,
        initial_head=head,
        model_factory=factory,
        payload_center=pickup["center"] if pickup else None,
        correction=correction,
        stall_elbow=stall_elbow,
    )
    env.frames = [] if collect else None
    try:
        _, info = env.reset(seed=1900000)
        if info.get("initialization_failed"):
            return {"status": "initialization_failed", "trace": []}, None
        initial_forearm = geometry_diagnostic(env.model, env.data)
        control = env.control
        runtime = TaskPoseCommandRuntime(env)
        client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=runtime.command_message({}))

        def send(action):
            runtime.dispatch_command(action)
            client._state = runtime.command_message({})

        client.send_message = send
        start_s = float(env.data.time)
        start = control.base_pose().copy()
        base_rotation = env.data.xmat[control.base].reshape(3, 3)

        def world_pose(pose):
            return np.r_[
                env.data.xpos[control.base] + base_rotation @ pose[:3], (base_rotation @ pose[3:].reshape(3, 2)).ravel()
            ]

        target = world_pose(approach)
        if two_waypoints:
            path = TaskPoseTrajectory(
                [0.0, duration],
                [start, start],
                [control.ee_pose(), target],
                [control.grip_reference] * 2,
                base_mask=[True] * 3,
            )
        else:
            path = approach_schedule(control, target, duration, swing_radius, entry_arm_delta)
        if pickup:
            pinch, target = map(world_pose, (pickup["pinch"], pickup["lift"]))
            pinch_s = duration + 5
            close_s = pinch_s + 1
            lift_s = close_s + 2
            path = TaskPoseTrajectory(
                [*path.times, duration + 1, pinch_s, close_s, lift_s, lift_s + pickup["lift_seconds"]],
                [start] * (len(path.times) + 5),
                [*path.hand, path.hand[-1], pinch, pinch, pinch, target],
                [*path.gripper, *([pickup["open_rad"]] * 3), *([env.profile.gripper_closed] * 2)],
                base_mask=path.base_mask,
            )
        command = send_task_pose_trajectory(client, path, motion_timeout_s=cfg.horizon + 1)
        if stall_elbow:
            control.limits[2] = control.positions()[2]
        trace = []
        contacts = {}
        update = env.monitor.update
        bilateral = 0.0
        bilateral_tick = False
        pickup_blocker = None
        closure_started = False
        lift_started = False
        lift_start_height = float(env.data.xpos[env.payload, 2])
        peak_sock_lift = 0.0

        def record_contacts(data, payload=None):
            nonlocal bilateral_tick
            evidence = update(data, payload)
            bilateral_tick |= evidence["bilateral_contact"]
            for a, b, _, force in evidence["contacts"]:
                if force > 0.1 and (a in env.monitor.robot_geoms or b in env.monitor.robot_geoms):
                    pair = (env.model.geom(a).name, env.model.geom(b).name)
                    contacts[pair] = max(contacts.get(pair, 0.0), float(force))
            return evidence

        env.monitor.update = record_contacts
        for _ in range(int(np.ceil(cfg.horizon / env.profile.dt)) + 2):
            contacts.clear()
            bilateral_tick = False
            if stall_elbow and env.stall_demand_s >= 0.6:
                pickup_blocker = "elbow_no_progress"
                cancel_reactive_goal(client, command)
                receipt = command_receipt(client, command)
                break
            if pickup:
                elapsed = control.trajectory_elapsed()
                if elapsed >= lift_s:
                    if not lift_started and bilateral < 0.25:
                        pickup_blocker = "no_sustained_bilateral_grasp"
                    elif lift_started and bilateral == 0:
                        pickup_blocker = "grasp_lost_during_lift"
                    lift_started = pickup_blocker is None
                if elapsed >= close_s and not closure_started:
                    hand = control.ee_pose()
                    if np.linalg.norm(hand[:3] - pinch[:3]) > 0.006 or np.linalg.norm(hand[3:] - pinch[3:]) > 0.05:
                        pickup_blocker = "pinch_pose_not_reached"
                    closure_started = pickup_blocker is None
                if pickup_blocker is not None:
                    cancel_reactive_goal(client, command)
                    receipt = command_receipt(client, command)
                    break
                if closure_started:
                    control.set_reference(control.reference, phase="close", grip=env.profile.gripper_closed)
                elif elapsed >= duration - 3:
                    control.set_reference(control.reference, phase="reach", grip=pickup["open_rad"])
                if elapsed >= path.end:
                    error, angle = env._hand_pose_error()
                    at_pose = error < cfg.hand_position_tolerance and angle < cfg.hand_orientation_tolerance
                    if control.holding and not at_pose:
                        control.resume_tracking_from_hold()
                    elif not control.holding and at_pose and bilateral >= 0.25 and whole_body_stopped(control):
                        control.hold_position()
            runtime.advance()
            bilateral = bilateral + env.profile.dt if bilateral_tick else 0.0
            client._state = runtime.command_message({})
            receipt = command_receipt(client, command)
            trace.append(
                {
                    "t_s": float(env.data.time) - start_s,
                    "hand_xyz_m": control.ee_pose()[:3].tolist(),
                    "target_xyz_m": control.hand_goal[:3].tolist(),
                    "arm_rad": control.positions().tolist(),
                    "base_xyt_m_rad": control.base_pose().tolist(),
                    "contacts": [[*pair, force] for pair, force in contacts.items()],
                }
            )
            if pickup:
                peak_sock_lift = max(peak_sock_lift, float(env.data.xpos[env.payload, 2]) - lift_start_height)
                trace[-1].update(
                    {
                        "schedule_s": elapsed,
                        "grip_rad": float(env.data.qpos[control.grip_q]),
                        "grip_reference_rad": control.grip_reference,
                        "sock_xyz_m": env.data.xpos[env.payload].tolist(),
                        "bilateral_contact": bilateral_tick,
                    }
                )
            if receipt["status"] in TERMINAL:
                break
        else:
            cancel_reactive_goal(client, command)
            receipt = command_receipt(client, command)
        outcome = receipt.get("result", {})
        position_error, orientation_error = env._hand_pose_error()
        base_error = control.base_pose() - start
        result = {
            "status": receipt["status"],
            "code": outcome.get("code", outcome.get("failure")),
            "hold_confirmed": outcome.get("stop_confirmed", False),
            "final_hand_error_m": float(np.linalg.norm(control.ee_pose()[:3] - target[:3])),
            "base_travel_m": float(np.linalg.norm(control.base_pose()[:2] - start[:2])),
            "settling": {
                "hand_position_error_m": position_error,
                "hand_orientation_error_rad": orientation_error,
                "base_heading_error_rad": float(np.arctan2(np.sin(base_error[2]), np.cos(base_error[2]))),
                "base_heading_tolerance_rad": env.active_base_heading_tolerance,
                "settled_s": env.settle_time,
            },
            "initial_forearm": initial_forearm,
            "model_fingerprint": env.fingerprint,
            "trajectory": path.record(),
            "trace": trace,
        }
        if pickup:
            sock_lift = float(env.data.xpos[env.payload, 2]) - lift_start_height
            result.update(
                {
                    "sock_lift_m": sock_lift,
                    "peak_sock_lift_m": peak_sock_lift,
                    "bilateral_contact_at_end": bilateral >= 0.25,
                    "pickup_succeeded": result["status"] == "succeeded"
                    and result["hold_confirmed"]
                    and sock_lift >= pickup["minimum_lift"]
                    and bilateral >= 0.25,
                }
            )
            if pickup_blocker is not None:
                result["code"] = pickup_blocker
        qualified = (
            result["status"] == "succeeded"
            and result["hold_confirmed"]
            and all(np.linalg.norm(row["base_xyt_m_rad"][:2] - start[:2]) <= 0.01 for row in trace)
            and all(not row["contacts"] for row in trace)
        )
        return result, env.frames if qualified else None
    finally:
        env.close()


def save_arm_bank(output, frames, replay_path):
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "arm-train.npz"
    np.savez_compressed(
        dataset, **{key: np.asarray([row[key] for row in frames], np.float32) for key in (*ACTOR_KEYS, "action")}
    )
    training = {
        "dataset": dataset.name,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "training_episodes": [
            {"id": "herman-sock:train:synthetic-open-approach", "start": 0, "stop": len(frames), "weight": 1.0}
        ],
    }
    save(
        output / "report.json",
        {
            "scope": "simulated teacher approach only",
            "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            "families": {"arm": {"train": training}, "base": {"train": {}}, "coupled": {"train": {}}},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--target", nargs=3, type=float, metavar=("X", "Y", "Z"), required=True, help="Pinch point in base_link, metres"
    )
    parser.add_argument(
        "--stand-off", type=float, default=0.055, help="Metres toward the shoulder. Negative goes beyond the pinch."
    )
    parser.add_argument("--clearance", type=float, default=0.07)
    parser.add_argument("--lift", type=float, default=0.06)
    parser.add_argument("--pitch", type=float, default=30.0, help="Downward hand approach pitch, degrees")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds for the approach schedule")
    parser.add_argument("--swing-radius", type=float, help="Optional outward arc radius from the shoulder, metres")
    parser.add_argument(
        "--entry-arm-delta", nargs=5, type=float, help="Opening waypoint as five joint offsets in radians"
    )
    parser.add_argument(
        "--box",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "DX", "DY", "DZ"),
        help="Approximate support box center and half extents in base_link",
    )
    parser.add_argument("--staged-capture", type=Path, help="Additional replay from a previously observed open pose")
    parser.add_argument("--arm-bank", type=Path, help="Save teacher frames from a passing staged arm-only approach")
    parser.add_argument("--legacy-forearm-box", action="store_true", help="Use the archived forearm collision box")
    parser.add_argument("--pickup", action="store_true", help="Rehearse closure and lifting of a simulated sock proxy")
    parser.add_argument("--sock-half-size", nargs=3, type=float, default=[0.011, 0.012, 0.012])
    parser.add_argument("--grip-open", type=float, default=0.67, help="Pickup opening in radians")
    parser.add_argument("--lift-forward", type=float, default=0.0, help="Pickup lift's forward travel in metres")
    parser.add_argument("--lift-duration", type=float, default=4.0, help="Seconds for the pickup lift")
    parser.add_argument("--lift-pitch", type=float, help="Final hand pitch in degrees during the lift")
    parser.add_argument(
        "--pinch-offset", nargs=2, type=float, default=[0.019, 0.013], metavar=("HAND_FORWARD", "WORLD_UP")
    )
    parser.add_argument("--correction-actor", type=Path, help="Evaluate a small clipped arm correction in simulation")
    parser.add_argument(
        "--stall-elbow", action="store_true", help="Simulate elbow position target stuck at its reset value"
    )
    parser.add_argument("--two-waypoints", action="store_true", help="Use one approach waypoint before the pinch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new report path")
    if args.arm_bank is not None and (args.staged_capture is None or args.arm_bank.exists()):
        parser.error("arm-bank requires staged-capture and a fresh bank directory")
    if args.pickup and (args.box is None or args.arm_bank is not None):
        parser.error("pickup requires an explicit support box and a separate report from teacher export")
    if args.pickup and not np.isfinite([args.grip_open, args.lift_forward, *args.pinch_offset]).all():
        parser.error("pickup requires finite grip opening, pinch offsets and lift-forward travel")
    if args.pickup and not (np.isfinite(args.lift_duration) and args.lift_duration > 0):
        parser.error("pickup lift duration must be positive and finite")
    if not np.isfinite(args.stand_off) or not all(
        np.isfinite(v) and v > 0 for v in (args.clearance, args.lift, args.duration)
    ):
        parser.error("finite stand-off and positive finite clearance, lift, and duration required")
    if not np.isfinite(args.pitch) or abs(args.pitch) > 90:
        parser.error("pitch must be in [-90, 90] degrees")
    if args.lift_pitch is not None and (not np.isfinite(args.lift_pitch) or abs(args.lift_pitch) > 90):
        parser.error("lift pitch must be in [-90, 90] degrees")
    if args.swing_radius is not None and (not np.isfinite(args.swing_radius) or args.swing_radius <= 0):
        parser.error("swing-radius must be finite and positive")
    if args.entry_arm_delta is not None and not np.isfinite(args.entry_arm_delta).all():
        parser.error("opening joint offsets must be finite")
    if (args.correction_actor or args.stall_elbow) and (not args.pickup or args.arm_bank):
        parser.error("correction and stall experiments require pickup mode")
    if args.two_waypoints and (not args.pickup or args.swing_radius is not None or args.entry_arm_delta is not None):
        parser.error("two-waypoint mode requires pickup with no opening or swing waypoint")
    correction = None
    if args.correction_actor:
        from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle

        correction = WholeBodyBundle(args.correction_actor, robot="innate_mars")
    capture, description = load_capture(args.capture)
    arm_origin = ET.fromstring(description).find("./joint[@name='joint1']/origin")
    origin = [] if arm_origin is None else arm_origin.get("xyz", "").split()
    if len(origin) != 3 or not np.isfinite(pivot := np.asarray(origin, dtype=float)).all():
        parser.error("capture must declare a finite joint1 origin")
    arm = capture["samples"]["arm"]
    positions = dict(zip(arm["names"], arm["position"], strict=True))
    pose = [positions[f"joint{i}"] for i in range(1, 6)]
    audit = capture_audit(capture, description)
    head = audit["head_rad"]
    grip_range = None
    if args.pickup:
        limits = audit["published_joint_limits"]
        driver, mirror = limits.get("joint6"), limits.get("joint6M")
        if driver is None or mirror is None:
            parser.error("pickup requires the captured gripper and mimic limits")
        grip_range = [max(driver[0], -mirror[1]), min(driver[1], -mirror[0])]
        if not grip_range[0] <= 0 < args.grip_open <= grip_range[1]:
            parser.error("grip opening and closure must be inside the captured gripper limits")
    scene = scene_voxels(project_depth(capture)[0])
    if args.box:
        scene.append({"position": args.box[:3], "half_size": args.box[3:], "rgba": [0.55, 0.45, 0.35, 1.0]})
    approach, close, lift = targets(
        args.target, pivot[:2], stand_off=args.stand_off, clearance=args.clearance, lift=args.lift, pitch=args.pitch
    )
    pickup = None
    if args.pickup:
        center = np.asarray(args.target, dtype=float)
        center[2] = args.box[2] + args.box[5] + args.sock_half_size[2]
        offset = close[3:].reshape(3, 2)[:, 0] * args.pinch_offset[0]
        offset[2] += args.pinch_offset[1]
        close[:3] = center + offset
        lift[:3] = close[:3] + [0, 0, args.lift]
        lift[:3] += args.lift_forward * close[3:].reshape(3, 2)[:, 0]
        if args.lift_pitch is not None:
            _, _, lift_pose = targets(
                args.target,
                pivot[:2],
                stand_off=args.stand_off,
                clearance=args.clearance,
                lift=args.lift,
                pitch=args.lift_pitch,
            )
            lift[3:] = lift_pose[3:]
        lift_rise = lift[2] - close[2]
        if lift_rise <= 0:
            parser.error("pickup must specify an upward lift")
        pickup = {
            "center": center.tolist(),
            "pinch": close,
            "lift": lift,
            "open_rad": args.grip_open,
            "lift_seconds": args.lift_duration,
            "minimum_lift": min(0.04, 0.8 * lift_rise),
        }
    report = {
        "scope": "simulator only",
        "hardware_motion_commands_sent": 0,
        "elbow_target_stalled": args.stall_elbow,
        "two_waypoints": args.two_waypoints,
        "correction_actor_sha256": correction.manifest["weights_sha256"] if correction else None,
        "forearm_collision": "proxy_box" if args.legacy_forearm_box else "cad_hull",
        "complete": False,
        "capture_sha256": hashlib.sha256((args.capture / "capture.json").read_bytes()).hexdigest(),
        "initial_head_rad": head,
        "parameter_frame": "base_link",
        "initial_arm_rad": pose,
        "parameters": {
            "pinch_xyz_m": args.target,
            "stand_off_m": args.stand_off,
            "clearance_m": args.clearance,
            "lift_m": args.lift,
            "pitch_deg": args.pitch,
            "duration_s": args.duration,
            "swing_radius_m": args.swing_radius,
            "entry_arm_delta_rad": args.entry_arm_delta,
            "approximate_support_box": args.box,
        },
        "planned_stages": [
            {"stage": "approach", "hand_pose": approach.tolist()},
            {"stage": "reach_pinch", "hand_pose": close.tolist(), "after": "confirmed_open_gripper_approach"},
            {"stage": "close_gripper", "after": "measured_pinch_pose", "requested": args.pickup},
            {"stage": "lift", "hand_pose": lift.tolist(), "after": "verified_grasp", "requested": args.pickup},
        ],
        "approach_runs": {},
    }
    if pickup:
        report["sock_proxy"] = {"center": pickup["center"], "half_size": args.sock_half_size}
        report["parameters"].update(
            grip_open_rad=args.grip_open,
            pinch_offset_m=args.pinch_offset,
            lift_forward_m=args.lift_forward,
            lift_pitch_deg=args.lift_pitch,
            lift_duration_s=args.lift_duration,
            minimum_retained_lift_m=pickup["minimum_lift"],
        )
    initials = [("measured", pose, positions["joint6"])]
    if args.staged_capture:
        staged, _ = load_capture(args.staged_capture)
        source = staged["samples"]["arm"]
        positions = dict(zip(source["names"], source["position"], strict=True))
        alternate = [positions[f"joint{i}"] for i in range(1, 6)]
        alternate[0] -= 0.04
        initials.append(("synthetic_open_start", alternate, 0.349))
    factory = scene_factory(
        capture,
        scene,
        forearm_collision=report["forearm_collision"],
        head_range_rad=audit["published_joint_limits"].get("joint_head"),
        grip_range_rad=grip_range,
        scene_payload_half_size=args.sock_half_size if args.pickup else None,
    )
    save(args.output, report)
    arm_frames = None
    for name, joints, grip in initials:
        try:
            run_record, frames = run(
                joints,
                head,
                grip,
                approach,
                args.duration,
                args.swing_radius,
                factory=factory,
                collect=args.arm_bank is not None and name == "synthetic_open_start",
                pickup=pickup,
                entry_arm_delta=args.entry_arm_delta,
                correction=correction,
                stall_elbow=args.stall_elbow,
                two_waypoints=args.two_waypoints,
            )
            if frames:
                arm_frames = frames
        except ValueError as exc:
            run_record = {"status": "rejected", "reason": str(exc), "trace": []}
        report["approach_runs"][name] = {"initial_arm_rad": joints, "initial_grip_rad": grip, **run_record}
        save(args.output, report)
        print(name, run_record["status"], run_record.get("code"), flush=True)
    report["complete"] = True
    report["exported_teacher_frames"] = len(arm_frames or [])
    save(args.output, report)
    if arm_frames:
        save_arm_bank(args.arm_bank, arm_frames, args.output)


if __name__ == "__main__":
    main()
