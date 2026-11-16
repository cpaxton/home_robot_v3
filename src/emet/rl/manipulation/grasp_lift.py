"""Reset-only, physically grasped lift-off tasks with private TAMP witnesses.

The actor starts before lift-off, with ordinary measured history/pose preview.
Offline FK/IK/planning supplies targets, never actor actions or a fallback.
"""

import mujoco
import numpy as np

from emet.motion.arm_rrt import plan_arm_joint_path
from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, segment
from emet.rl.manipulation.grasp_setup import support_margin
from emet.rl.manipulation.safety import retention_error
from emet.rl.manipulation.tamp_cases import _executor
from emet.rl.manipulation.tracking import TaskFrameFK


class SupportedGraspReadiness:
    """Require a quiet half-second relative grasp, not contact presence alone."""

    def __init__(self, dt):
        self.window = int(np.ceil(0.5 / dt)) + 1
        self.samples = []
        self.last = {"ready": False}

    def update(self, relative, *, bilateral, grip_velocity, supported):
        if not bilateral or not supported or not np.isfinite(grip_velocity):
            self.samples = []
        else:
            self.samples.append((tuple(np.asarray(x).copy() for x in relative), abs(float(grip_velocity))))
            self.samples = self.samples[-self.window :]
        translation = rotation = speed = float("inf")
        if self.samples:
            errors = [retention_error(self.samples[0][0], pose) for pose, _ in self.samples]
            translation = max(e[0] for e in errors)
            rotation = max(e[1] for e in errors)
            speed = max(v for _, v in self.samples)
        ready = len(self.samples) == self.window and translation < 0.001 and rotation < 0.005 and speed < 0.01
        self.last = {
            "ready": ready,
            "samples": len(self.samples),
            "translation_m": translation if np.isfinite(translation) else None,
            "rotation_rad": rotation if np.isfinite(rotation) else None,
            "grip_speed_rad_s": speed if np.isfinite(speed) else None,
        }
        return ready


class LiftCollisionGuard:
    """Freeze departure permissions at admission; check fresh measured state."""

    def __init__(self, executor, data):
        self.checker = executor.collision
        self.model = executor.model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, data)
        self.context = self.checker.payload_support_departure(self.model, self.data)
        self.departure = self.context.__enter__()
        self.contacts = []

    def collides(self, measured):
        mujoco.mj_copyData(self.data, self.model, measured)
        # Collision checking must use the actual measured object pose. Nominal
        # rigid attachment is appropriate for planning, not measured feedback.
        self.checker.set_payload(self.model, self.data, self.checker.payload_body, self.checker.payload_parent)
        result = self.checker.configuration_collides(self.model, self.data)
        self.contacts = list(self.checker.last_contacts)
        return result

    def close(self):
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None


def prepare(env, *, goal_delta=None):
    """Admit a full smooth lift and hold, without moving live simulation."""
    from emet.rl.manipulation.env import ResetMotionUnreachable

    control, profile = env.control, env.profile
    if env.grasp_lift_guard is not None:
        env.grasp_lift_guard.close()
        env.grasp_lift_guard = None
    base = control.base_pose().copy()
    height = float(env.np_random.uniform(0.8, 1.0) * env.config.grasp_lift_height)
    record = {
        "accepted": False,
        "kind": "supported_grasp_to_lift",
        "payload_object": env.payload_name,
        "requested_lift_m": height,
        "initial_support_load_N": env.payload_support_load(),
        "initial_object_height_m": float(env.data.xpos[env.payload, 2]),
        "initial_hand": control.ee_pose().tolist(),
        "base": base.tolist(),
        "dynamic_feasibility_verified": False,
        "actor_joint_reference_hint": False,
        "reset_attempts": list(env.grasp_lift_reset_attempts),
        "initial_state": {k: getattr(env.data, k).copy().tolist() for k in ("qpos", "qvel", "ctrl")},
        "initial_time": float(env.data.time),
    }
    env.grasp_lift_admission = record
    guard = None

    def reject(reason):
        if guard is not None:
            record["contacts"] = list(guard.checker.last_contacts)
            guard.close()
        record["reason"] = reason
        env.trajectory_rejections.append(dict(record))
        env.initialization_failed = True
        env.initialization_failure_reason = reason
        env.trajectory_admission = dict(record)
        control.stop()

    if record["initial_support_load_N"] < 0.1 * env.model.body_mass[env.payload] * 9.81:
        reject("grasp_lift_not_initially_supported")
        return
    try:
        goal = env._vertical_reference(height)
    except ResetMotionUnreachable:
        reject("grasp_lift_unreachable")
        return
    if goal_delta is not None:
        delta = np.asarray(goal_delta, dtype=float)
        if delta.shape != goal.shape or not np.isfinite(delta).all() or np.any(abs(delta) > 0.03):
            raise ValueError("invalid reset-only lift goal perturbation")
        goal = goal + delta
        record["goal_joint_delta_rad"] = delta.tolist()
    if np.any(goal < control.limits[:, 0] + 0.02) or np.any(goal > control.limits[:, 1] - 0.02):
        reject("grasp_lift_joint_margin")
        return
    executor = _executor(env)
    guard = LiftCollisionGuard(executor, env.data)
    path, error = executor.plan_joint_target(goal)
    if error:
        reject("grasp_lift_plan:" + error)
        return
    # RRT sampling vertices are geometric checks, not commanded stops. A
    # separately certified direct connection can be retimed as one min-jerk
    # segment instead of braking at every 0.025-rad planner sample.
    direct = plan_arm_joint_path(
        env.model,
        executor.data,
        joint_names=executor.joint_names,
        q_start=control.positions(),
        q_goal=goal,
        collision=executor.collision,
        planner="linear",
        step_size=0.01,
    )
    if direct.success:
        path = [control.positions().copy(), goal.copy()]
    record["direct_shortcut_certified"] = bool(direct.success)
    parts, elapsed = [], 0.0
    if env.config.jerk_limited_lift_reference:
        from emet.rl.manipulation.jerk_reference import compile_joint_path

        trajectory, certificate = compile_joint_path(
            path,
            np.asarray(profile.speed) * 0.35,
            np.asarray(profile.acceleration) * 0.35,
            2 * np.asarray(profile.acceleration),
        )
        parts, elapsed = trajectory.segments, trajectory.end
        record["arm_derivative_certificate"] = certificate
    else:
        for a, b in zip(path[:-1], path[1:], strict=True):
            if np.max(abs(np.asarray(a) - b)) < 1e-10:
                continue
            part = segment(a, b, np.asarray(profile.speed) * 0.5, np.asarray(profile.acceleration) * 0.5, start=elapsed)
            parts.append(part)
            elapsed += part.duration
    if not parts or elapsed + 2 > env.config.horizon:
        record["required_duration_s"] = elapsed + 2
        reject("grasp_lift_horizon")
        return
    trajectory = ArmTrajectory(parts)
    fk = TaskFrameFK(env.model, env.data, control)
    if goal_delta is not None:
        end_hand = fk(base, goal)
        record["perturbed_hand_lift_m"] = float(end_hand[2] - control.ee_pose()[2])
        if record["perturbed_hand_lift_m"] < 0.045:
            reject("grasp_lift_replay_insufficient_height")
            return
    minimum = float("inf")
    for t in np.linspace(0, elapsed, max(2, int(np.ceil(elapsed / profile.dt)) + 1)):
        q = trajectory.sample(t)[0]
        fk(base, q)
        if executor.collision.configuration_collides(env.model, fk.data):
            reject("grasp_lift_path_collision")
            return
        minimum = min(minimum, support_margin(env.model, fk.data, profile, env.payload)["margin_m"])
        if minimum < 0.005:
            reject("grasp_lift_support_margin")
            return
    # All state assignment is target/guard state; qpos/qvel/ctrl/time stay put.
    env.grasp_lift_guard = guard
    env.arm_trajectory = trajectory
    control.set_base_bias_compensation(False)
    control.set_pose_targets(base, np.ones(3, dtype=bool), control.ee_pose(), True)
    env._advance_dynamic_reference()
    record.update(
        accepted=True,
        reason="timed_supported_grasp_lift",
        goal=goal.tolist(),
        duration_s=elapsed,
        minimum_sampled_margin_m=minimum,
        departure_contacts=guard.departure,
        trajectory=trajectory.record(),
        goal_hand=fk(base, goal).tolist(),
    )
    env.trajectory_admission = dict(record)


def prepare_phase_reset(env):
    """Physically execute a verified training prefix; never teleport a held load.

    Admission includes the whole prefix's contact, retention and base limits.
    The original grasp reference is retained. Scored time restarts at handoff,
    while the reference clock records the elapsed prefix explicitly.
    """
    from emet.rl.manipulation.safety import relative_pose, retention_error

    mode = str(env.np_random.choice(["start", "mid", "late", "hold"], p=env.config.grasp_lift_phase_weights))
    env.grasp_lift_start_mode = mode
    fraction = {"start": 0.0, "hold": 1.0}.get(mode)
    if fraction is None:
        fraction = float(env.np_random.uniform(*((0.15, 0.6) if mode == "mid" else (0.7, 0.95))))
    record, c = env.grasp_lift_admission, env.control
    record["phase_reset"] = {"mode": mode, "requested_fraction": fraction, "teacher_assistance": "reset_prefix_only"}
    if mode == "start":
        return
    timestamp = float(env.data.time)
    origin_position = env.data.xpos[c.base].copy()
    origin_rotation = env.data.xmat[c.base].reshape(3, 3).copy()
    last_velocity = c.velocities().copy()
    last_base = c.base_pose().copy()
    last_base_velocity = np.zeros(2)
    acceleration = base_acceleration = None
    failed = None
    for _ in range(int(np.ceil(fraction * env.arm_trajectory.end / env.profile.dt))):
        env._advance_dynamic_reference()
        env._tick(env.teacher_action(), score=True)
        if env._physics_invalid:
            failed = "invalid_physics_lift_prefix"
            break
        translation, rotation = retention_error(env.retention_reference, relative_pose(env.data, c.ee, env.payload))
        position = env.data.xpos[c.base].copy()
        body_rotation = env.data.xmat[c.base].reshape(3, 3).copy()
        angle = float(np.arccos(np.clip((np.trace(origin_rotation.T @ body_rotation) - 1) / 2, -1, 1)))
        failure = (
            env._physics_invalid
            or env.monitor.wheel_lift
            or env.monitor.peak_force > env.profile.unsafe_force
            or translation > 0.02
            or rotation > 0.1
            or np.linalg.norm(position - origin_position) > 0.01
            or angle > 0.02
            or env.grasp_lift_guard.collides(env.data)
        )
        if failure:
            failed = "unsafe_physical_lift_prefix"
            break
        velocity = c.velocities().copy()
        acceleration = (velocity - last_velocity) / env.profile.dt
        base = c.base_pose().copy()
        delta = base - last_base
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        base_velocity = np.array([np.linalg.norm(delta[:2]), delta[2]]) / env.profile.dt
        base_acceleration = (base_velocity - last_base_velocity) / env.profile.dt
        last_velocity, last_base, last_base_velocity = velocity, base, base_velocity
        c.observe()  # Warm measured/command history at the policy cadence.
    record["phase_reset"].update(
        actual_prefix_s=float(env.data.time) - timestamp,
        prefix_force_N=env.monitor.peak_force,
        prefix_failure=failed,
    )
    if failed:
        env.initialization_failed = True
        env.initialization_failure_reason = failed
        record["accepted"] = False
        record["reason"] = failed
        env.grasp_lift_guard.close()
        env.control.stop()
        env.trajectory_admission = dict(record)
        return
    env.reference_time_offset = float(env.data.time) - timestamp
    env.start_time = float(env.data.time)
    env.last_velocity = last_velocity
    env.last_base_pose = last_base
    env.last_base_velocity = last_base_velocity
    env.last_acceleration = acceleration if acceleration is not None else np.zeros(env.profile.arm_size)
    env.last_base_acceleration = base_acceleration if base_acceleration is not None else np.zeros(2)
    # Retention reference and support-departure contract stay frozen. Prefix
    # faults were rejected before resetting scored contact-load statistics.
    env.monitor.reset()
    env._advance_dynamic_reference()
    record["supported_grasp_initial_state"] = record["initial_state"]
    record["initial_state"] = {key: getattr(env.data, key).copy().tolist() for key in ("qpos", "qvel", "ctrl")}
    record["initial_time"] = float(env.data.time)
    env.trajectory_admission = dict(record)


DEFAULT_EXPORT_SEED = 33_000_000


def export_cases(config, destination, *, samples=8, seed=DEFAULT_EXPORT_SEED, policy=None, teacher=False):
    """Export every admission/rejection and optionally measured lift rollouts."""
    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    from emet.rl.manipulation.carry_benchmark import source_hashes
    from emet.rl.manipulation.env import ManipulationEnv
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    if not 1 <= samples <= 100 or (policy is not None and teacher):
        raise ValueError("1–100 cases and one controller required")
    sampling_config = asdict(config)
    config = replace(config, scenario="carry", grasp_lift_probability=1, carry_push_probability=0)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "supported grasp lift development; no qualification",
        "config": asdict(config),
        "sampling_config": sampling_config,
        "source_hashes": source_hashes(),
        "cases": [],
        "complete": False,
        "safety_promoted": False,
        "controller": "teacher" if teacher else "policy" if policy else "geometry_only",
    }
    env = ManipulationEnv(config)
    try:
        if policy is not None and (policy.robot != config.robot or policy.manifest["fingerprint"] != env.fingerprint):
            raise ValueError("lift export policy/model mismatch")
        for i in range(samples):
            obs, info = env.reset(seed=seed + i)
            row = {
                "seed": seed + i,
                "initialization_failed": info["initialization_failed"],
                "admission": env.grasp_lift_admission,
                "rejection_reason": env.initialization_failure_reason,
                "fingerprint": env.fingerprint,
                "randomization": env.randomization,
                "trace": [],
            }
            report["cases"].append(row)
            if not info["initialization_failed"]:
                fk = TaskFrameFK(env.model, env.data, env.control)
                remaining = max(env.profile.dt, env.arm_trajectory.end - env.reference_time_offset)
                times = np.linspace(0, remaining, max(2, int(np.ceil(remaining / env.profile.dt)) + 1))
                hands = np.array(
                    [
                        fk(env.control.base_goal, env.arm_trajectory.sample(t + env.reference_time_offset)[0])
                        for t in times
                    ]
                )
                schedule = TaskPoseTrajectory(
                    times,
                    np.tile(env.control.base_goal, (len(times), 1)),
                    hands,
                    np.full(len(times), env.control.grip_reference),
                    base_mask=np.ones(3, dtype=bool),
                )
                row["task_pose_trajectory"] = schedule.record()
                if policy is not None or teacher:
                    for _ in range(round(config.horizon / env.profile.dt) + 1):
                        action = env.teacher_action() if teacher else policy(obs)
                        obs, _, done, truncated, info = env.step(action)
                        row["trace"].append(
                            dict(
                                time_s=float(env.data.time - env.start_time),
                                hand=env.control.ee_pose().tolist(),
                                goal=env.control.hand_goal.tolist(),
                                base=env.control.base_pose().tolist(),
                                goal_base=env.control.base_goal.tolist(),
                                joint_velocity=env.control.velocities().tolist(),
                                measured_body_velocity=env.data.qvel[env.control.base_dof : env.control.base_dof + 6]
                                .copy()
                                .tolist(),
                                support_load_N=env.payload_support_load(),
                                object_height_m=float(env.data.xpos[env.payload, 2]),
                                **{
                                    k: info[k]
                                    for k in [
                                        "code",
                                        "ee_position_error",
                                        "ee_orientation_error",
                                        "settled_seconds",
                                        "peak_contact_force",
                                        "wheel_lift",
                                    ]
                                },
                            )
                        )
                        if done or truncated:
                            break
            row["final_info"] = info
            print(
                "LIFT_CASE",
                seed + i,
                "admitted",
                not row["initialization_failed"],
                "outcome",
                info.get("code", "geometry_only"),
                flush=True,
            )
            (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["complete"] = True
        if report["source_hashes"] != source_hashes():
            raise RuntimeError("sources changed during lift distribution export")
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        env.close()
    plot_cases(report, destination / "targets.png")
    return report


def plot_cases(report, destination):
    """Equal-scale target paths and actual pose-error traces, if available."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 5), layout="constrained")
    target = figure.add_subplot(121, projection="3d")
    errors = figure.add_subplot(122)
    clouds = []
    for case in report["cases"]:
        if case["initialization_failed"]:
            continue
        hand = np.array(case["task_pose_trajectory"]["hand"])
        clouds.append(hand[:, :3])
        target.plot(*hand[:, :3].T, label=str(case["seed"]))
        trace = case["trace"]
        if trace:
            errors.plot(
                [r["time_s"] for r in trace], [1000 * r["ee_position_error"] for r in trace], label=str(case["seed"])
            )
    if clouds:
        all_points = np.vstack(clouds)
        center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
        radius = max(float(np.ptp(all_points, axis=0).max()) / 2, 0.001) * 1.1
        target.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            zlim=(center[2] - radius, center[2] + radius),
        )
        target.set_box_aspect((1, 1, 1))
    target.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)", title="Admitted hand trajectories")
    errors.axhline(3, color="green", ls="--", label="endpoint requirement")
    errors.set(xlabel="time (s)", ylabel="hand position error (mm)", title=report["controller"])
    if clouds:
        target.legend(fontsize=6)
    errors.legend(fontsize=6)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main():
    import argparse
    import json
    from pathlib import Path

    from emet.rl.manipulation.env import ManipulationConfig
    from emet.rl.manipulation.policy import ManipulationPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/benchmarks/mars_grasp_lift_curriculum_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_EXPORT_SEED)
    controllers = parser.add_mutually_exclusive_group()
    controllers.add_argument("--teacher", action="store_true")
    controllers.add_argument("--policy", type=Path)
    args = parser.parse_args()
    cfg = ManipulationConfig(**json.loads(args.spec.read_text())["environment"])
    actor = ManipulationPolicy(args.policy) if args.policy else None
    export_cases(cfg, args.output, samples=args.samples, seed=args.seed, policy=actor, teacher=args.teacher)


if __name__ == "__main__":
    main()
