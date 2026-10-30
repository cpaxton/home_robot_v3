"""Generate arm-path fixtures with the real TAMP executor's offline planner.

These are component tests, complementary to full public-tool oracle runs.
Planning operates on a copy; the live simulator state is never attached or moved.
"""

import argparse
import json
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.eval.physical_tamp import make_scene_checker
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv


def _executor(env):
    model, profile = env.model, env.profile
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, env.data)
    joints = tuple(name for group in profile.joint_groups for name in group)
    checker = make_scene_checker(model, {"object_body": env.payload_name, "gripper_bodies": profile.finger_bodies})
    if env.scenario in {"carry", "place", "disturbance"}:
        checker.set_payload(model, scratch, env.payload_name, profile.ee_body)
    executor = PhysicalPickPlaceExecutor(
        None,
        model=model,
        data=scratch,
        ee_body=profile.ee_body,
        joint_names=joints,
        collision=checker,
        synchronize=lambda _: None,
        command_joints=lambda _: False,
        coupled_groups=tuple(group for group in profile.joint_groups if len(group) > 1),
        base_body=profile.base_body,
        input_mode="privileged",
    )
    executor.payload_body = checker.payload_body
    return executor


def plan_case(env, target):
    profile = env.profile
    executor = _executor(env)
    goal = np.concatenate(
        [np.repeat(value / len(group), len(group)) for group, value in zip(profile.joint_groups, target, strict=True)]
    )
    path, error = executor.plan_joint_target(goal)
    return {
        "planner": "PhysicalPickPlaceExecutor.plan_joint_target",
        "geometry_source": "simulator_ground_truth",
        "scenario": env.scenario,
        "fingerprint": env.fingerprint,
        "joint_names": executor.joint_names,
        "goal": goal.tolist(),
        "waypoints": None if path is None else np.asarray(path).tolist(),
        "planning_error": error,
        "contacts": executor.collision.last_contacts,
        "initial_state": {name: getattr(env.data, name).tolist() for name in ("qpos", "qvel", "ctrl")},
        "initial_time": float(env.data.time),
        "randomization": env.randomization,
        "split": "generated_component_test",
        "execution_verified": False,
    }


def task_pose_trajectory(env, case, *, jerk_limited=False):
    """Convert an admitted TAMP arm path into timed task-frame commands.

    FK and retiming happen once on scratch data. Executing the resulting
    schedule requires poses and timestamps, not an online IK loop.
    This does not extend the joint executor's certificate to dynamic execution.
    """
    from emet.rl.manipulation.dynamic_trajectory import segment
    from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory

    if case["planning_error"] or case["fingerprint"] != env.fingerprint:
        raise ValueError("pose conversion requires an admitted plan for this model")
    if not np.allclose(case["initial_state"]["qpos"], env.data.qpos, atol=1e-10, rtol=0):
        raise ValueError("stale TAMP trajectory")
    names = list(case["joint_names"])
    waypoints = [
        np.array([sum(q[names.index(name)] for name in group) for group in env.profile.joint_groups])
        for q in case["waypoints"]
    ]
    fk = TaskFrameFK(env.model, env.data, env.control)
    base = env.control.base_pose().copy()
    if jerk_limited:
        from emet.rl.manipulation.jerk_pose_reference import FKPoseReference
        from emet.rl.manipulation.jerk_reference import compile_joint_path

        path, certificate = compile_joint_path(
            waypoints,
            np.asarray(env.profile.speed) * 0.35,
            np.asarray(env.profile.acceleration) * 0.35,
            2 * np.asarray(env.profile.acceleration),
        )
        return FKPoseReference(fk, base, path, env.control.grip_reference, certificate)
    times, hands = [0.0], [fk(base, waypoints[0])]
    elapsed = 0.0
    for q, goal in zip(waypoints[:-1], waypoints[1:], strict=True):
        part = segment(q, goal, np.asarray(env.profile.speed) * 0.5, np.asarray(env.profile.acceleration) * 0.5)
        for t in np.linspace(0, part.duration, max(2, int(np.ceil(part.duration / env.profile.dt)) + 1))[1:]:
            times.append(elapsed + float(t))
            hands.append(fk(base, part.sample(t)[0]))
        elapsed += part.duration
    if len(times) == 1:
        times.append(env.profile.dt)
        hands.append(hands[0].copy())
    return TaskPoseTrajectory(
        times,
        np.repeat(base[None], len(times), axis=0),
        hands,
        np.repeat(env.control.grip_reference, len(times)),
        base_mask=np.ones(3, dtype=bool),
        hand_mask=True,
    )


def execute_case(env, case, policy=None, *, hold_base=False):
    from emet.controller.manipulation import physical_pick_place
    from emet.rl.manipulation.safety import relative_pose, retention_error

    if case["planning_error"] or case["fingerprint"] != env.fingerprint:
        raise ValueError("execution requires a valid plan for the current model")
    if policy is not None and policy.manifest["fingerprint"] != env.fingerprint:
        raise ValueError("policy/model mismatch")
    if not np.allclose(case["initial_state"]["qpos"], env.data.qpos, atol=1e-10, rtol=0):
        raise ValueError("stale oracle fixture")
    env.control.action_mode = policy.action_mode if policy is not None else env.config.action_mode
    if policy is not None:
        env.control.pose_only_targets = policy.manifest.get("pose_only_targets", False)
        env.control.preview_enabled = "preview" in policy.normalization
        env.control.preview_sampler = env._sample_task_reference if env.control.preview_enabled else None
    executor = _executor(env)
    events, trace = [], []
    failure = None
    max_translation = max_rotation = 0.0
    max_hand_position = max_hand_orientation = 0.0

    def synchronize(data):
        mujoco.mj_copyData(data, env.model, env.data)

    def command(q):
        values = dict(zip(executor.joint_names, q, strict=True))
        reference = [sum(values[name] for name in group) for group in env.profile.joint_groups]
        env.control.set_reference(reference, phase=env.scenario)
        # These planner cases are stationary-base paths, even when the training
        # reset sampled an active base goal. Update hand observations from FK.
        env.control.set_pose_targets(
            env.control.base_goal if hold_base else env.control.base_pose(),
            np.ones(3, dtype=bool) if hold_base else np.zeros(3, dtype=bool),
            env._fk_hand_pose(env.control.base_goal if hold_base else env.control.base_pose(), reference),
            True,
        )
        return not env.control.stopped

    def sleep(duration):
        nonlocal failure, max_translation, max_rotation, max_hand_position, max_hand_orientation
        for _ in range(max(1, round(duration / env.profile.dt))):
            obs = env.control.observe()
            env._tick(env.teacher_action() if policy is None else policy(obs))
            position_error, angle_error = env._hand_pose_error()
            max_hand_position = max(max_hand_position, position_error)
            max_hand_orientation = max(max_hand_orientation, angle_error)
            translation, rotation = retention_error(
                env.retention_reference, relative_pose(env.data, env.control.ee, env.payload)
            )
            max_translation = max(max_translation, translation or 0)
            max_rotation = max(max_rotation, rotation or 0)
            failure = failure or (
                "invalid_physics"
                if env._physics_invalid
                else "wheel_lift"
                if env.monitor.wheel_lift
                else "unsafe_contact"
                if env.monitor.peak_force > env.profile.unsafe_force
                else "retention_lost"
                if max_translation > 0.02 or max_rotation > 0.1
                else None
            )
            if failure:
                env.control.stop()
            trace.append(
                {
                    "time": float(env.data.time),
                    "q": env.control.positions().tolist(),
                    "reference": env.control.reference.tolist(),
                    "failure": failure,
                    "ee_position_error": position_error,
                    "ee_orientation_error": angle_error,
                }
            )

    executor.synchronize, executor.command_joints, executor.event = (
        synchronize,
        command,
        lambda **row: events.append(row),
    )
    previous_time = physical_pick_place.time
    try:
        physical_pick_place.time = SimpleNamespace(monotonic=lambda: float(env.data.time), sleep=sleep)
        result = executor._execute_path(env.scenario, [np.asarray(q) for q in case["waypoints"]])
        # Endpoint agreement alone can occur while moving. Independently require
        # one second of settled physical hold with the same joint tolerance.
        held = 0.0
        for _ in range(100 if result.success else 0):
            sleep(env.profile.dt)
            error = np.max(abs(env.control.positions() - env.control.reference))
            position_error, angle_error = env._hand_pose_error()
            held = (
                held + env.profile.dt
                if error <= executor.joint_tolerance
                and position_error < env.config.hand_position_tolerance
                and angle_error < env.config.hand_orientation_tolerance
                and np.max(abs(env.control.velocities())) < 0.03
                else 0
            )
            if failure or held >= 1:
                break
        return {
            "execution_verified": bool(result.success and held >= 1 and failure is None),
            "executor_result": asdict(result),
            "controller_failure": failure,
            "settled_hold_seconds": held,
            "events": events,
            "trace": trace,
            "max_hand_tracking_position_error": max_hand_position,
            "max_hand_tracking_orientation_error": max_hand_orientation,
            "settled_hand_position_error": env._hand_pose_error()[0],
            "settled_hand_orientation_error": env._hand_pose_error()[1],
            "max_payload_translation": max_translation,
            "max_payload_rotation": max_rotation,
            "contact_metrics": env.monitor.metrics(),
        }
    finally:
        physical_pick_place.time = previous_time


def execute_pose_case(env, case, policy=None):
    """Replay one admitted GT path through public command identity/receipts.

    Component coverage only: it does not discover a task, grasp an object, or
    establish end-to-end public-tool acceptance. No deadline or goal relaxation.
    """
    import threading

    from emet.core.command_client import close_command_session, command_receipt
    from emet.rl.manipulation.command_runtime import TaskPoseCommandRuntime, send_task_pose_trajectory

    trajectory = task_pose_trajectory(env, case)
    server = TaskPoseCommandRuntime(env, policy=policy)
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=server.command_message({}))

    def send(action):
        server.dispatch_command(action)
        client._state = server.command_message({})

    client.send_message = send
    action = send_task_pose_trajectory(client, trajectory, motion_timeout_s=env.config.horizon)
    receipt = None
    trace = []
    try:
        for _ in range(int(np.ceil(env.config.horizon / env.profile.dt)) + 1):
            server.advance()
            client._state = server.command_message({})
            receipt = command_receipt(client, action)
            trace.append(
                {
                    "time": float(env.data.time),
                    "base": env.control.base_pose().tolist(),
                    "hand": env.control.ee_pose().tolist(),
                    "status": receipt["status"],
                }
            )
            if receipt["status"] in {"succeeded", "failed", "cancelled", "rejected"}:
                break
    finally:
        close_command_session(client)
    result = receipt.get("result", {}) if receipt else {}
    return {
        "execution_interface": "task_pose_command_protocol",
        "execution_verified": bool(receipt and receipt["status"] == "succeeded" and result.get("stop_confirmed")),
        "command_receipt": receipt,
        "trace": trace,
        "controller_failure": None
        if receipt and receipt["status"] == "succeeded"
        else result.get("failure", result.get("code", "command_incomplete")),
        "contact_metrics": env.monitor.metrics(),
        "task_pose_trajectory": trajectory.record(),
    }


def generate(
    *,
    episodes=10,
    seed=600000,
    execute=False,
    policy=None,
    robot="stretch",
    emit_pose_trajectories=False,
    execution_interface="direct",
):
    if execution_interface not in {"direct", "command"}:
        raise ValueError("unknown execution interface")
    if execution_interface == "command" and policy is not None and policy.action_mode != "direct_velocity":
        raise ValueError("pose commands require a direct-velocity policy")
    if episodes < 1:
        raise ValueError("positive episode count required")
    if robot not in {"stretch", "innate_mars"}:
        raise ValueError("robot must be stretch or innate_mars")
    rows = []
    for scenario in ("reach", "carry"):
        env = ManipulationEnv(
            ManipulationConfig(
                robot=robot,
                scenario=scenario,
                action_mode=policy.action_mode
                if policy
                else "direct_velocity"
                if execution_interface == "command"
                else "residual",
                hand_position_tolerance=policy.manifest.get("environment", {}).get("hand_position_tolerance", 0.015)
                if policy
                else 0.015,
                hand_orientation_tolerance=policy.manifest.get("environment", {}).get("hand_orientation_tolerance", 0.1)
                if policy
                else 0.1,
                trajectory_preview=execution_interface == "command"
                or (policy is not None and "preview" in policy.normalization),
                tracking_reward=policy is not None
                and policy.manifest.get("environment", {}).get("tracking_reward", False),
            )
        )
        try:
            for index in range(episodes):
                episode_seed = seed + index + (10000 if scenario == "carry" else 0)
                _, info = env.reset(seed=episode_seed)
                if info["initialization_failed"]:
                    rows.append({"seed": episode_seed, "scenario": scenario, "initialization_failed": True})
                    continue
                rng = np.random.default_rng(episode_seed)
                target = env.control.positions() + rng.uniform(-1, 1, 5) * [0.03, 0.02, 0.12, 0.08, 0.08]
                target[0] = max(target[0], env.control.positions()[0])
                target = np.clip(target, env.control.limits[:, 0] + 0.01, env.control.limits[:, 1] - 0.01)
                row = {"seed": episode_seed, **plan_case(env, target)}
                if emit_pose_trajectories and row["planning_error"] is None:
                    row["task_pose_trajectory"] = task_pose_trajectory(env, row).record()
                if execute and row["planning_error"] is None:
                    row.update(
                        execute_pose_case(env, row, policy)
                        if execution_interface == "command"
                        else execute_case(env, row, policy)
                    )
                rows.append(row)
        finally:
            env.close()
    return {
        "robot": robot,
        "cases": rows,
        "scope": "GT TAMP arm planner component cases; not end-to-end task success",
        "execution_supported": True,
        "execution_gate": None,
    }


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=600000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--emit-pose-trajectories", action="store_true")
    parser.add_argument("--execution-interface", choices=["direct", "command"], default="direct")
    parser.add_argument("--robot", choices=["stretch", "innate_mars"], default="stretch")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = None
    if args.policy:
        from emet.rl.manipulation.policy import ManipulationPolicy

        policy = ManipulationPolicy(args.policy, robot=args.robot)
    report = generate(
        episodes=args.episodes,
        seed=args.seed,
        execute=args.execute,
        policy=policy,
        robot=args.robot,
        emit_pose_trajectories=args.emit_pose_trajectories,
        execution_interface=args.execution_interface,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
