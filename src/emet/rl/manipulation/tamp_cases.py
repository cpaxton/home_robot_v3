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


def execute_case(env, case, policy=None):
    from emet.controller.manipulation import physical_pick_place
    from emet.rl.manipulation.safety import relative_pose, retention_error

    if case["planning_error"] or case["fingerprint"] != env.fingerprint:
        raise ValueError("execution requires a valid plan for the current model")
    if policy is not None and policy.manifest["fingerprint"] != env.fingerprint:
        raise ValueError("policy/model mismatch")
    if not np.allclose(case["initial_state"]["qpos"], env.data.qpos, atol=1e-10, rtol=0):
        raise ValueError("stale oracle fixture")
    executor = _executor(env)
    events, trace = [], []
    failure = None
    max_translation = max_rotation = 0.0

    def synchronize(data):
        mujoco.mj_copyData(data, env.model, env.data)

    def command(q):
        values = dict(zip(executor.joint_names, q, strict=True))
        reference = [sum(values[name] for name in group) for group in env.profile.joint_groups]
        env.control.set_reference(reference, phase=env.scenario)
        return not env.control.stopped

    def sleep(duration):
        nonlocal failure, max_translation, max_rotation
        for _ in range(max(1, round(duration / env.profile.dt))):
            obs = env.control.observe()
            env._tick(np.zeros(6) if policy is None else policy(obs))
            translation, rotation = retention_error(
                env.retention_reference, relative_pose(env.data, env.control.ee, env.payload)
            )
            max_translation = max(max_translation, translation or 0)
            max_rotation = max(max_rotation, rotation or 0)
            failure = failure or (
                "wheel_lift"
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
            held = (
                held + env.profile.dt
                if error <= executor.joint_tolerance and np.max(abs(env.control.velocities())) < 0.03
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
            "max_payload_translation": max_translation,
            "max_payload_rotation": max_rotation,
            "contact_metrics": env.monitor.metrics(),
        }
    finally:
        physical_pick_place.time = previous_time


def generate(*, episodes=10, seed=600000, execute=False, policy=None):
    if episodes < 1:
        raise ValueError("positive episode count required")
    rows = []
    for scenario in ("reach", "carry"):
        env = ManipulationEnv(ManipulationConfig(scenario=scenario))
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
                if execute and row["planning_error"] is None:
                    row.update(execute_case(env, row, policy))
                rows.append(row)
        finally:
            env.close()
    return {"cases": rows, "scope": "GT TAMP arm planner component cases; not end-to-end task success"}


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=600000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = None
    if args.policy:
        from emet.rl.manipulation.policy import ManipulationPolicy

        policy = ManipulationPolicy(args.policy, robot="stretch")
    report = generate(episodes=args.episodes, seed=args.seed, execute=args.execute, policy=policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
