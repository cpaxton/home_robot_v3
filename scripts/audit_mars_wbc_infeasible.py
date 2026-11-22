"""Send reachable and deliberately blocked hand requests through the simulated WBC runtime."""

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from emet.core.command_client import command_receipt, update_reactive_goal
from emet.core.command_tracker import TERMINAL
from emet.rl.manipulation.command_runtime import TaskPoseCommandRuntime, send_task_pose_trajectory
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle


def make_path(env, case, *, elbow_delta=0.025):
    control = env.control
    base = control.base_pose()
    hand = control.ee_pose()
    target = hand.copy()
    if case == "reachable":
        joints = control.positions().copy()
        joints[2] += elbow_delta
        target = TaskFrameFK(env.model, env.data, control)(base, joints)
    elif case == "inside_chassis":
        target[:3] = env.data.geom("rl_base_link").xpos
    elif case == "below_floor":
        target[2] = -0.035
    else:
        raise ValueError(f"Unknown case {case}")
    return TaskPoseTrajectory([0, 3], [base, base], [hand, target], [control.grip_reference] * 2, base_mask=[True] * 3)


def run(case, seed, config, actor=None, *, retarget=False):
    env = ManipulationEnv(config, mesh_storage="binary")
    try:
        _, initial = env.reset(seed=seed)
        if initial["initialization_failed"]:
            raise RuntimeError("Could not initialize the common simulator start")
        start = float(env.data.time)
        first_robot_contact_s = None
        robot_peak_force = robot_impulse = 0.0
        update = env.monitor.update

        def observe_contact(data, payload=None):
            nonlocal first_robot_contact_s, robot_peak_force, robot_impulse
            evidence = update(data, payload)
            # The runtime's aggregate also counts the bookkeeping payload
            # resting on its support. Record only contacts involving the robot.
            total = sum(
                force
                for a, b, _, force in evidence["contacts"]
                if a in env.monitor.robot_geoms or b in env.monitor.robot_geoms
            )
            if first_robot_contact_s is None and total > 0.1:
                first_robot_contact_s = float(data.time) - start
            robot_peak_force = max(robot_peak_force, total)
            robot_impulse += total * env.model.opt.timestep
            return evidence

        env.monitor.update = observe_contact
        runtime = TaskPoseCommandRuntime(env, policy=actor)
        client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=runtime.command_message({}))

        def send(action):
            runtime.dispatch_command(action)
            client._state = runtime.command_message({})

        client.send_message = send
        path = make_path(env, "reachable", elbow_delta=0.12) if retarget else make_path(env, case)
        action = send_task_pose_trajectory(client, path, motion_timeout_s=12)
        last = command_receipt(client, action)
        retarget_time_s = None
        for _ in range(round(env.config.horizon / env.profile.dt) + 2):
            runtime.advance()
            client._state = runtime.command_message({})
            last = command_receipt(client, action)
            if last["status"] in TERMINAL:
                break
            elapsed_s = float(env.data.time) - start
            if retarget and retarget_time_s is None and elapsed_s >= 0.6:
                path = make_path(env, case)
                replacement = update_reactive_goal(client, action, {"trajectory": path.record()})
                receipt = command_receipt(client, replacement)
                if receipt["status"] != "succeeded":
                    raise RuntimeError(f"Replacement was not applied: {receipt}")
                retarget_time_s = elapsed_s
        if retarget and retarget_time_s is None:
            raise RuntimeError(f"Initial waypoint ended before replacement: {last}")
        outcome = last.get("result", {})
        measured = outcome.get("measurements", {})
        elapsed_s = float(env.data.time) - start
        stop_confirmed = outcome.get("stop_confirmed", False)
        return {
            "case": f"retarget_{case}" if retarget else case,
            "controller": "feedback" if actor is None else "learned",
            "retarget_at_s": retarget_time_s,
            "target_hand_xyz_m": path.hand[-1, :3].tolist(),
            "status": last["status"],
            "reason": last.get("reason"),
            "controller_code": outcome.get("code"),
            "failure": outcome.get("failure"),
            "stop_confirmed": stop_confirmed,
            "first_robot_contact_over_0_1_n_s": first_robot_contact_s,
            "first_contact_to_confirmed_hold_s": (
                elapsed_s - first_robot_contact_s if stop_confirmed and first_robot_contact_s is not None else None
            ),
            "peak_sim_robot_contact_total_n": robot_peak_force,
            "sim_robot_contact_impulse_ns": robot_impulse,
            "end_hand_error_m": measured.get("ee_position_error"),
            "simulated_elapsed_s_including_stop": elapsed_s,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1900000)
    parser.add_argument("--actor", type=Path, help="Match a whole-body actor against the feedback controller")
    args = parser.parse_args()
    actor = WholeBodyBundle(args.actor, robot="innate_mars") if args.actor else None
    config = (
        ManipulationConfig(**actor.manifest["environment"])
        if actor is not None
        else ManipulationConfig(
            robot="innate_mars",
            action_mode="direct_velocity",
            trajectory_preview=True,
        )
    )
    config = replace(config, scenario="reach", randomization_scale=0, horizon=10)
    cases = (
        ("reachable", False),
        ("inside_chassis", False),
        ("below_floor", False),
        ("inside_chassis", True),
        ("below_floor", True),
    )
    rows = [
        run(case, args.seed, config, policy, retarget=retarget)
        for policy in ((None, actor) if actor is not None else (None,))
        for case, retarget in cases
    ]
    report = {
        "scope": "simulation only",
        "actor_weights_sha256": actor.manifest["weights_sha256"] if actor is not None else None,
        "seed": args.seed,
        "cases": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(row["controller"], row["case"], row["status"], row["controller_code"], row["stop_confirmed"])


if __name__ == "__main__":
    main()
