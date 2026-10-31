"""Diagnose checkpoint startup, target sensitivity and actual public TAMP inputs.

Simulation only. Stationary diagnostics use the existing arm certificate's
1 cm / 0.02 rad base envelope, independent force/wheel guards and measured stops.
Offline phase/target predictions never command motion or establish feasibility.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.eval.mars_tamp_control import CANONICAL_SUITE, fixture
from emet.eval.mars_tamp_sequences import manifest, run_sequence
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.safety import whole_body_stopped

PHASES = ("reach", "pregrasp", "close", "carry", "place", "release", "disturbance")


def sources():
    return {**manifest(), "eval/mars_policy_audit.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def describe_input(policy, observation):
    """Report normalized support and clipping without changing actor inputs."""
    fields, width = history_layout(
        MARS, retention_context=policy.manifest["environment"].get("retention_context", False)
    )
    if observation["history"].shape != (MARS.history_steps, width):
        raise ValueError("audit history layout mismatch")
    labels = {}
    for name, columns in fields.items():
        for column in range(columns.start, columns.stop):
            labels[column] = f"{name}/{column - columns.start}"
    for i, phase in enumerate(PHASES):
        labels[fields["phase"].start + i] = "phase/" + phase
    result = {}
    for key, (mean, var) in policy.normalization.items():
        values = np.asarray(observation[key])
        if values.shape != mean.shape or not np.isfinite(values).all():
            raise ValueError("invalid audit observation")
        z = (values - mean) / np.sqrt(var + policy.manifest["epsilon"])
        ranks = np.argsort(np.max(abs(z), axis=0))[-8:][::-1]
        result[key] = {
            "max_abs_z": float(np.max(abs(z))),
            "clipped_fraction": float(np.mean(abs(z) > policy.manifest["clip"])),
            "largest_columns": [
                {
                    "column": int(i),
                    "name": labels[int(i)] if key == "history" else f"preview/{i}",
                    "value": float(values[-1, i]),
                    "mean": float(mean[-1, i]),
                    "std": float(np.sqrt(var[-1, i] + policy.manifest["epsilon"])),
                    "last_z": float(z[-1, i]),
                }
                for i in ranks
            ],
        }
    phase = fields["phase"]
    result["inherited_phase_means"] = dict(zip(PHASES, policy.mean[-1, phase].tolist(), strict=True))
    return result


def phase_predictions(policy, observation):
    """Counterfactual input audit only; neither inference changes physics."""
    fields, _ = history_layout(MARS, retention_context=policy.manifest["environment"].get("retention_context", False))
    result = {}
    for index, phase in enumerate(PHASES):
        value = {k: np.asarray(v).copy() for k, v in observation.items()}
        value["history"][:, fields["phase"]] = 0
        value["history"][:, fields["phase"].start + index] = 1
        result[phase] = policy(value).tolist()
    return result


def exploration_support(policy, action):
    """Describe training noise around the deterministic actor's current output."""
    std = policy.model.policy.log_std.detach().cpu().exp().numpy()
    if std.shape != (MARS.action_size,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid actor exploration standard deviation")
    action = np.asarray(action)
    return {
        "normalized_action_std": std.tolist(),
        "deterministic_action": action.tolist(),
        "distance_from_zero_in_std": (abs(action) / std).tolist(),
        "coordinates_with_zero_outside_two_std": int(np.count_nonzero(abs(action) > 2 * std)),
        "scope": "diagnostic, not a proof that PPO cannot recover; execution remains deterministic",
    }


class ZeroVelocity:
    controller_name = "diagnostic_zero_velocity"

    def __call__(self, observation):
        return np.zeros(MARS.action_size)


def stationary_trial(
    case, directory, evidence, *, selector, warmup_s=0.0, duration_s=2.0, joint_delta=None, wheel_feedback_gain=None
):
    """Matched initial fixtures, no goal motion; never relax a failed envelope."""
    if selector not in {"checkpoint_policy", "zero", "reference", "explicit_hold"}:
        raise ValueError("unknown stationary controller selector")
    if selector == "explicit_hold" and joint_delta is not None:
        raise ValueError("an explicit hold cannot track a displaced target")
    if wheel_feedback_gain is not None and (
        not np.isfinite(wheel_feedback_gain) or not 0 <= wheel_feedback_gain <= 2
    ):
        raise ValueError("diagnostic wheel feedback gain must be in [0, 2]")
    env, admission = fixture(case)
    try:
        if not admission["accepted"]:
            return {"selector": selector, "fixture_admission": admission, "accepted": False}
        robot = MarsSimRobot.from_checkpoint(env, directory, evidence=evidence, record_observations=True)
        actor = robot.policy
        if wheel_feedback_gain is not None:
            robot.servo.base_velocity_feedback_gain = wheel_feedback_gain
        initial = {"qpos": env.data.qpos.tolist(), "qvel": env.data.qvel.tolist(), "ctrl": env.data.ctrl.tolist()}
        # Explicitly request the current pose. Otherwise startup's idle gate
        # correctly suppresses inference, which would hide raw actor hold bias.
        target = robot.servo.positions().copy()
        if joint_delta is not None:
            from emet.controller.task.tamp.mars_native_adapter import MarsNativeTableAdapter

            delta = np.asarray(joint_delta, dtype=float)
            if delta.shape != target.shape or not np.isfinite(delta).all():
                raise ValueError("invalid nearby joint witness")
            target += delta
            executor = MarsNativeTableAdapter(robot, model=env.model)._executor(env.payload_name)
            path, error = executor.plan_joint_target(target)
            if error or path is None:
                return {"selector": selector, "accepted": False, "reason": error}
            import mujoco

            from emet.rl.manipulation.grasp_setup import support_margin

            for q in path:
                executor.data.qpos[executor.qadr] = q
                mujoco.mj_kinematics(env.model, executor.data)
                if support_margin(env.model, executor.data, MARS, None)["margin_m"] < 0.005:
                    return {"selector": selector, "accepted": False, "reason": "nearby_static_support_margin"}
        robot.command_arm_reference(target, blocking=False)
        origin = robot.servo.base_pose().copy()
        origin_position = env.data.xpos[robot.servo.base].copy()
        origin_rotation = env.data.xmat[robot.servo.base].reshape(3, 3).copy()
        start_time = float(env.data.time)

        def tick():
            robot.step(MARS.dt)
            distance = np.linalg.norm(env.data.xpos[robot.servo.base] - origin_position)
            rotation = env.data.xmat[robot.servo.base].reshape(3, 3)
            angle = np.arccos(np.clip((np.trace(origin_rotation.T @ rotation) - 1) / 2, -1, 1))
            if not robot.failure and (distance > 0.01 or angle > 0.02):
                robot.stop_and_confirm("diagnostic_base_envelope")

        # Zero-velocity prehistory is an explicitly labelled ablation. It never
        # invokes a teacher or reinitializes policy history before scored motion.
        if warmup_s:
            robot.policy = ZeroVelocity()
            for _ in range(round(warmup_s / MARS.dt)):
                tick()
                if robot.failure:
                    break
        robot.policy = actor if selector == "checkpoint_policy" else ZeroVelocity() if selector == "zero" else None
        if selector == "explicit_hold":
            robot.policy = ZeroVelocity()
            robot.servo.hold_position()
        scored_start = len(robot.trace)
        calls_start = robot.policy_calls
        for _ in range(round(duration_s / MARS.dt)):
            if robot.failure:
                break
            tick()
        trace = robot.trace[scored_start:]
        hand_errors = [np.linalg.norm(np.asarray(t["hand"])[:3] - np.asarray(t["hand_goal"])[:3]) for t in trace]
        base_errors = [np.linalg.norm(np.asarray(t["base"])[:2] - origin[:2]) for t in trace]
        actor.verify_unchanged()
        current = robot.servo.ee_pose()
        goal = robot.servo.hand_goal
        a, b = current[3:].reshape(3, 2), goal[3:].reshape(3, 2)
        ra = np.column_stack([a, np.cross(a[:, 0], a[:, 1])])
        rb = np.column_stack([b, np.cross(b[:, 0], b[:, 1])])
        angle = float(np.arccos(np.clip((np.trace(rb.T @ ra) - 1) / 2, -1, 1)))
        endpoint = float(np.linalg.norm(current[:3] - goal[:3]))
        return {
            "selector": selector,
            "warmup_s": warmup_s,
            "wheel_feedback_gain_override": wheel_feedback_gain,
            "accepted": True,
            "initial_state": initial,
            "policy_admission": actor.admission,
            "failure": robot.failure,
            "stop_confirmed": robot.stop_confirmed,
            "quiet_at_end": whole_body_stopped(robot.servo),
            "joint_delta": None if joint_delta is None else np.asarray(joint_delta).tolist(),
            "hand_error_at_end_m": endpoint,
            "orientation_error_at_end_rad": angle,
            "goal_reached_at_end": not robot.failure
            and endpoint <= 0.005
            and angle <= 0.05
            and whole_body_stopped(robot.servo),
            "completed_duration": not robot.failure and len(trace) == round(duration_s / MARS.dt),
            "scored_policy_calls": robot.policy_calls - calls_start if selector == "checkpoint_policy" else 0,
            "max_base_drift_m": float(max(base_errors, default=0)),
            "max_hand_error_m": float(max(hand_errors, default=0)),
            "sim_duration_s": float(env.data.time) - start_time,
            "warmup_trace": robot.trace[:scored_start],
            "trace": trace,
            "contact": env.monitor.metrics(),
        }
    finally:
        env.close()


def target_predictions(case, directory, evidence):
    """Probe announced goals with cold history; do not execute any prediction."""
    env, admission = fixture(case)
    try:
        if not admission["accepted"]:
            return {"accepted": False, "fixture_admission": admission}
        robot = MarsSimRobot.from_checkpoint(env, directory, evidence=evidence)
        c = robot.servo
        base, hand = c.base_pose(), c.ee_pose()
        before = env.data.qpos.copy(), env.data.ctrl.copy(), float(env.data.time)
        rows = []
        for displacement in (-0.1, 0.0, 0.1):
            goal = base.copy()
            goal[:2] += displacement * np.array([np.cos(base[2]), np.sin(base[2])])
            c.set_pose_targets(goal, np.ones(3, dtype=bool), hand, False)
            obs = c.observe()
            # Offline cold-start input variant; live history is never cleared.
            obs["history"] = np.repeat(obs["history"][-1:], MARS.history_steps, axis=0)
            rows.append({"forward_displacement_m": displacement, "action": robot.policy(obs).tolist()})
        unchanged = (
            np.array_equal(before[0], env.data.qpos)
            and np.array_equal(before[1], env.data.ctrl)
            and before[2] == float(env.data.time)
        )
        if not unchanged:
            raise RuntimeError("offline target probe changed physics")
        return {"not_executed": True, "physics_unchanged": unchanged, "base_only_predictions": rows}
    finally:
        env.close()


def run(spec, case):
    from emet.rl.manipulation.policy import ManipulationPolicy

    report = {
        "scope": __doc__,
        "source_hashes": sources(),
        "spec": spec,
        "case": case,
        "safety_promoted": False,
        "training_started": False,
    }
    episode = run_sequence(
        case,
        spec["program"],
        policy_directory=spec["checkpoint"],
        policy_evidence=spec["training_evidence"],
        record_observations=True,
    )
    report["tamp_episode"] = episode
    if not episode.get("learned_policy_evaluated"):
        return report
    actor = ManipulationPolicy(
        spec["checkpoint"], robot="innate_mars", fingerprint=episode["policy_admission"]["training_fingerprint"]
    )
    samples = [
        t["actor_observation"] for t in episode["trace"] if "actor_observation" in t and t["action_source"] == "policy"
    ]
    first = {k: np.asarray(v) for k, v in samples[0].items()}
    report["first_input"] = describe_input(actor, first)
    report["phase_predictions_not_executed"] = phase_predictions(actor, first)
    first_policy_trace = next(t for t in episode["trace"] if t["action_source"] == "policy")
    report["exploration_support"] = exploration_support(actor, first_policy_trace["normalized_action"])
    report["stationary_trials"] = [
        stationary_trial(case, spec["checkpoint"], spec["training_evidence"], selector=selector, warmup_s=warmup)
        for selector, warmup in (
            ("reference_teacher", 0.0),
            ("zero", 0.0),
            ("checkpoint_policy", 0.0),
            ("checkpoint_policy", 1.0),
        )
    ]
    report["target_predictions"] = target_predictions(case, spec["checkpoint"], spec["training_evidence"])
    if report["source_hashes"] != sources():
        raise RuntimeError("audit sources changed during execution")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/benchmarks/mars_tamp_policy_development_v1.json"))
    parser.add_argument("--case", default="left:20g:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    spec = json.loads(args.spec.read_text())
    case = next(c for c in json.loads(CANONICAL_SUITE.read_text())["cases"] if c["id"] == args.case)
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(spec, case)
    (args.output / "report.json.gz").write_bytes(
        gzip.compress((json.dumps(result, indent=2, allow_nan=False) + "\n").encode(), mtime=0)
    )
    for row in result.get("stationary_trials", []):
        print(
            row["selector"],
            row["warmup_s"],
            row["completed_duration"],
            row["max_base_drift_m"],
            row["max_hand_error_m"],
            row["failure"],
            flush=True,
        )


if __name__ == "__main__":
    main()
