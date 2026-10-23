"""Seedable timed references and scratch-only geometric admission.

Admission checks sampled arm references, not closed-loop dynamic feasibility.
Physics rollout gates remain mandatory, including after an unexpected retarget.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Segment:
    start: float
    duration: float
    coefficients: np.ndarray

    def sample(self, time):
        t = np.clip(time - self.start, 0, self.duration)
        c = self.coefficients
        q = sum(c[k] * t**k for k in range(6))
        v = sum(k * c[k] * t ** (k - 1) for k in range(1, 6))
        a = sum(k * (k - 1) * c[k] * t ** (k - 2) for k in range(2, 6))
        j = sum(k * (k - 1) * (k - 2) * c[k] * t ** (k - 3) for k in range(3, 6))
        if time > self.start + self.duration:
            v, a, j = np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
        return q, v, a, j


def segment(q, goal, speed, acceleration, *, start=0, velocity=None, accel=None, time_scale=1):
    q, goal = np.asarray(q), np.asarray(goal)
    v = np.zeros_like(q) if velocity is None else np.asarray(velocity)
    a = np.zeros_like(q) if accel is None else np.asarray(accel)
    speed, acceleration = np.asarray(speed), np.asarray(acceleration)
    if (
        q.ndim != 1
        or not len(q)
        or any(x.shape != q.shape for x in (goal, v, a, speed, acceleration))
        or not all(np.isfinite(x).all() for x in (q, goal, v, a, speed, acceleration))
        or np.any(speed <= 0)
        or np.any(acceleration <= 0)
        or not np.isfinite(start)
        or start < 0
        or not np.isfinite(time_scale)
        or time_scale <= 0
    ):
        raise ValueError("invalid trajectory boundary conditions or limits")
    jerk = 4 * acceleration
    duration = (
        max(
            0.5,
            float(np.max(1.875 * abs(goal - q) / (0.7 * speed))),
            float(np.max(np.sqrt(5.774 * abs(goal - q) / (0.7 * acceleration)))),
            float(np.max(np.cbrt(60 * abs(goal - q) / jerk))),
        )
        * time_scale
    )
    for _ in range(30):
        t = duration
        c = np.zeros((6, len(q)))
        c[:3] = q, v, a / 2
        matrix = np.array([[t**3, t**4, t**5], [3 * t**2, 4 * t**3, 5 * t**4], [6 * t, 12 * t**2, 20 * t**3]])
        c[3:] = np.linalg.solve(matrix, np.array([goal - q - v * t - a * t * t / 2, -v - a * t, -a]))
        result = Segment(start, duration, c)
        samples = [result.sample(start + u) for u in np.linspace(0, duration, 101)]
        if all(
            np.all(abs(s[1]) <= speed + 1e-8)
            and np.all(abs(s[2]) <= acceleration + 1e-8)
            and np.all(abs(s[3]) <= jerk + 1e-8)
            for s in samples
        ):
            return result
        duration *= 1.15
    raise ValueError("cannot retime reference within derivative bounds")


@dataclass
class ArmTrajectory:
    segments: list[Segment]
    replan_time: float | None = None

    @property
    def end(self):
        return self.segments[-1].start + self.segments[-1].duration

    def sample(self, time):
        selected = self.segments[0]
        for part in self.segments[1:]:
            if time >= part.start:
                selected = part
        # Initial pauses hold the start configuration with zero derivatives.
        if time < selected.start:
            q = selected.sample(selected.start)[0]
            return q, np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
        return selected.sample(time)

    def record(self):
        return {
            "replan_time": self.replan_time,
            "segments": [
                {"start": s.start, "duration": s.duration, "coefficients": s.coefficients.tolist()}
                for s in self.segments
            ],
        }


def build(q, goal, profile, rng, *, replan_time=None, replan_goal=None):
    pause = float(rng.uniform(0, 0.35))
    first = segment(q, goal, profile.speed, profile.acceleration, start=pause, time_scale=float(rng.uniform(1, 1.5)))
    parts = [first]
    if replan_time is not None:
        position, velocity, acceleration, _ = first.sample(max(pause, replan_time))
        if replan_time < pause:
            position, velocity, acceleration = np.asarray(q), np.zeros_like(q), np.zeros_like(q)
        parts.append(
            segment(
                position,
                replan_goal,
                profile.speed,
                profile.acceleration,
                start=replan_time,
                velocity=velocity,
                accel=acceleration,
                time_scale=float(rng.uniform(1, 1.5)),
            )
        )
    return ArmTrajectory(parts, replan_time)


def admit(env, trajectory):
    """Use TAMP collision semantics on copies; preserve rejected contact evidence."""
    from emet.eval.physical_tamp import make_scene_checker

    checker = make_scene_checker(
        env.model, {"object_body": env.payload_name, "gripper_bodies": env.profile.finger_bodies}
    )
    scratch = mujoco.MjData(env.model)
    mujoco.mj_copyData(scratch, env.model, env.data)
    times = list(np.arange(0, trajectory.end, 0.025)) + [trajectory.end]
    samples = [(t, trajectory.sample(t)[0]) for t in times]
    samples += [(s.start + s.duration, s.sample(s.start + s.duration)[0]) for s in trajectory.segments]
    for index, (time, arm) in enumerate(samples):
        if np.any(arm < env.control.limits[:, 0]) or np.any(arm > env.control.limits[:, 1]):
            return {"accepted": False, "reason": "joint_limits", "time": float(time)}
        for group, value in zip(env.control.q, arm, strict=True):
            scratch.qpos[group] = value / len(group)
        if checker.configuration_collides(env.model, scratch):
            return {
                "accepted": False,
                "reason": "collision",
                "time": float(time),
                "contacts": list(checker.last_contacts),
            }
        if (
            env.profile.robot == "innate_mars"
            and env.control.base_mask.any()
            and env.control.hand_mask
            and (index % 4 == 0 or index >= len(times))
        ):
            # Conservative reset-time sweep of the teacher's yaw compensation.
            # This rejects a reference for this execution strategy, not the goal
            # as universally infeasible. The actor never receives oracle contacts.
            address = int(env.control.q[0][0])
            lo, hi = env.control.limits[0]
            for yaw in np.linspace(lo, hi, int(np.ceil((hi - lo) / 0.04)) + 1):
                scratch.qpos[address] = yaw
                if checker.configuration_collides(env.model, scratch):
                    return {
                        "accepted": False,
                        "reason": "base_compensation_collision",
                        "time": float(time),
                        "compensation_joint1": float(yaw),
                        "contacts": list(checker.last_contacts),
                    }
    if env.config.base_only_min_distance and not env.control.hand_mask:
        from emet.rl.manipulation.tracking import TaskFrameFK

        witness = TaskFrameFK(env.model, env.data, env.control)
        witness(env.control.base_goal, trajectory.sample(trajectory.end)[0])
        if checker.configuration_collides(env.model, witness.data):
            return {"accepted": False, "reason": "base_target_collision", "contacts": list(checker.last_contacts)}
    return {
        "accepted": True,
        "reason": "sampled_geometric_path",
        "samples": len(times),
        "dynamic_feasibility": "unverified_until_rollout",
    }


def plot_report(report_path, output):
    """Plot admitted moving references for inspection, with explicit derivative units."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    report = json.loads(Path(report_path).read_text())
    rows = [r for r in report["episodes"] if r["initial"].get("trajectory")]
    if not rows:
        raise ValueError("report has no dynamic references")
    fig, axes = plt.subplots(4, min(3, len(rows)), figsize=(13, 10), squeeze=False, sharex="col")
    for col, row in enumerate(rows[:3]):
        record = row["initial"]["trajectory"]
        path = ArmTrajectory(
            [Segment(s["start"], s["duration"], np.asarray(s["coefficients"])) for s in record["segments"]],
            record["replan_time"],
        )
        times = np.linspace(0, path.end + 0.5, 300)
        values = np.array([path.sample(t) for t in times])
        for derivative, label in enumerate(("Position", "Velocity", "Acceleration", "Jerk")):
            ax = axes[derivative, col]
            for j in range(values.shape[2]):
                ax.plot(times, values[:, derivative, j], label=f"joint {j + 1}")
            if path.replan_time is not None:
                ax.axvline(path.replan_time, color="black", linestyle="--", alpha=0.5)
            ax.set_ylabel(
                label
                + (
                    " (rad" + (f"/s^{derivative}" if derivative else "") + ")"
                    if report["robot"] == "innate_mars"
                    else " (joint units)"
                )
            )
            ax.grid(alpha=0.2)
        axes[0, col].set_title(f"Seed {row['seed']} · success={row['is_success']}")
        axes[-1, col].set_xlabel("Simulation time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Nominal reference targets; dashed line marks replanning. Execution remains closed loop.")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot dynamic references from a local evaluation report")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_report(args.report, args.output)
