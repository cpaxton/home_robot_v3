"""Simulation-only command/measurement traces and paired hold interventions."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.policy import ManipulationPolicy

MODES = ("policy", "policy_feedback_off", "zero", "explicit_hold", "latched_hold", "zero_feedback_off")


def run(policy, *, seed=250000, horizon=12.0, switch_time=6.0):
    cfg = replace(ManipulationConfig(**policy.manifest["environment"]), horizon=horizon)
    result = {}
    for mode in MODES:
        env = ManipulationEnv(cfg)
        try:
            obs, _ = env.reset(seed=seed)
            c = env.control
            c.base_velocity_feedback_gain = 2.0  # Reproduce the historical extra loop.
            c.base_feedback_at_rest = True
            frames = []
            for _ in range(round(horizon / env.profile.dt)):
                elapsed = float(env.data.time - env.start_time)
                active = elapsed >= switch_time - 1e-9
                raw = policy(obs)
                action = raw.copy()
                if active and mode == "policy_feedback_off":
                    c.base_velocity_feedback_gain = 0
                if active and mode not in {"policy", "policy_feedback_off"}:
                    action[:] = 0
                    if mode == "explicit_hold" and not c.holding:
                        c.hold_position()
                    if mode == "latched_hold" and not c.stopped:
                        c.stop()  # Diagnostic fault-stop latch, not an automatic runtime mode.
                    if mode == "zero_feedback_off":
                        c.base_velocity_feedback_gain = 0
                obs, _, done, truncated, info = env.step(action)
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, c.base, spatial, 0)
                forward = env.data.xmat[c.base].reshape(3, 3)[:, 0]
                wheel_dofs = [env.model.body_dofadr[env.model.body(name).id] for name in env.profile.wheel_bodies[:2]]
                frames.append(
                    {
                        "time": float(env.data.time - env.start_time),
                        "intervention": active,
                        "raw_policy_action": raw.tolist(),
                        "submitted_action": action.tolist(),
                        "base_command": c.base_command.tolist(),
                        "arm_velocity_command": c.velocity.tolist(),
                        "arm_measured_velocity": c.velocities().tolist(),
                        "arm_position_target": c.target.tolist(),
                        "arm_position_measured": c.positions().tolist(),
                        "arm_actuator_target": env.data.ctrl[c.act].tolist(),
                        "wheel_actuator_target": env.data.ctrl[c.wheel_act].tolist(),
                        "wheel_measured_velocity": env.data.qvel[wheel_dofs].tolist(),
                        "base_measured_velocity": [float(spatial[3:] @ forward), float(spatial[2])],
                        "base_pose_delta_velocity": env.last_base_velocity.tolist(),
                        "hand_position_error": info["ee_position_error"],
                        "hand_orientation_error": info["ee_orientation_error"],
                        "base_heading_error": info["base_heading_error_rad"],
                        "stopped": c.stopped,
                        "holding": c.holding,
                        "terminal": bool(done or truncated),
                    }
                )
                if done or truncated:
                    break
            # Exclude the slew-down transient and timeout's stop-command rewrite.
            tail = [f for f in frames if f["time"] >= switch_time + 2 and not f["terminal"]]
            summary = dict(info)
            for name in [
                "base_command",
                "base_measured_velocity",
                "base_pose_delta_velocity",
                "arm_velocity_command",
                "arm_measured_velocity",
            ]:
                values = np.asarray([f[name] for f in tail])
                summary[name + "_rms"] = np.sqrt(np.mean(values**2, axis=0)).tolist() if len(values) else None
                summary[name + "_abs_max"] = np.max(np.abs(values), axis=0).tolist() if len(values) else None
            summary["tail_hand_position_error_max"] = max((f["hand_position_error"] for f in tail), default=None)
            result[mode] = {"frames": frames, "summary": summary}
        finally:
            env.close()
    return {
        "scope": "paired simulation interventions; latched_hold is an explicit diagnostic stop, not policy success",
        "policy_sha256": policy.manifest["weights_sha256"],
        "seed": seed,
        "switch_time": switch_time,
        "horizon": horizon,
        "cases": result,
        "hardware_commanded": False,
    }


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(report["cases"]), figsize=(19, 10), sharex=True, layout="constrained")
    for col, (mode, case) in enumerate(report["cases"].items()):
        frames = [f for f in case["frames"] if not f["terminal"]]
        t = [f["time"] for f in frames]
        for row, axis in enumerate([0, 1]):
            axes[row, col].plot(t, [f["base_command"][axis] for f in frames], label="command")
            axes[row, col].plot(t, [f["base_measured_velocity"][axis] for f in frames], label="measured", alpha=0.7)
        axes[2, col].plot(t, [max(abs(v) for v in f["arm_velocity_command"]) for f in frames], label="command")
        axes[2, col].plot(
            t, [max(abs(v) for v in f["arm_measured_velocity"]) for f in frames], label="measured", alpha=0.7
        )
        axes[3, col].plot(t, [1000 * f["hand_position_error"] for f in frames])
        axes[0, col].set_title(mode)
        axes[3, col].set_xlabel("Time (s)")
        for ax in axes[:, col]:
            ax.axvline(report["switch_time"], color="black", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.25)
    for ax, name in zip(
        axes[:, 0],
        ["Base forward (m/s)", "Base yaw (rad/s)", "Arm max speed (joint units/s)", "Hand error (mm)"],
        strict=True,
    ):
        ax.set_ylabel(name)
    axes[0, 0].legend()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=250000)
    args = parser.parse_args()
    report = run(ManipulationPolicy(args.policy, robot="innate_mars"), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report, args.output / "commands-vs-measurements.png")
    for mode, case in report["cases"].items():
        s = case["summary"]
        print(
            mode,
            {
                k: s[k]
                for k in [
                    "base_command_rms",
                    "base_measured_velocity_rms",
                    "arm_measured_velocity_rms",
                    "tail_hand_position_error_max",
                    "unsafe_contact",
                    "wheel_lift",
                ]
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
