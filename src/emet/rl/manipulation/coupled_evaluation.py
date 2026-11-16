"""Bounded teacher readouts and target/trace plots for coupled tracking."""

import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv, load_suite
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.regression_gate import ROOT, source_state


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def teacher_episode(env, case):
    """Ordinary resolved-rate teacher, without consulting the private witness."""
    e = env.unwrapped
    trace, ticks, info = [], 0, {}
    initial = np.r_[e.data.qpos.copy(), e.data.qvel.copy()]
    for _ in range(int(np.ceil(e.config.horizon / e.profile.dt))):
        action = task_pose_teacher_action(e.control)
        _, _, done, truncated, info = env.step(action)
        ticks += 1
        trace.append(
            {
                "time_s": ticks * e.profile.dt,
                "base": e.control.base_pose().tolist(),
                "hand": e.control.ee_pose().tolist(),
                "base_target": e.control.base_goal.tolist(),
                "hand_target": e.control.hand_goal.tolist(),
                "action": action.tolist(),
                "position_error_m": info["ee_position_error"],
                "orientation_error_rad": info["ee_orientation_error"],
            }
        )
        if done or truncated:
            break
    e.control.stop()
    stop_ticks = 0
    while not whole_body_stopped(e.control) and stop_ticks < 16 and not e._physics_invalid:
        e._tick(np.zeros(e.profile.action_size))
        stop_ticks += 1
    stopped = whole_body_stopped(e.control) and not e._physics_invalid
    final = e._info()
    safety = {k: bool(final[k]) for k in ("wheel_lift", "unsafe_contact", "retention_lost")}
    safety["invalid_physics"] = bool(e._physics_invalid)
    base_delta = (e.control.base_goal - e.control.base_pose()) * e.control.base_mask
    base_delta[2] = np.arctan2(np.sin(base_delta[2]), np.cos(base_delta[2]))
    pose_ok = (
        (
            not e.control.hand_mask
            or (
                final["ee_position_error"] <= e.config.hand_position_tolerance
                and final["ee_orientation_error"] <= e.config.hand_orientation_tolerance
            )
        )
        and np.linalg.norm(base_delta[:2]) <= e.config.base_xy_tolerance
        and abs(base_delta[2]) <= e.config.base_heading_tolerance
    )
    return {
        "tracking_success": bool(info.get("is_success")) and pose_ok,
        "post_stop_pose_in_contract": bool(pose_ok),
        "code": info.get("code"),
        "control_ticks": ticks,
        "stop_ticks": stop_ticks,
        "stop_confirmed": stopped,
        "retarget_delivered": bool(info.get("retarget_delivered")),
        "retarget_expected": "retarget" in case,
        "safety_flags": safety,
        "final_metrics": final,
        "initial_state_sha256": digest(initial.tolist()),
        "model_fingerprint": e.fingerprint,
        "trace": trace,
    }


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 4))
    base_axes = figure.add_subplot(131)
    hand_axes = figure.add_subplot(132, projection="3d")
    error_axes = figure.add_subplot(133)
    for row in report["cases"]:
        if not row["accepted"]:
            continue
        reference = row["reference"]
        base, hand = np.array(reference["base"]), np.array(reference["hand"])
        label = row["case"]["id"].split(":", 1)[1]
        line = base_axes.plot(*base[:, :2].T, label=label)[0]
        color = line.get_color()
        for index in (0, -1):
            base_axes.quiver(
                *base[index, :2],
                0.005 * np.cos(base[index, 2]),
                0.005 * np.sin(base[index, 2]),
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
            )
        hand_axes.plot(*hand[:, :3].T, color=color)
        hand_axes.scatter(*hand[-1, :3], color=color, marker="x")
        if teacher := row.get("teacher"):
            trace = teacher["trace"]
            measured = np.array([t["hand"][:3] for t in trace])
            hand_axes.plot(*measured.T, color=color, linestyle="--", alpha=0.6)
            error_axes.plot([t["time_s"] for t in trace], [1000 * t["position_error_m"] for t in trace], color=color)
    base_axes.set(xlabel="World x (m)", ylabel="World y (m)", title="Base targets and headings")
    base_axes.set_aspect("equal", adjustable="datalim")
    hand_axes.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Hand: reference / measured dashed")
    error_axes.set(xlabel="Physical time (s)", ylabel="Hand position error (mm)", title="Teacher tracking")
    base_axes.legend(fontsize=5)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def export_distribution(robot, output, *, suite_path, teacher=False):
    suite = load_suite(suite_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = source_state(ROOT)
    (output / "sources.json").write_text(json.dumps(before, indent=2) + "\n")
    env = CoupledTrackingEnv(robot, suite["robots"][robot])
    report = {
        "format": "coupled-tracking-readout-v1",
        "robot": robot,
        "suite_sha256": digest(suite),
        "source_state_digest": digest(before),
        "cases": [],
        "complete": False,
        "teacher_requested": bool(teacher),
        "ready_for_training": False,
        "new_ppo_steps": 0,
        "hardware_used": False,
    }
    try:
        for case in env.cases:
            try:
                env.reset(options={"case_id": case["id"]})
                row = {
                    "case": case,
                    "accepted": True,
                    "admission": env.admission,
                    "reference": env.unwrapped.control.task_trajectory.record(),
                }
                if teacher:
                    row["teacher"] = teacher_episode(env, case)
            except (ValueError, RuntimeError) as exc:
                row = {"case": case, "accepted": False, "reason": str(exc)}
            report["cases"].append(row)
            (output / "references.json").write_text(
                json.dumps(report, indent=2, default=json_value, allow_nan=False) + "\n"
            )
            print(
                case["id"],
                row["accepted"],
                row.get("teacher", {}).get("tracking_success"),
                row.get("reason"),
                flush=True,
            )
    finally:
        env.close()
    if source_state(ROOT) != before:
        raise RuntimeError("source changed during coupled readout")
    report["complete"] = True
    report["ready_for_training"] = bool(
        teacher
        and all(
            r["accepted"]
            and r["teacher"]["tracking_success"]
            and r["teacher"]["stop_confirmed"]
            and not any(r["teacher"]["safety_flags"].values())
            and (not r["teacher"]["retarget_expected"] or r["teacher"]["retarget_delivered"])
            for r in report["cases"]
        )
    )
    (output / "references.json").write_text(json.dumps(report, indent=2, default=json_value, allow_nan=False) + "\n")
    plot_report(report, output / "references.svg")
    print(
        json.dumps(
            {
                "cases": len(report["cases"]),
                "admitted": sum(r["accepted"] for r in report["cases"]),
                "teacher_successes": sum(r.get("teacher", {}).get("tracking_success", False) for r in report["cases"]),
                "ready_for_training": report["ready_for_training"],
            }
        ),
        flush=True,
    )
    return report
