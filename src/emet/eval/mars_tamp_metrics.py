"""Measured tracking/motion diagnostics for public-tool Mars executions."""

from collections import Counter

import numpy as np


def motion_metrics(trace):
    if not trace:
        return {"samples": 0}
    times = np.asarray([x["time"] for x in trace])
    hand = np.asarray([x["hand"] for x in trace])
    target = np.asarray([x["hand_goal"] for x in trace])

    def rotations(poses):
        columns = poses[:, 3:].reshape(-1, 3, 2)
        return np.concatenate([columns, np.cross(columns[:, :, 0], columns[:, :, 1])[:, :, None]], axis=2)

    position = np.linalg.norm(hand[:, :3] - target[:, :3], axis=1)
    relative = np.einsum("nji,njk->nik", rotations(target), rotations(hand))
    angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    measured = np.asarray([x["joint_velocity"] for x in trace])
    commanded = np.asarray([x["commanded_joint_velocity"] for x in trace])
    result = {
        "samples": len(trace),
        "duration_s": float(times[-1] - times[0]),
        "hand_position_rms_m": float(np.sqrt(np.mean(position**2))),
        "hand_position_max_m": float(position.max()),
        "hand_rotation_rms_rad": float(np.sqrt(np.mean(angle**2))),
        "hand_rotation_max_rad": float(angle.max()),
        "arm_speed_max_rad_s": float(np.max(abs(measured))),
        "arm_velocity_command_error_rms_rad_s": float(np.sqrt(np.mean((measured - commanded) ** 2))),
        "scope": "all recorded execution samples, including failed segments and stop transients; no smoothing",
    }
    if len(times) >= 3:
        dt = np.diff(times)
        if np.any(dt <= 0):
            raise ValueError("motion diagnostics require strictly increasing sample times")
        acceleration = np.diff(measured, axis=0) / dt[:, None]
        jerk = np.diff(acceleration, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
        result.update(
            arm_acceleration_max_rad_s2=float(np.max(abs(acceleration))),
            arm_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))),
            arm_jerk_max_rad_s3=float(np.max(abs(jerk))),
        )
    return result


def summarize(rows):
    stages = Counter()
    strata = {}
    for row in rows:
        case = row["case"]
        key = f"{case['layout']}:{case['mass_kg'] * 1000:g}g"
        entry = strata.setdefault(
            key, {"cases": 0, "admitted": 0, "planned": 0, "execution_started": 0, "successes": 0}
        )
        entry["cases"] += 1
        entry["admitted"] += row.get("fixture_admission", {}).get("accepted") is True
        entry["successes"] += row["success"] is True
        calls = row.get("calls", [])
        entry["planned"] += any(x["tool"] == "plan_pick_place" and x["result"]["status"] == "ok" for x in calls)
        execution = next((x["result"] for x in calls if x["tool"] == "execute_pick_place_plan"), None)
        entry["execution_started"] += bool(execution and execution.get("data", {}).get("execution_started"))
        if not row["success"]:
            if execution:
                stage = execution.get("data", {}).get("failed_stage") or "verification"
                code = row.get("robot_failure") or execution["code"]
            elif row.get("exception"):
                stage, code = "exception", row["exception"]
            elif not row.get("fixture_admission", {}).get("accepted"):
                stage, code = "fixture", row.get("fixture_admission", {}).get("reason", "unknown")
            else:
                stage, code = "planning", "no_admitted_plan"
            stages[f"{stage}:{code}"] += 1
    for entry in strata.values():
        entry["end_to_end_success_rate"] = entry["successes"] / entry["cases"]
    return {"strata": strata, "failure_stages": dict(stages)}


def plot_case(row, output):
    """Export references, measured paths and error/load plots for one case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = row.get("trace", [])
    if not trace:
        raise ValueError("case has no measured execution trace")
    t = np.asarray([x["time"] for x in trace])
    base = np.asarray([x["base"] for x in trace])
    hand = np.asarray([x["hand"] for x in trace])
    target = np.asarray([x["hand_goal"] for x in trace])
    figure = plt.figure(figsize=(12, 8))
    ee_axes = figure.add_subplot(221, projection="3d")
    ee_axes.plot(*target[:, :3].T, label="active target", linestyle="--")
    ee_axes.plot(*hand[:, :3].T, label="measured")
    ee_axes.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)", title="Hand")
    ee_axes.legend()
    base_axes = figure.add_subplot(222)
    base_axes.plot(base[:, 0], base[:, 1], label="measured")
    record = row.get("task_pose_trajectory")
    if record:
        planned = np.asarray(record["base"])
        base_axes.plot(planned[:, 0], planned[:, 1], "--", label="planned transport")
    for name in ("source_support", "destination_support"):
        point = row["fixture_admission"][name]
        base_axes.scatter(*point[:2], marker="s", label=name.replace("_", " "))
    base_axes.set(xlabel="world x (m)", ylabel="world y (m)", title="Base and supports", aspect="equal")
    base_axes.legend(fontsize=8)
    errors = figure.add_subplot(223)
    errors.plot(t, np.linalg.norm(hand[:, :3] - target[:, :3], axis=1) * 1000)
    errors.axhline(5, color="r", linestyle="--", label="5 mm tracking reference")
    errors.set(xlabel="simulation time (s)", ylabel="position error (mm)")
    errors.legend(fontsize=8)
    forces = figure.add_subplot(224)
    forces.plot(t, [x["force"] for x in trace], label="running peak contact load")
    forces.axhline(30, color="r", linestyle="--", label="unsafe contact threshold")
    forces.set(xlabel="simulation time (s)", ylabel="force (N)")
    forces.legend(fontsize=8)
    figure.suptitle(f"{row['case']['id']} — {'PASS' if row['success'] else 'FAIL'} — oracle/teacher simulation")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
