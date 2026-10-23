"""Read-only diagnostics around a single simulated carry rollout; no hardware.

Use --case-id from a frozen carry suite. JSON contains measured/commanded poses,
grip torque, contact wrenches, support signals and the authoritative env outcome.
Optional HTML overlays executed paths and failure signals in a standalone viewer.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from emet.rl.manipulation.carry_benchmark import case_config, digest, source_hashes
from emet.rl.manipulation.env import ManipulationEnv
from emet.rl.manipulation.safety import relative_pose


def active_payload_contacts(model, data, payload):
    contacts = []
    for index, contact in enumerate(data.contact[: data.ncon]):
        a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
        if payload not in (a, b):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        contacts.append(
            {
                "bodies": [model.body(a).name, model.body(b).name],
                "position_world_m": contact.pos.tolist(),
                "contact_frame": contact.frame.tolist(),
                "contact_wrench": force.tolist(),
                "distance_m": float(contact.dist),
            }
        )
    return contacts


def trace_case(suite, case, policy_path=None):
    if suite["config"].get("robot") != "innate_mars":
        raise ValueError("this trace tool requires a Mars suite")
    sources = source_hashes()
    env = ManipulationEnv(case_config(suite, case))
    try:
        policy = None
        if policy_path:
            from emet.rl.manipulation.policy import ManipulationPolicy

            policy = ManipulationPolicy(policy_path, robot="innate_mars", fingerprint=env.fingerprint)
        obs, info = env.reset(seed=case["seed"])
        rows = []
        initial_qpos = env.data.qpos.tolist()
        for _ in range(round(env.config.horizon / env.profile.dt) + 1):
            obs, _, done, truncated, info = env.step(env.teacher_action() if policy is None else policy(obs))
            control = env.control
            contacts = active_payload_contacts(env.model, env.data, env.payload)
            translation, rotation = relative_pose(env.data, control.ee, env.payload)
            rows.append(
                {
                    "time_s": info["time"],
                    "hand_measured": control.ee_pose().tolist(),
                    "hand_commanded": control.hand_goal.tolist(),
                    "base_measured": control.base_pose().tolist(),
                    "base_commanded": control.base_goal.tolist(),
                    "base_velocity_command": control.base_command.tolist(),
                    "arm_measured": control.positions().tolist(),
                    "arm_reference": control.reference.tolist(),
                    "arm_velocity": control.velocities().tolist(),
                    "arm_actuator_target": control.target.tolist(),
                    "grip_measured_rad": float(env.data.qpos[control.grip_q]),
                    "grip_target_rad": control.grip_target,
                    "grip_torque_Nm": float(env.data.actuator_force[control.grip_act]),
                    "payload_world_m": env.data.xpos[env.payload].tolist(),
                    "payload_relative_translation_m": translation.tolist(),
                    "payload_relative_rotation": rotation.tolist(),
                    "peak_retention_rotation_rad": info["max_payload_rotation"],
                    "peak_retention_translation_m": info["max_payload_translation"],
                    "peak_wheel_gap_m": info["max_wheel_gap"],
                    "contacts": contacts,
                }
            )
            if done or truncated:
                break
        if sources != source_hashes():
            raise RuntimeError("control source changed during trace")
        return {
            "case": case,
            "suite_sha256": digest(suite),
            "source_hashes": sources,
            "config": asdict(env.config),
            "fingerprint": env.fingerprint,
            "initial_qpos": initial_qpos,
            "controller": "teacher" if policy is None else policy.manifest["weights_sha256"],
            "grasp_setup": env.randomization.get("grasp_setup"),
            "scheduled_push": {
                "step": env.disturbance_start,
                "steps": env.disturbance_steps,
                "wrench": env.disturbance_wrench.tolist(),
            },
            "final": info,
            "trace": rows,
            "scope": "simulator measured execution; contact forces are not hardware sensors",
        }
    finally:
        env.close()


def export_html(report, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = report["trace"]
    t = [r["time_s"] for r in rows]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene", "rowspan": 2}, {"secondary_y": True}], [None, {"secondary_y": True}]],
        subplot_titles=("Executed world paths", "Retention (peak since reset)", "Contact / support"),
    )
    for key, label in (
        ("hand_commanded", "Commanded hand"),
        ("hand_measured", "Measured hand"),
        ("payload_world_m", "Payload"),
    ):
        points = np.array([r[key][:3] for r in rows])
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines+markers", marker={"size": 2}, name=label
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=t, y=[1000 * r["peak_retention_translation_m"] for r in rows], name="Retention translation (mm)"),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t, y=[r["peak_retention_rotation_rad"] for r in rows], name="Retention rotation (rad)"),
        row=1,
        col=2,
        secondary_y=True,
    )
    # A normal trace avoids Plotly's mixed scene/XY add_hline bug, which
    # otherwise tries to read an xaxis property from the Scatter3d traces.
    fig.add_trace(
        go.Scatter(
            x=[t[0], t[-1]],
            y=[0.1, 0.1],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="0.1 rad retention limit",
        ),
        row=1,
        col=2,
        secondary_y=True,
    )
    for index in range(3):
        fig.add_trace(
            go.Scatter(x=t, y=[1000 * r["peak_wheel_gap_m"][index] for r in rows], name=f"Wheel {index} peak gap (mm)"),
            row=2,
            col=2,
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=[sum(c["contact_wrench"][0] for c in r["contacts"]) for r in rows],
            name="Payload contact normals (N)",
        ),
        row=2,
        col=2,
        secondary_y=True,
    )
    fig.update_yaxes(title_text="mm", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="rad", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="mm", row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="N", row=2, col=2, secondary_y=True)
    fig.update_xaxes(title_text="simulator seconds", row=2, col=2)
    fig.update_layout(
        title=f"{report['case']['id']} — {report['final'].get('code')}; simulator only",
        scene={"aspectmode": "data"},
        height=800,
    )
    fig.write_html(output, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/benchmarks/mars_carry_v2.json"))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh trace output; do not overwrite evidence")
    suite = json.loads(args.suite.read_text())
    case = next(c for c in suite["cases"] if c["id"] == args.case_id)
    report = trace_case(suite, case, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.html:
        export_html(report, args.output.with_suffix(".html"))
    print(report["case"]["id"], report["final"].get("code"))


if __name__ == "__main__":
    main()
