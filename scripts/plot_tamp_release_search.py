# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Plot measured native release refinement separately from hypothetical targets."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from emet.controller.manipulation.native_scene import model_signature
from emet.robots.stretch import STRETCH_ARM_CHAIN


def measurements(directory, model):
    rows = [json.loads(line) for line in (directory / "contact.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["object_body"] == "object2"]
    place = next(r for r in rows if r["event"]["phase"] == "place_complete")
    events = json.loads((directory / "adapter_events.json").read_text())
    end = next(i for i, e in enumerate(events) if e["phase"] == "place_complete")
    command = next(e["command"] for e in reversed(events[:end]) if e["phase"] == "place" and "command" in e)
    names = STRETCH_ARM_CHAIN.joint_names[:-1]
    qadr = [model.joint(n).qposadr[0] for n in names]
    data = mujoco.MjData(model)
    selected = [place] + [
        r
        for r in rows
        if r["event"]["phase"]
        in ("release_adjustment", "release_adjustment_complete", "release", "release_open_verified")
    ]
    records = []
    for row in selected:
        data.qpos[:] = row["qpos"]
        mujoco.mj_kinematics(model, data)
        measured = float(data.body(STRETCH_ARM_CHAIN.ee_body).xpos[2])
        command = row["event"].get("command", command)
        data.qpos[qadr] = command
        mujoco.mj_kinematics(model, data)
        records.append(
            {
                "time_from_place_s": row["sim_time"] - place["sim_time"],
                "measured_ee_z_m": measured,
                "command_fk_z_m": float(data.body(STRETCH_ARM_CHAIN.ee_body).xpos[2]),
                "phase": row["event"]["phase"],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.before / "fixture/scene.xml"))
    other = mujoco.MjModel.from_xml_path(str(args.after / "fixture/scene.xml"))
    if model_signature(model) != model_signature(other):
        raise ValueError("incomparable_release_models")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    all_records = []
    for ax, label, directory in zip(
        axes, ("Small nominal corrections", "Measured-error reserve"), (args.before, args.after), strict=True
    ):
        records = measurements(directory, model)
        for row in records:
            all_records.append({"condition": label, **row})
        ax.plot(
            [r["time_from_place_s"] for r in records],
            [r["measured_ee_z_m"] for r in records],
            "o-",
            color="#206c9c",
            label="Measured EE height",
        )
        ax.step(
            [r["time_from_place_s"] for r in records],
            [r["command_fk_z_m"] for r in records],
            "--",
            where="post",
            color="#ae681b",
            label="Command FK (other joints measured)",
        )
        for row in records:
            if row["phase"] == "release_open_verified":
                ax.axvline(row["time_from_place_s"], color="#38864c", alpha=0.7, label="Measured opening verified")
        summary = json.loads((directory / "summary.json").read_text())
        ax.set_title(label + "\nTask " + ("passed" if summary["passed"] else "failed"))
        ax.set_xlabel("Simulated time since cylinder place completion (s)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("EE height in world coordinates (m)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=2)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    with args.output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    plt.close(fig)


if __name__ == "__main__":
    main()
