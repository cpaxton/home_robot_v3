#!/usr/bin/env python3
"""Export paper figures from terminal TAMP admission and replay ledgers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from emet.eval.clutter_fixture import moved_placements
from emet.eval.tamp_clutter import placement_obstacle_disks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-id", action="append", default=[])
    args = parser.parse_args()
    inputs = [args.admission / "ledger.json", args.replay / "ledger.json"]
    admissions, replays = [json.loads(p.read_text()) for p in inputs]
    if any(r["status"] in ("pending", "running") for rows in (admissions, replays) for r in rows.values()):
        parser.error("Use terminal runs only; partial results are not paper figures")
    admitted = {k: r for k, r in admissions.items() if r["status"] == "fixture_admitted"}
    if set(replays) != set(admitted):
        parser.error("Replay must cover exactly the admitted registry")
    for key, row in replays.items():
        expected = admitted[key]["result"]["resolved_episode"]["fixture"]["sha256"]
        if row.get("result", {}).get("fixture_sha256") != expected:
            parser.error(f"Unverified replay identity: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    robots = ["rby1", "stretch", "innate_mars", "nori"]
    labels = ["RBY1\nLatch", "Stretch\nOracle", "Mars\nOracle", "Nori\nOracle"]
    buckets = ["Admitted", "Layout rejected", "Missing receptacle", "Reference failed", "Infrastructure error"]
    counts = {robot: Counter() for robot in robots}
    successes = Counter()
    for key, row in admissions.items():
        result = row.get("result", {})
        robot = result.get("robot") or key.split("_scene")[0].removeprefix("tamp_v2_")
        if row["status"] == "fixture_admitted":
            bucket = "Admitted"
        elif result.get("error") == "no_valid_fixture_within_budget":
            bucket = "Layout rejected"
        elif result.get("error") == "missing_receptacle":
            bucket = "Missing receptacle"
        elif result.get("error") == "reference_execution_failed":
            bucket = "Reference failed"
        else:
            bucket = "Infrastructure error"
        counts[robot][bucket] += 1
        if key in replays and replays[key].get("task_success"):
            successes[robot] += 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    bottom = np.zeros(len(robots))
    colors = ["#0072B2", "#B9BDC4", "#E69F00", "#CC79A7", "#D55E00"]
    for bucket, color in zip(buckets, colors, strict=True):
        values = np.array([counts[r][bucket] for r in robots])
        axes[0].bar(labels, values, bottom=bottom, label=bucket, color=color, width=0.65)
        bottom += values
    axes[0].set(title=f"Admission coverage: {len(admitted)}/{len(admissions)}", ylabel="Candidate tasks")
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    n = np.array([counts[r]["Admitted"] for r in robots])
    passed = np.array([successes[r] for r in robots])
    axes[1].bar(labels, passed, color="#009E73", label="Success", width=0.65)
    axes[1].bar(labels, n - passed, bottom=passed, color="#D55E00", label="Failure", width=0.65)
    for i, (p, total) in enumerate(zip(passed, n, strict=True)):
        axes[1].text(i, total + 0.2, f"{p}/{total}" if total else "Not scored", ha="center", fontsize=9)
    axes[1].set(title=f"Fresh MCTS: {sum(passed)}/{sum(n)}", ylabel="Admitted tasks evaluated", ylim=(0, max(n) + 3))
    axes[1].legend(frameon=False)
    fig.suptitle("Scene-validated TAMP controls", fontsize=13)
    export(fig, args.output_dir / "tamp_coverage_and_success")
    with (args.output_dir / "tamp_counts.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["robot", *buckets, "mcts_successes"])
        for robot in robots:
            writer.writerow([robot, *[counts[robot][b] for b in buckets], successes[robot]])
    example_files = []
    for key in args.example_id:
        episode = admitted[key]["result"]["resolved_episode"]
        fixture = episode["fixture"]
        source = args.admission / key / f"{key}.initial.json"
        example_files.append(source)
        snapshot = json.loads(source.read_text())
        before = moved_placements(snapshot["placements"], {r["body"]: r["pos"] for r in fixture["clutter"]})
        after = moved_placements(before, fixture["reference"]["final_positions"])
        selected = {r["body"] for r in fixture["clutter"]}
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), layout="constrained")
        for ax, scene, label in zip(
            axes, [before, after], ["Frozen initial layout", "Executed reference outcome"], strict=True
        ):
            for xy, radius, body in placement_obstacle_disks(scene):
                color = "#D55E00" if body in selected else "#9A9FA7"
                ax.add_patch(Circle(xy, radius, color=color, alpha=0.5))
                ax.add_patch(Circle(xy, radius + fixture["clearance_m"], fill=False, color=color, alpha=0.25, lw=0.5))
            ax.scatter(*fixture["robot_start_xy"], marker="s", color="#0072B2", label="Original start", zorder=5)
            if fixture["goal_xy"] is not None:
                ax.scatter(*fixture["goal_xy"], marker="*", s=110, color="#009E73", label="Goal", zorder=5)
            ax.scatter(
                *np.asarray(scene[fixture["bin_body"]]["pos"])[:2],
                marker="x",
                color="black",
                label="Receptacle",
                zorder=5,
            )
            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set(title=label, xlabel="World X (m)", ylabel="World Y (m)")
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle(f"{episode['robot']} · {episode['mode']} · {episode['n_objects']} objects", fontsize=12)
        export(fig, args.output_dir / key)
    provenance = {
        "inputs": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs + example_files},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "examples": args.example_id,
        "scope": "GT/MCTS oracle and latch controls; not physical or learned-agent acceptance",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output_dir / "captions.md").write_text(
        "Admission coverage and fresh MCTS outcomes are shown separately. Rejected candidates are not scored planner failures. "
        "Success on admitted tasks is conditional on an executed reference witness and is not a general reliability estimate. "
        "RBY1 uses kinematic latch; other robots use oracle teleport. These are not physical or learned-agent results.\n\n"
        "Layout panels depict the benchmark disk approximation, including clearance outlines; they are not renderings of collision meshes. "
        "The right panel shows measured reference object positions, not the tested MCTS rollout. Unselected scene bodies retain their initial poses.\n"
    )
    print(args.output_dir)


def export(fig, path):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
