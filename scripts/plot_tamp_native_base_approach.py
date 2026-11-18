# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Render the recorded original-reference cube retention comparison on CPU."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem for PDF and PNG")
    args = parser.parse_args()
    record = json.loads(args.data.read_text())
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.6, 2.8), layout="constrained")
    for series in record["series"]:
        points = series["points"]
        t0 = points[0]["sim_time"]
        label = (f"{series['label']}: {series['completed_transfers']}/"
                 f"{series['declared_transfers']} transfers, "
                 f"task {'passed' if series['task_passed'] else 'stopped'}")
        ax.plot([p["sim_time"]-t0 for p in points], [p["rotation_rad"] for p in points], ".-",
                color="#287b70" if series["task_passed"] else "#c55a38", linewidth=1.5,
                markersize=3, label=label)
    limit = record["rotation_tolerance_rad"]
    ax.axhline(limit, color="#6b6b6b", ls="--", lw=1, label=f"Unchanged retention limit ({limit:g} rad)")
    ax.set(xlabel="Simulation time from first cube lift command (s)",
           ylabel="Original-grasp rotation error (rad)", ylim=(-.005, limit*1.12))
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output.with_suffix("."+suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
