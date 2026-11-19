#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Reproduce the mobile release figure from the compact versioned JSON evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("docs/experiments/mobile_release_20261117.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    data = json.loads(args.index.read_text())["release_fix"]
    slacks = np.asarray(data["candidate_support_slack_m"]) * 1000
    lowered = np.array([data["old_final_lowered_gap_m"], data["new_final_lowered_gap_m"]]) * 1000
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    ax = axes[0]
    ax.bar(
        ["Candidate 1\nold first choice", "Candidate 2\nselected", "Candidate 3"],
        slacks,
        color=["#a8aeb5", "#247a9b", "#a8aeb5"],
        width=0.6,
    )
    for i, value in enumerate(slacks):
        ax.text(i, value + 0.2, f"{value:.2f}", ha="center")
    ax.set(ylabel="Certified bottleneck slack (mm)", title="Same certified candidate set", ylim=(0, 10))
    ax = axes[1]
    positions = np.arange(2)
    for shift, values, label, color in [
        (-0.15, lowered + data["clearance_m"] * 1000, "Measured payload gap", "#247a9b"),
        (0.15, lowered, "Conservative lowered probe", "#ce682f"),
    ]:
        ax.bar(positions + shift, values, width=0.3, label=label, color=color)
    ax.axhline(
        -data["support_penetration_limit_m"] * 1000, color="#b53b3b", ls="--", lw=1, label="Unchanged −1 mm limit"
    )
    ax.set(
        xticks=positions,
        xticklabels=["Old: 2 corrections\nrelease rejected", "Ranked: 1 correction\nrelease verified"],
        ylabel="Payload bottom above tabletop (mm)",
        title="Final release decision",
        ylim=(-1.7, 3.8),
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Mobile cylinder release: native state + provided RGBD collision map", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
