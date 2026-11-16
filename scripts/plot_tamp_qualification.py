# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Plot complete-task receipts and transfer counts without pooling evidence tiers."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def panel(ax, cohorts, labels, names):
    states, annotations = [], []
    for cohort in cohorts:
        states.append([2 if c["passed"] else 1 if c["done"] else 0 for c in cohort])
        annotations.append([f"{c['done']}/{c['total']}" for c in cohort])
    ax.imshow(np.array(states).T, cmap=ListedColormap(["#efe0d4", "#f2cb75", "#65aa99"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for row in range(len(names)):
        for col in range(len(labels)):
            ax.text(col, row, annotations[col][row], ha="center", va="center", fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-before", type=Path, required=True)
    p.add_argument("--native-after", type=Path, required=True)
    p.add_argument("--agent-before", type=Path, required=True)
    p.add_argument("--agent-after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    native = [json.loads(path.read_text()) for path in (args.native_before, args.native_after)]
    agents = [json.loads(path.read_text()) for path in (args.agent_before, args.agent_after)]
    ncases = [[c for c in r["cases"] if c["population"] != "expected_rejection"] for r in native]
    assert [c["id"] for c in ncases[0]] == [c["id"] for c in ncases[1]]
    assert [c["case_id"] for c in agents[0]["cases"]] == [c["id"] for c in agents[1]["cases"]]

    def normalize(rows):
        return [
            {"passed": c["task_passed"], "done": c["completed_transfers"], "total": c["declared_transfers"]}
            for c in rows
        ]

    prior_agent = [
        {"passed": c["task_passed"], "done": c["completed_steps"], "total": c["planned_steps"]}
        for c in agents[0]["cases"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    panel(
        axes[0],
        [normalize(rows) for rows in ncases],
        ["Initial\n1/8 tasks", f"Replay\n{native[1]['aggregate']['positive_tasks_passed']}/8 tasks"],
        [
            "Control",
            "X +15 mm",
            "X −30 mm",
            "X +30 mm",
            "Y −12 mm, yaw +0.12",
            "Y +12 mm, yaw −0.12",
            "Swap X −10 mm",
            "Swap X +25 mm",
        ],
    )
    axes[0].set_title("(a) Scripted native tools\nSame eight development tasks", fontsize=10)
    panel(
        axes[1],
        [prior_agent, normalize(agents[1]["cases"])],
        ["Nov 1\n2/4 tasks", f"Nov 13\n{agents[1]['aggregate']['tasks_passed']}/4 tasks"],
        ["Two goals", "Swap", "Two goals, X +15 mm", "Swap, X +15 mm"],
    )
    axes[1].set_title("(b) Actual run agent + Caliban\nSame four development tasks", fontsize=10)
    fig.legend(
        handles=[
            Patch(facecolor=c, label=t)
            for c, t in [
                ("#efe0d4", "No completed transfer"),
                ("#f2cb75", "Partial task"),
                ("#65aa99", "Complete task receipt + goals"),
            ]
        ],
        loc="lower center",
        fontsize=8,
        ncol=1,
        frameon=False,
    )
    fig.text(
        0.5,
        0.18,
        "Cell labels: completed / declared transfers. One live trial per cell; no held-out reliability estimate.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.21, 1, 1), w_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
