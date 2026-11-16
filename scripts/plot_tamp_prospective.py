# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Plot prospective native/agent receipts without pooling recovery trials."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_tamp_qualification import panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    native = [c for c in record["native"]["cases"] if c["population"] == "prospective"]
    agents = record["agent"]["cases"]
    assert [c["id"] for c in native] == [c["id"] for c in agents]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    def rows(cases):
        return [
            {"passed": c["task_passed"], "done": c["completed_transfers"], "total": c["declared_transfers"]}
            for c in cases
        ]

    panel(
        ax,
        [rows(native), rows(agents)],
        ["Native public tools", "Actual run agent + remote VLM"],
        [
            "Two goals: (+22,+7) mm, yaw +.08",
            "Two goals: (−24,−9) mm, yaw −.08",
            "Swap: (+12.5,+6) mm, yaw +.07",
            "Swap: (−17.5,−7) mm, yaw −.07",
        ],
    )
    ax.set_title("Four prospectively frozen within-family layouts", fontsize=10, pad=12)
    fig.legend(
        handles=[
            Patch(facecolor=c, label=t)
            for c, t in [
                ("#efe0d4", "No completed transfer"),
                ("#f2cb75", "Incomplete task"),
                ("#65aa99", "Successful receipt + independent goals"),
            ]
        ],
        loc="lower center",
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
