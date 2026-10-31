#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Plot measured moving-mask and gripper audit diagnostics, including failure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def head_panel(path, *, mask):
    capture = np.load(path)
    rgb = capture["rgb"].copy()
    if mask:
        pixels = capture["predicted_mask"]
        rgb[pixels] = (0.45 * rgb[pixels] + 0.55 * np.array([40, 255, 70])).astype(np.uint8)
    upright = np.rot90(rgb, -1)
    # Fixed presentation crop only; the tracker always receives the full image.
    return upright[470:, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Root containing moving-v3, contact-v1, head-contact-v3."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    args = parser.parse_args()
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for name, label in [("front-lift", "view A"), ("opposite-lift", "view B")]:
        rows = read(args.input / "moving-v3" / name / "result.json")
        axes[0, 0].plot(
            [r["index"] for r in rows],
            [1000 * r["scorer_center_error_m"] for r in rows],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[0, 0].set(
        title="(a) Rendered stage: 24 cm translation + 8 cm lift",
        xlabel="Capture index",
        ylabel="Center error vs scorer (mm)",
        ylim=(0, 0.2),
    )
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(alpha=0.25)
    samples = sum((read(args.input / "contact-v1" / f"{name}-envelope.json") for name in ("object1", "object2")), [])
    for contact, label, color in [(False, "No target contact", "#527aa7"), (True, "Target contact", "#df8333")]:
        selected = [(i, r) for i, r in enumerate(samples) if r["target_contact"] == contact]
        axes[0, 1].scatter(
            [i for i, r in selected],
            [1000 * r["max_corner_escape_m"] for i, r in selected],
            s=12,
            color=color,
            label=f"{label} ({len(selected)})",
        )
    axes[0, 1].set(
        title="(b) Whole-gripper envelope: sampled containment",
        xlabel="Snapshot index across two contact controls",
        ylabel="Corner escape (mm)",
        ylim=(-0.1, 0.3),
    )
    axes[0, 1].text(
        0.5,
        0.82,
        "175/175 contained; no sampled joint-limit overrun\nConditional model intervals; not a hardware certificate",
        ha="center",
        transform=axes[0, 1].transAxes,
        fontsize=9,
    )
    axes[0, 1].legend(loc="lower right", fontsize=8)
    rows = read(args.input / "head-contact-v3" / "tracking.json")
    loss = next(row for row in rows if row["state"] == "unknown")
    axes[1, 0].imshow(head_panel(args.input / "head-contact-v3" / "capture-000.npz", mask=True))
    axes[1, 0].set_title("(c) Stretch RGBD: initial mask and shape candidate")
    axes[1, 1].imshow(head_panel(args.input / "head-contact-v3" / f"capture-{loss['index']:03}.npz", mask=False))
    axes[1, 1].set_title("(d) Stop: shape no longer observable near gripper")
    for ax in axes[1]:
        ax.axis("off")
    fig.suptitle("Observed-input development diagnostics — not observed task success", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.paper_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190)
    fig.savefig(args.paper_output)
    plt.close(fig)


if __name__ == "__main__":
    main()
