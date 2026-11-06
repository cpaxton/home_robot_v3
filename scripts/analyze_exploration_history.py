#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Audit recorded RGB-D histories and export sensor-only path/view panels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(source, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from PIL import Image

    manifest = json.loads((source / "manifest.json").read_text())
    records = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
    events = (
        [json.loads(line) for line in (source / "events.jsonl").read_text().splitlines()]
        if (source / "events.jsonl").is_file()
        else []
    )
    if not records or [r["sequence"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Missing, duplicate, or reordered frames")
    if any(b["timestamp_ns"] <= a["timestamp_ns"] for a, b in zip(records, records[1:], strict=False)):
        raise ValueError("Capture clock did not advance")
    for record in records:
        prefix = source / f"frame-{record['sequence']:05d}"
        if (
            sha256(prefix.with_suffix(".png")) != record["rgb_sha256"]
            or sha256(prefix.with_suffix(".npz")) != record["depth_sha256"]
        ):
            raise ValueError("Changed RGB/depth capture")
        with np.load(prefix.with_suffix(".npz")) as packet:
            if int(packet["timestamp_ns"]) != record["timestamp_ns"]:
                raise ValueError("Depth packet and RGB capture timestamps differ")
    accepted = [r for r in records if r["odometry"]["ok"]]
    if not accepted:
        raise ValueError("No accepted pose: retain this failed collection for infrastructure diagnosis")
    xy = np.array([r["odometry"]["camera_pose"] for r in accepted])[:, :2, 3]
    outcome = {
        "case": source.name,
        "source_commit": manifest["source_commit"],
        "status": manifest["status"],
        "acquisition_seed": manifest.get("acquisition_seed"),
        "mapping_rng_recorded": "mapping_rng_sha256" in manifest,
        "frames": len(records),
        "accepted_registration_frames": len(accepted),
        "reported_collisions": sum(r["collided"] for r in records),
        "estimated_camera_path_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        "final_estimated_displacement_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "attempted_primitives": manifest["attempted_primitives"],
        "completed_motion_receipts": sum(e["stage"] == "motion_finished" and e["receipt"]["success"] for e in events),
        "budget": manifest["budgets"],
        "wall_s": manifest["wall_s"],
        "manifest_sha256": sha256(source / "manifest.json"),
        "frames_sha256": sha256(source / "frames.jsonl"),
        "analysis_driver_sha256": sha256(Path(__file__)),
        "scope": "sensor collection and local odometry, not target coverage or agent success",
    }
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    indices = np.linspace(0, len(records) - 1, min(8, len(records)), dtype=int)
    for ax, index in zip(axes.flat, indices, strict=False):
        record = records[index]
        ax.imshow(Image.open(source / f"frame-{record['sequence']:05d}.png"))
        ax.set_title(f"Frame {record['sequence']}: {record['command'] or 'capture'}", fontsize=9)
        ax.axis("off")
    ax = axes.flat[8]
    if (source / "navigation.npz").is_file():
        with np.load(source / "navigation.npz") as maps:
            explored, obstacles = maps["explored"], maps["obstacles"]
            origin, res = maps["grid_origin"][:2], float(maps["grid_resolution_m"])
            pixels = np.zeros(explored.shape, dtype=np.uint8)
            pixels[explored] = 1
            pixels[obstacles] = 2
            extent = [
                (-origin[0] - 0.5) * res,
                (pixels.shape[0] - origin[0] - 0.5) * res,
                (-origin[1] - 0.5) * res,
                (pixels.shape[1] - origin[1] - 0.5) * res,
            ]
            ax.imshow(
                pixels.T, origin="lower", extent=extent, cmap=ListedColormap([".65", "white", ".15"]), vmin=0, vmax=2
            )
            cells = np.argwhere(explored | obstacles)
            bounds = np.vstack(((cells - origin) * res, xy))
            ax.set_xlim(bounds[:, 0].min() - 0.3, bounds[:, 0].max() + 0.3)
            ax.set_ylim(bounds[:, 1].min() - 0.3, bounds[:, 1].max() + 0.3)
            outcome.update(
                observed_cells=int(explored.sum()),
                obstacle_cells=int(obstacles.sum()),
                navigation_sha256=sha256(source / "navigation.npz"),
            )
    ax.plot(xy[:, 0], xy[:, 1], color="C0", linewidth=1)
    ax.scatter(*xy[0], marker="o", label="start")
    ax.scatter(*xy[-1], marker="x", label="end")
    ax.set(xlabel="Local X (m)", ylabel="Local Y (m)", title="Observed map and estimated camera path")
    ax.set_aspect("equal")
    ax.legend()
    fig.suptitle(
        f"{source.name}: {manifest['status']} | {len(records)} frames | {outcome['estimated_camera_path_m']:.2f} m estimated camera path",
        fontsize=14,
    )
    fig.text(
        0.02,
        0.015,
        "Uniformly sampled raw RGB; not target-selected. Gray: unknown, white: observed, dark: obstacle/hazard. No global pose or target annotations.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{source.name}.{suffix}", dpi=140)
    plt.close(fig)
    (output / f"{source.name}.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
