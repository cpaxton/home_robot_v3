#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Static 3D figure: voxel subsample + graph nodes from an episode checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_graph_ckpt(ckpt_dir: Path) -> tuple[np.ndarray, list[dict]]:
    nodes_path = ckpt_dir / "graph_eqa" / "nodes.json"
    if not nodes_path.is_file():
        nodes_path = ckpt_dir / "memory" / "graph_eqa" / "nodes.json"
    if not nodes_path.is_file():
        raise FileNotFoundError(f"no graph nodes.json under {ckpt_dir}")
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    if isinstance(nodes, dict) and "nodes" in nodes:
        nodes = nodes["nodes"]
    pts: list[np.ndarray] = []
    meta: list[dict] = []
    for n in nodes:
        xyz = n.get("xyz") or n.get("position")
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=float).reshape(-1)[:3]
        pts.append(arr)
        meta.append(
            {
                "is_frontier": bool(n.get("is_frontier")),
                "is_viewpoint": bool(n.get("is_viewpoint")),
                "labels": n.get("labels") or [],
            }
        )
    if not pts:
        raise ValueError("empty graph")
    return np.stack(pts, axis=0), meta


def _load_voxel_pts(bundle: Path, max_pts: int) -> np.ndarray | None:
    for rel in ("voxel_history.npz", "memory/voxel_history.npz", "explored_points.npy"):
        p = bundle / rel
        if not p.is_file():
            continue
        if p.suffix == ".npz":
            data = np.load(p)
            key = "points" if "points" in data else data.files[0]
            pts = np.asarray(data[key], dtype=float).reshape(-1, 3)
        else:
            pts = np.load(p)
        if pts.size == 0:
            continue
        if len(pts) > max_pts:
            idx = np.linspace(0, len(pts) - 1, max_pts, dtype=int)
            pts = pts[idx]
        return pts[:, :3]
    return None


def render_figure(
    bundle: Path,
    *,
    output: Path,
    max_voxels: int = 4000,
    elev: float = 35,
    azim: float = -60,
) -> Path:
    ckpt = bundle / "checkpoint"
    if not ckpt.is_dir():
        ckpt = bundle
    graph_xyz, meta = _load_graph_ckpt(ckpt)
    vox = _load_voxel_pts(bundle, max_voxels)

    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    if vox is not None:
        ax.scatter(vox[:, 0], vox[:, 2], vox[:, 1], s=1, c="#4a6fa5", alpha=0.25, label="voxels")

    for i, m in enumerate(meta):
        color = "#e74c3c" if m["is_frontier"] else "#2ecc71" if m["is_viewpoint"] else "#f1c40f"
        size = 28 if m["is_frontier"] else 18
        ax.scatter(
            graph_xyz[i, 0],
            graph_xyz[i, 2],
            graph_xyz[i, 1],
            s=size,
            c=color,
            edgecolors="k",
            linewidths=0.3,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(bundle.name)
    ax.legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-voxels", type=int, default=4000)
    args = parser.parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    out = args.output or (bundle / "dynagraph_3d.png")
    path = render_figure(bundle, output=out, max_voxels=args.max_voxels)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
