# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Matplotlib figure export for TAMP pick-place episodes (paper-oriented)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _savefig(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    return path


def save_topdown_figure(
    out_dir: Path | str,
    *,
    base_path_xyt: Sequence[Sequence[float]] | None = None,
    object_xy: Sequence[float] | None = None,
    receptacle_xy: Sequence[float] | None = None,
    grasp_xy: Sequence[float] | None = None,
    title: str = "TAMP pick-place (top-down)",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if base_path_xyt:
        arr = np.asarray(base_path_xyt, dtype=np.float64).reshape(-1, 3)
        ax.plot(arr[:, 0], arr[:, 1], "-o", color="#1f4e79", markersize=3, label="base path")
        ax.scatter(arr[0, 0], arr[0, 1], c="#2ca02c", s=60, zorder=5, label="start")
        ax.scatter(arr[-1, 0], arr[-1, 1], c="#d62728", s=60, zorder=5, label="end")
    if object_xy is not None:
        o = np.asarray(object_xy, dtype=np.float64).reshape(2)
        ax.scatter(o[0], o[1], marker="s", c="#ff7f0e", s=80, label="object")
    if receptacle_xy is not None:
        r = np.asarray(receptacle_xy, dtype=np.float64).reshape(2)
        ax.scatter(r[0], r[1], marker="D", c="#9467bd", s=80, label="receptacle")
    if grasp_xy is not None:
        g = np.asarray(grasp_xy, dtype=np.float64).reshape(2)
        ax.scatter(g[0], g[1], marker="*", c="#e377c2", s=120, label="grasp")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    path = _savefig(fig, out / "topdown.png")
    plt.close(fig)
    return path


def save_ee_path_figure(
    out_dir: Path | str,
    *,
    planned_xyz: Sequence[Sequence[float]] | None = None,
    executed_xyz: Sequence[Sequence[float]] | None = None,
    targets: dict[str, Sequence[float]] | None = None,
    title: str = "EE path (XZ)",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if planned_xyz:
        p = np.asarray(planned_xyz, dtype=np.float64).reshape(-1, 3)
        ax.plot(p[:, 0], p[:, 2], "-", color="#1f77b4", label="planned")
    if executed_xyz:
        e = np.asarray(executed_xyz, dtype=np.float64).reshape(-1, 3)
        ax.plot(e[:, 0], e[:, 2], "--", color="#ff7f0e", label="executed")
    if targets:
        for name, xyz in targets.items():
            t = np.asarray(xyz, dtype=np.float64).reshape(3)
            ax.scatter(t[0], t[2], s=50, label=name)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    path = _savefig(fig, out / "ee_path_xz.png")
    plt.close(fig)
    return path


def save_joint_traj_figure(
    out_dir: Path | str,
    *,
    waypoints: Sequence[Sequence[float]],
    joint_names: Sequence[str] | None = None,
    title: str = "Commanded joint waypoints",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    w = np.asarray(waypoints, dtype=np.float64)
    if w.ndim != 2 or w.size == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no waypoints", ha="center")
        ax.axis("off")
        path = _savefig(fig, out / "joint_traj.png")
        plt.close(fig)
        return path
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    t = np.arange(w.shape[0])
    names = list(joint_names) if joint_names is not None else [f"q{i}" for i in range(w.shape[1])]
    for i in range(w.shape[1]):
        label = names[i] if i < len(names) else f"q{i}"
        ax.plot(t, w[:, i], label=label, linewidth=1.0)
    ax.set_xlabel("waypoint index")
    ax.set_ylabel("joint angle (rad)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    path = _savefig(fig, out / "joint_traj.png")
    plt.close(fig)
    return path


def save_plan_tree_figure(
    out_dir: Path | str,
    *,
    expanded_nodes: Sequence[str],
    chosen_grasp_index: int | None = None,
    title: str = "TAMP search trace",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    nodes = list(expanded_nodes)
    fig_h = max(3.0, 0.35 * max(1, len(nodes)) + 1.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, max(1, len(nodes)))
    ax.axis("off")
    ax.set_title(title)
    for i, text in enumerate(nodes):
        y = len(nodes) - 1 - i
        color = "#d62728" if chosen_grasp_index is not None and f"grasp[{chosen_grasp_index}]" in text else "#1f77b4"
        if text.startswith("chosen_grasp"):
            color = "#2ca02c"
        ax.text(0.05, y, text, fontsize=9, color=color, family="monospace", va="center")
        if i > 0:
            ax.plot([0.02, 0.02], [y + 0.35, y + 0.65], color="#888888", linewidth=1.0)
    path = _savefig(fig, out / "plan_tree.png")
    plt.close(fig)
    return path


def write_tamp_figure_bundle(
    out_dir: Path | str,
    *,
    plan: Any | None = None,
    base_path_xyt: Sequence[Sequence[float]] | None = None,
    object_xy: Sequence[float] | None = None,
    receptacle_xy: Sequence[float] | None = None,
    grasp_xy: Sequence[float] | None = None,
    planned_ee_xyz: Sequence[Sequence[float]] | None = None,
    joint_waypoints: Sequence[Sequence[float]] | None = None,
    joint_names: Sequence[str] | None = None,
    targets: dict[str, Sequence[float]] | None = None,
) -> dict[str, Path]:
    """Write the standard TAMP figure set; returns map of name → PNG path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["topdown"] = save_topdown_figure(
        out,
        base_path_xyt=base_path_xyt,
        object_xy=object_xy,
        receptacle_xy=receptacle_xy,
        grasp_xy=grasp_xy,
    )
    paths["ee_path"] = save_ee_path_figure(
        out,
        planned_xyz=planned_ee_xyz,
        targets=targets,
    )
    if joint_waypoints is not None:
        paths["joint_traj"] = save_joint_traj_figure(out, waypoints=joint_waypoints, joint_names=joint_names)
    if plan is not None:
        paths["plan_tree"] = save_plan_tree_figure(
            out,
            expanded_nodes=getattr(plan, "expanded_nodes", []) or [],
            chosen_grasp_index=getattr(plan, "chosen_grasp_index", None),
        )
    return paths
