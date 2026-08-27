# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared ground-truth / Rerun helpers for Dynagraph and LazyGraph CLIs."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from emet.memory.graph_eqa.sim_ground_truth_graph import (
    ground_truth_alignment_report,
    gt_pose_sanity_report,
    read_sim_object_placements,
)
from emet.utils.geometry import nav_xyt_to_world_xyt


def ensure_ground_truth_ready(agent: Any, *, context: str) -> None:
    """Populate GT graph + Rerun immediately; fail fast when session has no placements."""
    session = agent.robot.get_emet_session()
    if session is None:
        raise click.ClickException(
            f"Ground-truth mode ({context}): no emet_session from the ZMQ server. "
            "Start emet serve mujoco (default, --scene robocasa, or --scene ithor …) with the "
            "same --port-offset as this client, then retry."
        )
    n_bodies = agent.refresh_ground_truth()
    if n_bodies == 0:
        runtime = session.get("runtime_kind", "?")
        raise click.ClickException(
            f"Ground-truth mode ({context}): emet_session has no sim_object_placements "
            f"(runtime_kind={runtime!r}). Restart the sim server from this branch with the same "
            "--port-offset — servers started before the ground-truth feature do not publish placements."
        )
    n_nodes = len(agent.graph_memory.get_nodes()) if agent.graph_memory is not None else 0
    n_boxes = sum(
        1
        for n in (agent.graph_memory.get_nodes() if agent.graph_memory else [])
        if getattr(n, "extent_half", None) is not None
    )
    click.echo(f"Ground truth: {n_bodies} sim bodies → {n_nodes} graph nodes ({n_boxes} with 3D bounds).")
    click.echo(
        "Rerun: «Graph (ground truth)» column — nodes at world/dynagraph/nodes, boxes at world/dynagraph/bboxes."
    )
    placements = read_sim_object_placements(agent.robot.get_emet_session())
    if agent.graph_memory is not None and placements:
        click.echo(ground_truth_alignment_report(agent.graph_memory, placements))
    session = agent.robot.get_emet_session()
    try:
        obs = agent.robot.get_observation()
        gps = np.asarray(obs.gps, dtype=np.float64).reshape(-1)
        comp = np.asarray(obs.compass, dtype=np.float64).ravel()
        local = np.array([float(gps[0]), float(gps[1]), float(comp[0]) if comp.size else 0.0])
        robot_world = nav_xyt_to_world_xyt(local, session)
    except Exception:
        robot_world = None
    click.echo(gt_pose_sanity_report(placements, robot_world_xyt=robot_world, session=session))


def print_graph_nav_rerun_help(
    *,
    product: str,
    enabled: bool,
    headless: bool,
    ground_truth: bool = False,
    compare_to_gt: bool = False,
) -> None:
    """Graph-nav Rerun hints (web URL is printed from RerunVisualizer after rr.serve)."""
    if not enabled:
        click.echo("Rerun visualization is disabled (--no-rerun).")
        return
    if headless:
        click.echo("Rerun headless: no auto-open browser (use the URL printed when the viewer started).")
    if ground_truth:
        click.echo(
            "Ground-truth mode: use «Graph (ground truth)» for labeled nodes and 3D boxes "
            "(world/dynagraph/nodes, world/dynagraph/bboxes)."
        )
        return
    click.echo(
        f"{product}: 3D world view + graph node list; full tree text is export/stdout only (not live Rerun)."
    )
    if compare_to_gt:
        click.echo("Compare mode: green sim reference under «Sim GT (reference)» (world/dynagraph/ground_truth/).")
