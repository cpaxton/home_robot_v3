# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI: offline grid search for GraphObjectFusion vs sim GT + calibration frames."""

from __future__ import annotations

import json
from pathlib import Path

import click

from emet.memory.graph_eqa.graph_object_fusion.calibrate import (
    grid_search_fusion_config,
    load_calibration_frames_jsonl,
    write_fusion_config_yaml,
)
from emet.simulation.mujoco_gt_objects import load_gt_scene_json


@click.command("tune-graph-fusion")
@click.option("--gt", "gt_path", required=True, type=click.Path(exists=True), help="GT scene JSON")
@click.option(
    "--frames",
    "frames_path",
    required=True,
    type=click.Path(exists=True),
    help="Calibration frames JSONL from dynagraph --calibration-export",
)
@click.option(
    "--report",
    "report_path",
    default=None,
    type=click.Path(),
    help="Write full grid report JSON (default: alongside --gt)",
)
@click.option(
    "--write-config",
    "write_config_path",
    default=None,
    type=click.Path(),
    help="Write tuned graph_object_fusion YAML to this path",
)
@click.option(
    "--min-recall",
    default=0.85,
    type=float,
    show_default=True,
    help="Minimum spatial_recall (geometry) to accept a grid point",
)
@click.option(
    "--min-label-recall",
    default=None,
    type=float,
    help="Optional minimum label_recall (taxonomy diagnostic); off by default",
)
def main(
    gt_path: str,
    frames_path: str,
    report_path: str | None,
    write_config_path: str | None,
    min_recall: float,
    min_label_recall: float | None,
) -> None:
    """Grid-search fusion parameters on one environment (no sim)."""
    gt = load_gt_scene_json(gt_path)
    frames = load_calibration_frames_jsonl(frames_path)
    if not frames:
        raise click.ClickException(f"No frames in {frames_path}")

    best_cfg, report, _ = grid_search_fusion_config(
        frames,
        gt,
        min_recall=min_recall,
        min_label_recall=min_label_recall,
    )

    rep_dest = Path(report_path) if report_path else Path(gt_path).with_name("fusion_tune_report.json")
    rep_dest.parent.mkdir(parents=True, exist_ok=True)
    rep_dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    best = report["best"]
    click.echo(
        f"Best config (spatial_recall={best.get('spatial_recall', 0):.3f}, "
        f"label_recall={best.get('label_recall', 0):.3f}, "
        f"nodes={best.get('node_count', 0):.0f}):"
    )
    click.echo(
        f"  spatial_merge_xy_m={best_cfg.spatial_merge_xy_m} "
        f"embedding_min_cosine={best_cfg.embedding_min_cosine} "
        f"bounds_3d_iou_min={best_cfg.bounds_3d_iou_min}"
    )
    click.echo(f"Wrote report -> {rep_dest}")

    if write_config_path:
        out = write_fusion_config_yaml(write_config_path, best_cfg)
        click.echo(f"Wrote config -> {out}")


if __name__ == "__main__":
    main()
