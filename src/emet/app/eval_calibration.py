# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""CLI: evaluate calibration frames vs sim GT (spatial recall first)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from emet.memory.graph_eqa.graph_object_fusion.calibrate import (
    load_calibration_frames_jsonl,
    replay_frames_with_fusion,
)
from emet.memory.graph_eqa.graph_object_fusion.config import load_graph_object_fusion_config
from emet.memory.graph_eqa.graph_object_fusion.evaluate import (
    format_calibration_eval_report,
    score_detections_vs_gt,
    score_fused_graph_vs_gt,
)
from emet.simulation.mujoco_gt_objects import load_gt_scene_json


@click.command("eval-calibration")
@click.option("--gt", "gt_path", required=True, type=click.Path(exists=True), help="GT scene JSON")
@click.option(
    "--frames",
    "frames_path",
    required=True,
    type=click.Path(exists=True),
    help="Calibration frames JSONL from dynagraph --calibration-export",
)
@click.option("--match-xy-m", default=0.55, type=float, show_default=True)
@click.option("--bounds-iou-min", default=0.08, type=float, show_default=True)
@click.option(
    "--report",
    "report_path",
    default=None,
    type=click.Path(),
    help="Write full metrics JSON (default: alongside --gt)",
)
@click.option(
    "--fusion-config",
    "fusion_config_path",
    default=None,
    type=click.Path(exists=True),
    help="Optional: replay frames with this graph_object_fusion YAML and score fused nodes",
)
def main(
    gt_path: str,
    frames_path: str,
    match_xy_m: float,
    bounds_iou_min: float,
    report_path: str | None,
    fusion_config_path: str | None,
) -> None:
    """Score calibration detections vs sim GT (geometry-first, labels diagnostic)."""
    gt = load_gt_scene_json(gt_path)
    frames = load_calibration_frames_jsonl(frames_path)
    if not frames:
        raise click.ClickException(f"No frames in {frames_path}")

    metrics = score_detections_vs_gt(
        gt,
        frames,
        match_xy_m=match_xy_m,
        bounds_iou_min=bounds_iou_min,
    )
    payload: dict = {"raw": metrics}

    if fusion_config_path:
        cfg = load_graph_object_fusion_config(fusion_config_path)
        mem = replay_frames_with_fusion(frames, cfg)
        n_raw = sum(len(fr.get("detections", [])) for fr in frames)
        fused = score_fused_graph_vs_gt(
            mem,
            gt,
            match_xy_m=match_xy_m,
            bounds_iou_min=bounds_iou_min,
            n_raw_detections=n_raw,
        )
        payload["fused"] = fused

    rep_dest = Path(report_path) if report_path else Path(gt_path).with_name("calibration_eval.json")
    rep_dest.parent.mkdir(parents=True, exist_ok=True)
    rep_dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    click.echo(format_calibration_eval_report(metrics))
    if fusion_config_path:
        click.echo("")
        click.echo("### After fusion replay")
        fused = payload["fused"]
        click.echo(
            f"- spatial_recall: {fused.get('spatial_recall', 0):.3f}  "
            f"label_recall: {fused.get('label_recall', 0):.3f}  "
            f"nodes: {int(fused.get('n_fused_nodes', 0))}"
        )
    click.echo(f"\nWrote report -> {rep_dest}")


if __name__ == "__main__":
    main()
