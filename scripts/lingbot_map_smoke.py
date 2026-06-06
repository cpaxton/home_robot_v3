#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Evaluate LingBot-Map predictions vs sim GT; optional DA3 baseline and Rerun."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import click
import cv2
import numpy as np

from emet.perception.depth.lingbot_eval import (
    evaluate_da3_depth_rmse,
    evaluate_lingbot_vs_gt,
    load_episode,
    load_lingbot_predictions,
    load_rgb,
)


def _depth_colorize(depth: np.ndarray, near: float = 0.15, far: float = 6.0) -> np.ndarray:
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    finite = d[d > 1e-6]
    if finite.size > 50:
        near = float(np.percentile(finite, 5))
        far = float(np.percentile(finite, 95))
        far = max(far, near + 0.5)
    u8 = (np.clip((d - near) / (far - near), 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _maybe_run_lingbot_infer(
    episode: Path,
    lingbot_output: Path,
    *,
    checkpoint: Path | None,
    keyframe_interval: int | None,
    skip_infer: bool,
    use_sdpa: bool,
) -> None:
    if skip_infer and (lingbot_output / "lingbot_predictions.jsonl").is_file():
        return
    repo = Path(__file__).resolve().parents[1]
    py = Path(os.environ.get("LINGBOT_MAP_VENV", repo / ".venv-lingbot-map")) / "bin" / "python"
    ckpt = checkpoint or Path(os.environ.get("LINGBOT_MAP_CHECKPOINT", ""))
    if not py.is_file():
        raise FileNotFoundError(f"Missing LingBot venv python: {py} (run ./scripts/install_lingbot_map.sh)")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt} (set LINGBOT_MAP_CHECKPOINT)")
    cmd = [
        str(py),
        "-m",
        "emet_lingbot_map",
        "infer",
        "--episode",
        str(episode),
        "--output",
        str(lingbot_output),
        "--checkpoint",
        str(ckpt),
    ]
    if keyframe_interval is not None:
        cmd.extend(["--keyframe-interval", str(keyframe_interval)])
    if use_sdpa:
        cmd.append("--use-sdpa")
    click.echo("Running: " + " ".join(cmd))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        click.echo(proc.stdout)
        click.echo(proc.stderr, err=True)
        raise SystemExit(proc.returncode)


@click.command()
@click.option("--episode", type=click.Path(exists=True, file_okay=False), required=True)
@click.option(
    "--lingbot-output",
    type=click.Path(file_okay=False),
    default=None,
    help="LingBot prediction dir (default: <episode>/lingbot)",
)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--keyframe-interval", type=int, default=2, show_default=True)
@click.option(
    "--use-sdpa/--flashinfer", default=True, show_default=True, help="SDPA avoids FlashInfer JIT (needs ninja)."
)
@click.option("--skip-infer", is_flag=True, help="Skip LingBot infer if predictions exist")
@click.option("--skip-da3", is_flag=True, help="Skip DA3 baseline (faster)")
@click.option("--da3-model-id", default="depth-anything/DA3-SMALL", show_default=True)
@click.option("--rerun", is_flag=True, help="Log comparison to Rerun")
@click.option("--rerun-spawn", is_flag=True, help="Open native Rerun window")
@click.option("--max-rerun-frames", type=int, default=80, show_default=True)
def main(
    episode: str,
    lingbot_output: str | None,
    checkpoint: str | None,
    keyframe_interval: int,
    use_sdpa: bool,
    skip_infer: bool,
    skip_da3: bool,
    da3_model_id: str,
    rerun: bool,
    rerun_spawn: bool,
    max_rerun_frames: int,
) -> None:
    """Compare LingBot depth/pose to sim sensor GT on a recorded Mars episode."""
    ep = Path(episode)
    lb_out = Path(lingbot_output) if lingbot_output else ep / "lingbot"
    ckpt = Path(checkpoint) if checkpoint else None

    _maybe_run_lingbot_infer(
        ep,
        lb_out,
        checkpoint=ckpt,
        keyframe_interval=keyframe_interval,
        skip_infer=skip_infer,
        use_sdpa=use_sdpa,
    )

    metrics = evaluate_lingbot_vs_gt(ep, lb_out)
    click.echo("LingBot vs sim GT:")
    for k, v in metrics.items():
        click.echo(f"  {k}: {v}")

    da3_metrics = None
    if not skip_da3:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            da3_metrics = evaluate_da3_depth_rmse(ep, model_id=da3_model_id, device=device)
            click.echo("DA3 vs sim GT (depth only):")
            for k, v in da3_metrics.items():
                click.echo(f"  {k}: {v}")
        except Exception as e:
            click.echo(f"DA3 baseline skipped: {e}", err=True)

    report = {"lingbot": metrics, "da3": da3_metrics}
    report_path = lb_out / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(f"Wrote {report_path}")

    if rerun:
        try:
            import rerun as rr
        except ImportError as e:
            raise SystemExit("rerun-sdk required for --rerun") from e

        rr.init("lingbot_map_smoke", spawn=rerun_spawn)
        if not rerun_spawn:
            rr.serve(open_browser=not os.environ.get("RERUN_HEADLESS", "").strip())

        ep_data = load_episode(ep)
        preds = load_lingbot_predictions(lb_out)
        scale = float(metrics.get("depth_scale_median", 1.0) or 1.0)

        for i, fr in enumerate(ep_data.frames[:max_rerun_frames]):
            if fr.frame_idx not in preds:
                continue
            rgb = load_rgb(fr)
            rr.set_time_sequence("frame", i)
            rr.log("gt/rgb", rr.Image(rgb))
            pred_d = preds[fr.frame_idx]["depth"] * scale
            h, w = rgb.shape[:2]
            if pred_d.shape[:2] != (h, w):
                pred_d = cv2.resize(pred_d, (w, h), interpolation=cv2.INTER_LINEAR)
            rr.log("lingbot/depth_colormap", rr.Image(_depth_colorize(pred_d)))
            if fr.camera_pose is not None:
                gt_c = np.asarray(fr.camera_pose[:3, 3], dtype=np.float32)
                lb_c = np.asarray(preds[fr.frame_idx]["camera_pose"][:3, 3], dtype=np.float32)
                rr.log(
                    "trajectory/gt",
                    rr.Points3D(positions=gt_c.reshape(1, 3), colors=[[0, 200, 0]], radii=0.05),
                )
                rr.log(
                    "trajectory/lingbot",
                    rr.Points3D(positions=lb_c.reshape(1, 3), colors=[[200, 80, 0]], radii=0.05),
                )


if __name__ == "__main__":
    main()
