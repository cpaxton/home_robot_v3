#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
"""Frozen Habitat RGB comparison with explicitly approximate XY/FOV coverage.

No simulator, depth reconstruction, or ground-truth labels feed selection.
Target XY comes from the recorded investigate hypothesis. Unoccluded 2D FOV
coverage is an optimistic proxy, NOT measured surface coverage. Optional VLM
calls must run through the serial GPU job queue.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from emet.memory.view_coverage import CoverageGrid, depth_surface_cells, gaussian_coverage_objective
from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views


def build_case(source, out, sigma, depth_replay=None, max_images=4, max_depth_m=4.0):
    out.mkdir(parents=True, exist_ok=False)
    decision = source / "eqa_decisions/iter_001"
    current_pixels = np.asarray(Image.open(decision / "image_1.png").convert("RGB"))
    rows = [json.loads(line) for line in (source / "metadata.jsonl").read_text().splitlines()]
    pixels = {r["frame_idx"]: np.asarray(Image.open(source / r["image"]).convert("RGB")) for r in rows}
    matches = [r for r in rows if np.array_equal(pixels[r["frame_idx"]], current_pixels)]
    if not matches:
        raise ValueError("assessed image does not exactly match a saved render")
    current = matches[-1]
    rows = [r for r in rows if r["frame_idx"] <= current["frame_idx"]]
    replay = None
    if depth_replay is not None:
        replay = json.loads((depth_replay / "replay.json").read_text())
        if Path(replay["source"]).resolve() != source.resolve():
            raise ValueError("depth replay belongs to a different episode")
        replay_rows = {r["frame_idx"]: r for r in replay["frames"]}
        # Always use co-rendered RGB with depth, never pair old RGB with new depth.
        pixels = {
            r["frame_idx"]: np.asarray(Image.open(depth_replay / replay_rows[r["frame_idx"]]["rgb"]).convert("RGB"))
            for r in rows
        }
    trace = [json.loads(line) for line in (source / "agentic_trace.jsonl").read_text().splitlines()]
    targets = [r["look_at_xy"] for r in trace if r.get("look_at_xy") is not None]
    if len(targets) != 1:
        raise ValueError("diagnostic requires one unambiguous recorded target")
    target = np.asarray(targets[0])
    # Fixed world-aligned grid: avoid counting repeated XY cells twice.
    resolution = 0.2
    lo = np.floor((target - 3 * sigma) / resolution).astype(int)
    hi = np.ceil((target + 3 * sigma) / resolution).astype(int)
    indices = np.array([(x, y) for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1)])
    centers = (indices + 0.5) * resolution
    keys = [f"{x},{y}" for x, y in indices]
    objective = gaussian_coverage_objective(
        dict(zip(keys, centers, strict=True)),
        target_xy=tuple(target),
        sigma_m=sigma,
        scope_id=str(source),
        visibility_source=f"rendered_depth_endpoints_stride4_range{max_depth_m:g}"
        if replay
        else "unoccluded_base_xy_yaw_fov120_range4_proxy",
    )
    frames = []
    saved = []
    for row in rows:
        i = row["frame_idx"]
        x, y, yaw = row["pose_xyt"]
        delta = centers - [x, y]
        angles = np.arctan2(delta[:, 1], delta[:, 0]) - yaw
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        mask = (np.linalg.norm(delta, axis=1) <= 4) & (np.abs(angles) <= np.pi / 3)
        cells = frozenset(k for k, keep in zip(keys, mask, strict=True) if keep)
        pose = np.eye(4)
        pose[:3, :3] = [[-np.sin(yaw), 0, np.cos(yaw)], [np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0]]
        pose[:3, 3] = [x, y, 0]
        if replay is not None:
            with np.load(depth_replay / replay_rows[i]["geometry"], allow_pickle=False) as geometry:
                pose = geometry["camera_pose"]
                cells = depth_surface_cells(
                    geometry["depth"],
                    geometry["intrinsics"],
                    pose,
                    grid=CoverageGrid(resolution),
                    max_depth_m=max_depth_m,
                    pixel_stride=4,
                )
                cells = cells & frozenset(keys)  # Same ROI for weighted/uniform coverage.
        frames.append(ViewFrame(str(i), str(source), float(i), pixels[i], camera_pose=pose, visible_cells=cells))
        path = out / f"frame-{i:04d}.png"
        Image.fromarray(pixels[i]).save(path)
        saved.append(
            {
                "view_id": str(i),
                "scope_id": str(source),
                "timestamp_s": float(i),
                "image_path": path.name,
                "camera_pose": pose.tolist(),
                "visible_cells": sorted(cells),
            }
        )
    settings = {"max_images": max_images, "max_candidates": len(frames), "max_age_s": len(frames) + 1}
    selections = {}
    for policy in ["recency", "pose", "coverage", "weighted_coverage"]:
        selections[policy] = select_views(
            frames,
            current_id=str(current["frame_idx"]),
            config=SelectionConfig(policy=policy, **settings),
            coverage_objective=objective,
        )
    selections["current_only"] = select_views(
        frames,
        current_id=str(current["frame_idx"]),
        config=SelectionConfig(policy="recency", **{**settings, "max_images": 1}),
        coverage_objective=objective,
    )
    manifest = {
        "source": str(source),
        "target_xy": target.tolist(),
        "sigma_m": sigma,
        "coverage_max_depth_m": max_depth_m if replay else None,
        "assumptions": "Sequence indices disable age pruning. Target is an unverified recorded hypothesis. "
        + (
            "RGB and depth co-rendered; actual optical poses. Replay differences recorded separately."
            if replay
            else "Base-yaw/FOV proxy without occlusion."
        ),
        "replay": replay,
        "frames": saved,
        "current_id": str(current["frame_idx"]),
        "selection_config": settings,
        "coverage_objective": {
            "scope_id": str(source),
            "cell_weights": dict(objective.cell_weights),
            "visibility_source": objective.visibility_source,
        },
        "image_sha256": {
            r["image_path"]: hashlib.sha256((out / r["image_path"]).read_bytes()).hexdigest() for r in saved
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "selections.json").write_text(json.dumps({k: v.to_dict() for k, v in selections.items()}, indent=2))
    sheet = Image.new("RGB", (1280, 280 * len(selections)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (policy, selection) in enumerate(selections.items()):
        draw.text((5, j * 280 + 2), policy + " | " + objective.visibility_source, fill="black")
        for k, v in enumerate(selection.views):
            rgb = Image.fromarray(v.frame.rgb)
            rgb.thumbnail((320, 240))
            sheet.paste(rgb, (320 * k, 280 * j + 20))
            draw.text(
                (320 * k + 4, 280 * j + 262),
                f"{v.role} frame {v.frame.view_id}; new weight {v.features['weighted_coverage']:.1f}",
                fill="black",
            )
    sheet.save(out / "comparison.jpg")
    meta = json.loads((decision / "meta.json").read_text())
    return meta["question"], selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument(
        "--depth-replay", action="append", type=Path, help="One replay directory per episode, same order"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigma-m", type=float, default=1.5)
    parser.add_argument("--max-images", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--coverage-max-depth-m", type=float, default=4.0)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    replays = args.depth_replay or [None] * len(args.episode)
    if len(replays) != len(args.episode):
        parser.error("supply one --depth-replay per --episode")
    cases = [
        (p.name, *build_case(p, args.output / p.name, args.sigma_m, replay, args.max_images, args.coverage_max_depth_m))
        for p, replay in zip(args.episode, replays, strict=True)
    ]
    if not args.infer:
        return
    import torch

    from emet.core.parameters import get_parameters
    from emet.eval.agentic_vlm_assess import assess_view_with_vlm
    from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

    params = get_parameters("configs/emet/default.yaml")
    (args.output / "eqa_config.json").write_text(json.dumps(dict(params.get("eqa") or {}), indent=2, default=str))
    _, client = build_graph_eqa_vlm_clients(parameters=params, device="cuda")
    with (args.output / "responses.jsonl").open("x") as output:
        for name, question, selections in cases:
            for repeat in range(args.repeats):
                for policy, selection in selections.items():
                    torch.manual_seed(repeat)
                    np.random.seed(repeat)
                    # No fabricated seconds from sequence-only metadata in prompts.
                    labels = "\n".join(
                        f"Image {i}: "
                        + ("current assessment view" if i == 1 else "earlier context; not proof of current visibility")
                        for i in range(1, len(selection.views) + 1)
                    )
                    bundle = SimpleNamespace(views=selection.views, prompt_labels=lambda: labels)
                    result = assess_view_with_vlm(
                        client, question=question, rgb=selection.views[0].frame.rgb, is_mcq=True, selected_views=bundle
                    )
                    if not result.raw:
                        raise RuntimeError("empty inference response; not a scored abstention")
                    row = {
                        "case": name,
                        "policy": policy,
                        "repeat": repeat,
                        "views": [v.frame.view_id for v in selection.views],
                        **result.to_dict(),
                    }
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
