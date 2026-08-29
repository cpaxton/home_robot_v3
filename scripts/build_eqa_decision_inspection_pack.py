#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build a best-effort inspection pack for bundles without ``eqa_decisions/``."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nearest_frame(metadata: list[dict], xy: tuple[float, float]) -> dict | None:
    best = None
    best_d = float("inf")
    for row in metadata:
        pose = row.get("pose_xyt") or row.get("base_pose_xyt")
        if not pose:
            continue
        d = math.hypot(float(pose[0]) - xy[0], float(pose[1]) - xy[1])
        if d < best_d:
            best_d = d
            best = row
    return best


def build_inspection_pack(bundle_dir: Path, *, out_name: str = "eqa_decisions_retro") -> Path:
    bundle_dir = bundle_dir.resolve()
    out = bundle_dir / out_name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    metrics = _load_json(bundle_dir / "metrics.json") if (bundle_dir / "metrics.json").is_file() else {}
    history = []
    hist_path = bundle_dir / "eqa_history.json"
    if hist_path.is_file():
        history = _load_json(hist_path).get("iterations") or []

    meta_rows = []
    meta_path = bundle_dir / "metadata.jsonl"
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("frame_idx") is not None:
                meta_rows.append(row)

    seen_from = []
    seen_path = bundle_dir / "dynagraph" / "seen_from.json"
    if seen_path.is_file():
        seen_from = _load_json(seen_path).get("seen_from") or []

    clock_views: list[dict] = []
    for edge in seen_from:
        if "clock" not in str(edge.get("object_label") or "").lower():
            continue
        vp = edge.get("viewpoint_xyz") or []
        if len(vp) < 2:
            continue
        frame = _nearest_frame(meta_rows, (float(vp[0]), float(vp[1])))
        clock_views.append(
            {
                "object_obs_id": edge.get("object_obs_id"),
                "viewpoint_obs_id": edge.get("viewpoint_obs_id"),
                "object_xyz": edge.get("object_xyz"),
                "viewpoint_xyz": vp,
                "nearest_frame_idx": frame.get("frame_idx") if frame else None,
                "nearest_frame_relpath": frame.get("image") if frame else None,
            }
        )

    key_dir = out / "key_views"
    key_dir.mkdir()
    links: list[dict] = []
    for i, cv in enumerate(clock_views):
        rel = cv.get("nearest_frame_relpath")
        if not rel:
            continue
        src = bundle_dir / rel
        if not src.is_file():
            continue
        dst = key_dir / f"clock_view_{i + 1}_obs{cv.get('viewpoint_obs_id')}_frame{cv.get('nearest_frame_idx')}.png"
        shutil.copy2(src, dst)
        links.append({"copy": str(dst.relative_to(out)), **cv})

    crops_dir = bundle_dir / "dynagraph" / "crops"
    if crops_dir.is_dir():
        crop_out = out / "clock_crops"
        crop_out.mkdir()
        for crop in sorted(crops_dir.glob("*clock*.png")):
            shutil.copy2(crop, crop_out / crop.name)

    summary = {
        "bundle": str(bundle_dir),
        "question_id": metrics.get("question_id"),
        "question": metrics.get("question"),
        "gold_answer_letter": metrics.get("gold_answer_letter"),
        "predicted_answer": metrics.get("predicted_answer"),
        "eqa_iterations": metrics.get("eqa_iterations"),
        "prompt_obs_count": (metrics.get("graph_health") or {}).get("prompt_obs_count"),
        "note": (
            "Retro pack: full per-iteration VLM prompts were not saved on this run. "
            "Use key_views/ + clock_crops/ + eqa_history.json. Future runs write eqa_decisions/."
        ),
        "clock_views": links,
        "eqa_history_lines": len(history),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if hist_path.is_file():
        shutil.copy2(hist_path, out / "eqa_history.json")
    if (bundle_dir / "raw_eqa.txt").is_file():
        shutil.copy2(bundle_dir / "raw_eqa.txt", out / "raw_eqa.txt")

    readme = out / "README.md"
    readme.write_text(
        "# Retro EQA inspection pack\n\n"
        "This bundle predates per-iteration ``eqa_decisions/`` export.\n\n"
        "## q33-style clock miss\n\n"
        "- **Image 1** in every iteration cited the **wall clock at obs 68** "
        "(viewpoint ``(-0.37, 2.77)``).\n"
        "- See ``key_views/`` for the head-camera frame at that pose.\n"
        "- ``clock_crops/046_clock.png`` is the detector bbox crop, not the full VLM frame.\n"
        "- ``eqa_history.json`` has 20 one-line summaries (no obs ids on old runs).\n\n"
        "New runs: ``eqa_decisions/iter_NNN/{prompt.txt,meta.json,image_*}.``\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument(
        "-o",
        "--out-name",
        default="eqa_decisions_retro",
        help="Subdirectory name under the bundle (default: eqa_decisions_retro)",
    )
    args = parser.parse_args()
    out = build_inspection_pack(args.bundle_dir, out_name=args.out_name)
    print(json.dumps({"inspection_pack": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
