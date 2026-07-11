#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Batch-render top-down map figures from episode bundles for paper / tuning runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _scene_id_from_bundle(bundle: Path) -> str | None:
    for rel in ("metrics.json", "metadata.jsonl", "diagnostics_manifest.json"):
        path = bundle / rel
        if not path.is_file():
            continue
        if rel == "metadata.jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                scene = row.get("scene") or row.get("scene_id")
                if scene:
                    return str(scene)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            scene = data.get("scene") or data.get("scene_id")
            if scene:
                return str(scene)
    return None


def _discover_bundles(episodes_root: Path, run_id: str) -> list[Path]:
    out: list[Path] = []
    if not episodes_root.is_dir():
        return out
    seen: set[Path] = set()
    for parent in episodes_root.glob(f"*{run_id}*"):
        if not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "explored_2d.npy").is_file() and child not in seen:
                seen.add(child)
                out.append(child)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=Path.home() / ".cache/habitat_eqa/episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ~/runs/emet/dynagraph_tuning/<run-id>/figures/maps",
    )
    parser.add_argument("--write-map-video", action="store_true")
    parser.add_argument("--max-bundles", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--min-side", type=int, default=512, help="Upscale small crops (default 512; use 1024 for legacy exports)")
    parser.add_argument("--no-filter-islands", action="store_true")
    parser.add_argument("--with-gt", action="store_true", help="Write GT navmesh when scene id is known")
    parser.add_argument(
        "--with-overlay",
        action="store_true",
        help="Write GT+agent overlay when scene id is known",
    )
    parser.add_argument("--scene-id", default="", help="Override HM3D scene id for all bundles")
    args = parser.parse_args()

    out_dir = args.output_dir or (
        Path.home() / "runs" / "emet" / "dynagraph_tuning" / args.run_id / "figures" / "maps"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO))
    from scripts.render_episode_map_overlay import render_bundle

    bundles = _discover_bundles(args.episodes_root.expanduser(), args.run_id)[: args.max_bundles]
    manifest: list[dict[str, object]] = []
    for bundle in bundles:
        sub = out_dir / bundle.name
        sub.mkdir(parents=True, exist_ok=True)
        try:
            scene_id = args.scene_id.strip() or _scene_id_from_bundle(bundle)
            paths = render_bundle(
                bundle,
                scene_id=scene_id,
                max_side=max(256, int(args.max_side)),
                min_side=max(128, int(args.min_side)),
                filter_islands=not args.no_filter_islands,
                write_gt=bool(args.with_gt and scene_id),
                write_overlay=bool(args.with_overlay and scene_id),
            )
            for key, src in paths.items():
                src_path = Path(src)
                if src_path.is_file():
                    dst = sub / src_path.name
                    shutil.copy2(src_path, dst)
                    paths[key] = str(dst)
            if args.write_map_video:
                from emet.eval.episode_diagnostics import _write_map_exploration_mp4

                mp4 = _write_map_exploration_mp4(bundle, prefer_overlay=True)
                if mp4:
                    dst_mp4 = sub / Path(mp4).name
                    shutil.copy2(mp4, dst_mp4)
                    paths["topdown_exploration"] = str(dst_mp4)
            manifest.append({"bundle": str(bundle), "output_dir": str(sub), "ok": True, "paths": paths})
            print(f"ok {bundle.name} -> {sub}")
        except Exception as exc:
            manifest.append({"bundle": str(bundle), "output_dir": str(sub), "ok": False, "error": str(exc)})
            print(f"FAIL {bundle.name}: {exc}")

    manifest_path = out_dir / "maps_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({len(manifest)} bundles)")


if __name__ == "__main__":
    main()
