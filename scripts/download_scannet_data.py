#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Download ScanNet meshes (and optional posed ``.sens`` RGB-D) for SQA3D embodied replay."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from emet.benchmarks.sqa3d.scannet.config import (
    SQA3D_MIN_FILETYPES,
    SQA3D_SENS_FILETYPES,
    collect_scenes_from_question_slice,
    collect_sqa3d_scene_ids,
    default_download_script,
    default_scannet_root,
    scene_assets_present,
    scene_mesh_path,
    scene_replay_assets_present,
    scene_sens_path,
    scene_sens_present,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _print_instructions(scannet_root: Path) -> None:
    print(
        f"""
ScanNet layout for SQA3D embodied eval

  SCANNET_ROOT={scannet_root}
    scans/<scene_id>/<scene_id>_vh_clean_2.ply
    scans/<scene_id>/<scene_id>.txt
    scans/<scene_id>/<scene_id>.sens   (optional posed RGB-D; large)

Terms of Use: http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf

Quick start (downloads meshes for SQA3D val scenes, can be large):
  uv run python scripts/download_scannet_data.py --accept-tos --scenes-from-sqa3d --split val --limit 5

Smoke (one scene mesh, ~few MB):
  uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00

Posed RGB-D replay (adds ~hundreds of MB per scene):
  uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00 --with-sens

Run embodied episode (after SQA3D + ScanNet data):
  uv run emet sqa3d run-episode --question-id 220602000000 --mock-llm --replay-mode auto
"""
    )


def _run_download_script(
    script: Path,
    scannet_root: Path,
    scene_id: str,
    file_type: str,
    *,
    accept_tos: bool,
) -> int:
    cmd = [
        sys.executable,
        str(script),
        "-o",
        str(scannet_root),
        "--id",
        scene_id,
        "--type",
        file_type,
        "--skip_existing",
    ]
    # TOS prompt; .sens also prompts to confirm v1 sens download (Enter = include).
    stdin = "\n\n" if accept_tos and file_type == ".sens" else ("\n" if accept_tos else None)
    proc = subprocess.run(
        cmd,
        input=stdin,
        text=True,
        check=False,
    )
    return int(proc.returncode)


def _missing_file_types(
    scene_id: str,
    scannet_root: Path,
    file_types: tuple[str, ...],
) -> tuple[str, ...]:
    from emet.benchmarks.sqa3d.scannet.config import scene_scan_dir

    scan_dir = scene_scan_dir(scene_id, scannet_root)
    missing: list[str] = []
    for ft in file_types:
        path = scan_dir / f"{scene_id}{ft}"
        if not path.is_file():
            missing.append(ft)
    return tuple(missing)


def _download_scene(
    script: Path,
    scannet_root: Path,
    scene_id: str,
    *,
    accept_tos: bool,
    file_types: tuple[str, ...] = SQA3D_MIN_FILETYPES,
) -> bool:
    ok = True
    for ft in _missing_file_types(scene_id, scannet_root, file_types):
        code = _run_download_script(script, scannet_root, scene_id, ft, accept_tos=accept_tos)
        if code != 0:
            print(f"WARN: download failed scene={scene_id} type={ft} exit={code}")
            ok = False
    return ok


def _scene_complete(scene_id: str, scannet_root: Path, *, with_sens: bool) -> bool:
    replay_mode = "sens" if with_sens else "auto"
    return scene_replay_assets_present(scene_id, scannet_root, replay_mode=replay_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ScanNet assets for SQA3D")
    parser.add_argument("--scannet-root", type=Path, default=None, help="SCANNET_ROOT output directory")
    parser.add_argument("--download-script", type=Path, default=None, help="Path to download-scannet.py")
    parser.add_argument("--scene", action="append", default=[], help="Single scene id (repeatable)")
    parser.add_argument("--scenes-from-sqa3d", action="store_true", help="Download all scenes referenced by split")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sqa3d-data-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of scenes when using --scenes-from-sqa3d")
    parser.add_argument("--question-start", type=int, default=None, help="With --scenes-from-sqa3d: slice start index")
    parser.add_argument(
        "--question-end", type=int, default=None, help="With --scenes-from-sqa3d: slice end index (exclusive)"
    )
    parser.add_argument(
        "--with-sens",
        action="store_true",
        help="Also download posed RGB-D .sens files (large; required for --replay-mode sens)",
    )
    parser.add_argument("--accept-tos", action="store_true", help="Auto-accept ScanNet Terms of Use prompt")
    parser.add_argument("--verify", action="store_true", help="Only report which scenes are present")
    parser.add_argument("--instructions", action="store_true")
    args = parser.parse_args()

    scannet_root = args.scannet_root or default_scannet_root()
    file_types = SQA3D_SENS_FILETYPES if args.with_sens else SQA3D_MIN_FILETYPES

    if args.instructions:
        _print_instructions(scannet_root)
        return

    scenes: list[str] = list(args.scene)
    if args.scenes_from_sqa3d:
        if args.question_start is not None and args.question_end is not None:
            scenes = collect_scenes_from_question_slice(
                args.split,
                args.question_start,
                args.question_end,
                data_dir=args.sqa3d_data_dir,
            )
        else:
            scenes = collect_sqa3d_scene_ids(args.split, data_dir=args.sqa3d_data_dir, limit=args.limit)

    if not scenes:
        parser.print_help()
        _print_instructions(scannet_root)
        return

    if args.verify:
        present_mesh = sum(1 for s in scenes if scene_assets_present(s, scannet_root))
        present_sens = sum(1 for s in scenes if scene_sens_present(s, scannet_root))
        print(f"SCANNET_ROOT={scannet_root}")
        print(f"scenes_requested={len(scenes)} mesh={present_mesh} sens={present_sens}")
        for s in scenes[:20]:
            mesh = scene_mesh_path(s, scannet_root)
            sens = scene_sens_path(s, scannet_root)
            print(f"  {s}: mesh={mesh.is_file()} sens={sens.is_file()}")
        if len(scenes) > 20:
            print(f"  ... ({len(scenes) - 20} more)")
        return

    if not args.accept_tos:
        print("ScanNet download requires accepting the Terms of Use.")
        print("Re-run with --accept-tos (see http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf)")
        return

    script = args.download_script or default_download_script()
    if not script.is_file():
        raise FileNotFoundError(f"download-scannet.py not found: {script}")

    scannet_root.mkdir(parents=True, exist_ok=True)
    print(f"SCANNET_ROOT={scannet_root}")
    print(f"Downloading {len(scenes)} scene(s) file_types={file_types}...")
    ok_count = 0
    for scene_id in scenes:
        if _scene_complete(scene_id, scannet_root, with_sens=args.with_sens):
            print(f"  {scene_id}: already present")
            ok_count += 1
            continue
        if _download_scene(script, scannet_root, scene_id, accept_tos=True, file_types=file_types):
            ok_count += 1
    label = "mesh+sens" if args.with_sens else "mesh"
    print(f"Done: {ok_count}/{len(scenes)} scenes with {label} on disk")


if __name__ == "__main__":
    main()
