#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SQA3D + ScanNet embodied smoke: one scene, mock LLM, score batch JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = "scene0380_00"
DEFAULT_QUESTION_ID = 220602000000
DEFAULT_SPLIT = "train"


def _run(cmd: list[str], *, cwd: Path = REPO) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="SQA3D ScanNet embodied smoke test")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--question-id", type=int, default=DEFAULT_QUESTION_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sqa3d_scannet_smoke"))
    parser.add_argument("--download-scannet", action="store_true", help="Fetch mesh if missing")
    args = parser.parse_args()

    from emet.benchmarks.sqa3d.scannet.config import default_scannet_root, scene_assets_present

    scannet_root = default_scannet_root()
    if not scene_assets_present(args.scene, scannet_root):
        if not args.download_scannet:
            print(
                f"ScanNet mesh missing for {args.scene} under {scannet_root}\n"
                "Re-run with --download-scannet or:\n"
                f"  uv run python scripts/download_scannet_data.py --accept-tos --scene {args.scene}",
                file=sys.stderr,
            )
            return 1
        dl = _run(
            [
                sys.executable,
                "scripts/download_scannet_data.py",
                "--accept-tos",
                "--scene",
                args.scene,
                "--scannet-root",
                str(scannet_root),
            ]
        )
        if dl.returncode != 0:
            print(dl.stdout + dl.stderr, file=sys.stderr)
            return dl.returncode

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "episode.jsonl"

    ep = _run(
        [
            sys.executable,
            "-m",
            "emet.cli",
            "sqa3d",
            "run-episode",
            "--split",
            args.split,
            "--question-id",
            str(args.question_id),
            "--mock-llm",
            "--max-planning-steps",
            "3",
            "--output",
            str(jsonl),
        ]
    )
    print(ep.stdout)
    if ep.returncode != 0:
        print(ep.stderr, file=sys.stderr)
        return ep.returncode

    ev = _run(
        [
            sys.executable,
            "-m",
            "emet.cli",
            "eval-sqa3d",
            "-p",
            str(jsonl),
            "--split",
            args.split,
            "-o",
            str(out_dir / "eval.json"),
        ]
    )
    print(ev.stdout)
    if ev.returncode != 0:
        print(ev.stderr, file=sys.stderr)
        return ev.returncode

    summary = json.loads((out_dir / "eval.json").read_text(encoding="utf-8"))
    em = float(summary["qa"]["em@1"])
    print(f"\nSmoke OK: em@1={em:.4f} episodes={int(summary['qa']['n_questions'])}")
    return 0 if em >= 1.0 else 2


if __name__ == "__main__":
    sys.exit(main())
