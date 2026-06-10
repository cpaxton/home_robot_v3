#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Aggregate SQA3D batch JSONL files to CSV (parallel to OVMM ``aggregate_*.csv``)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SQA3D episode JSONL sweeps to CSV.")
    parser.add_argument(
        "jsonl",
        nargs="*",
        help="Episode JSONL paths (default: all *.jsonl under --input-dir)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to scan for *.jsonl when no positional paths are given",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="configs/sqa3d/benchmark.yaml",
        help="Benchmark YAML for default output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for aggregate CSV/JSON (default from benchmark.yaml)",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="aggregate_sqa3d.csv",
        help="Output CSV filename inside --output-dir",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="aggregate_sqa3d.json",
        help="Optional mirror JSON summary filename",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate question_id rows")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None, help="Split for coverage stats")
    parser.add_argument("--data-dir", type=str, default=None, help="SQA3D data dir for coverage stats")
    parser.add_argument("--scannet-root", type=str, default=None, help="ScanNet root for runnable coverage")
    parser.add_argument(
        "--replay-mode",
        choices=("auto", "sens", "mesh"),
        default="auto",
        help="Replay mode for runnable coverage",
    )
    return parser.parse_args()


def _collect_jsonl_paths(args: argparse.Namespace) -> list[Path]:
    if args.jsonl:
        return [Path(p).expanduser().resolve() for p in args.jsonl]
    input_dir = Path(args.input_dir or "").expanduser()
    if not input_dir.is_dir():
        raise SystemExit("Provide JSONL paths or --input-dir")
    return sorted(input_dir.glob("*.jsonl"))


def main() -> int:
    from emet.benchmarks.sqa3d.aggregate import aggregate_sqa3d_jsonl_paths
    from emet.benchmarks.sqa3d.benchmark_config import load_sqa3d_benchmark_config

    args = _parse_args()
    paths = _collect_jsonl_paths(args)
    if not paths:
        print("No JSONL files found.", file=sys.stderr)
        return 1

    bench = load_sqa3d_benchmark_config(args.benchmark)
    output_dir = Path(args.output_dir or bench.paths.output_dir).expanduser().resolve()
    csv_path = output_dir / args.csv_name
    json_path = output_dir / args.json_name

    rows = aggregate_sqa3d_jsonl_paths(
        paths,
        dedupe=not args.no_dedupe,
        split=args.split,
        data_dir=Path(args.data_dir).expanduser() if args.data_dir else None,
        scannet_root=Path(args.scannet_root).expanduser() if args.scannet_root else None,
        replay_mode=args.replay_mode,
        output_csv=csv_path,
        output_json=json_path,
    )
    for row in rows:
        print(
            f"{row.get('source_jsonl', '')}\t"
            f"method={row.get('method')} replay={row.get('replay_backend')} "
            f"em@1={row.get('em@1'):.3f} n={row.get('n_episodes')} infra={row.get('n_infra')}"
        )
    print(f"Wrote {len(rows)} rows to {csv_path} (JSON: {json_path})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
