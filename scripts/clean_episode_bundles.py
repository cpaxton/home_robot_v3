#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Retention-based cleanup of Habitat HM-EQA episode debug bundles.

Each full-113 sweep writes large per-episode debug artifacts (RGB frame PNGs,
MP4s, topdown maps) under ``~/.cache/habitat_eqa/episodes/<run_tag>/`` — on the
order of 2–45 GB per method. The scored results live separately in
``~/.cache/habitat_eqa/results/*.jsonl`` (tens of MB), so old bundles can be
pruned without losing any benchmark numbers.

This script keeps the newest ``--keep N`` runs per prefix (a prefix is the
``subset_paper113_YYYYMMDD_HHMMSS_dynagraph``-style stem) and deletes the rest:

    # dry run (default)
    uv run python scripts/clean_episode_bundles.py

    # keep the newest 2 runs per prefix, delete the rest
    uv run python scripts/clean_episode_bundles.py --keep 2 --apply

    # aggressive: drop everything older than 14 days, regardless of prefix
    uv run python scripts/clean_episode_bundles.py --max-age-days 14 --apply

    # free space report only
    uv run python scripts/clean_episode_bundles.py --report

Never touches ``~/.cache/habitat_eqa/results/`` (the scored jsonl).
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

from emet.habitat.episode_debug import default_episodes_root


def _prefix(name: str) -> str:
    """Group bundles of one sweep together.

    Sweep bundles look like ``subset_paper113_20260813_104004_dynagraph_qwen3_vl``
    and ``..._static_graph_qwen3_vl`` (same run, one dir per method). Group by the
    run stem: everything through the first 8-digit ``YYYYMMDD`` (or the whole name
    when no timestamp is present, e.g. ``cli_episode_q0000``).
    """
    m = re.search(r"(\d{8})", name)
    if m:
        return name[: m.start() + 8]
    return name


def gather(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=int, default=2, help="keep newest N bundle dirs per prefix (default 2)")
    ap.add_argument("--max-age-days", type=float, default=0, help="also delete any bundle older than N days")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--report", action="store_true", help="print per-bundle sizes and exit")
    args = ap.parse_args()

    root = default_episodes_root()
    dirs = gather(root)
    if not dirs:
        print(f"no bundle dirs under {root}")
        return 0

    if args.report:
        total = 0
        for p in dirs:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            total += sz
            print(f"{sz / 1e9:8.2f} GB  {p.name}")
        print(f"total: {total / 1e9:.2f} GB across {len(dirs)} bundles")
        return 0

    now = time.time()
    # Group bundles of the same sweep (per _prefix) so keep=N is per RUN, not per
    # per-method dir (dynagraph + static_graph of one sweep stay together).
    by_prefix: dict[str, list[Path]] = {}
    for p in dirs:
        by_prefix.setdefault(_prefix(p.name), []).append(p)

    doomed: list[tuple[Path, str]] = []
    kept: list[Path] = []
    for _pre, ps in by_prefix.items():
        ps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for i, p in enumerate(ps):
            age_d = (now - p.stat().st_mtime) / 86400.0
            if i >= args.keep:
                doomed.append((p, f"beyond keep={args.keep} runs"))
            elif args.max_age_days > 0 and age_d > args.max_age_days:
                doomed.append((p, f"older than {args.max_age_days:.0f}d"))
            else:
                kept.append(p)

    if not doomed:
        print(f"nothing to clean ({len(dirs)} bundles, keep={args.keep})")
        return 0

    freed = 0
    for p, why in doomed:
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        freed += sz
        if args.apply:
            print(f"DELETE {sz / 1e9:7.2f} GB  {p.name}  ({why})")
            shutil.rmtree(p, ignore_errors=True)
        else:
            print(f"would   {sz / 1e9:7.2f} GB  {p.name}  ({why})")
    print(
        f"freed: {freed / 1e9:.2f} GB  ({len(doomed)} bundles)  "
        f"[{'APPLIED' if args.apply else 'dry-run; use --apply to delete'}]"
    )
    print(f"kept {len(kept)} bundles; results/*.jsonl untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
