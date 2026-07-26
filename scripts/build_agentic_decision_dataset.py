#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026

"""Build a causal, scene-disjoint agentic EQA decision dataset from bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.eval.agentic_dataset import mine_evidence_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Bundle root containing agentic_trace.jsonl files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--split-salt", default="emet-agentic-v1")
    parser.add_argument(
        "--require-view-labels",
        action="store_true",
        help="Keep only HM3D semantic-sensor view labels (exclude distance proxies/unlabeled)",
    )
    args = parser.parse_args()
    manifest = mine_evidence_dataset(
        args.root,
        args.output,
        salt=args.split_salt,
        require_view_labels=args.require_view_labels,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
