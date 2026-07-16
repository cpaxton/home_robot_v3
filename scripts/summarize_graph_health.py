#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Summarize Dynagraph graph-health from Habitat metrics.json or dynamic-explore cycles.

Examples::

    uv run python scripts/summarize_graph_health.py ~/.cache/habitat_eqa/episodes/RUN/q0094_dynagraph/metrics.json
    uv run python scripts/summarize_graph_health.py ~/runs/emet/dynamic_exploration/*/lifelong.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from emet.memory.graph_eqa.graph_stats import classify_graph_failure, graph_health_from_checkpoint_nodes


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_health(label: str, health: dict[str, Any]) -> None:
    cls = health.get("failure_class") or classify_graph_failure(health)
    print(
        f"{label}: class={cls} "
        f"obj={health.get('n_object')} vp={health.get('n_viewpoint')} "
        f"fr={health.get('n_frontier')} obs={health.get('n_obs')} "
        f"singleton_frac={health.get('singleton_frac')} "
        f"mean_support={health.get('mean_support')} "
        f"prompt_nodes={health.get('prompt_node_count')}"
    )
    top = health.get("top_labels") or []
    if top:
        print("  top_labels:", ", ".join(f"{t['label']}×{t['count']}" for t in top[:5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path, help="metrics.json / lifelong.json / graph.json")
    args = ap.parse_args()
    for path in args.paths:
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            continue
        data = _load(path)
        if path.name == "graph.json" or ("nodes" in data and isinstance(data.get("nodes"), list)):
            health = graph_health_from_checkpoint_nodes(
                list(data.get("nodes") or []),
                n_obs=len(data.get("observations") or []) if data.get("observations") is not None else None,
            )
            health["failure_class"] = classify_graph_failure(health)
            _print_health(str(path), health)
            continue
        if "graph_health" in data:
            _print_health(str(path), dict(data["graph_health"]))
            continue
        if "cycle_results" in data:
            for row in data.get("cycle_results") or []:
                h = dict(row.get("graph_health") or {})
                if not h and row.get("export_dir"):
                    gpath = Path(row["export_dir"]) / "graph.json"
                    if gpath.is_file():
                        g = _load(gpath)
                        h = graph_health_from_checkpoint_nodes(list(g.get("nodes") or []))
                h["failure_class"] = classify_graph_failure(h)
                _print_health(f"{path} cycle={row.get('cycle')}", h)
            continue
        print(f"unrecognized payload: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
