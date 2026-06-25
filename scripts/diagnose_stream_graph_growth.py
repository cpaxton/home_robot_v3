#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
"""Break down graph node growth during ``emet stream`` (object vs frontier vs viewpoint)."""

from __future__ import annotations

import argparse
import sys

from emet.app.stream_agent_factory import (
    create_dynagraph_agent,
    resolve_stream_dynav_config,
    stream_agent_update,
)


def _breakdown(graph_memory) -> dict[str, int]:
    nodes = graph_memory.get_nodes()
    frontier = [n for n in nodes if getattr(n, "is_frontier", False)]
    viewpoint = [n for n in nodes if n.is_viewpoint]
    obj = [n for n in nodes if not n.is_viewpoint and not getattr(n, "is_frontier", False)]
    return {
        "total": len(nodes),
        "object": len(obj),
        "frontier": len(frontier),
        "viewpoint": len(viewpoint),
    }


def _fusion_summary(agent) -> str:
    fusion = getattr(agent, "_graph_object_fusion", None)
    if fusion is None:
        return "fusion=disabled"
    cfg = fusion.config
    return (
        f"fusion=on fallback_xy={cfg.fallback_spatial_merge_xy_m:.2f} "
        f"strict_xy={cfg.spatial_merge_xy_m:.2f} embed_min={cfg.embedding_min_cosine:.2f}"
    )


def run_session(
    *,
    host: str,
    max_steps: int,
    use_sensor_perception: bool,
) -> list[dict]:
    dynav = resolve_stream_dynav_config(
        "innate_mars",
        host,
        "dynav_config.yaml",
        dynav_from_default=True,
    )
    agent, dynav_resolved = create_dynagraph_agent(
        robot="innate_mars",
        host=host,
        port_offset=0,
        dynav_config=dynav,
        enable_rerun=False,
        headless=True,
        allow_missing_depth=True,
        use_sensor_perception=use_sensor_perception,
        use_instance_graph=True,
    )
    if not agent.robot.start():
        raise RuntimeError(f"ZMQ connect failed to {host}")

    gm = agent.graph_memory
    print(f"dynav={dynav_resolved!r} sensor_perception={use_sensor_perception} {_fusion_summary(agent)}")
    print(f"frontier_enabled={getattr(gm, 'frontier_nodes_enabled', '?')}")

    rows: list[dict] = []
    prev = _breakdown(gm)
    rows.append({"step": 0, **prev, "d_total": 0, "d_object": 0, "d_frontier": 0, "d_viewpoint": 0})

    for step in range(1, max_steps + 1):
        stream_agent_update(agent, "dynagraph")
        cur = _breakdown(gm)
        row = {
            "step": step,
            **cur,
            "d_total": cur["total"] - prev["total"],
            "d_object": cur["object"] - prev["object"],
            "d_frontier": cur["frontier"] - prev["frontier"],
            "d_viewpoint": cur["viewpoint"] - prev["viewpoint"],
        }
        rows.append(row)
        print(
            f"  step {step}: total={cur['total']} (+{row['d_total']}) "
            f"object={cur['object']} (+{row['d_object']}) "
            f"frontier={cur['frontier']} (+{row['d_frontier']}) "
            f"viewpoint={cur['viewpoint']} (+{row['d_viewpoint']})"
        )
        prev = cur

    agent.robot.stop()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="herman")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("both", "default", "no_sensor"),
        default="both",
        help="default=VLMP+instances; no_sensor=instance graph only",
    )
    args = parser.parse_args()

    modes: list[tuple[str, bool]] = []
    if args.mode in ("both", "default"):
        modes.append(("default (VLM+instances)", True))
    if args.mode in ("both", "no_sensor"):
        modes.append(("no_sensor_perception", False))

    for label, use_sensor in modes:
        print(f"\n=== {label} ===")
        try:
            run_session(host=args.host, max_steps=args.max_steps, use_sensor_perception=use_sensor)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
