# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Lifelong checkpoint graph.json invalidation after world fuzz."""

from __future__ import annotations

import json
from pathlib import Path

from emet.eval.dynamic_exploration_runner import invalidate_checkpoint_nodes_near_moves


def test_invalidate_checkpoint_nodes_near_moves(tmp_path: Path):
    graph = {
        "nodes": [
            {
                "node_id": 1,
                "labels": ["obj_main"],
                "xyz": [1.0, 0.0, 0.5],
                "obs_id": 1,
                "is_viewpoint": False,
                "is_frontier": False,
            },
            {
                "node_id": 2,
                "labels": ["sink"],
                "xyz": [4.0, 4.0, 0.5],
                "obs_id": 2,
                "is_viewpoint": False,
                "is_frontier": False,
            },
            {
                "node_id": 3,
                "labels": ["viewpoint"],
                "xyz": [0.0, 0.0, 1.0],
                "obs_id": 1,
                "is_viewpoint": True,
                "is_frontier": False,
            },
        ],
        "observations": [
            {"obs_id": 1, "labels": ["obj_main"]},
            {"obs_id": 2, "labels": ["sink"]},
        ],
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    n = invalidate_checkpoint_nodes_near_moves(
        tmp_path,
        [{"target": "obj_main", "old_pos": [1.0, 0.0, 0.5], "pos": [2.5, 1.0, 0.5]}],
    )
    assert n >= 1
    data = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    labels = [" ".join(n.get("labels") or []) for n in data["nodes"] if not n.get("is_viewpoint")]
    assert any("sink" in lb for lb in labels)
    assert not any("obj_main" in lb for lb in labels)
