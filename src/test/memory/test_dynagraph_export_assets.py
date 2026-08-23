# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode, GraphObservation
from emet.memory.headless_export import export_dynagraph_visual_assets


def test_export_dynagraph_visual_assets(tmp_path: Path) -> None:
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[10:30, 12:40] = 200
    mem._observations.append(GraphObservation(obs_id=1, rgb=rgb, xyz=np.array([1.0, 0.0, 0.5]), labels=["mug"]))
    mem._nodes.append(
        GraphNode(
            node_id=1,
            labels=["mug"],
            xyz=np.array([1.0, 0.0, 0.5]),
            obs_id=1,
            bbox_xyxy=(12, 10, 40, 30),
        )
    )
    mem._nodes.append(
        GraphNode(
            node_id=2,
            labels=["view img 1"],
            xyz=np.array([0.5, 0.0, 0.0]),
            obs_id=1,
            is_viewpoint=True,
        )
    )
    mem._edges.append((1, 2, "seen_from"))

    export_dynagraph_visual_assets(mem, tmp_path)

    seen = json.loads((tmp_path / "dynagraph" / "seen_from.json").read_text(encoding="utf-8"))
    assert len(seen["seen_from"]) == 1
    assert seen["seen_from"][0]["object_node_id"] == 1
    assert (tmp_path / "dynagraph" / "gallery.md").is_file()
    crops = list((tmp_path / "dynagraph" / "crops").glob("*"))
    assert len(crops) >= 1
