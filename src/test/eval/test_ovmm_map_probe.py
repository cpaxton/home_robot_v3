# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU tests for emet ovmm probe-map (offline graph query)."""

from __future__ import annotations

import json
from pathlib import Path

from emet.eval.ovmm_map_probe import graph_hits_for_query, probe_graph, unique_labels
from emet.memory.format import GRAPH_FILENAME


def _write_graph(tmp_path: Path) -> Path:
    blob = {
        "nodes": [
            {"node_id": 1, "labels": ["jar", "bottle"], "xyz": [1.0, 0.0, 0.9]},
            {"node_id": 2, "labels": ["cabinet", "counter"], "xyz": [2.0, 0.1, 1.0]},
            {"node_id": 3, "labels": ["view img 4"], "xyz": [0.0, 0.0, 0.0]},
        ],
        "edges": [],
    }
    path = tmp_path / GRAPH_FILENAME
    path.write_text(json.dumps(blob), encoding="utf-8")
    return tmp_path


def test_graph_hits_jar_and_cabinet_not_cab_word(tmp_path: Path) -> None:
    nodes = [
        {"node_id": 1, "labels": ["jar"], "xyz": [0.0, 0.0, 0.0]},
        {"node_id": 2, "labels": ["cabinet"], "xyz": [1.0, 0.0, 0.0]},
    ]
    jar = graph_hits_for_query(nodes, "jar")
    assert jar["n_hits"] == 1
    cab = graph_hits_for_query(nodes, "cab")
    # Live find uses substring match: "cab" is inside "cabinet".
    assert cab["n_hits"] == 1
    cabinet = graph_hits_for_query(nodes, "cabinet")
    assert cabinet["n_hits"] == 1
    missing = graph_hits_for_query(nodes, "apple")
    assert missing["n_hits"] == 0


def test_probe_graph_reads_json(tmp_path: Path) -> None:
    root = _write_graph(tmp_path)
    report = probe_graph(root, ["jar", "cabinet", "apple"])
    assert report["n_nodes"] == 3
    by_q = {row["query"]: row["n_hits"] for row in report["queries"]}
    assert by_q["jar"] == 1
    assert by_q["cabinet"] == 1
    assert by_q["apple"] == 0
    labels = unique_labels([{"labels": ["Cabinet", "cabinet"]}, {"labels": ["jar"]}])
    assert {x.lower() for x in labels} == {"cabinet", "jar"}
