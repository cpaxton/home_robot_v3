# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Headless-friendly export: save machine-readable memory + human-readable text report
# (GraphEQA unified format and OpenVocab scene graph).

from __future__ import annotations

from pathlib import Path
from typing import Any

from emet.memory.floor_metrics import (
    compute_explored_floor_metrics,
    format_floor_metrics_summary,
    merge_spawn_floor_map,
    write_floor_metrics_json,
)
from emet.memory.format import SCENE_GRAPH_REPORT_TXT
from emet.memory.graph_eqa import format_scene_graph_pretty


def export_graph_eqa_dir(
    graph_memory: Any,
    voxel_map: Any,
    path: str,
    *,
    title: str = "Scene graph",
    robot: str | None = None,
    environment: dict[str, Any] | None = None,
    spawn_floor_map: dict[str, Any] | None = None,
) -> str:
    """
    Save GraphEQA memory via MemoryBackend (manifest, graph.json, frames, …) and write
    ``scene_graph_report.txt`` for logs / headless inspection.

    Returns:
        The same pretty-print string written to disk (caller may print to stdout).
    """
    from emet.memory.backend import get_memory_backend

    backend = get_memory_backend(
        "graph_eqa",
        graph_memory=graph_memory,
        voxel_map=voxel_map,
    )
    backend.save(path)
    text = format_scene_graph_pretty(graph_memory, title=title)
    floor_metrics = compute_explored_floor_metrics(
        voxel_map,
        robot=robot,
        environment=environment,
    )
    floor_metrics = merge_spawn_floor_map(floor_metrics, spawn_floor_map)
    write_floor_metrics_json(path, floor_metrics)
    floor_summary = format_floor_metrics_summary(floor_metrics)
    text = f"{text.rstrip()}\n\n--- Explored floor ---\n{floor_summary}\n"
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / SCENE_GRAPH_REPORT_TXT
    report_path.write_text(text, encoding="utf-8")
    return text


def export_open_vocab_scene_graph_dir(scene_graph: Any, path: str) -> str:
    """
    Save OpenVocabSceneGraph (scene_graph.json, node_tensors.pt, crops, …) and write
    ``scene_graph_report.txt`` from ``to_string()``.

    Returns:
        The report string written to disk.
    """
    scene_graph.save(path)
    text = scene_graph.to_string()
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / SCENE_GRAPH_REPORT_TXT
    report_path.write_text(text, encoding="utf-8")
    return text
