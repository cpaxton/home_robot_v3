# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Headless-friendly export: save machine-readable memory + human-readable text report
# (GraphEQA unified format and OpenVocab scene graph).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emet.memory.format import (
    GT_ALIGNMENT_REPORT_TXT,
    SCENE_GRAPH_REPORT_TXT,
    SIM_GT_PLACEMENTS_FILENAME,
)
from emet.memory.graph_eqa import format_scene_graph_pretty


def export_graph_eqa_dir(
    graph_memory: Any,
    voxel_map: Any,
    path: str,
    *,
    title: str = "Scene graph",
    ground_truth_mode: bool = False,
    sim_object_placements: dict[str, Any] | None = None,
    gt_alignment_report_text: str | None = None,
) -> str:
    """
    Save GraphEQA memory via MemoryBackend (manifest, graph.json, frames, …) and write
    ``scene_graph_report.txt`` for logs / headless inspection.

    When ``ground_truth_mode`` is set, also writes ``sim_object_placements.json`` and
    optional ``gt_alignment_report.txt``.

    Returns:
        The same pretty-print string written to disk (caller may print to stdout).
    """
    from emet.memory.backend import get_memory_backend

    backend = get_memory_backend(
        "graph_eqa",
        graph_memory=graph_memory,
        voxel_map=voxel_map,
    )
    backend.save(
        path,
        ground_truth_mode=ground_truth_mode,
        sim_object_placements=sim_object_placements,
    )
    text = format_scene_graph_pretty(graph_memory, title=title)
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / SCENE_GRAPH_REPORT_TXT
    report_path.write_text(text, encoding="utf-8")

    if sim_object_placements:
        from emet.memory.graph_eqa.sim_ground_truth_graph import placements_to_json_dict

        placements_path = out / SIM_GT_PLACEMENTS_FILENAME
        placements_path.write_text(
            json.dumps(placements_to_json_dict(sim_object_placements), indent=2),
            encoding="utf-8",
        )
    if gt_alignment_report_text:
        (out / GT_ALIGNMENT_REPORT_TXT).write_text(gt_alignment_report_text, encoding="utf-8")

    return text


def export_dynagraph_episode(
    graph_memory: Any,
    voxel_map: Any,
    path: str,
    *,
    title: str = "Scene graph (Dynagraph export)",
    ground_truth_mode: bool = False,
    sim_object_placements: dict[str, Any] | None = None,
    gt_alignment_report_text: str | None = None,
) -> str:
    """Full Dynagraph episode export (graph + frames + optional GT sidecars)."""
    return export_graph_eqa_dir(
        graph_memory,
        voxel_map,
        path,
        title=title,
        ground_truth_mode=ground_truth_mode,
        sim_object_placements=sim_object_placements,
        gt_alignment_report_text=gt_alignment_report_text,
    )


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
