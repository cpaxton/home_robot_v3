# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Headless-friendly export: save machine-readable memory + human-readable text report
# (GraphEQA unified format and OpenVocab scene graph).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from emet.memory.floor_metrics import (
    compute_explored_floor_metrics,
    format_floor_metrics_summary,
    merge_spawn_floor_map,
    write_floor_metrics_json,
)
from emet.memory.format import SCENE_GRAPH_REPORT_TXT
from emet.memory.graph_eqa import format_scene_graph_pretty


def export_dynagraph_visual_assets(graph_memory: Any, export_dir: str | Path) -> None:
    """
    Write detector crops, mosaic, ``seen_from`` edges, and a gallery index under ``export_dir/dynagraph/``.

    Rerun live logging skips these by default to avoid viewer OOM; use this folder for offline review.
    """
    from emet.visualization.rerun import (
        _mosaic_labeled_images,
        _rgb_to_uint8,
        build_dynagraph_export_gallery_markdown,
        collect_dynagraph_crop_entries,
        format_dynagraph_node_label,
    )

    root = Path(export_dir)
    dg = root / "dynagraph"
    crops_dir = dg / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    mosaic_entries, crop_relpath = collect_dynagraph_crop_entries(graph_memory)
    from emet.visualization.rerun import _obs_rgb_by_id, dynagraph_node_rgb_crop

    obs_rgb = _obs_rgb_by_id(graph_memory)
    for n in graph_memory.get_nodes():
        if int(n.node_id) not in crop_relpath:
            continue
        rgb = dynagraph_node_rgb_crop(n, obs_rgb)
        if rgb is None:
            continue
        out_path = root / crop_relpath[int(n.node_id)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            Image.fromarray(_rgb_to_uint8(rgb)).save(out_path)
        except ImportError:
            np.save(out_path.with_suffix(".npy"), _rgb_to_uint8(rgb))

    mosaic = _mosaic_labeled_images(mosaic_entries)
    if mosaic is not None:
        try:
            from PIL import Image

            Image.fromarray(mosaic).save(dg / "crops_mosaic.png")
        except ImportError:
            np.save(dg / "crops_mosaic.npy", mosaic)

    nodes = graph_memory.get_nodes()
    node_by_id = {int(n.node_id): n for n in nodes}
    get_edges = getattr(graph_memory, "get_edges", None)
    edges = list(get_edges()) if get_edges is not None else []
    seen_from_rows: list[dict[str, Any]] = []
    for a, b, rel in edges:
        if rel != "seen_from":
            continue
        obj = node_by_id.get(int(a))
        vp = node_by_id.get(int(b))
        row: dict[str, Any] = {
            "object_node_id": int(a),
            "viewpoint_node_id": int(b),
            "relation": rel,
        }
        if obj is not None:
            row["object_label"] = (obj.labels[0] if obj.labels else "").strip()
            row["object_xyz"] = [float(x) for x in np.asarray(obj.xyz).reshape(3)]
            row["object_obs_id"] = int(obj.obs_id)
        if vp is not None:
            row["viewpoint_xyz"] = [float(x) for x in np.asarray(vp.xyz).reshape(3)]
            row["viewpoint_obs_id"] = int(vp.obs_id)
            row["viewpoint_label"] = format_dynagraph_node_label(vp)
        seen_from_rows.append(row)

    (dg / "seen_from.json").write_text(
        json.dumps({"seen_from": seen_from_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    obj_nodes = [n for n in nodes if not getattr(n, "is_viewpoint", False)]
    (dg / "gallery.md").write_text(
        build_dynagraph_export_gallery_markdown(obj_nodes, crop_relpath_by_node_id=crop_relpath),
        encoding="utf-8",
    )


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
    export_dynagraph_visual_assets(graph_memory, path)
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
