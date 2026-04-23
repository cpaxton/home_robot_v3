# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.

"""Offline: rebuild GraphEQA graph from a saved memory directory (frames + optional detections)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import click
import numpy as np

from emet.core.parameters import get_parameters
from emet.memory.format import (
    SCENE_GRAPH_REPORT_TXT,
    GraphBlob,
    GraphEdgeView,
    GraphNodeView,
    MemoryManifest,
    MemoryState,
    load_memory,
    save_memory,
)
from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.graph_eqa.graph_observation_pipeline import (
    apply_instance_items_to_graph,
    dense_world_xyz_from_frame,
    frameblob_to_labels_xyz,
)
from emet.memory.graph_eqa.instance_observations import DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M
from emet.memory.graph_eqa.pretty_print import format_scene_graph_pretty


def _make_dedup_skips(gm: GraphEQAMemory, threshold: float):
    def _dedup(label: str, xyz: np.ndarray) -> bool:
        if threshold <= 0:
            return False
        lb = label.strip().lower()
        for n in gm.get_nodes():
            if not n.labels:
                continue
            nl = (n.labels[0] or "").strip().lower()
            if nl != lb:
                continue
            if float(np.linalg.norm(n.xyz[:2] - xyz[:2])) < threshold:
                return True
        return False

    return _dedup


def _merge_yolo_instance(fr: Any, detector: Any) -> Any:
    """Run YoloE and attach instance + classes to a new ``FrameBlob``."""
    from emet.memory.format import FrameBlob

    if fr.rgb is None:
        return None
    rgb = np.asarray(fr.rgb)
    depth = np.asarray(fr.depth) if fr.depth is not None else None
    _sem, inst, task = detector.predict(rgb, depth=depth, draw_instance_predictions=False)
    ic = task.get("instance_classes")
    if ic is None:
        return None
    wx = dense_world_xyz_from_frame(fr)
    if wx is None:
        return None
    scores = task.get("instance_scores")
    return FrameBlob(
        camera_pose=fr.camera_pose,
        base_pose=fr.base_pose,
        camera_K=fr.camera_K,
        rgb=fr.rgb,
        depth=fr.depth,
        feats=fr.feats,
        world_xyz=wx,
        instance=np.asarray(inst, dtype=np.int64),
        instance_classes=np.asarray(ic, dtype=np.int64),
        instance_scores=np.asarray(scores, dtype=np.float32) if scores is not None else None,
        detections=fr.detections,
        info=fr.info,
    )


def reprocess_memory_directory(
    input_dir: str,
    output_dir: str,
    *,
    parameters: dict | Any,
    run_detector: bool = False,
    require_cache: bool = False,
    min_depth: float = 0.1,
    max_depth: float = 4.0,
) -> tuple[GraphEQAMemory, str]:
    """Load frames from disk, rebuild graph nodes, write directory + report. Returns (memory, report text)."""
    state = load_memory(input_dir)
    if isinstance(parameters, dict):
        dedup_m = float(parameters.get("graph_instance_dedup_xy_m", DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M))
    else:
        dedup_m = float(parameters.get("graph_instance_dedup_xy_m", DEFAULT_GRAPH_INSTANCE_DEDUP_XY_M))

    gm = GraphEQAMemory(parameters=parameters, defer_llm_clients=True)
    dedup_skips = _make_dedup_skips(gm, dedup_m)

    detector: Any = None
    if run_detector:
        import torch

        from emet.perception.detection.yoloe import YoloEPerception

        det_conf = 0.05
        if isinstance(parameters, dict):
            det_conf = float(parameters.get("detection", {}).get("confidence_threshold", 0.05))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = YoloEPerception(confidence_threshold=det_conf, device=device, size="l")

    for fr in state.frames:
        if fr.rgb is None:
            continue
        rgb = np.asarray(fr.rgb)

        if fr.detections:
            for row in fr.detections:
                label = str(row.get("label_short") or row.get("label") or "object")
                xyz = np.asarray(row["xyz"], dtype=np.float64).ravel()[:3]
                if dedup_skips(label, xyz):
                    continue
                gm.add_observation(rgb, xyz, [label])
            continue

        if require_cache and fr.instance is None:
            continue

        fr_work = fr
        if fr.instance is None and run_detector and detector is not None:
            merged = _merge_yolo_instance(fr, detector)
            if merged is not None:
                fr_work = merged

        if fr_work.instance is None:
            continue

        items = frameblob_to_labels_xyz(
            fr_work,
            min_depth=min_depth,
            max_depth=max_depth,
            detection_model=detector,
            min_points=10,
        )
        if items:
            apply_instance_items_to_graph(gm, rgb, items, dedup_skips=dedup_skips)

    manifest = state.manifest or MemoryManifest()
    desc = getattr(manifest, "description", None) or ""
    manifest = replace(
        manifest,
        description=(desc + f" | reprocessed_from={input_dir}").strip(" |"),
        backend="graph_eqa",
    )
    graph_blob = GraphBlob(
        nodes=[
            GraphNodeView(
                node_id=n.node_id,
                labels=list(n.labels),
                xyz=list(np.ravel(n.xyz).tolist()),
                obs_id=n.obs_id,
                description=getattr(n, "description", None),
            )
            for n in gm.get_nodes()
        ],
        edges=[GraphEdgeView(id1=e[0], id2=e[1], relation=e[2]) for e in gm.get_edges()],
    )
    out_state = MemoryState(
        point_cloud=state.point_cloud,
        frames=state.frames,
        graph=graph_blob,
        text_descriptions=state.text_descriptions,
        user_messages=state.user_messages,
        grid_origin=state.grid_origin,
        grid_resolution=state.grid_resolution,
        obstacles_2d=state.obstacles_2d,
        explored_2d=state.explored_2d,
        manifest=manifest,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_memory(out_state, output_dir)
    report = format_scene_graph_pretty(gm, title="Scene graph (reprocessed)")
    Path(output_dir).joinpath(SCENE_GRAPH_REPORT_TXT).write_text(report, encoding="utf-8")
    return gm, report


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=str),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Output directory (default: INPUT_DIR_reprocessed next to input)",
)
@click.option(
    "--run-detector/--no-run-detector",
    default=False,
    help="Run YoloE on RGB-D when instance masks are missing (needs GPU for typical setups).",
)
@click.option(
    "--require-cache",
    is_flag=True,
    help="Only consume frames that already have detections JSON or instance+depth data.",
)
@click.option(
    "--config",
    type=str,
    default="dynav_config.yaml",
    help="Parameters YAML for dedup threshold and detector confidence.",
)
def main(
    input_dir: str,
    output: str | None,
    run_detector: bool,
    require_cache: bool,
    config: str,
) -> None:
    """Rebuild graph.json from saved frames (detections JSON and/or instance masks)."""
    in_path = Path(input_dir).resolve()
    out_path = Path(output) if output else in_path.parent / f"{in_path.name}_reprocessed"
    out_path = out_path.resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    parameters = get_parameters(config)
    _gm, report = reprocess_memory_directory(
        str(in_path),
        str(out_path),
        parameters=parameters,
        run_detector=run_detector,
        require_cache=require_cache,
        min_depth=float(parameters.get("min_depth", 0.1)),
        max_depth=float(parameters.get("max_depth", 4.0)),
    )
    print(report)
    print(f"Wrote reprocessed memory to {out_path}")


if __name__ == "__main__":
    main()
