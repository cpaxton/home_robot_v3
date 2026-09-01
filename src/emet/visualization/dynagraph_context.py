# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Always-on Dynagraph VLM-context panel for live Rerun debugging."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_stats import format_graph_node_breakdown
from emet.utils.logger import Logger
from emet.visualization.null_visualizer import visualizer_is_enabled

logger = Logger(__name__)

_CONTEXT_ENTITY = "world/dynagraph/context"
_MOSAIC_ENTITY = "world/dynagraph/context/mosaic"
_MAX_MD = 48_000
_MAX_PROMPT = 12_000
_MAX_ROUTER = 8_000
_MAX_NODES = 16


def _as_rgb_uint8(img: Any) -> np.ndarray | None:
    if img is None:
        return None
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4) or arr.size == 0:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _truncate(text: str, limit: int) -> str:
    raw = text.strip()
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "\n\n… _(truncated)_"


def _attached_image_lines(graph_memory: Any) -> list[str]:
    obs_ids = [int(x) for x in (graph_memory.last_eqa_obs_ids or [])]
    look_id = graph_memory.last_eqa_look_obs_id
    action_id = graph_memory.last_eqa_action_obs_id
    n_fallback = int(graph_memory.last_eqa_nav_fallback_count or 0)
    n_images = len(graph_memory.last_relevant_images or [])
    if not obs_ids and n_images <= 0:
        return ["_(no images attached to the last VLM call)_"]
    lines: list[str] = []
    n_show = max(len(obs_ids), n_images)
    for i in range(n_show):
        tags: list[str] = []
        if i < len(obs_ids):
            oid = obs_ids[i]
            tags.append(f"obs {oid}")
            if look_id is not None and int(look_id) == oid:
                tags.append("look")
            if action_id is not None and int(action_id) == oid:
                tags.append("action")
        elif n_fallback:
            tags.append("nav fallback / crop")
        else:
            tags.append("extra")
        lines.append(f"- **Image {i + 1}** — {', '.join(tags)}")
    return lines


def _object_node_lines(graph_memory: Any) -> list[str]:
    obj = [n for n in graph_memory.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    obj.sort(key=lambda n: (-int(n.support_count), int(n.node_id)))
    shown = obj[:_MAX_NODES]
    omitted = len(obj) - len(shown)
    lines = ["| id | label | img | xyz |", "| -: | ----- | --: | --- |"]
    for n in shown:
        labels = n.labels or []
        primary = str(labels[0]).strip() if labels else "object"
        xyz = np.asarray(n.xyz, dtype=np.float64).reshape(-1)
        xyz_s = f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})" if xyz.size >= 3 else "?"
        lines.append(f"| {int(n.node_id)} | {primary} | {int(n.obs_id)} | {xyz_s} |")
    if omitted > 0:
        lines.append("")
        lines.append(f"_(+{omitted} more object nodes)_")
    return lines


def build_dynagraph_context_markdown(graph_memory: Any | None, *, max_chars: int = _MAX_MD) -> str:
    """Compact dump of what the VLM actually saw (prompt, Image N, router state)."""
    if graph_memory is None:
        return "# Dynagraph context\n\n_(no graph memory)_\n"
    parsed = graph_memory.last_eqa_parsed or ("", "", False, "", "")
    answer = str(parsed[1] or "").strip() if len(parsed) > 1 else ""
    confident = bool(parsed[2]) if len(parsed) > 2 else False
    rag = graph_memory.last_eqa_spatial_rag or {}
    decision = graph_memory.last_agentic_decision
    n_attached = len(graph_memory.last_eqa_obs_ids or [])
    lines = [
        "# Dynagraph context",
        "",
        "What the last EQA / router call actually received. Heavy crop/edge streams stay off; this panel is always on.",
        "",
        f"- **Graph:** {format_graph_node_breakdown(graph_memory)}",
        f"- **Prompt nodes:** {graph_memory.last_eqa_prompt_node_count} · **attached obs:** {n_attached}",
        f"- **Regions (spatial RAG):** {graph_memory.last_eqa_prompt_regions}",
        f"- **Last answer:** {answer or '_(none)_'} · confident={confident}",
        f"- **look_obs:** {graph_memory.last_eqa_look_obs_id} · **action_obs:** {graph_memory.last_eqa_action_obs_id}",
        "",
        "## Attached images (Image N in the prompt)",
        *_attached_image_lines(graph_memory),
        "",
        "## Object nodes (top by support)",
        *_object_node_lines(graph_memory),
    ]
    if rag:
        lines.extend(
            [
                "",
                "## Spatial RAG",
                f"- regions={rag.get('n_regions')} nodes={rag.get('n_nodes')} radius_m={rag.get('radius_m')}",
                f"- seed_node_ids={rag.get('seed_node_ids')}",
            ]
        )
    prompt = _truncate(graph_memory.last_eqa_prompt_text or "", _MAX_PROMPT)
    lines.extend(["", "## Last EQA prompt", ""])
    if prompt:
        lines.append("```")
        lines.append(prompt)
        lines.append("```")
    else:
        lines.append("_(no EQA prompt yet — run a question / query_answer)_")
    router = _truncate(graph_memory.last_router_state_text or "", _MAX_ROUTER)
    lines.extend(["", "## Last router state", ""])
    if router:
        lines.append("```")
        lines.append(router)
        lines.append("```")
    else:
        lines.append("_(no agentic router turn yet)_")
    if isinstance(decision, dict) and decision:
        bits = ", ".join(f"{k}={decision[k]!r}" for k in list(decision)[:8])
        lines.extend(["", "## Last agentic decision", bits])
    raw = _truncate(graph_memory.last_eqa_raw or "", 4000)
    if raw:
        lines.extend(["", "## Last model raw (truncated)", "", "```", raw, "```"])
    text = "\n".join(lines).strip() + "\n"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n… _(truncated for Rerun)_\n"
    return text


def _eqa_context_mosaic_entries(graph_memory: Any) -> list[tuple[str, np.ndarray]]:
    obs_ids = [int(x) for x in (graph_memory.last_eqa_obs_ids or [])]
    images = list(graph_memory.last_relevant_images or [])
    entries: list[tuple[str, np.ndarray]] = []
    if images:
        for i, img in enumerate(images):
            arr = _as_rgb_uint8(img)
            if arr is None:
                continue
            if i < len(obs_ids):
                cap = f"I{i + 1} obs{obs_ids[i]}"
            else:
                cap = f"I{i + 1}"
            entries.append((cap, arr))
        return entries
    rgb_fn = getattr(graph_memory, "_eqa_rgb_for_obs", None)
    if rgb_fn is None:
        return entries
    for i, oid in enumerate(obs_ids):
        arr = _as_rgb_uint8(rgb_fn(int(oid)))
        if arr is None:
            continue
        entries.append((f"I{i + 1} obs{oid}", arr))
    return entries


def _eqa_context_mosaic(entries: list[tuple[str, np.ndarray]]) -> np.ndarray | None:
    """Tile EQA Image-N thumbnails. Isolated so tests can stub it without importing rerun-sdk."""
    # Deferred: emet.visualization.rerun pulls rerun-sdk native extensions.
    from emet.visualization.rerun import _mosaic_labeled_images

    return _mosaic_labeled_images(entries, thumb_max=160, cols=min(4, len(entries)))


def log_vlm_context_to_visualizer(visualizer: Any, graph_memory: Any | None) -> None:
    """Write ``world/dynagraph/context`` markdown + Image-N mosaic (no-op if Rerun is off)."""
    if not visualizer_is_enabled(visualizer):
        return
    try:
        visualizer.log_text(_CONTEXT_ENTITY, build_dynagraph_context_markdown(graph_memory))
        if graph_memory is None:
            return
        entries = _eqa_context_mosaic_entries(graph_memory)
        key = tuple(cap for cap, _ in entries)
        if key == getattr(visualizer, "_eqa_context_mosaic_key", None):
            return
        visualizer._eqa_context_mosaic_key = key
        if not entries:
            visualizer.clear_identity(_MOSAIC_ENTITY)
            return
        mosaic = _eqa_context_mosaic(entries)
        if mosaic is None:
            return
        visualizer.log_custom_2d_image(_MOSAIC_ENTITY, mosaic)
    except Exception as err:
        logger.warning(f"VLM-context Rerun log failed: {err}")


def send_graph_memory_rerun_blueprint(
    visualizer: Any,
    *,
    graph_label: str = "Graph 3D",
    summary_label: str = "Context (VLM)",
    extra_right_column: Any | None = None,
) -> None:
    """Live layout: world 3D, monologue, cameras, graph + VLM context + EQA mosaic."""
    if not visualizer_is_enabled(visualizer):
        return
    try:
        # Deferred: rerun-sdk native extensions.
        import rerun as rr
        import rerun.blueprint as rrb

        graph_column = rrb.Vertical(
            rrb.Spatial3DView(name=graph_label, origin="world/dynagraph"),
            rrb.TextDocumentView(name=summary_label, origin="world/dynagraph/context"),
            rrb.Spatial2DView(name="EQA images", origin="world/dynagraph/context/mosaic"),
        )
        right_columns: list[Any] = [graph_column]
        if extra_right_column is not None:
            right_columns.append(extra_right_column)
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world", contents="world/**"),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="robot_monologue"),
                rrb.Spatial2DView(name="relevant image", origin="/observation_similar_to_text"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="world/head_camera/rgb"),
                rrb.Spatial2DView(name="ee_rgb", origin="world/ee_camera/rgb"),
                rrb.Spatial2DView(name="map_topdown", origin="world/map_snapshot/topdown"),
            ),
            rrb.Vertical(*right_columns),
            column_shares=[3, 1, 1, 1],
        )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(main, rrb.TimePanel(state=True)),
                collapse_panels=visualizer.collapse_panels,
            )
        )
    except Exception as err:
        logger.warning(f"Graph-memory Rerun blueprint failed: {err}")
