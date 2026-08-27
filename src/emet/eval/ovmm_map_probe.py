# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Query a saved scene map (``graph.json`` / ``voxel_map.pkl``) with no sim.

Iterate FindObj/FindRec phrases on ``~/.cache/emet/scene_maps/<key>/`` instead of
multi-hour Stretch agentic episodes. Graph matching uses the same
``category_matches`` + query variants as live find. ``--voxel`` loads SigLIP and
``localize_text`` on the pickle (still no MuJoCo).

Default dump phrases include ``cab`` / ``jar`` so substring false hits are
visible. Live camera verify is :mod:`emet.eval.ovmm_verify_probe`; GT body
selection is :mod:`emet.eval.ovmm_probe_targets`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emet.eval.ovmm_find_phase import _query_variants, category_matches
from emet.eval.ovmm_probe_targets import DEFAULT_MAP_QUERIES
from emet.eval.scene_map_cache import scene_cache_dir, scene_cache_root
from emet.memory.format import GRAPH_FILENAME, VOXEL_PICKLE_FILENAME

DEFAULT_KITCHEN_QUERIES = DEFAULT_MAP_QUERIES
DEFAULT_ROBOCASA_L1_KEY = "robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt"


def list_cached_maps(*, root: Path | str | None = None) -> list[dict[str, Any]]:
    """Return cache keys under the scene-map root that have a graph and/or voxel pickle."""
    base = scene_cache_root(root)
    rows: list[dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        graph = child / GRAPH_FILENAME
        voxel = child / VOXEL_PICKLE_FILENAME
        if not graph.is_file() and not voxel.is_file():
            continue
        rows.append(
            {
                "key": child.name,
                "dir": str(child),
                "has_graph": graph.is_file(),
                "has_voxel": voxel.is_file(),
            }
        )
    return rows


def resolve_map_dir(*, map_dir: Path | str | None = None, cache_key: str | None = None) -> Path:
    """Resolve a scene-map directory from ``--map`` or ``--cache-key``.

    Empty key falls back to the July Robocasa L1 Stretch GT dump.
    """
    if map_dir is not None:
        return Path(map_dir).expanduser().resolve()
    key = (cache_key or "").strip() or DEFAULT_ROBOCASA_L1_KEY
    return scene_cache_dir(key)


def load_graph_nodes(map_dir: Path | str) -> list[dict[str, Any]]:
    """Load ``graph.json`` nodes as ``{node_id, labels, xyz}`` dicts."""
    path = Path(map_dir) / GRAPH_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    raw = blob.get("nodes") if isinstance(blob, dict) else blob
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no nodes list")
    nodes: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        labels = [str(x) for x in (item.get("labels") or []) if str(x).strip()]
        xyz = item.get("xyz")
        nodes.append(
            {
                "node_id": item.get("node_id"),
                "labels": labels,
                "xyz": xyz,
            }
        )
    return nodes


def unique_labels(nodes: list[dict[str, Any]]) -> list[str]:
    """Sorted unique node labels (case-insensitive de-dupe, original spelling kept)."""
    seen: set[str] = set()
    out: list[str] = []
    for node in nodes:
        for lab in node.get("labels") or []:
            key = str(lab).strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(str(lab).strip())
    return sorted(out, key=str.lower)


def graph_hits_for_query(nodes: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """Match ``query`` the same way live find does (``category_matches`` + variants)."""
    variants = _query_variants(query)
    hits: list[dict[str, Any]] = []
    for node in nodes:
        labels = [str(lbl) for lbl in (node.get("labels") or [])]
        matched_labels = [lbl for lbl in labels if any(category_matches(v, lbl) for v in variants)]
        if not matched_labels:
            continue
        hits.append(
            {
                "node_id": node.get("node_id"),
                "xyz": node.get("xyz"),
                "matched_labels": matched_labels,
                "labels": labels[:12],
            }
        )
    return {
        "query": query,
        "variants": variants,
        "n_hits": len(hits),
        "hits": hits,
    }


def probe_graph(
    map_dir: Path | str,
    queries: list[str],
) -> dict[str, Any]:
    """CPU graph query: unique labels plus per-phrase hit lists."""
    nodes = load_graph_nodes(map_dir)
    labels = unique_labels(nodes)
    rows = [graph_hits_for_query(nodes, q) for q in queries if str(q).strip()]
    return {
        "map_dir": str(Path(map_dir).expanduser().resolve()),
        "n_nodes": len(nodes),
        "n_unique_labels": len(labels),
        "unique_labels": labels,
        "queries": rows,
    }


def probe_voxel(
    map_dir: Path | str,
    queries: list[str],
) -> dict[str, Any]:
    """Load ``voxel_map.pkl`` and run ``localize_text`` (SigLIP; no MuJoCo).

    Constructs the map with ``detection=None``, so YOLOE never runs — cosine-only.
    Live find attaches YOLOE; this path is a pickle replay, not a detector bakeoff.
    """
    from emet.core.parameters import get_parameters
    from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap
    from emet.mapping.voxel_localize import localize_text_xyz
    from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

    pkl = Path(map_dir) / VOXEL_PICKLE_FILENAME
    if not pkl.is_file():
        raise FileNotFoundError(f"missing {pkl}")
    parameters = get_parameters("dynav_config.yaml")
    encoder = get_shared_mask_siglip_encoder(
        version="so400m",
        device="cuda",
        feature_matching_threshold=0.14,
    )
    voxel_map = SparseVoxelMap(
        resolution=parameters["voxel_size"],
        local_radius=parameters["local_radius"],
        obs_min_height=parameters["obs_min_height"],
        obs_max_height=parameters["obs_max_height"],
        obs_min_density=parameters["obs_min_density"],
        grid_resolution=0.1,
        min_depth=parameters["min_depth"],
        max_depth=parameters["max_depth"],
        pad_obstacles=parameters.get("pad_obstacles", 0),
        encoder=encoder,
        detection=None,
        mllm=False,
        run_eqa=False,
        defer_eqa_vllm=True,
        log=str(Path(map_dir) / "probe_voxel"),
    )
    voxel_map.read_from_pickle(str(pkl))
    rows: list[dict[str, Any]] = []
    for query in queries:
        xyz, stats = localize_text_xyz(voxel_map, query, refresh=True)
        rows.append(
            {
                "query": query,
                "xyz": None if xyz is None else [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "max_cosine": stats.get("max_cosine") if isinstance(stats, dict) else None,
                "yoloe_hit": bool(stats.get("yoloe_hit")) if isinstance(stats, dict) else False,
            }
        )
    return {"map_dir": str(Path(map_dir).expanduser().resolve()), "queries": rows}
