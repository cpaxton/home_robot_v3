# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Spatial neighborhood RAG for GraphEQA / Dynagraph EQA prompts.

Instead of dumping a flat top-K node list with full (x,y,z) on every line, retrieve
keyword / preferred-obs seeds, expand planar neighbors, cluster into regions, and
emit compact REGION blocks for Qwen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from emet.memory.graph_eqa.geom import UnionFind as _UnionFind
from emet.memory.graph_eqa.geom import planar_dist as _planar_dist
from emet.memory.graph_eqa.geom import xy as _xy


@dataclass(frozen=True)
class SpatialRegion:
    """One spatially coherent group of object nodes for an EQA prompt."""

    region_id: int
    node_ids: tuple[int, ...]
    labels: tuple[str, ...]
    image_ids: tuple[int, ...]
    anchor_xy: tuple[float, float]
    score: float = 0.0


@dataclass
class SpatialRagResult:
    """Retrieval output for ``to_string`` / router state."""

    regions: list[SpatialRegion] = field(default_factory=list)
    kept_node_ids: set[int] = field(default_factory=set)
    frontier_nodes: list[Any] = field(default_factory=list)
    seed_node_ids: list[int] = field(default_factory=list)
    n_candidates: int = 0


def _label_list(node: Any) -> list[str]:
    out: list[str] = []
    for lab in list(getattr(node, "labels", None) or []):
        s = str(lab).strip()
        if s and s.lower() != "frontier":
            out.append(s)
    return out


def _keyword_hit(labels: Sequence[str], keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    blob = " ".join(labels).lower()
    hits = 0
    for kw in keywords:
        k = str(kw).strip().lower()
        if len(k) < 2:
            continue
        if k in blob:
            hits += 1
    return float(hits)


def select_seed_nodes(
    nodes: Sequence[Any],
    *,
    keywords: Sequence[str] | None = None,
    prefer_obs_ids: Sequence[int] | None = None,
    max_seeds: int = 12,
) -> list[Any]:
    """Pick object-node seeds for spatial expansion (not frontiers/viewpoints)."""
    prefer = {int(x) for x in (prefer_obs_ids or [])}
    kws = [str(k) for k in (keywords or []) if str(k).strip()]
    scored: list[tuple[float, Any]] = []
    for n in nodes:
        if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
            continue
        labels = _label_list(n)
        kw = _keyword_hit(labels, kws)
        prefer_bonus = 2.0 if int(getattr(n, "obs_id", -1) or -1) in prefer else 0.0
        support = float(getattr(n, "support_count", 1) or 1)
        score = 10.0 * kw + prefer_bonus + 0.1 * support
        if score <= 0.0 and not prefer_bonus:
            continue
        scored.append((score, n))
    scored.sort(key=lambda t: (-t[0], int(getattr(t[1], "node_id", 0))))
    if scored:
        return [n for _, n in scored[:max_seeds]]
    # Fallback: highest-support objects so RAG never returns empty on a populated graph.
    fallback: list[tuple[float, Any]] = []
    for n in nodes:
        if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
            continue
        support = float(getattr(n, "support_count", 1) or 1)
        fallback.append((support, n))
    fallback.sort(key=lambda t: (-t[0], int(getattr(t[1], "node_id", 0))))
    return [n for _, n in fallback[: max(1, min(max_seeds, 4))]]


def expand_neighbors(
    nodes: Sequence[Any],
    seeds: Sequence[Any],
    *,
    radius_m: float = 2.5,
    max_neighbors_per_seed: int = 8,
) -> list[Any]:
    """Union of seeds and planar neighbors within ``radius_m``."""
    by_id = {int(getattr(n, "node_id", i)): n for i, n in enumerate(nodes)}
    objects = [n for n in nodes if not getattr(n, "is_frontier", False) and not getattr(n, "is_viewpoint", False)]
    kept: dict[int, Any] = {}
    for seed in seeds:
        sid = int(getattr(seed, "node_id", -1))
        if sid in by_id:
            kept[sid] = by_id[sid]
        else:
            kept[sid] = seed
        sxy = _xy(seed)
        scored: list[tuple[float, Any]] = []
        for n in objects:
            nid = int(getattr(n, "node_id", -1))
            if nid == sid:
                continue
            d = _planar_dist(sxy, _xy(n))
            if d <= float(radius_m):
                scored.append((d, n))
        scored.sort(key=lambda t: t[0])
        for _, n in scored[:max_neighbors_per_seed]:
            kept[int(getattr(n, "node_id", -1))] = n
    return list(kept.values())


def cluster_into_regions(
    candidates: Sequence[Any],
    seeds: Sequence[Any],
    *,
    radius_m: float = 2.5,
    max_regions: int = 6,
    keywords: Sequence[str] | None = None,
) -> list[SpatialRegion]:
    """Greedy XY clustering of expanded candidates into prompt regions."""
    if not candidates:
        return []
    ids = [int(getattr(n, "node_id", i)) for i, n in enumerate(candidates)]
    by_id = {int(getattr(n, "node_id", i)): n for i, n in enumerate(candidates)}
    uf = _UnionFind(ids)
    for i, a in enumerate(candidates):
        axy = _xy(a)
        aid = ids[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if _planar_dist(axy, _xy(b)) <= float(radius_m):
                uf.union(aid, ids[j])

    groups: dict[int, list[Any]] = {}
    for nid in ids:
        root = uf.find(nid)
        groups.setdefault(root, []).append(by_id[nid])

    seed_ids = {int(getattr(s, "node_id", -1)) for s in seeds}
    kws = [str(k) for k in (keywords or []) if str(k).strip()]
    regions: list[SpatialRegion] = []
    for root, members in groups.items():
        labels: list[str] = []
        image_ids: list[int] = []
        score = 0.0
        for n in members:
            for lab in _label_list(n):
                if lab not in labels:
                    labels.append(lab)
            oid = int(getattr(n, "obs_id", -1) or -1)
            if oid >= 0 and oid not in image_ids:
                image_ids.append(oid)
            score += 10.0 * _keyword_hit(_label_list(n), kws)
            score += 0.1 * float(getattr(n, "support_count", 1) or 1)
            if int(getattr(n, "node_id", -1)) in seed_ids:
                score += 5.0
        # Anchor: highest keyword score member, else first
        anchor = max(
            members,
            key=lambda n: (
                _keyword_hit(_label_list(n), kws),
                float(getattr(n, "support_count", 1) or 1),
            ),
        )
        axy = _xy(anchor)
        regions.append(
            SpatialRegion(
                region_id=int(root),
                node_ids=tuple(int(getattr(n, "node_id", -1)) for n in members),
                labels=tuple(labels[:24]),
                image_ids=tuple(image_ids[:8]),
                anchor_xy=(axy[0], axy[1]),
                score=float(score) + 0.01 * len(members),
            )
        )
    regions.sort(key=lambda r: (-r.score, r.region_id))
    # Re-number for stable prompt display
    out: list[SpatialRegion] = []
    for i, r in enumerate(regions[: max(1, int(max_regions))], start=1):
        out.append(
            SpatialRegion(
                region_id=i,
                node_ids=r.node_ids,
                labels=r.labels,
                image_ids=r.image_ids,
                anchor_xy=r.anchor_xy,
                score=r.score,
            )
        )
    return out


def select_spatial_regions(
    nodes: Sequence[Any],
    *,
    keywords: Sequence[str] | None = None,
    prefer_obs_ids: Sequence[int] | None = None,
    radius_m: float = 2.5,
    max_regions: int = 6,
    max_nodes: int = 48,
    max_neighbors_per_seed: int = 8,
    max_frontiers: int = 4,
) -> SpatialRagResult:
    """End-to-end spatial RAG selection for an EQA SCENE_GRAPH block."""
    seeds = select_seed_nodes(nodes, keywords=keywords, prefer_obs_ids=prefer_obs_ids)
    expanded = expand_neighbors(
        nodes,
        seeds,
        radius_m=radius_m,
        max_neighbors_per_seed=max_neighbors_per_seed,
    )
    # Hard cap after expand: keep highest seed-affinity first
    if len(expanded) > max_nodes:
        prefer = {int(x) for x in (prefer_obs_ids or [])}
        kws = list(keywords or [])

        def _prio(n: Any) -> tuple[float, int]:
            kw = _keyword_hit(_label_list(n), kws)
            pref = 1.0 if int(getattr(n, "obs_id", -1) or -1) in prefer else 0.0
            return (-(10.0 * kw + 2.0 * pref), int(getattr(n, "node_id", 0)))

        expanded = sorted(expanded, key=_prio)[:max_nodes]
    regions = cluster_into_regions(
        expanded,
        seeds,
        radius_m=radius_m,
        max_regions=max_regions,
        keywords=keywords,
    )
    kept = {nid for r in regions for nid in r.node_ids}
    frontiers = [n for n in nodes if getattr(n, "is_frontier", False)]
    # Prefer keyword-scored frontiers
    frontiers_scored = sorted(
        frontiers,
        key=lambda n: (
            -_keyword_hit(_label_list(n), list(keywords or [])),
            -float(getattr(n, "frontier_cell_count", 0) or 0),
            int(getattr(n, "node_id", 0)),
        ),
    )[: max(0, int(max_frontiers))]
    return SpatialRagResult(
        regions=regions,
        kept_node_ids=kept,
        frontier_nodes=list(frontiers_scored),
        seed_node_ids=[int(getattr(s, "node_id", -1)) for s in seeds],
        n_candidates=len(expanded),
    )


def format_regions_for_prompt(
    result: SpatialRagResult,
    *,
    max_label_chars: int = 120,
) -> str:
    """Serialize regions (+ optional frontiers) for the EQA / router prompt."""
    if not result.regions and not result.frontier_nodes:
        return ""
    lines: list[str] = ["SCENE_GRAPH (spatial regions — views to look at, not the answer):"]
    for r in result.regions:
        label_str = ", ".join(r.labels) if r.labels else "object"
        if len(label_str) > max_label_chars:
            label_str = label_str[: max_label_chars - 3] + "..."
        near = f"Image {r.image_ids[0]}" if r.image_ids else "no image"
        lines.append(f"REGION {r.region_id} (near {near}): {label_str}")
        imgs = ", ".join(str(i) for i in r.image_ids) if r.image_ids else "-"
        lines.append(f"  anchor ({r.anchor_xy[0]:.2f}, {r.anchor_xy[1]:.2f}); images: {imgs}")
    for n in result.frontier_nodes:
        lbl = ", ".join(_label_list(n)) or "frontier"
        xy = _xy(n)
        oid = int(getattr(n, "obs_id", -1) or -1)
        lines.append(f"Frontier {getattr(n, 'node_id', '?')}: {lbl} at ({xy[0]:.2f}, {xy[1]:.2f}) [Image {oid}]")
    return "\n".join(lines)


def format_regions_compact(result: SpatialRagResult, *, max_chars: int = 900) -> str:
    """Shorter REGION summary for the agentic router state message."""
    text = format_regions_for_prompt(result)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
