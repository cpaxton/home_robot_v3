# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Graph node counts and health metrics for eval sweeps and stream diagnostics."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_LABEL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "with",
        "on",
        "in",
        "to",
        "for",
        "object",
        "item",
        "thing",
    }
)

# Common open-vocab drift pairs (kitchen / indoor). Symmetric via frozenset keys.
_LABEL_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"mug", "cup", "coffee cup", "coffee mug", "teacup"}),
    frozenset({"trash", "trash can", "garbage", "garbage can", "bin", "recycle", "recycling bin"}),
    frozenset({"fridge", "refrigerator", "freezer"}),
    frozenset({"sofa", "couch"}),
    frozenset({"tv", "television", "monitor"}),
    frozenset({"lamp", "table lamp", "floor lamp"}),
    frozenset({"chair", "armchair", "dining chair"}),
    frozenset({"table", "dining table", "coffee table", "side table"}),
    frozenset({"pillow", "cushion"}),
    frozenset({"bottle", "water bottle"}),
    frozenset({"plant", "potted plant", "houseplant"}),
)


def graph_node_breakdown(graph_memory: Any | None) -> dict[str, int]:
    """Return object / viewpoint / frontier / total node counts."""
    if graph_memory is None or not hasattr(graph_memory, "get_nodes"):
        return {"total": 0, "object": 0, "viewpoint": 0, "frontier": 0}
    nodes = graph_memory.get_nodes()
    frontier = sum(1 for n in nodes if getattr(n, "is_frontier", False))
    viewpoint = sum(1 for n in nodes if getattr(n, "is_viewpoint", False))
    obj = len(nodes) - frontier - viewpoint
    return {
        "total": len(nodes),
        "object": max(0, obj),
        "viewpoint": viewpoint,
        "frontier": frontier,
    }


def format_graph_node_breakdown(graph_memory: Any | None) -> str:
    """Compact status line: ``graph 11 obj / 12 vp / 1 fr (24 total)``."""
    b = graph_node_breakdown(graph_memory)
    return (
        f"graph {b['object']} obj / {b['viewpoint']} vp / {b['frontier']} fr "
        f"({b['total']} total)"
    )


def format_graph_size_report(graph_memory: Any | None, *, verbose: bool = True) -> str:
    """Growth diagnostic for CHAT / lifelong load (object count is the main signal).

    Example: ``graph size: 42 obj / 8 vp / 2 fr (52 total), 31 obs, singleton=61%, …``.
    """
    h = graph_health_metrics(graph_memory)
    line = (
        f"graph size: {h['n_object']} obj / {h['n_viewpoint']} vp / {h['n_frontier']} fr "
        f"({h['n_total']} total), {h['n_obs']} obs"
    )
    if int(h["n_object"]) > 0:
        line += (
            f", singleton={float(h['singleton_frac']):.0%}, "
            f"mean_support={float(h['mean_support']):.1f}"
        )
    if verbose and h.get("top_labels"):
        tops = ", ".join(f"{t['label']}×{t['count']}" for t in h["top_labels"][:5])
        line += f"; top: {tops}"
    return line


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


def label_tokens(label: str) -> set[str]:
    """Content tokens for open-vocab label comparison."""
    norm = _normalize_label(label)
    if not norm:
        return set()
    parts = re.split(r"[^a-z0-9]+", norm)
    return {p for p in parts if p and p not in _LABEL_STOPWORDS and len(p) > 1}


def _synonym_group_id(label: str) -> int | None:
    norm = _normalize_label(label)
    if not norm:
        return None
    for i, group in enumerate(_LABEL_SYNONYM_GROUPS):
        for g in group:
            if norm == g or g in norm or norm in g:
                return i
    return None


def labels_compatible_for_dedup(a: str, b: str) -> bool:
    """True when two detector/VLM labels should count as the same instance for XY dedup.

    Exact match, substring, shared content tokens, or known synonym groups all match.
    Distinct furniture nouns with no shared tokens do not (``mug`` vs ``chair``).
    """
    la = _normalize_label(a)
    lb = _normalize_label(b)
    if not la or not lb:
        return False
    if la == lb:
        return True
    if la in lb or lb in la:
        return True
    ta, tb = label_tokens(la), label_tokens(lb)
    if ta and tb and (ta & tb):
        return True
    ga, gb = _synonym_group_id(la), _synonym_group_id(lb)
    return ga is not None and ga == gb


def _object_nodes(graph_memory: Any) -> list[Any]:
    return [
        n
        for n in graph_memory.get_nodes()
        if not getattr(n, "is_viewpoint", False) and not getattr(n, "is_frontier", False)
    ]


def _primary_label(node: Any) -> str:
    labels = getattr(node, "labels", None) or []
    if not labels:
        return ""
    return _normalize_label(str(labels[0]))


def _label_entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = float(len(labels))
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * math.log(p, 2)
    return float(ent)


def graph_health_metrics(
    graph_memory: Any | None,
    *,
    prompt_node_count: int | None = None,
    prompt_obs_count: int | None = None,
) -> dict[str, Any]:
    """Shared graph-quality snapshot for Habitat + dynamic-explore exports.

    Object-node count (not total) is the primary health signal. Singletons and
    label entropy help distinguish fragmentation from blowup.
    """
    breakdown = graph_node_breakdown(graph_memory)
    out: dict[str, Any] = {
        **{f"n_{k}": int(v) for k, v in breakdown.items()},
        "n_obs": 0,
        "mean_support": 0.0,
        "singleton_frac": 0.0,
        "n_singletons": 0,
        "label_entropy": 0.0,
        "top_labels": [],
        "prompt_node_count": prompt_node_count,
        "prompt_obs_count": prompt_obs_count,
    }
    if graph_memory is None or not hasattr(graph_memory, "get_nodes"):
        return out

    obs = getattr(graph_memory, "_observations", None) or getattr(graph_memory, "observations", None)
    if obs is not None:
        try:
            out["n_obs"] = int(len(obs))
        except TypeError:
            out["n_obs"] = 0
    if prompt_node_count is None:
        prompt_node_count = getattr(graph_memory, "last_eqa_prompt_node_count", None)
    if prompt_obs_count is None:
        prompt_obs_count = len(getattr(graph_memory, "last_eqa_obs_ids", None) or [])
    out["prompt_node_count"] = prompt_node_count
    out["prompt_obs_count"] = int(prompt_obs_count) if prompt_obs_count is not None else None

    objects = _object_nodes(graph_memory)
    if not objects:
        return out
    supports = [max(1, int(getattr(n, "support_count", 1) or 1)) for n in objects]
    singletons = sum(1 for s in supports if s <= 1)
    labels = [_primary_label(n) for n in objects if _primary_label(n)]
    top = Counter(labels).most_common(8)
    out["mean_support"] = float(sum(supports) / len(supports))
    out["n_singletons"] = int(singletons)
    out["singleton_frac"] = float(singletons / len(supports))
    out["label_entropy"] = _label_entropy(labels)
    out["top_labels"] = [{"label": lb, "count": int(c)} for lb, c in top]
    return out


def graph_health_from_checkpoint_nodes(
    nodes: list[dict[str, Any]],
    *,
    n_obs: int | None = None,
) -> dict[str, Any]:
    """Health metrics from exported ``graph.json`` node dicts (dynamic-explore cycles)."""
    frontier = sum(1 for n in nodes if n.get("is_frontier"))
    viewpoint = sum(1 for n in nodes if n.get("is_viewpoint"))
    objects = [
        n
        for n in nodes
        if not n.get("is_frontier") and not n.get("is_viewpoint")
    ]
    supports = [max(1, int(n.get("support_count", 1) or 1)) for n in objects]
    labels: list[str] = []
    for n in objects:
        labs = n.get("labels") or []
        if labs:
            labels.append(_normalize_label(str(labs[0])))
    singletons = sum(1 for s in supports if s <= 1)
    top = Counter(labels).most_common(8)
    return {
        "n_total": len(nodes),
        "n_object": len(objects),
        "n_viewpoint": viewpoint,
        "n_frontier": frontier,
        "n_obs": int(n_obs) if n_obs is not None else None,
        "mean_support": float(sum(supports) / len(supports)) if supports else 0.0,
        "n_singletons": int(singletons),
        "singleton_frac": float(singletons / len(supports)) if supports else 0.0,
        "label_entropy": _label_entropy(labels),
        "top_labels": [{"label": lb, "count": int(c)} for lb, c in top],
    }


def classify_graph_failure(
    health: dict[str, Any],
    *,
    blowup_obj: int = 200,
    thin_obj: int = 1,
    fragment_singleton_frac: float = 0.7,
) -> str:
    """Coarse failure class for diagnosis logs (not a graded metric)."""
    n_obj = int(health.get("n_object") or health.get("object") or 0)
    n_obs = health.get("n_obs")
    singleton_frac = float(health.get("singleton_frac") or 0.0)
    prompt_nodes = health.get("prompt_node_count")
    if n_obj <= 0 and (n_obs is None or int(n_obs or 0) <= 0):
        return "empty_graph"
    if n_obj < thin_obj:
        return "thin_graph"
    if n_obj >= blowup_obj or (prompt_nodes is not None and int(prompt_nodes) >= blowup_obj):
        return "blowup"
    if n_obj >= 8 and singleton_frac >= fragment_singleton_frac:
        return "fragmentation"
    return "ok"
