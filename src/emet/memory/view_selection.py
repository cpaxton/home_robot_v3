# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Model-free, bounded selection of camera evidence for a shared robot agent.

Inputs are retained RGB snapshots, not object instances. Camera poses use one
world frame and optical +Z forward. Timestamps must share an acquisition clock.
Optional embeddings must share an encoder/preprocessing version; visible cells
must share a map resolution/origin. ``scope_id`` identifies that shared episode
and map frame. This module does not acquire frames, infer saliency, or run models.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import combinations
from time import perf_counter
from types import MappingProxyType
from typing import Literal

import numpy as np

from emet.memory.view_coverage import CoverageObjective

Policy = Literal["recency", "pose", "embedding", "coverage", "weighted_coverage", "quality_coverage", "hybrid"]
FEATURES = ("recency", "pose", "embedding", "coverage", "relevance", "saliency")


@dataclass(frozen=True)
class ViewFrame:
    """Owned, immutable evidence snapshot; no localization or presence claim."""

    view_id: str
    scope_id: str
    timestamp_s: float
    rgb: np.ndarray = field(repr=False, compare=False)
    camera_pose: np.ndarray | None = field(default=None, repr=False, compare=False)
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    embedding_space: str | None = None
    visible_cells: frozenset[str] | None = None
    saliency: float | None = None
    cell_quality: Mapping[str, float] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if (
            not isinstance(self.view_id, str)
            or not self.view_id
            or not isinstance(self.scope_id, str)
            or not self.scope_id
            or not math.isfinite(self.timestamp_s)
        ):
            raise ValueError("view_id, scope_id and finite acquisition timestamp are required")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("rgb must be nonempty HxWx3 uint8")
        for name in ("rgb", "camera_pose", "embedding"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=None if name == "rgb" else np.float64, copy=True)
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} must be finite")
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        pose = self.camera_pose
        if pose is not None:
            if (
                pose.shape != (4, 4)
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(abs(np.linalg.det(pose[:3, :3])), 1, atol=1e-4)
            ):
                raise ValueError("camera_pose must be an orthonormal world-from-optical transform")
        if self.embedding is not None:
            if (
                self.embedding.ndim != 1
                or not self.embedding.size
                or not 0 < np.linalg.norm(self.embedding) < float("inf")
            ):
                raise ValueError("embedding must be a nonzero vector")
            if not self.embedding_space:
                raise ValueError("embedding_space must identify encoder and preprocessing")
        if self.saliency is not None and (not math.isfinite(self.saliency) or not 0 <= self.saliency <= 1):
            raise ValueError("saliency must be in [0, 1]")
        if self.saliency is not None:
            object.__setattr__(self, "saliency", float(self.saliency))
        if self.visible_cells is not None:
            if any(not isinstance(cell, str) for cell in self.visible_cells):
                raise ValueError("visible_cells must contain stable string map-cell IDs")
            object.__setattr__(self, "visible_cells", frozenset(self.visible_cells))
        if self.cell_quality is not None:
            quality = dict(self.cell_quality)
            if self.visible_cells is None or set(quality) != set(self.visible_cells):
                raise ValueError("cell_quality must describe exactly the visible cells")
            if any(not math.isfinite(q) or not 0 <= q <= 1 for q in quality.values()):
                raise ValueError("cell quality must be finite and in [0, 1]")
            object.__setattr__(self, "cell_quality", MappingProxyType(quality))


@dataclass(frozen=True)
class SelectionConfig:
    policy: Policy = "pose"
    spatial_labels: bool = False
    max_images: int = 4
    max_candidates: int = 32
    max_age_s: float = 60.0
    recency_tau_s: float = 20.0
    translation_scale_m: float = 1.0
    angle_scale_deg: float = 60.0
    # Experimental weights, not tuned defaults or claimed optimal settings.
    weights: tuple[float, ...] = (0.1, 0.3, 0.2, 0.2, 0.15, 0.05)

    def __post_init__(self):
        if self.policy not in (
            "recency",
            "pose",
            "embedding",
            "coverage",
            "weighted_coverage",
            "quality_coverage",
            "hybrid",
        ):
            raise ValueError("unknown view selection policy")
        if not isinstance(self.max_images, int) or not isinstance(self.max_candidates, int):
            raise ValueError("image and candidate budgets must be integers")
        if not 1 <= self.max_images <= self.max_candidates:
            raise ValueError("require 1 <= max_images <= max_candidates")
        for value in (self.max_age_s, self.recency_tau_s, self.translation_scale_m, self.angle_scale_deg):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("time and distance scales must be finite and positive")
        if len(self.weights) != len(FEATURES) or any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("one finite nonnegative weight is required per feature")
        if sum(self.weights) <= 0:
            raise ValueError("at least one feature weight must be positive")


@dataclass(frozen=True)
class SelectedView:
    frame: ViewFrame
    role: str
    age_s: float
    score: float | None
    features: dict[str, float | None]

    def label(self, slot: int) -> str:
        if self.role == "current":
            return f"Image {slot} [{self.frame.view_id}]: current assessment view; use for current visibility."
        role = "arrival before final turn" if self.role == "pre_turn" else "earlier context"
        return (
            f"Image {slot} [{self.frame.view_id}]: {role}, {self.age_s:.2f}s before the current view. "
            "Historical evidence, not proof of current visibility or manipulation geometry."
        )


@dataclass
class ViewSelection:
    views: list[SelectedView]
    excluded: dict[str, str]
    metrics: dict
    config: SelectionConfig

    def prompt_labels(self) -> str:
        if not self.config.spatial_labels:
            return "\n".join(view.label(i) for i, view in enumerate(self.views, 1))
        return spatial_packet_labels(
            [
                {"label": view.label(i), "timestamp_s": view.frame.timestamp_s, "camera_pose": view.frame.camera_pose}
                for i, view in enumerate(self.views, 1)
            ]
        )

    def to_dict(self) -> dict:
        """JSON-safe audit record; pixels/embeddings stay outside the manifest."""
        return {
            "config": asdict(self.config),
            "prompt_labels": self.prompt_labels(),
            "views": [
                {
                    "view_id": v.frame.view_id,
                    "scope_id": v.frame.scope_id,
                    "role": v.role,
                    "age_s": v.age_s,
                    "timestamp_s": v.frame.timestamp_s,
                    "score": v.score,
                    "features": v.features,
                    "label": v.label(i),
                }
                for i, v in enumerate(self.views, 1)
            ],
            "excluded": self.excluded,
            "metrics": self.metrics,
        }


def spatial_packet_labels(rows) -> str:
    """Describe same-world captures without confusing display order with time.

    Input timestamps must share a clock and poses a world frame (the selector
    enforces both). Sequence ranks are packet-local, not global sensor IDs;
    simultaneous captures share a rank. Missing poses remain explicitly unknown.
    """
    times = sorted({float(row["timestamp_s"]) for row in rows})
    if not all(math.isfinite(t) for t in times):
        raise ValueError("Capture timestamps must be finite")
    lines = [
        "Capture sequence is chronological within this packet (1 = earliest); "
        "image display order is different. Positions are camera XYZ in meters in "
        "one shared world frame; directions are camera optical-forward unit vectors "
        "in that frame. These are camera poses, not object locations."
    ]
    for row in rows:
        sequence = times.index(float(row["timestamp_s"])) + 1
        pose = row.get("camera_pose")
        geometry = "camera position/direction unknown"
        if pose is not None:
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError("Camera pose must be a finite 4x4 transform")
            xyz = ", ".join(f"{x:.2f}" for x in pose[:3, 3])
            forward = ", ".join(f"{x:.2f}" for x in pose[:3, 2])
            geometry = f"camera XYZ=({xyz}) m; viewing direction=({forward})"
        lines.append(f"{row['label']} Capture sequence={sequence}; {geometry}.")
    return "\n".join(lines)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))


def _pose_distance(a: ViewFrame, b: ViewFrame) -> tuple[float, float] | None:
    if a.camera_pose is None or b.camera_pose is None:
        return None
    distance = float(np.linalg.norm(a.camera_pose[:3, 3] - b.camera_pose[:3, 3]))
    # Viewing direction, not roll: rotating the image alone adds no view coverage.
    angle = math.degrees(math.acos(_cosine(a.camera_pose[:3, 2], b.camera_pose[:3, 2])))
    return distance, angle


def select_views(
    frames: list[ViewFrame],
    *,
    current_id: str,
    pre_turn_id: str | None = None,
    config: SelectionConfig = SelectionConfig(),
    query_embedding: np.ndarray | None = None,
    embedding_space: str | None = None,
    coverage_objective: CoverageObjective | None = None,
) -> ViewSelection:
    """Pin current/pre-turn evidence, then greedy diversity within a recent pool.

    Missing features score zero (never fabricated); recency and ID break ties.
    Coverage is marginal visible-cell union, not target recall or inferred free
    space. A caller must explicitly identify the true pre-turn capture.
    """
    started = perf_counter()
    by_id = {f.view_id: f for f in frames}
    if len(by_id) != len(frames):
        raise ValueError("duplicate view IDs")
    if current_id not in by_id:
        raise ValueError("current view must be retained")
    if pre_turn_id is not None and pre_turn_id not in by_id:
        raise ValueError("pre-turn view must be retained when specified")
    current = by_id[current_id]
    weighted_policy = config.policy in ("weighted_coverage", "quality_coverage")
    if weighted_policy and coverage_objective is None:
        raise ValueError("weighted_coverage requires an explicit coverage objective")
    if coverage_objective is not None and coverage_objective.scope_id != current.scope_id:
        raise ValueError("coverage objective and current view must share scope")
    excluded = {}
    eligible = []
    for frame in frames:
        age = current.timestamp_s - frame.timestamp_s
        reason = (
            "different_scope"
            if frame.scope_id != current.scope_id
            else "future_frame"
            if age < 0
            else "too_old"
            if age > config.max_age_s
            else None
        )
        if reason:
            excluded[frame.view_id] = reason
        else:
            eligible.append(frame)
    if pre_turn_id in excluded:
        raise ValueError(f"invalid pre-turn view: {excluded[pre_turn_id]}")
    # World axes may be left-handed (e.g. Habitat X,Z,Y voxel coordinates).
    # Distances and dot products remain valid, but one scope must not mix bases.
    handedness = {int(np.sign(np.linalg.det(f.camera_pose[:3, :3]))) for f in eligible if f.camera_pose is not None}
    if len(handedness) > 1:
        raise ValueError("camera poses in one scope must share world-basis handedness")
    pins = [current] + ([by_id[pre_turn_id]] if pre_turn_id and pre_turn_id != current_id else [])
    pool = pins + sorted(
        (f for f in eligible if f.view_id not in {p.view_id for p in pins}),
        key=lambda f: (-f.timestamp_s, f.view_id),
    )
    for frame in pool[config.max_candidates :]:
        excluded[frame.view_id] = "candidate_budget"
    pool = pool[: config.max_candidates]
    spaces = {(f.embedding_space, f.embedding.shape) for f in pool if f.embedding is not None}
    if len(spaces) > 1:
        raise ValueError("incompatible candidate embedding spaces or dimensions")
    if query_embedding is not None:
        query_embedding = np.asarray(query_embedding, dtype=float)
        if (
            query_embedding.ndim != 1
            or not query_embedding.size
            or not np.isfinite(query_embedding).all()
            or not 0 < np.linalg.norm(query_embedding) < float("inf")
            or not embedding_space
        ):
            raise ValueError("query embedding requires a finite nonzero vector and embedding_space")
        if spaces and (embedding_space, query_embedding.shape) not in spaces:
            raise ValueError("query embedding space does not match candidate embeddings")
    # Exact duplicate pixels cannot use another slot, regardless of pose metadata.
    unique = []
    hashes = set()
    for frame in pool:
        digest = (frame.rgb.shape, hashlib.sha256(frame.rgb.tobytes()).digest())
        if digest in hashes:
            excluded[frame.view_id] = "duplicate_pixels"
        else:
            hashes.add(digest)
            unique.append(frame)
    coverage_universe = set().union(*(f.visible_cells or () for f in unique))
    available_quality: dict[str, float] = {}
    quality_available = all(f.cell_quality is not None for f in unique)
    if quality_available:
        for f in unique:
            for cell, quality in (f.cell_quality or {}).items():
                available_quality[cell] = max(quality, available_quality.get(cell, 0))
    if weighted_policy and any(f.visible_cells is None for f in unique):
        raise ValueError("weighted_coverage requires visible_cells for every candidate (empty is allowed)")
    covered: set[str] = set()
    best_quality: dict[str, float] = {}
    if config.policy == "quality_coverage" and any(f.cell_quality is None for f in unique):
        raise ValueError("quality_coverage requires cell_quality for every candidate")
    selected: list[SelectedView] = []

    def features(frame):
        pose = [_pose_distance(frame, v.frame) for v in selected]
        pose = [p for p in pose if p is not None]
        distances = [
            (1 - _cosine(frame.embedding, v.frame.embedding)) / 2
            for v in selected
            if frame.embedding is not None and v.frame.embedding is not None
        ]
        return {
            "recency": math.exp(-(current.timestamp_s - frame.timestamp_s) / config.recency_tau_s),
            "pose": min(min(1.0, max(d / config.translation_scale_m, a / config.angle_scale_deg)) for d, a in pose)
            if pose
            else None,
            "embedding": min(distances) if distances else None,
            "coverage": len(frame.visible_cells - covered) / max(1, len(coverage_universe))
            if frame.visible_cells is not None
            else None,
            "relevance": (1 + _cosine(frame.embedding, query_embedding)) / 2
            if frame.embedding is not None and query_embedding is not None
            else None,
            "saliency": frame.saliency,
            "weighted_coverage": coverage_objective.marginal(frame.visible_cells, covered)
            if coverage_objective is not None and frame.visible_cells is not None
            else None,
            "quality_coverage": coverage_objective.quality_marginal(frame.cell_quality, best_quality)
            if coverage_objective is not None and frame.cell_quality is not None
            else None,
        }

    while unique and len(selected) < config.max_images:
        pinned = next((f for f in pins if f in unique), None)
        if pinned is not None:
            frame, score = pinned, None
            values = features(frame)
        else:

            def rank(f):
                values = features(f)
                score = (
                    sum(w * (values[k] or 0) for k, w in zip(FEATURES, config.weights, strict=True))
                    / sum(config.weights)
                    if config.policy == "hybrid"
                    else values[config.policy] or 0
                )
                return score, f.timestamp_s

            # Stable view ID tie-break independent of input enumeration order.
            frame = max(sorted(unique, key=lambda f: f.view_id), key=rank)
            score, _ = rank(frame)
            values = features(frame)
            if weighted_policy and score <= 0:
                break  # Up to N images: do not pad with zero-information frames.
        role = "current" if frame.view_id == current_id else "pre_turn" if frame.view_id == pre_turn_id else "context"
        selected.append(SelectedView(frame, role, current.timestamp_s - frame.timestamp_s, score, values))
        covered.update(frame.visible_cells or ())
        for cell, quality in (frame.cell_quality or {}).items():
            best_quality[cell] = max(quality, best_quality.get(cell, 0))
        unique.remove(frame)
    remaining_reason = (
        "no_marginal_coverage" if weighted_policy and len(selected) < config.max_images else "image_budget"
    )
    excluded.update({f.view_id: remaining_reason for f in unique})
    pairs = [p for a, b in combinations(selected, 2) if (p := _pose_distance(a.frame, b.frame)) is not None]
    embedding_pairs = [
        (1 - _cosine(a.frame.embedding, b.frame.embedding)) / 2
        for a, b in combinations(selected, 2)
        if a.frame.embedding is not None and b.frame.embedding is not None
    ]
    metrics = {
        "input_count": len(frames),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "pre_turn_retained": any(v.role == "pre_turn" for v in selected),
        "selected_pixel_count": sum(int(v.frame.rgb.shape[0] * v.frame.rgb.shape[1]) for v in selected),
        "max_age_s": max(v.age_s for v in selected),
        "pose_pair_count": len(pairs),
        "mean_translation_m": float(np.mean([d for d, _ in pairs])) if pairs else None,
        "mean_view_angle_deg": float(np.mean([a for _, a in pairs])) if pairs else None,
        "mean_embedding_distance": float(np.mean(embedding_pairs)) if embedding_pairs else None,
        "coverage_cells": len(covered),
        "candidate_coverage_fraction": len(covered) / len(coverage_universe) if coverage_universe else None,
        "feature_support": {name: sum(v.features[name] is not None for v in selected) for name in FEATURES},
        "selection_ms": (perf_counter() - started) * 1000,
    }
    if coverage_objective is not None:
        candidate_weight = coverage_objective.weight(coverage_universe)
        selected_weight = coverage_objective.weight(covered)
        candidate_quality_weight = (
            coverage_objective.quality_marginal(available_quality, {}) if quality_available else None
        )
        selected_quality_weight = coverage_objective.quality_marginal(best_quality, {}) if quality_available else None
        metrics["weighted_coverage"] = {
            "scope_id": coverage_objective.scope_id,
            "visibility_source": coverage_objective.visibility_source,
            "objective_weight": coverage_objective.weight(coverage_objective.cell_weights),
            "candidate_weight": candidate_weight,
            "selected_weight": selected_weight,
            "candidate_fraction": selected_weight / candidate_weight if candidate_weight else None,
            "selected_cells": sorted(covered & coverage_objective.cell_weights.keys()),
            "selected_quality_weight": selected_quality_weight,
            "candidate_quality_weight": candidate_quality_weight,
            "quality_candidate_fraction": selected_quality_weight / candidate_quality_weight
            if candidate_quality_weight and selected_quality_weight is not None
            else None,
        }
    return ViewSelection(selected, excluded, metrics, config)
