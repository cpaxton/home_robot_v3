# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Configuration for GraphObjectFusion (YAML + dataclass).

The config is split into *policy blocks* so merge decisions are explicit:

- ``gates`` — which evidence signals gate a merge, in decreasing evidence
  strength (identity → 3D-bounds → embedding → spatial). A candidate merges
  into a node when the first active gate that matches passes.
- ``labels`` — when a label match is required, plus extra synonym /
  incompatible label pairs beyond the built-in synonym groups.
- ``keep`` — which node survives a merge and how its state blends.
- ``admission`` — thresholds for accepting a detection into the graph at all.
- ``growth`` — duplicate/scale control (per-episode object-node cap, temporal
  window) so repeated YoloE detections cannot flood the shared scene graph.

Legacy flat keys (``spatial_merge_xy_m``, ``bounds_3d_iou_merge_min``, …) still
parse and remain readable as properties so old YAML files and existing callers
keep working.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import yaml

from emet.utils.config import resolve_config_yaml_path


@dataclass
class IdentityGate:
    """Merge only on an exact, persistent identity (tracked instances / HM3D ids)."""

    on: bool = True


@dataclass
class BoundsGate:
    """Merge on 3D bounds overlap (duplicate views of the same physical object)."""

    on: bool = True
    # Hard floor when both sides carry bounds_3d (checked alongside spatial gates).
    iou_floor: float = 0.08
    # Merge purely on bounds overlap at/above this IoU even when the spatial/
    # centroid gates fail (0 = off). High overlap = same object, disjoint = distinct.
    iou_merge_min: float = 0.3


@dataclass
class EmbeddingGate:
    """Appearance similarity gate (SigLIP crop embeddings)."""

    on: bool = True
    min_cosine: float = 0.62
    blend_alpha: float = 0.35
    # Encode each instance bbox crop with the shared SigLIP encoder so repeated
    # detections of the same object carry an appearance embedding.
    use_siglip_crops: bool = True
    # When a candidate and a nearby node look this similar, merge even if the
    # label gate fails (YoloE class strings drift frame-to-frame). Proximity is
    # still enforced by the spatial gate so identical-but-distinct instances that
    # sit apart stay separate. 0 = disabled.
    appearance_merge_min_cosine: float = 0.9


@dataclass
class SpatialGate:
    """Centroid-proximity gates (the classic XY / 3D merge thresholds)."""

    on: bool = True
    xy_m: float = 0.42
    centroid_3d_m: float = 0.55
    # Merge to the nearest object node within this XY radius when the strict
    # spatial / embedding / bounds gates all fail (0 = disabled).
    fallback_xy_m: float = 0.0


@dataclass
class FusionGates:
    """Ordered merge evidence blocks (identity > bounds > embedding > spatial)."""

    identity: IdentityGate = field(default_factory=IdentityGate)
    bounds: BoundsGate = field(default_factory=BoundsGate)
    embedding: EmbeddingGate = field(default_factory=EmbeddingGate)
    spatial: SpatialGate = field(default_factory=SpatialGate)


@dataclass
class FusionLabels:
    """Label compatibility policy for merges."""

    require_match: bool = False
    # When either side is a countable YoloE/HM3D instance, require compatible labels.
    require_match_for_instances: bool = True
    # Extra synonym groups (in addition to the built-in ones in graph_stats).
    synonyms: list[list[str]] = field(default_factory=list)
    # Label pairs that must never merge (e.g. ["person", "lamp"]).
    incompatible: list[list[str]] = field(default_factory=list)


@dataclass
class FusionKeep:
    """Merge decision: which node survives and how state blends."""

    prefer_support: bool = True
    union_labels: bool = True
    union_bounds: bool = True
    blend_embedding: bool = True
    update_xyz: bool = True


@dataclass
class FusionAdmission:
    """Acceptance thresholds before a detection may become/merge a graph node."""

    instance_min_confidence: float = 0.12
    instance_min_mask_points: int = 25
    max_candidates_per_frame: int = 64
    match_xy_m: float = 0.55


@dataclass
class FusionGrowth:
    """Duplicate / scale control (the 2026-09-02 flood regression)."""

    # Hard cap on object nodes per episode; 0 = unlimited. Stops the instance
    # graph from flooding the shared graph with singleton nodes.
    max_object_nodes: int = 0
    # Only merge into a node whose last_seen is within this many steps; 0 = off.
    temporal_window_steps: int = 0


@dataclass(init=False)
class GraphObjectFusionConfig:
    """Merge policy for instance detections → graph nodes.

    ``use_instance_nodes`` is the master switch: when false, YoloE/instance
    detections never become scene-graph entries (count/FIND recall only), which
    is the pre-instance-graph behavior of the published 49.6% baseline.
    """

    enabled: bool = False
    use_instance_nodes: bool = True
    gates: FusionGates = field(default_factory=FusionGates)
    labels: FusionLabels = field(default_factory=FusionLabels)
    keep: FusionKeep = field(default_factory=FusionKeep)
    admission: FusionAdmission = field(default_factory=FusionAdmission)
    growth: FusionGrowth = field(default_factory=FusionGrowth)

    def __init__(
        self,
        *,
        enabled: bool = False,
        use_instance_nodes: bool = True,
        gates: FusionGates | dict[str, Any] | None = None,
        labels: FusionLabels | dict[str, Any] | None = None,
        keep: FusionKeep | dict[str, Any] | None = None,
        admission: FusionAdmission | dict[str, Any] | None = None,
        growth: FusionGrowth | dict[str, Any] | None = None,
        **legacy: Any,
    ) -> None:
        mapping: dict[str, Any] = {"enabled": enabled, "use_instance_nodes": use_instance_nodes}
        for key, val in (
            ("gates", gates),
            ("labels", labels),
            ("keep", keep),
            ("admission", admission),
            ("growth", growth),
        ):
            if val is None:
                continue
            mapping[key] = asdict(val) if not isinstance(val, dict) else dict(val)
        if legacy:
            # Legacy flat-key kwargs (spatial_merge_xy_m, bounds_3d_iou_merge_min, …).
            mapping.update(_coerce_legacy_flat_keys(legacy))
        merged = decode_graph_object_fusion_config(mapping)
        self.enabled = merged.enabled
        self.use_instance_nodes = merged.use_instance_nodes
        self.gates = merged.gates
        self.labels = merged.labels
        self.keep = merged.keep
        self.admission = merged.admission
        self.growth = merged.growth

    @classmethod
    def _build(
        cls,
        *,
        enabled: bool,
        use_instance_nodes: bool,
        gates: FusionGates,
        labels: FusionLabels,
        keep: FusionKeep,
        admission: FusionAdmission,
        growth: FusionGrowth,
    ) -> GraphObjectFusionConfig:
        obj = object.__new__(cls)
        obj.enabled = enabled
        obj.use_instance_nodes = use_instance_nodes
        obj.gates = gates
        obj.labels = labels
        obj.keep = keep
        obj.admission = admission
        obj.growth = growth
        return obj

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> GraphObjectFusionConfig:
        """Decode a mapping (nested or legacy flat) into a config."""
        return decode_graph_object_fusion_config(raw)

    def __setattr__(self, name: str, value: Any) -> None:
        dotted = _LEGACY_FLAT_KEYS.get(name)
        if dotted:
            parts = dotted.split(".")
            node = self
            for part in parts[:-1]:
                node = getattr(node, part)
            object.__setattr__(node, parts[-1], value)
            return
        object.__setattr__(self, name, value)

    # --- Legacy flat-key properties (map onto the nested blocks) ----------
    @property
    def spatial_merge_xy_m(self) -> float:
        return self.gates.spatial.xy_m

    @property
    def min_centroid_dist_m(self) -> float:
        return self.gates.spatial.centroid_3d_m

    @property
    def fallback_spatial_merge_xy_m(self) -> float:
        return self.gates.spatial.fallback_xy_m

    @property
    def bounds_3d_iou_min(self) -> float:
        return self.gates.bounds.iou_floor

    @property
    def bounds_3d_iou_merge_min(self) -> float:
        return self.gates.bounds.iou_merge_min

    @property
    def embedding_min_cosine(self) -> float:
        return self.gates.embedding.min_cosine

    @property
    def embedding_blend_alpha(self) -> float:
        return self.gates.embedding.blend_alpha

    @property
    def require_label_match(self) -> bool:
        return self.labels.require_match

    @property
    def require_label_match_for_instances(self) -> bool:
        return self.labels.require_match_for_instances

    @property
    def max_candidates(self) -> int:
        return self.admission.max_candidates_per_frame

    @property
    def instance_min_confidence(self) -> float:
        return self.admission.instance_min_confidence

    @property
    def instance_min_mask_points(self) -> int:
        return self.admission.instance_min_mask_points

    @property
    def match_xy_m(self) -> float:
        return self.admission.match_xy_m


# Legacy flat key → nested dotted path (for YAML back-compat).
_LEGACY_FLAT_KEYS: dict[str, str] = {
    "spatial_merge_xy_m": "gates.spatial.xy_m",
    "min_centroid_dist_m": "gates.spatial.centroid_3d_m",
    "fallback_spatial_merge_xy_m": "gates.spatial.fallback_xy_m",
    "bounds_3d_iou_min": "gates.bounds.iou_floor",
    "bounds_3d_iou_merge_min": "gates.bounds.iou_merge_min",
    "embedding_min_cosine": "gates.embedding.min_cosine",
    "embedding_blend_alpha": "gates.embedding.blend_alpha",
    "require_label_match": "labels.require_match",
    "require_label_match_for_instances": "labels.require_match_for_instances",
    "max_candidates": "admission.max_candidates_per_frame",
    "instance_min_confidence": "admission.instance_min_confidence",
    "instance_min_mask_points": "admission.instance_min_mask_points",
    "match_xy_m": "admission.match_xy_m",
}


def _coerce_legacy_flat_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite legacy flat keys into the nested schema (nested keys win)."""
    out: dict[str, Any] = {}
    legacy_only: dict[str, Any] = {}
    for key, val in raw.items():
        if key in _LEGACY_FLAT_KEYS:
            legacy_only[key] = val
        else:
            out[key] = val
    if not legacy_only:
        return out
    for flat, dotted in _LEGACY_FLAT_KEYS.items():
        if flat not in legacy_only:
            continue
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = legacy_only[flat]
    return out


def _decode_block(block_type: type[Any], raw: Any) -> Any:
    if raw is None:
        return block_type()
    if isinstance(raw, block_type):
        return raw
    if not isinstance(raw, dict):
        return block_type()
    try:
        return draccus.decode(block_type, raw)
    except Exception:
        return block_type()


def decode_graph_object_fusion_config(raw: dict[str, Any] | None) -> GraphObjectFusionConfig:
    """Decode a config mapping (nested or legacy-flat) into ``GraphObjectFusionConfig``.

    Assembles via ``_build`` so decoding never re-enters ``__init__``.
    """
    if not isinstance(raw, dict) or not raw:
        return GraphObjectFusionConfig()
    coerced = _coerce_legacy_flat_keys(raw)
    return GraphObjectFusionConfig._build(
        enabled=bool(coerced.get("enabled", False)),
        use_instance_nodes=bool(coerced.get("use_instance_nodes", True)),
        gates=_decode_block(FusionGates, coerced.get("gates")),
        labels=_decode_block(FusionLabels, coerced.get("labels")),
        keep=_decode_block(FusionKeep, coerced.get("keep")),
        admission=_decode_block(FusionAdmission, coerced.get("admission")),
        growth=_decode_block(FusionGrowth, coerced.get("growth")),
    )


def load_graph_object_fusion_config(path: str | None = None) -> GraphObjectFusionConfig:
    """Load standalone YAML or return defaults.

    ``EMET_GRAPH_FUSION_CONFIG`` overrides the default file so eval sweeps can run a
    range of thresholds (e.g. ``gates.bounds.iou_merge_min``) without editing the
    packaged default.
    """
    if not path:
        env = os.environ.get("EMET_GRAPH_FUSION_CONFIG", "").strip()
        if env:
            path = env
    if not path:
        default = Path(__file__).resolve().parents[3] / "config" / "agents" / "default_graph_object_fusion.yaml"
        if default.is_file():
            path = str(default)
        else:
            return GraphObjectFusionConfig()
    full = Path(resolve_config_yaml_path(path))
    with full.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    subset = raw.get("graph_object_fusion", raw)
    if not isinstance(subset, dict):
        return GraphObjectFusionConfig()
    return decode_graph_object_fusion_config(subset)
