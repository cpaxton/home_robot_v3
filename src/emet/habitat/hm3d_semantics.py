# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""HM3D semantic instance → category labels for Habitat EQA."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emet.habitat.config import (
    default_hm3d_data_path,
    hm3d_scene_glb_path,
)

_SKIP_CATEGORY_NAMES = frozenset(
    {
        "unknown",
        "void",
        "unlabeled",
        "wall",
        "ceiling",
        "floor",
        "misc",
    }
)


def _clean_category_name(raw: str) -> str:
    name = raw.split("/")[-1].replace("_", " ").strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


def _instance_index_from_object_id(object_id: object) -> int | None:
    text = str(object_id)
    if "_" not in text:
        return None
    tail = text.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


@dataclass
class Hm3dSemanticLabeler:
    """Map Habitat semantic-sensor instance ids to HM3D category names."""

    instance_to_label: dict[int, str]

    @classmethod
    def from_semantic_scene(cls, semantic_scene) -> Hm3dSemanticLabeler | None:
        if semantic_scene is None:
            return None
        mapping: dict[int, str] = {}
        for obj in semantic_scene.objects:
            idx = _instance_index_from_object_id(obj.id)
            if idx is None:
                continue
            label = _clean_category_name(obj.category.name())
            if label and label not in _SKIP_CATEGORY_NAMES:
                mapping[idx] = label
        return cls(instance_to_label=mapping) if mapping else None

    def labels_from_frame(
        self,
        semantic: np.ndarray,
        depth: np.ndarray | None = None,
        *,
        max_labels: int = 8,
        min_pixels: int = 120,
    ) -> list[str]:
        """Return distinct object category names visible in the semantic frame."""
        sem = np.asarray(semantic)
        if sem.ndim == 3:
            sem = sem[..., 0]
        valid = np.ones(sem.shape, dtype=bool)
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            valid &= (d > 0.05) & (d < 8.0) & np.isfinite(d)
        pixels = sem[valid].reshape(-1)
        if pixels.size == 0:
            return []
        counts = Counter(int(v) for v in pixels if int(v) > 0)
        labels: list[str] = []
        seen: set[str] = set()
        for inst_id, count in counts.most_common():
            if count < min_pixels:
                continue
            label = self.instance_to_label.get(inst_id)
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
            if len(labels) >= max_labels:
                break
        return labels


def hm3d_instance_items_from_obs(
    labeler: Hm3dSemanticLabeler,
    obs,
    *,
    max_instances: int = 8,
    min_pixels: int = 120,
) -> list[tuple[str, np.ndarray]]:
    """Per-instance (label, world xyz) for object-centric graph nodes."""
    if obs.semantic is None or obs.depth is None or obs.camera_pose is None:
        return []
    sem = np.asarray(obs.semantic)
    depth = np.asarray(obs.depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = (depth > 0.05) & (depth < 8.0) & np.isfinite(depth)
    try:
        obs.compute_xyz(scaling=1.0)
        pts_world = obs.get_xyz_in_world_frame(scaling=1.0)
    except Exception:
        return []
    if pts_world is None:
        return []
    flat_pts = pts_world.reshape(-1, 3)
    flat_sem = sem.reshape(-1)
    flat_valid = valid.reshape(-1)
    counts = Counter(int(v) for v, ok in zip(flat_sem, flat_valid, strict=False) if ok and int(v) > 0)
    items: list[tuple[str, np.ndarray]] = []
    seen_labels: set[str] = set()
    for inst_id, count in counts.most_common():
        if count < min_pixels:
            break
        label = labeler.instance_to_label.get(inst_id)
        if not label or label in seen_labels:
            continue
        mask = flat_valid & (flat_sem == inst_id)
        sel = flat_pts[mask]
        if sel.shape[0] < min_pixels:
            continue
        xyz = np.median(sel, axis=0).astype(np.float64)
        items.append((label, xyz))
        seen_labels.add(label)
        if len(items) >= max_instances:
            break
    return items


def hm3d_annotated_scene_dataset_config(hm3d_root: Path | None = None, *, split: str = "train") -> Path | None:
    """Return ``hm3d_annotated_basis.scene_dataset_config.json`` if present."""
    root = hm3d_root or default_hm3d_data_path()
    candidates = [
        root / "scene_datasets" / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json",
        root / "scene_datasets" / "hm3d" / split / "hm3d_annotated_basis.scene_dataset_config.json",
        root / "scene_datasets" / "hm3d" / split / f"hm3d_annotated_{split}_basis.scene_dataset_config.json",
        root / "versioned_data" / "hm3d-0.2" / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json",
        root / "versioned_data" / "hm3d-0.2" / "hm3d" / split / "hm3d_annotated_basis.scene_dataset_config.json",
        root
        / "versioned_data"
        / "hm3d-0.2"
        / "hm3d"
        / split
        / f"hm3d_annotated_{split}_basis.scene_dataset_config.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(root.rglob("hm3d_annotated*basis.scene_dataset_config.json"))
    if not matches:
        return None
    split_hits = [p for p in matches if f"/{split}/" in p.as_posix()]
    return split_hits[0] if split_hits else matches[0]


def hm3d_scene_has_semantic_assets(scene_id: str, hm3d_root: Path | None = None) -> bool:
    """True when ``<short_id>.semantic.glb`` exists next to the scene basis mesh."""
    glb = hm3d_scene_glb_path(scene_id, hm3d_root)
    return hm3d_semantic_glb_for_basis(glb).is_file()


def hm3d_placements_from_semantic_scene(semantic_scene) -> dict[str, dict] | None:
    """
    Build find-phase GT placements from HM3D ``semantic_scene.objects``.

    Positions are Habitat Y-up world coordinates. Each entry includes ``frame: habitat_yup``
    and axis-aligned ``bounds`` for bounds-aware scoring (XZ horizontal plane).
    """
    if semantic_scene is None:
        return None
    placements: dict[str, dict] = {}
    for obj in semantic_scene.objects:
        idx = _instance_index_from_object_id(obj.id)
        if idx is None:
            continue
        label = _clean_category_name(obj.category.name())
        if not label or label in _SKIP_CATEGORY_NAMES:
            continue
        aabb = obj.aabb
        center = np.array([float(aabb.center()[i]) for i in range(3)], dtype=np.float64)
        mn = np.array([float(aabb.min[i]) for i in range(3)], dtype=np.float64)
        mx = np.array([float(aabb.max[i]) for i in range(3)], dtype=np.float64)
        body = f"hm3d_{label}_{idx}"
        placements[body] = {
            "cat": label,
            "pos": center,
            "bounds": np.stack([mn, mx]),
            "frame": "habitat_yup",
        }
    return placements or None


def hm3d_semantic_glb_for_basis(basis_glb: Path) -> Path:
    short = basis_glb.name.split(".")[0]
    if short.endswith("_basis"):
        short = short[: -len("_basis")]
    return basis_glb.parent / f"{short}.semantic.glb"
