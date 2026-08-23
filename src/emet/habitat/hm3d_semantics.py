# Copyright (c) Chris Paxton 2026
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
    default_hm3d_scene_dir,
    hm3d_scene_glb_path,
    questions_csv_path,
)
from emet.habitat.hmeqa_enrich_labels import HMEQA_PAPER_QUESTION_COUNT

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

    def visibility_for_phrase(
        self,
        semantic: np.ndarray,
        phrase: str,
        depth: np.ndarray | None = None,
        *,
        min_pixels: int = 1,
    ) -> dict[str, object]:
        """Return view-level semantic visibility for an open-vocabulary phrase.

        This is stronger supervision than proximity-to-GT: it labels whether a
        matching HM3D semantic instance is actually rendered in this camera view,
        together with its visible pixel fraction, bounding box, and median range.
        """
        from emet.memory.graph_eqa.graph_memory import label_matches_relevant_object

        sem = np.asarray(semantic)
        if sem.ndim == 3:
            sem = sem[..., 0]
        valid = np.ones(sem.shape, dtype=bool)
        depth_m = None
        if depth is not None:
            depth_m = np.asarray(depth, dtype=np.float32)
            if depth_m.ndim == 3:
                depth_m = depth_m[..., 0]
            valid &= (depth_m > 0.05) & (depth_m < 8.0) & np.isfinite(depth_m)

        matching_ids = [
            int(instance_id)
            for instance_id, label in self.instance_to_label.items()
            if label_matches_relevant_object(phrase, label)
        ]
        mask = valid & np.isin(sem, matching_ids)
        pixel_count = int(mask.sum())
        image_pixels = int(valid.sum())
        bbox = None
        median_depth_m = None
        if pixel_count >= int(min_pixels):
            ys, xs = np.nonzero(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            if depth_m is not None:
                median_depth_m = float(np.median(depth_m[mask]))
        return {
            "gt_view_label_available": True,
            "gt_in_view": bool(pixel_count >= int(min_pixels)),
            "gt_visible_pixels": pixel_count,
            "gt_visible_fraction": (pixel_count / image_pixels) if image_pixels else 0.0,
            "gt_bbox_xyxy": bbox,
            "gt_median_depth_m": median_depth_m,
            "gt_matching_instance_ids": matching_ids,
        }


@dataclass(frozen=True)
class Hm3dInstanceItem:
    """One HM3D semantic instance with a scene-stable identity."""

    label: str
    xyz: np.ndarray
    identity_key: str


def hm3d_instance_items_from_obs(
    labeler: Hm3dSemanticLabeler,
    obs,
    *,
    max_instances: int = 8,
    min_pixels: int = 120,
    with_instance_ids: bool = False,
) -> list[tuple[str, np.ndarray] | Hm3dInstanceItem]:
    """Per-instance graph rows, optionally carrying scene-stable semantic IDs.

    The legacy two-tuple return remains the default. The identity-carrying mode is
    used by GraphEQA so two same-category instances in one frame remain distinct.
    """
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
    items: list[tuple[str, np.ndarray] | Hm3dInstanceItem] = []
    seen_labels: set[str] = set()
    for inst_id, count in counts.most_common():
        if count < min_pixels:
            break
        label = labeler.instance_to_label.get(inst_id)
        if not label or (not with_instance_ids and label in seen_labels):
            continue
        mask = flat_valid & (flat_sem == inst_id)
        sel = flat_pts[mask]
        if sel.shape[0] < min_pixels:
            continue
        xyz = np.median(sel, axis=0).astype(np.float64)
        if with_instance_ids:
            items.append(Hm3dInstanceItem(label, xyz, f"hm3d:{int(inst_id)}"))
        else:
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


@dataclass(frozen=True)
class HMEQASemanticsCoverage:
    """HM-EQA question/scene overlap with on-disk HM3D-Semantics assets."""

    hm3d_train_root: Path
    hm3d_data_root: Path
    questions_csv: Path
    paper_question_count: int
    questions_with_semantics: tuple[int, ...]
    questions_without_semantics: tuple[int, ...]
    questions_missing_glb: tuple[int, ...]
    scenes_with_semantics: tuple[str, ...]
    scenes_without_semantics: tuple[str, ...]
    train_scene_count: int
    train_scenes_with_semantics: int
    semantic_glb_count: int
    annotated_config_present: bool

    @property
    def unique_paper_scenes(self) -> int:
        return len(self.scenes_with_semantics) + len(self.scenes_without_semantics)


def compute_hmeqa_semantics_coverage(
    *,
    hm3d_root: Path | None = None,
    hm3d_data_root: Path | None = None,
    questions_path: Path | None = None,
    paper_question_count: int = HMEQA_PAPER_QUESTION_COUNT,
) -> HMEQASemanticsCoverage:
    """Summarize which HM-EQA paper questions have ``*.semantic.glb`` on disk."""
    from emet.habitat.datasets import load_hmeqa_questions

    train_root = hm3d_root or default_hm3d_scene_dir()
    data_root = hm3d_data_root or default_hm3d_data_path()
    csv_path = questions_path or questions_csv_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"HM-EQA questions CSV not found: {csv_path}")

    questions = load_hmeqa_questions(csv_path)[:paper_question_count]
    with_sem: list[int] = []
    without_sem: list[int] = []
    missing_glb: list[int] = []
    scenes_with: set[str] = set()
    scenes_without: set[str] = set()

    for q in questions:
        glb = hm3d_scene_glb_path(q.scene, train_root)
        if not glb.is_file():
            missing_glb.append(q.index)
            continue
        if hm3d_semantic_glb_for_basis(glb).is_file():
            with_sem.append(q.index)
            scenes_with.add(q.scene)
        else:
            without_sem.append(q.index)
            scenes_without.add(q.scene)

    train_scene_dirs = [p for p in train_root.iterdir() if p.is_dir()] if train_root.is_dir() else []
    train_with_sem = sum(1 for p in train_scene_dirs if any(p.glob("*.semantic.glb")))
    sem_glbs = list(data_root.rglob("*.semantic.glb")) if data_root.is_dir() else []
    cfg = hm3d_annotated_scene_dataset_config(data_root, split="train")

    return HMEQASemanticsCoverage(
        hm3d_train_root=train_root,
        hm3d_data_root=data_root,
        questions_csv=csv_path,
        paper_question_count=paper_question_count,
        questions_with_semantics=tuple(with_sem),
        questions_without_semantics=tuple(without_sem),
        questions_missing_glb=tuple(missing_glb),
        scenes_with_semantics=tuple(sorted(scenes_with)),
        scenes_without_semantics=tuple(sorted(scenes_without)),
        train_scene_count=len(train_scene_dirs),
        train_scenes_with_semantics=train_with_sem,
        semantic_glb_count=len(sem_glbs),
        annotated_config_present=bool(cfg and cfg.is_file()),
    )


def hmeqa_annotated_question_ids(
    *,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    paper_question_count: int = HMEQA_PAPER_QUESTION_COUNT,
) -> list[int]:
    """Paper HM-EQA indices whose scene has ``*.semantic.glb`` (GraphEQA sim parity subset)."""
    cov = compute_hmeqa_semantics_coverage(
        hm3d_root=hm3d_root,
        questions_path=questions_path,
        paper_question_count=paper_question_count,
    )
    return list(cov.questions_with_semantics)


def format_hmeqa_semantics_coverage_report(cov: HMEQASemanticsCoverage) -> str:
    """Human-readable coverage summary for CLI / docs."""
    q_annot = len(cov.questions_with_semantics)
    q_unannot = len(cov.questions_without_semantics)
    q_missing = len(cov.questions_missing_glb)
    s_annot = len(cov.scenes_with_semantics)
    s_unannot = len(cov.scenes_without_semantics)
    train_pct = 100.0 * cov.train_scenes_with_semantics / cov.train_scene_count if cov.train_scene_count else 0.0
    lines = [
        f"HM-EQA paper questions (0–{cov.paper_question_count - 1}): {cov.paper_question_count}",
        f"  with HM3D GT semantics (.semantic.glb): {q_annot}",
        f"  mesh only (no semantic.glb — HM3DSem never annotated this scene): {q_unannot}",
        f"  missing basis.glb: {q_missing}",
        f"Unique HM-EQA scenes: {s_annot + s_unannot} ({s_annot} annotated, {s_unannot} not in HM3DSem)",
        f"HM3D train split: {cov.train_scenes_with_semantics}/{cov.train_scene_count} scenes "
        f"with semantic.glb ({train_pct:.1f}%; HM3DSem annotates 145 train scenes total)",
        f"Total *.semantic.glb under {cov.hm3d_data_root}: {cov.semantic_glb_count}",
        f"Annotated scene_dataset config: {'present' if cov.annotated_config_present else 'MISSING'}",
    ]
    if q_unannot and cov.train_scenes_with_semantics >= 140:
        lines.append(
            "Note: remaining gaps are dataset coverage, not a failed download. "
            "Explore-EQA / HM-EQA uses 49 train scenes; only 14 overlap HM3DSem."
        )
    elif cov.train_scenes_with_semantics < 140:
        lines.append(
            "Action: fetch train semantics — "
            "uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d-semantics train"
        )
    return "\n".join(lines)
