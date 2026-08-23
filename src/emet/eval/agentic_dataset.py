# Copyright (c) Chris Paxton 2026

"""Mine causal, scene-disjoint decision records from agentic EQA bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ACTION_TOOLS = frozenset(
    {
        "inspect_graph",
        "navigate_to_obs",
        "explore_frontier",
        "look_around",
        "verify_siglip",
        "submit_answer",
        "abstain_unverified",
    }
)


@dataclass
class EvidenceRecord:
    """One causal decision point; no future trace rows are included."""

    episode_id: str
    scene: str
    split: str
    question_id: int
    question: str
    gold_answer_letter: str
    step_id: str
    round: int
    action_taken: str
    action_args: dict[str, Any] = field(default_factory=dict)
    phrase: str = ""
    obs_id: int | None = None
    rgb_path: str | None = None
    robot_xyt: list[float] | None = None
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    prior_verifies: list[dict[str, Any]] = field(default_factory=list)
    nav_budget_left: int | None = None
    verify_scores: dict[str, float | None] = field(default_factory=dict)
    decision: str | None = None
    gt: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    label_source: str = "unlabeled"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scene_split(
    scene: str,
    *,
    salt: str = "emet-agentic-v1",
    train_pct: int = 70,
    val_pct: int = 15,
) -> str:
    """Stable scene-level split; every trajectory from one scene stays together."""
    if not scene:
        return "unknown"
    bucket = int(hashlib.sha256(f"{salt}:{scene}".encode()).hexdigest()[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _episode_dirs(root: Path) -> Iterable[Path]:
    for trace in sorted(root.rglob("agentic_trace.jsonl")):
        yield trace.parent


def _resolve_rgb(episode_dir: Path, obs_id: int | None) -> str | None:
    """Resolve an observation frame without using a future frame."""
    if obs_id is None:
        return None
    candidates = (
        episode_dir / "images" / f"rgb_{obs_id:04d}.png",
        episode_dir / "frames" / f"frame_{obs_id:04d}.png",
        episode_dir / "frames" / f"rgb_{obs_id:04d}.png",
    )
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    return None


def _view_gt(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    keys = (
        "gt_body_key",
        "gt_xyz",
        "gt_dist_m",
        "gt_present",
        "gt_in_view",
        "gt_visible_pixels",
        "gt_visible_fraction",
        "gt_bbox_xyxy",
        "gt_median_depth_m",
        "gt_matching_instance_ids",
    )
    gt = {key: row.get(key) for key in keys if key in row}
    if row.get("gt_view_label_available"):
        return gt, "hm3d_semantic_sensor"
    if "gt_present" in row:
        return gt, "placement_distance_proxy"
    return gt, "unlabeled"


def mine_episode_records(
    episode_dir: Path,
    *,
    salt: str = "emet-agentic-v1",
) -> list[EvidenceRecord]:
    """Mine action records while exposing only trace-prefix state."""
    metrics = _read_json(episode_dir / "metrics.json")
    trace = _read_jsonl(episode_dir / "agentic_trace.jsonl")
    if not trace:
        return []
    scene = str(metrics.get("scene") or trace[0].get("scene") or "")
    qid = int(metrics.get("question_id", trace[0].get("question_id", -1)))
    question = str(metrics.get("question") or trace[0].get("question") or "")
    split = scene_split(scene, salt=salt)
    episode_id = episode_dir.parent.name + "/" + episode_dir.name
    prior_verifies: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    records: list[EvidenceRecord] = []

    for index, row in enumerate(trace):
        tool = str(row.get("tool") or "")
        if tool == "inspect_graph":
            hypotheses = list(row.get("hypotheses") or [])
        if tool not in _ACTION_TOOLS:
            continue
        obs_id = row.get("obs_id")
        obs_id = int(obs_id) if obs_id is not None else None
        gt, label_source = _view_gt(row)
        verify_scores = {
            "selected": row.get("sim"),
            "siglip_full": row.get("full_frame_sim"),
            "siglip_dense": row.get("dense_sim"),
            "siglip_voxel": row.get("voxel_sim"),
            "detector": row.get("detector_score"),
            "crop_siglip": row.get("crop_siglip_sim"),
            "vlm_verify": row.get("vlm_verify_score"),
        }
        record = EvidenceRecord(
            episode_id=episode_id,
            scene=scene,
            split=split,
            question_id=qid,
            question=question,
            gold_answer_letter=str(metrics.get("gold_answer_letter") or ""),
            step_id=f"r{int(row.get('round', 0))}_{index}_{tool}",
            round=int(row.get("round", 0)),
            action_taken=tool,
            action_args=dict(row.get("args") or {}),
            phrase=str(row.get("phrase") or ""),
            obs_id=obs_id,
            rgb_path=_resolve_rgb(episode_dir, obs_id),
            robot_xyt=list(row.get("xyt")) if row.get("xyt") is not None else None,
            hypotheses=list(hypotheses),
            prior_verifies=list(prior_verifies),
            nav_budget_left=row.get("nav_budget_left"),
            verify_scores=verify_scores,
            decision=row.get("decision"),
            gt=gt,
            outcome={
                "correct": metrics.get("correct") if tool in ("submit_answer", "abstain_unverified") else None,
                "verified": row.get("verified"),
                "nav_success": row.get("nav_success", row.get("ok")),
            },
            label_source=label_source,
        )
        records.append(record)
        if tool == "verify_siglip":
            prior_verifies.append(
                {
                    "obs_id": obs_id,
                    "phrase": record.phrase,
                    "decision": record.decision,
                    "scores": verify_scores,
                    "gt": gt,
                }
            )
    return records


def mine_evidence_dataset(
    root: str | Path,
    output_jsonl: str | Path,
    *,
    salt: str = "emet-agentic-v1",
    require_view_labels: bool = False,
) -> dict[str, Any]:
    """Mine all bundles under *root* and write JSONL plus a manifest."""
    root_p = Path(root).expanduser().resolve()
    output = Path(output_jsonl).expanduser().resolve()
    records: list[EvidenceRecord] = []
    for episode_dir in _episode_dirs(root_p):
        records.extend(mine_episode_records(episode_dir, salt=salt))
    if require_view_labels:
        records = [r for r in records if r.label_source == "hm3d_semantic_sensor"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record.to_dict(), default=str) + "\n" for record in records),
        encoding="utf-8",
    )
    scenes_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set(), "unknown": set()}
    labeled = 0
    for record in records:
        scenes_by_split.setdefault(record.split, set()).add(record.scene)
        labeled += int(record.label_source == "hm3d_semantic_sensor")
    scene_owners: dict[str, set[str]] = {}
    for split, scenes in scenes_by_split.items():
        for scene in scenes:
            scene_owners.setdefault(scene, set()).add(split)
    leakage = sorted(scene for scene, owners in scene_owners.items() if len(owners) > 1)
    manifest = {
        "schema_version": 1,
        "root": str(root_p),
        "output": str(output),
        "n_records": len(records),
        "n_view_labeled": labeled,
        "records_by_split": {
            split: sum(1 for record in records if record.split == split) for split in sorted(scenes_by_split)
        },
        "scenes_by_split": {
            split: sorted(scene for scene in scenes if scene) for split, scenes in sorted(scenes_by_split.items())
        },
        "scene_leakage": leakage,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
