# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Crash-safe HM-EQA unit completion, aggregation, and run finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emet.eval.hmeqa_launch import (
    HmeqaRunManifestError,
    hmeqa_run_config_digest,
    load_hmeqa_run_manifest,
    normalize_hmeqa_artifact_profile,
)
from emet.habitat.metrics import episode_run_completed

HMEQA_UNIT_COMPLETE_SCHEMA = "emet.hmeqa.unit_complete"
HMEQA_UNIT_COMPLETE_VERSION = 1
HMEQA_DONE_SCHEMA = "emet.hmeqa.done"
HMEQA_DONE_VERSION = 1

UNIT_ROW_FILENAME = "episode.json"
UNIT_COMPLETE_FILENAME = "COMPLETE.json"
RUN_DONE_FILENAME = "DONE"

_ARMS = frozenset({"classic", "agentic"})
_COPY_FILES = (
    "topdown_map.png",
    "topdown_map_overlay.png",
    "topdown_gt_navmesh.png",
    "explored_2d.npy",
    "obstacles_2d.npy",
    "grid_meta.json",
    "trajectory.jsonl",
    "observations_history.jsonl",
    "voxel_debug.pkl",
    "spawn_record.json",
    "metrics.json",
    "diagnostics_manifest.json",
    "floor_metrics.json",
    "floor_area.jsonl",
    "floor_area_growth.png",
    "topdown_exploration.mp4",
    "episode_rgb.mp4",
    "agentic_trace.jsonl",
    "agentic_summary.json",
    "world_evidence.json",
    "attempt_ledger.json",
    "room_events.json",
    "eqa_history.json",
    "raw_eqa.txt",
    "scene_graph_report.txt",
    "frontier_nodes.json",
)
_COPY_DIRS = (
    "maps",
    "dynagraph",
    "frontier_picks",
    "world_evidence_views",
    "compact_memory",
    "graph_checkpoint",
)


class HmeqaCompletionError(ValueError):
    """Raised when an episode or completion marker is not trustworthy."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HmeqaCompletionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(text: str, *, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, HmeqaCompletionError) as exc:
        raise HmeqaCompletionError(f"{label} is not one valid JSON value: {exc}") from exc


def _load_json(path: Path, *, expected: type, label: str | None = None) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HmeqaCompletionError(f"cannot read {label or path}: {exc}") from exc
    value = _loads_strict(text, label=label or str(path))
    if not isinstance(value, expected):
        raise HmeqaCompletionError(f"{label or path} must contain a JSON {expected.__name__}")
    return value


def read_pending_row(path: Path) -> dict[str, Any]:
    """Read exactly one JSON object from a pending per-unit result."""
    row = _load_json(Path(path), expected=dict, label=f"pending row {path}")
    return dict(row)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise HmeqaCompletionError(f"cannot hash {path}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(payload)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with tmp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _atomic_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )


def _manifest(out_dir: Path) -> dict[str, Any]:
    try:
        manifest = load_hmeqa_run_manifest(Path(out_dir), require_resumable=True)
    except HmeqaRunManifestError as exc:
        raise HmeqaCompletionError(str(exc)) from exc
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise HmeqaCompletionError("run manifest config is missing")
    actual = hmeqa_run_config_digest(config)
    if manifest.get("config_digest") != actual:
        raise HmeqaCompletionError("run manifest config_digest is corrupt")
    return manifest


def planned_units(manifest: Mapping[str, Any]) -> list[tuple[str, int]]:
    try:
        arms = list(manifest["config"]["evaluation"]["arms"])
        qids = list(manifest["config"]["ids"]["question_ids"])
    except (KeyError, TypeError) as exc:
        raise HmeqaCompletionError(f"run manifest has no valid arm/QID plan: {exc}") from exc
    units: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for arm in arms:
        arm_s = str(arm)
        if arm_s not in _ARMS:
            raise HmeqaCompletionError(f"invalid planned arm {arm_s!r}")
        for qid in qids:
            if isinstance(qid, bool) or not isinstance(qid, int) or qid < 0:
                raise HmeqaCompletionError(f"invalid planned question id {qid!r}")
            unit = (arm_s, qid)
            if unit in seen:
                raise HmeqaCompletionError(f"duplicate planned unit {arm_s} q{qid}")
            seen.add(unit)
            units.append(unit)
    if not units:
        raise HmeqaCompletionError("run manifest plans no units")
    return units


def _unit_name(arm: str, qid: int) -> str:
    if arm not in _ARMS:
        raise HmeqaCompletionError(f"arm must be classic or agentic; got {arm!r}")
    if isinstance(qid, bool) or not isinstance(qid, int) or qid < 0:
        raise HmeqaCompletionError(f"question id must be a non-negative integer; got {qid!r}")
    return f"{arm}_q{qid}"


def unit_bundle_dir(out_dir: Path, arm: str, qid: int) -> Path:
    return Path(out_dir).expanduser().resolve() / "bundles" / _unit_name(arm, qid)


def hmeqa_debug_run_tag(
    out_dir: Path,
    arm: str,
    qid: int,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    frozen = dict(manifest or _manifest(Path(out_dir)))
    out = Path(out_dir).expanduser().resolve()
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in out.name)[:64] or "run"
    identity = _sha256_bytes(f"{out}\0{frozen['config_digest']}\0{arm}\0{qid}".encode()).split(":", 1)[1][:12]
    return f"hmeqa_h2h_{arm}_q{qid:04d}_{slug}_{identity}"


def expected_debug_bundle_dir(
    out_dir: Path,
    arm: str,
    qid: int,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    frozen = dict(manifest or _manifest(Path(out_dir)))
    try:
        data_dir = Path(str(frozen["config"]["inputs"]["data_dir"])).expanduser().resolve(strict=False)
    except (KeyError, TypeError) as exc:
        raise HmeqaCompletionError(f"manifest data_dir is invalid: {exc}") from exc
    episodes_root = data_dir.parent / "episodes"
    tag = hmeqa_debug_run_tag(out_dir, arm, qid, manifest=frozen)
    return Path(os.path.abspath(episodes_root / tag / f"q{qid:04d}_dynagraph"))


def _assert_no_symlink_path(path: Path, *, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise HmeqaCompletionError(f"{path} escapes expected root {root}") from exc
    current = root
    if current.is_symlink():
        raise HmeqaCompletionError(f"artifact root is a symlink: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise HmeqaCompletionError(f"artifact path contains a symlink: {current}")


def _validate_source_path(raw: Any, expected: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise HmeqaCompletionError("episode row debug_bundle_dir is missing")
    supplied = Path(os.path.abspath(os.path.expanduser(raw)))
    if supplied != expected:
        raise HmeqaCompletionError(f"debug_bundle_dir {supplied} does not equal expected run bundle {expected}")
    if not expected.is_dir():
        raise HmeqaCompletionError(f"expected debug bundle does not exist: {expected}")
    root = expected.parent.parent
    _assert_no_symlink_path(expected, root=root)
    try:
        resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise HmeqaCompletionError(f"cannot resolve debug bundle {expected}: {exc}") from exc
    if resolved != expected:
        raise HmeqaCompletionError(f"debug bundle resolves outside its canonical path: {expected} -> {resolved}")
    return expected


def validate_episode_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    qid: int,
    method: str = "dynagraph",
    expected_debug_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate identity and completion without treating a wrong answer as failure."""
    value = dict(row)
    _unit_name(arm, qid)
    for key in ("h2h_arm", "arm"):
        if key in value and str(value[key]) != arm:
            raise HmeqaCompletionError(f"row {key}={value[key]!r} does not match arm {arm!r}")
    if value.get("dataset") != "hmeqa":
        raise HmeqaCompletionError(f"row dataset must be 'hmeqa'; got {value.get('dataset')!r}")
    if isinstance(value.get("question_id"), bool) or value.get("question_id") != qid:
        raise HmeqaCompletionError(f"row question_id {value.get('question_id')!r} does not match qid {qid}")
    if value.get("method") != method:
        raise HmeqaCompletionError(f"row method {value.get('method')!r} does not match {method!r}")
    error = value.get("error", "")
    if error not in ("", None):
        raise HmeqaCompletionError(f"row contains runtime error: {error}")
    if not isinstance(value.get("correct"), bool):
        raise HmeqaCompletionError("row correct must be boolean (False is a valid completed answer)")
    if not isinstance(value.get("success"), bool):
        raise HmeqaCompletionError("row success must be boolean")
    if not episode_run_completed(value):
        raise HmeqaCompletionError("row does not represent a completed EQA episode")
    if expected_debug_dir is not None:
        _validate_source_path(value.get("debug_bundle_dir"), expected_debug_dir)
    return value


def _safe_relative_path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise HmeqaCompletionError(f"{label} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise HmeqaCompletionError(f"{label} is unsafe: {raw!r}")
    return path


def _validate_jsonl(path: Path, *, label: str, require_row: bool) -> list[Any]:
    if not path.is_file():
        raise HmeqaCompletionError(f"required {label} is missing: {path}")
    rows: list[Any] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        rows.append(_loads_strict(line, label=f"{label} line {index}"))
    if require_row and not rows:
        raise HmeqaCompletionError(f"{label} contains no rows")
    return rows


def _require_file(root: Path, relative: str, *, nonempty: bool = True) -> Path:
    path = root / relative
    _assert_no_symlink_path(path, root=root)
    if not path.is_file():
        raise HmeqaCompletionError(f"required artifact is missing: {relative}")
    if nonempty and path.stat().st_size <= 0:
        raise HmeqaCompletionError(f"required artifact is empty: {relative}")
    return path


def _validate_png(path: Path, *, relative: str) -> None:
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise HmeqaCompletionError(f"{relative} is not a PNG")


def _validate_mp4(path: Path, *, relative: str) -> None:
    header = path.read_bytes()[:32]
    if b"ftyp" not in header:
        raise HmeqaCompletionError(f"{relative} is not an MP4")


def _validate_compact_memory(root: Path) -> None:
    compact = root / "compact_memory"
    manifest = _load_json(
        _require_file(root, "compact_memory/manifest.json"),
        expected=dict,
    )
    if manifest.get("checkpoint_profile") != "graph_only":
        raise HmeqaCompletionError("compact_memory must use checkpoint_profile='graph_only'")
    if manifest.get("has_frames") not in (False, 0) or manifest.get("has_point_cloud") not in (False, 0):
        raise HmeqaCompletionError("compact_memory graph_only profile contains dense frames/point cloud")
    graph = _load_json(_require_file(compact, "graph.json"), expected=dict)
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise HmeqaCompletionError("compact_memory/graph.json must contain nodes and edges lists")


def _validate_world_evidence(root: Path, *, require_rgb_dir: bool) -> None:
    world = _load_json(_require_file(root, "world_evidence.json"), expected=dict)
    if not isinstance(world.get("schema_version"), int):
        raise HmeqaCompletionError("world_evidence.json schema_version must be an integer")
    for field in ("entities", "places", "views", "events", "rooms", "frontiers"):
        if not isinstance(world.get(field), list):
            raise HmeqaCompletionError(f"world_evidence.json {field} must be a list")
    if require_rgb_dir:
        views_dir = root / "world_evidence_views"
        _assert_no_symlink_path(views_dir, root=root)
        if not views_dir.is_dir():
            raise HmeqaCompletionError("required artifact is missing: world_evidence_views/")
    for index, view in enumerate(world["views"]):
        if not isinstance(view, Mapping):
            raise HmeqaCompletionError(f"world_evidence view {index} must be an object")
        rgb_file = view.get("rgb_file")
        if rgb_file:
            relative = _safe_relative_path(rgb_file, label=f"world_evidence view {index} rgb_file")
            _require_file(root, relative.as_posix())


def validate_snapshot_bundle(
    root: Path,
    *,
    row: Mapping[str, Any],
    arm: str,
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    require_full_frames: bool,
) -> None:
    """Validate schemas and required content in a source or staged snapshot."""
    bundle = Path(root)
    if not bundle.is_dir():
        raise HmeqaCompletionError(f"snapshot bundle is missing: {bundle}")
    artifacts = normalize_hmeqa_artifact_profile(profile)
    metrics = _load_json(_require_file(bundle, "metrics.json"), expected=dict)
    if metrics != dict(row):
        raise HmeqaCompletionError("metrics.json does not exactly match the validated episode row")
    history = _load_json(_require_file(bundle, "eqa_history.json"), expected=dict)
    if not isinstance(history.get("iterations"), list):
        raise HmeqaCompletionError("eqa_history.json iterations must be a list")
    _require_file(bundle, "raw_eqa.txt", nonempty=False)
    _require_file(bundle, "scene_graph_report.txt")
    frontiers = _load_json(_require_file(bundle, "frontier_nodes.json"), expected=list)
    if any(not isinstance(item, Mapping) for item in frontiers):
        raise HmeqaCompletionError("frontier_nodes.json entries must be objects")

    diagnostic_outputs = (
        "export_map",
        "export_obstacle_grids",
        "export_trajectory",
        "export_rgb_frames",
        "export_video",
        "export_object_crops",
        "export_full_graph",
        "export_compact_memory",
        "export_voxel_history",
        "export_voxel_pickle",
        "export_gt_navmesh_map",
        "export_map_overlay",
        "export_map_video",
    )
    diagnostics: Mapping[str, Any] = {}
    if any(bool(artifacts[name]) for name in diagnostic_outputs):
        diagnostics = _load_json(
            _require_file(bundle, "diagnostics_manifest.json"),
            expected=dict,
        )
        if not diagnostics:
            raise HmeqaCompletionError("diagnostics_manifest.json must not be empty")

    if artifacts["export_map"]:
        path = _require_file(bundle, "topdown_map.png")
        _validate_png(path, relative="topdown_map.png")
    if artifacts["export_obstacle_grids"]:
        _require_file(bundle, "obstacles_2d.npy")
        _require_file(bundle, "explored_2d.npy")
        _load_json(_require_file(bundle, "grid_meta.json"), expected=dict)
    if artifacts["export_trajectory"]:
        _validate_jsonl(
            _require_file(bundle, "trajectory.jsonl"),
            label="trajectory.jsonl",
            require_row=True,
        )
    if artifacts["export_rgb_frames"] and (require_full_frames or int(artifacts["snapshot_rgb_frames"]) > 0):
        frames = bundle / "frames"
        _assert_no_symlink_path(frames, root=bundle)
        if not frames.is_dir() or not any(path.is_file() for path in frames.glob("rgb_*.png")):
            raise HmeqaCompletionError("required artifact is missing: frames/rgb_*.png")
    if artifacts["export_video"]:
        path = _require_file(bundle, "episode_rgb.mp4")
        _validate_mp4(path, relative="episode_rgb.mp4")
    # Object-crop export is explicitly best-effort: scenes without usable
    # instance crops legitimately omit it. If the diagnostics inventory says
    # it was produced, validate it strictly.
    if diagnostics.get("object_crops_mosaic"):
        path = _require_file(bundle, "dynagraph/crops_mosaic.png")
        _validate_png(path, relative="dynagraph/crops_mosaic.png")
    if artifacts["export_compact_memory"]:
        _validate_compact_memory(bundle)
    if artifacts["export_voxel_history"]:
        _validate_jsonl(
            _require_file(bundle, "observations_history.jsonl"),
            label="observations_history.jsonl",
            require_row=True,
        )
    if artifacts["export_voxel_pickle"]:
        _require_file(bundle, "voxel_debug.pkl")
    if artifacts["export_gt_navmesh_map"]:
        path = _require_file(bundle, "topdown_gt_navmesh.png")
        _validate_png(path, relative="topdown_gt_navmesh.png")
    if artifacts["export_map_overlay"]:
        path = _require_file(bundle, "topdown_map_overlay.png")
        _validate_png(path, relative="topdown_map_overlay.png")
    if artifacts["export_map_video"]:
        path = _require_file(bundle, "topdown_exploration.mp4")
        _validate_mp4(path, relative="topdown_exploration.mp4")
    if artifacts["export_full_graph"]:
        graph_checkpoint = bundle / "graph_checkpoint"
        _assert_no_symlink_path(graph_checkpoint, root=bundle)
        if not graph_checkpoint.is_dir():
            raise HmeqaCompletionError("required artifact is missing: graph_checkpoint/")

    if arm == "agentic":
        variant = config.get("variant", {})
        if str(variant.get("graph_evidence_mode", "off")) != "off":
            _validate_world_evidence(
                bundle,
                require_rgb_dir=bool(artifacts["export_world_evidence_rgb"]),
            )
        if str(variant.get("attempt_ledger_mode", "off")) != "off":
            attempts = _load_json(_require_file(bundle, "attempt_ledger.json"), expected=list)
            if any(not isinstance(item, Mapping) for item in attempts):
                raise HmeqaCompletionError("attempt_ledger.json entries must be objects")
        if str(variant.get("room_history_mode", "off")) != "off":
            rooms = _load_json(_require_file(bundle, "room_events.json"), expected=list)
            if any(not isinstance(item, Mapping) for item in rooms):
                raise HmeqaCompletionError("room_events.json entries must be objects")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _copy_file(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    _fsync_file(destination)


def _copy_tree_without_symlinks(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise HmeqaCompletionError(f"snapshot source contains symlink: {path}")
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        if path.is_file():
            _fsync_file(path)
    for path in sorted(
        (item for item in destination.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_dir(path)
    _fsync_dir(destination)


def _copy_snapshot(source: Path, stage: Path, profile: Mapping[str, Any]) -> None:
    stage.mkdir(parents=True, exist_ok=False)
    for name in _COPY_FILES:
        src = source / name
        if src.is_symlink():
            raise HmeqaCompletionError(f"snapshot source contains symlink: {src}")
        if src.is_file():
            _copy_file(src, stage / name)
    for name in _COPY_DIRS:
        src = source / name
        if src.is_symlink():
            raise HmeqaCompletionError(f"snapshot source contains symlink: {src}")
        if src.is_dir():
            _copy_tree_without_symlinks(src, stage / name)
    sample_count = int(profile["snapshot_rgb_frames"])
    frames = source / "frames"
    if sample_count > 0 and frames.is_dir():
        if frames.is_symlink():
            raise HmeqaCompletionError(f"snapshot source contains symlink: {frames}")
        rgbs = sorted(frames.glob("rgb_*.png"))
        if rgbs:
            selected = {rgbs[index * (len(rgbs) - 1) // max(1, sample_count - 1)] for index in range(sample_count)}
            dst = stage / "frames"
            dst.mkdir()
            for image in sorted(selected):
                if image.is_symlink():
                    raise HmeqaCompletionError(f"snapshot source contains symlink: {image}")
                _copy_file(image, dst / image.name)


def _rewrite_paths(value: Any, *, source: Path, destination: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, source=source, destination=destination) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, source=source, destination=destination) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            relative = path.relative_to(source)
        except (ValueError, TypeError):
            return value
        return str(destination / relative)
    return value


def _canonical_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    canonical = dict(_rewrite_paths(dict(row), source=source, destination=destination))
    canonical["h2h_arm"] = arm
    canonical["source_debug_bundle_dir"] = str(source)
    canonical["debug_bundle_dir"] = str(destination)
    return canonical


def _snapshot_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HmeqaCompletionError(f"published snapshot contains symlink: {path}")
        if path.is_file() and path.name != UNIT_COMPLETE_FILENAME:
            hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    if UNIT_ROW_FILENAME not in hashes:
        raise HmeqaCompletionError(f"staged snapshot is missing {UNIT_ROW_FILENAME}")
    return hashes


def commit_pending_episode(
    out_dir: Path,
    *,
    arm: str,
    qid: int,
    pending_path: Path,
    exit_code: int,
    method: str = "dynagraph",
) -> dict[str, Any]:
    """Validate and atomically publish one pending episode plus its snapshot."""
    if int(exit_code) != 0:
        raise HmeqaCompletionError(f"episode child exited nonzero ({int(exit_code)}); row is not committable")
    out = Path(out_dir).expanduser().resolve()
    frozen = _manifest(out)
    if (arm, qid) not in set(planned_units(frozen)):
        raise HmeqaCompletionError(f"{arm} q{qid} is not planned by the run manifest")
    source_row = read_pending_row(Path(pending_path))
    expected_source = expected_debug_bundle_dir(out, arm, qid, manifest=frozen)
    source_row = validate_episode_row(
        source_row,
        arm=arm,
        qid=qid,
        method=method,
        expected_debug_dir=expected_source,
    )
    profile = normalize_hmeqa_artifact_profile(frozen["config"]["artifacts"])
    validate_snapshot_bundle(
        expected_source,
        row=source_row,
        arm=arm,
        profile=profile,
        config=frozen["config"],
        require_full_frames=True,
    )

    destination = unit_bundle_dir(out, arm, qid)
    if destination.exists():
        try:
            existing = validate_unit_marker(out, arm, qid, manifest=frozen)
        except HmeqaCompletionError:
            pass
        else:
            Path(pending_path).unlink(missing_ok=True)
            return existing
    bundles = destination.parent
    bundles.mkdir(parents=True, exist_ok=True)
    stage = bundles / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        _copy_snapshot(expected_source, stage, profile)
        canonical = _canonical_row(
            source_row,
            arm=arm,
            source=expected_source,
            destination=destination,
        )
        _atomic_json(stage / UNIT_ROW_FILENAME, canonical)
        _atomic_json(stage / "metrics.json", canonical)
        diagnostics_path = stage / "diagnostics_manifest.json"
        if diagnostics_path.is_file():
            diagnostics = _load_json(diagnostics_path, expected=dict)
            _atomic_json(
                diagnostics_path,
                _rewrite_paths(
                    diagnostics,
                    source=expected_source,
                    destination=destination,
                ),
            )
        validate_snapshot_bundle(
            stage,
            row=canonical,
            arm=arm,
            profile=profile,
            config=frozen["config"],
            require_full_frames=False,
        )
        hashes = _snapshot_hashes(stage)
        marker = {
            "schema": HMEQA_UNIT_COMPLETE_SCHEMA,
            "schema_version": HMEQA_UNIT_COMPLETE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arm": arm,
            "question_id": qid,
            "method": method,
            "config_digest": frozen["config_digest"],
            "artifact_profile_digest": _json_digest(profile),
            "source_debug_bundle_dir": str(expected_source),
            "row_file": UNIT_ROW_FILENAME,
            "row_sha256": hashes[UNIT_ROW_FILENAME],
            "artifact_hashes": hashes,
        }
        _atomic_json(stage / UNIT_COMPLETE_FILENAME, marker)
        _fsync_dir(stage)
        if destination.exists():
            # Only an invalid/incomplete directory reaches this point. A valid
            # published marker returned above and is never destructively replaced.
            shutil.rmtree(destination)
        os.replace(stage, destination)
        _fsync_dir(bundles)
        validated = validate_unit_marker(out, arm, qid, manifest=frozen)
        _atomic_jsonl_row(out / f"{_unit_name(arm, qid)}.jsonl", validated["row"])
        Path(pending_path).unlink(missing_ok=True)
        return validated
    finally:
        if stage.is_dir():
            shutil.rmtree(stage, ignore_errors=True)


def validate_unit_marker(
    out_dir: Path,
    arm: str,
    qid: int,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir).expanduser().resolve()
    frozen = dict(manifest or _manifest(out))
    bundle = unit_bundle_dir(out, arm, qid)
    marker_path = bundle / UNIT_COMPLETE_FILENAME
    marker = _load_json(marker_path, expected=dict)
    expected_fields = {
        "schema": HMEQA_UNIT_COMPLETE_SCHEMA,
        "schema_version": HMEQA_UNIT_COMPLETE_VERSION,
        "arm": arm,
        "question_id": qid,
        "method": "dynagraph",
        "config_digest": frozen["config_digest"],
    }
    for key, expected in expected_fields.items():
        if marker.get(key) != expected:
            raise HmeqaCompletionError(f"{marker_path} {key}={marker.get(key)!r}, expected {expected!r}")
    profile = normalize_hmeqa_artifact_profile(frozen["config"]["artifacts"])
    if marker.get("artifact_profile_digest") != _json_digest(profile):
        raise HmeqaCompletionError(f"{marker_path} artifact profile digest is stale or corrupt")
    expected_source = expected_debug_bundle_dir(out, arm, qid, manifest=frozen)
    if marker.get("source_debug_bundle_dir") != str(expected_source):
        raise HmeqaCompletionError(f"{marker_path} source debug bundle identity is wrong")
    if marker.get("row_file") != UNIT_ROW_FILENAME:
        raise HmeqaCompletionError(f"{marker_path} row_file is invalid")
    declared_hashes = marker.get("artifact_hashes")
    if not isinstance(declared_hashes, Mapping) or not declared_hashes:
        raise HmeqaCompletionError(f"{marker_path} artifact_hashes must be a non-empty object")
    actual_hashes = _snapshot_hashes(bundle)
    if dict(declared_hashes) != actual_hashes:
        raise HmeqaCompletionError(f"{marker_path} artifact hashes do not match the bundle")
    if marker.get("row_sha256") != actual_hashes.get(UNIT_ROW_FILENAME):
        raise HmeqaCompletionError(f"{marker_path} row hash does not match {UNIT_ROW_FILENAME}")
    row = _load_json(bundle / UNIT_ROW_FILENAME, expected=dict)
    row = validate_episode_row(row, arm=arm, qid=qid, method="dynagraph")
    if row.get("debug_bundle_dir") != str(bundle):
        raise HmeqaCompletionError(f"{UNIT_ROW_FILENAME} does not point at its published bundle")
    if row.get("source_debug_bundle_dir") != str(expected_source):
        raise HmeqaCompletionError(f"{UNIT_ROW_FILENAME} source bundle identity is wrong")
    validate_snapshot_bundle(
        bundle,
        row=row,
        arm=arm,
        profile=profile,
        config=frozen["config"],
        require_full_frames=False,
    )
    return {"marker": marker, "row": row, "marker_sha256": _sha256_file(marker_path)}


def unit_is_complete(out_dir: Path, arm: str, qid: int) -> bool:
    try:
        validate_unit_marker(out_dir, arm, qid)
    except HmeqaCompletionError:
        return False
    return True


def completed_unit_count(out_dir: Path) -> int:
    frozen = _manifest(Path(out_dir))
    return sum(unit_is_complete(out_dir, arm, qid) for arm, qid in planned_units(frozen))


def rebuild_aggregates(out_dir: Path) -> dict[str, str]:
    """Atomically rebuild convenience rows and per-arm JSONL from markers only."""
    out = Path(out_dir).expanduser().resolve()
    frozen = _manifest(out)
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    for arm, qid in planned_units(frozen):
        try:
            validated = validate_unit_marker(out, arm, qid, manifest=frozen)
        except HmeqaCompletionError:
            continue
        row = dict(validated["row"])
        arm_rows.setdefault(arm, []).append(row)
        _atomic_jsonl_row(out / f"{_unit_name(arm, qid)}.jsonl", row)
    hashes: dict[str, str] = {}
    for arm in frozen["config"]["evaluation"]["arms"]:
        path = out / f"{arm}.jsonl"
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in arm_rows.get(str(arm), [])
        ).encode("utf-8")
        _atomic_write(path, payload)
        hashes[path.name] = _sha256_file(path)
    return hashes


def _actual_marker_units(out: Path) -> set[str]:
    bundles = out / "bundles"
    if not bundles.is_dir():
        return set()
    return {marker.parent.name for marker in bundles.glob(f"*/{UNIT_COMPLETE_FILENAME}") if marker.is_file()}


def finalize_run(out_dir: Path) -> dict[str, Any]:
    """Publish atomic JSON ``DONE`` only after every planned marker validates."""
    out = Path(out_dir).expanduser().resolve()
    frozen = _manifest(out)
    units: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm, qid in planned_units(frozen):
        try:
            validated = validate_unit_marker(out, arm, qid, manifest=frozen)
        except HmeqaCompletionError:
            missing.append(_unit_name(arm, qid))
            continue
        units.append(
            {
                "arm": arm,
                "question_id": qid,
                "bundle": f"bundles/{_unit_name(arm, qid)}",
                "marker_sha256": validated["marker_sha256"],
                "row_sha256": validated["marker"]["row_sha256"],
            }
        )
    if missing:
        (out / RUN_DONE_FILENAME).unlink(missing_ok=True)
        raise HmeqaCompletionError(
            "cannot finalize incomplete HM-EQA run; missing/invalid markers: " + ", ".join(missing)
        )
    expected_names = {_unit_name(str(item["arm"]), int(item["question_id"])) for item in units}
    extras = sorted(_actual_marker_units(out) - expected_names)
    if extras:
        raise HmeqaCompletionError(f"cannot finalize run with unplanned completion markers: {', '.join(extras)}")
    aggregate_hashes = rebuild_aggregates(out)
    done = {
        "schema": HMEQA_DONE_SCHEMA,
        "schema_version": HMEQA_DONE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_digest": frozen["config_digest"],
        "artifact_profile_digest": _json_digest(frozen["config"]["artifacts"]),
        "unit_count": len(units),
        "units": units,
        "aggregate_hashes": aggregate_hashes,
    }
    _atomic_json(out / RUN_DONE_FILENAME, done)
    return done


def validate_done(out_dir: Path) -> bool:
    """Return whether ``DONE`` exactly binds the current marker and aggregate set."""
    try:
        out = Path(out_dir).expanduser().resolve()
        frozen = _manifest(out)
        done = _load_json(out / RUN_DONE_FILENAME, expected=dict)
        if done.get("schema") != HMEQA_DONE_SCHEMA or done.get("schema_version") != HMEQA_DONE_VERSION:
            return False
        if done.get("config_digest") != frozen["config_digest"]:
            return False
        if done.get("artifact_profile_digest") != _json_digest(frozen["config"]["artifacts"]):
            return False
        planned = planned_units(frozen)
        rows = done.get("units")
        if not isinstance(rows, list) or len(rows) != len(planned) or done.get("unit_count") != len(planned):
            return False
        expected_rows: list[dict[str, Any]] = []
        for arm, qid in planned:
            validated = validate_unit_marker(out, arm, qid, manifest=frozen)
            expected_rows.append(
                {
                    "arm": arm,
                    "question_id": qid,
                    "bundle": f"bundles/{_unit_name(arm, qid)}",
                    "marker_sha256": validated["marker_sha256"],
                    "row_sha256": validated["marker"]["row_sha256"],
                }
            )
        if rows != expected_rows:
            return False
        if _actual_marker_units(out) != {_unit_name(arm, qid) for arm, qid in planned}:
            return False
        aggregate_hashes = done.get("aggregate_hashes")
        if not isinstance(aggregate_hashes, Mapping):
            return False
        for name, digest in aggregate_hashes.items():
            path = out / str(name)
            if not path.is_file() or _sha256_file(path) != digest:
                return False
        return True
    except (HmeqaCompletionError, OSError, TypeError, ValueError):
        return False


def has_resume_state(out_dir: Path) -> bool:
    out = Path(out_dir)
    if (out / "run_manifest.json").is_file():
        return True
    pending = out / ".pending"
    if pending.is_dir() and any(pending.iterdir()):
        return True
    if _actual_marker_units(out):
        return True
    return any(out.glob("*_q*.jsonl"))


def reconcile_run(out_dir: Path) -> dict[str, Any]:
    """CPU-only rebuild/finalize from durable markers without invoking Habitat."""
    out = Path(out_dir).expanduser().resolve()
    frozen = _manifest(out)
    total = len(planned_units(frozen))
    aggregate_hashes = rebuild_aggregates(out)
    completed = completed_unit_count(out)
    done = None
    if completed == total:
        done = finalize_run(out)
    else:
        (out / RUN_DONE_FILENAME).unlink(missing_ok=True)
    return {
        "completed": completed,
        "total": total,
        "aggregate_hashes": aggregate_hashes,
        "done": done is not None,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tag = sub.add_parser("debug-run-tag")
    tag.add_argument("out", type=Path)
    tag.add_argument("arm", choices=sorted(_ARMS))
    tag.add_argument("qid", type=int)

    complete = sub.add_parser("is-complete")
    complete.add_argument("out", type=Path)
    complete.add_argument("arm", choices=sorted(_ARMS))
    complete.add_argument("qid", type=int)

    commit = sub.add_parser("commit")
    commit.add_argument("out", type=Path)
    commit.add_argument("arm", choices=sorted(_ARMS))
    commit.add_argument("qid", type=int)
    commit.add_argument("pending", type=Path)
    commit.add_argument("--exit-code", type=int, required=True)

    for name in ("count", "rebuild", "finalize", "validate-done", "reconcile"):
        command = sub.add_parser(name)
        command.add_argument("out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "debug-run-tag":
            print(hmeqa_debug_run_tag(args.out, args.arm, args.qid))
        elif args.command == "is-complete":
            return 0 if unit_is_complete(args.out, args.arm, args.qid) else 1
        elif args.command == "commit":
            commit_pending_episode(
                args.out,
                arm=args.arm,
                qid=args.qid,
                pending_path=args.pending,
                exit_code=args.exit_code,
            )
        elif args.command == "count":
            print(completed_unit_count(args.out))
        elif args.command == "rebuild":
            print(json.dumps(rebuild_aggregates(args.out), sort_keys=True))
        elif args.command == "finalize":
            print(json.dumps(finalize_run(args.out), sort_keys=True))
        elif args.command == "validate-done":
            return 0 if validate_done(args.out) else 1
        elif args.command == "reconcile":
            print(json.dumps(reconcile_run(args.out), sort_keys=True))
    except HmeqaCompletionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
