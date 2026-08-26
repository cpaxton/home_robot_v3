# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Per-episode debug bundles for Habitat HM-EQA (logs, graph snapshots, EQA history)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from emet.eval.hmeqa_launch import hmeqa_git_state
from emet.habitat.config import (
    default_habitat_eqa_data_dir,
    default_hm3d_scene_dir,
    questions_csv_path,
    scene_init_poses_csv_path,
)
from emet.habitat.metrics import EpisodeMetrics

ATTEMPT_LEDGER_FILENAME = "attempt_ledger.json"
ROOM_EVENTS_FILENAME = "room_events.json"
COMPACT_MEMORY_DIRNAME = "compact_memory"
HMEQA_BATCH_MANIFEST_SCHEMA = "emet.hmeqa.batch_manifest"
HMEQA_BATCH_MANIFEST_VERSION = 2


def default_episodes_root() -> Path:
    return default_habitat_eqa_data_dir().parent / "episodes"


def _git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


@lru_cache(maxsize=1)
def code_state_fingerprint() -> str:
    """Identify the code that is actually running, not just ``HEAD``.

    Runs launched from a dirty tree all report the same ``git_commit`` even though
    they execute different code, which made two 2026-07 HM-EQA sweeps look like
    nondeterminism from identical configs. Appending a hash of the uncommitted diff
    makes those runs distinguishable. Cached once per process, since the code was
    imported at start-up and later edits do not affect this process.
    """
    head = _git_head() or "nogit"
    try:
        root = Path(__file__).resolve().parents[3]
        r = subprocess.run(
            ["git", "-C", str(root), "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return f"{head}+unknown"
    if r.returncode != 0:
        return f"{head}+unknown"
    diff = r.stdout
    if not diff.strip():
        return f"{head}+clean"
    return f"{head}+dirty.{hashlib.sha1(diff.encode('utf-8')).hexdigest()[:12]}"


def coerce_parameters_dict(parameters: Any) -> dict[str, Any]:
    """Normalize :class:`~emet.core.parameters.Parameters` or a plain dict for JSON manifests."""
    if parameters is None:
        return {}
    if isinstance(parameters, dict):
        return parameters
    data = getattr(parameters, "data", None)
    if isinstance(data, dict):
        return dict(data)
    return {}


def harness_fingerprint_from_parameters(parameters: Any) -> dict[str, Any]:
    """Extract merge / explore / profile fields for episode + run manifests."""
    params = coerce_parameters_dict(parameters)
    fusion = params.get("graph_object_fusion") or {}
    if not isinstance(fusion, dict):
        fusion = {}
    block = params.get("dynagraph_harness") or {}
    if not isinstance(block, dict):
        block = {}
    return {
        "git_commit": _git_head(),
        "code_state": code_state_fingerprint(),
        "dynagraph_merge_xy_m": params.get("dynagraph_merge_xy_m"),
        "dynagraph_staleness_horizon": params.get("dynagraph_staleness_horizon"),
        "fallback_spatial_merge_xy_m": fusion.get("fallback_spatial_merge_xy_m"),
        "profile": block.get("profile"),
        "memory_summary": block.get("memory_summary"),
        "mcq_debias": block.get("mcq_debias"),
        "explore_when_uncovered": block.get("explore_when_uncovered"),
        "harness": block.get("harness"),
        "method": block.get("method"),
    }


def run_tag_from_output_jsonl(output_jsonl: Path | None) -> str:
    if output_jsonl is None:
        return "habitat_episode"
    return output_jsonl.stem


def _manifest_file_fingerprint(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"HM-EQA manifest input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.expanduser().resolve()),
        "sha256": f"sha256:{digest.hexdigest()}",
    }


def _manifest_parameters_digest(parameters: dict[str, Any]) -> str:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _hmeqa_behavior_environment() -> dict[str, str]:
    exact = {
        "EQA_HF_MODEL_ID",
        "EQA_VL_FAMILY",
        "EQA_VL_QUANTIZATION",
        "EMET_ATTEMPT_LEDGER_MAX",
        "EMET_ATTEMPT_LEDGER_PERSIST_ABSENT",
        "EMET_LLM_HOST",
        "EMET_OPENAI_BASE_URL",
        "EMET_VLM_FRONTIER_SCORING",
        "EMET_VL_ENDPOINT",
        "EMET_WORLD_SESSION_ID",
    }
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if value and (name.startswith(("EMET_EQA_", "EMET_DYNAGRAPH_", "EMET_HABITAT_")) or name in exact)
    }


def write_run_manifest(
    *,
    output_jsonl: Path,
    method: str,
    question_ids: list[int],
    mock_llm: bool,
    max_planning_steps: int,
    max_movement_step: int,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    eqa_vl_quantization: str | None,
    device: str | None,
    use_hm3d_semantics: bool | None = None,
    use_enrich_labels: bool = False,
    resume: bool,
    parameters: Any = None,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
) -> Path:
    """Create or validate the immutable manifest beside a batch results JSONL."""
    manifest_path = output_jsonl.parent / f"{output_jsonl.stem}_manifest.json"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    params = coerce_parameters_dict(parameters)
    eqa_cfg = {}
    raw = params.get("eqa", {}) or {}
    if isinstance(raw, dict):
        eqa_cfg = dict(raw)
    resolved_questions = Path(questions_path or questions_csv_path()).expanduser().resolve()
    resolved_init_poses = Path(init_poses_path or scene_init_poses_csv_path()).expanduser().resolve()
    resolved_hm3d_root = Path(hm3d_root or default_hm3d_scene_dir()).expanduser().resolve()
    if not resolved_hm3d_root.is_dir():
        raise FileNotFoundError(f"HM3D scene root is missing: {resolved_hm3d_root}")
    project_root = Path(__file__).resolve().parents[3]
    immutable = {
        "schema": HMEQA_BATCH_MANIFEST_SCHEMA,
        "schema_version": HMEQA_BATCH_MANIFEST_VERSION,
        "run_tag": output_jsonl.stem,
        "output_jsonl": str(output_jsonl.expanduser().resolve()),
        "episodes_root": str(default_episodes_root() / output_jsonl.stem),
        "method": method,
        "question_ids": question_ids,
        "mock_llm": mock_llm,
        "max_planning_steps": max_planning_steps,
        "max_movement_step": max_movement_step,
        "eqa_vl_family": eqa_vl_family,
        "eqa_hf_model_id": eqa_hf_model_id or eqa_cfg.get("vl_hf_model_id"),
        "eqa_vl_quantization": eqa_vl_quantization or eqa_cfg.get("vl_quantization"),
        "device": device,
        "use_hm3d_semantics": use_hm3d_semantics,
        "use_enrich_labels": bool(use_enrich_labels),
        "git_commit": _git_head(),
        "code_state": code_state_fingerprint(),
        "git": hmeqa_git_state(project_root),
        "harness": harness_fingerprint_from_parameters(params),
        "graph_eqa_frontier_nodes": params.get("graph_eqa_frontier_nodes"),
        "parameters_sha256": _manifest_parameters_digest(params),
        "behavior_environment": _hmeqa_behavior_environment(),
        "external_inputs": {
            "questions": _manifest_file_fingerprint(resolved_questions),
            "scene_init_poses": _manifest_file_fingerprint(resolved_init_poses),
            # Record the canonical asset root without claiming to hash the large meshes.
            "hm3d_root": str(resolved_hm3d_root),
        },
        "export_full_graph": os.environ.get("HABITAT_EQA_EXPORT_GRAPH", "").strip().lower()
        in ("1", "true", "yes", "on"),
        "export_diagnostics_map": os.environ.get("EMET_EVAL_EXPORT_MAP", os.environ.get("HABITAT_EQA_EXPORT_MAP", "1"))
        .strip()
        .lower()
        in ("1", "true", "yes", "on", ""),
    }
    now = datetime.now(timezone.utc).isoformat()
    if not resume and (manifest_path.exists() or (output_jsonl.exists() and output_jsonl.stat().st_size > 0)):
        raise ValueError(
            f"refusing to append to an existing HM-EQA run at {output_jsonl}; use --resume or choose a new output path"
        )
    if resume and manifest_path.is_file():
        if not output_jsonl.is_file():
            raise ValueError(
                f"cannot resume {output_jsonl}: manifest exists but the results JSONL is missing; "
                "restore the output or choose a new output path"
            )
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot resume {output_jsonl}: invalid manifest {manifest_path}: {exc}") from exc
        mismatches = [key for key, value in immutable.items() if not isinstance(prev, dict) or prev.get(key) != value]
        if mismatches:
            raise ValueError(f"cannot resume {output_jsonl}: manifest mismatch in {', '.join(mismatches)}")
        payload = dict(prev)
        payload["updated_at"] = now
        payload["last_invocation_resume"] = True
    else:
        # A zero-byte JSONL is the durable placeholder created before a new
        # batch manifest; it is safe to complete that interrupted initialization.
        if resume and output_jsonl.exists() and output_jsonl.stat().st_size > 0:
            raise ValueError(
                f"cannot resume {output_jsonl}: {manifest_path.name} is missing; "
                "refusing to mix historical rows with a new configuration"
            )
        payload = {
            **immutable,
            "started_at": now,
            "updated_at": now,
            "last_invocation_resume": bool(resume),
        }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return manifest_path


def _frontier_node_summaries(graph_memory: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in graph_memory.get_nodes():
        if not getattr(node, "is_frontier", False):
            continue
        out.append(
            {
                "node_id": int(node.node_id),
                "labels": list(node.labels or []),
                "xyz": [float(node.xyz[0]), float(node.xyz[1]), float(node.xyz[2])],
                "description": str(node.description or ""),
                "obs_id": int(node.obs_id),
            }
        )
    return out


def _save_agentic_evidence_artifacts(
    episode_dir: Path,
    graph_memory: Any,
    *,
    include_world_evidence_rgb: bool,
) -> None:
    """Persist policy-sidecar evidence needed for shadow/agent audits."""
    world_evidence = getattr(graph_memory, "world_evidence", None)
    if world_evidence is not None and bool(getattr(world_evidence, "enabled", False)):
        save_world_evidence = getattr(world_evidence, "save", None)
        if not callable(save_world_evidence):
            raise TypeError("enabled world_evidence store does not provide save()")
        save_world_evidence(episode_dir, include_rgb=include_world_evidence_rgb)

    export_attempt_ledger = getattr(graph_memory, "export_attempt_ledger", None)
    if callable(export_attempt_ledger):
        attempt_rows = list(export_attempt_ledger() or [])
        (episode_dir / ATTEMPT_LEDGER_FILENAME).write_text(
            json.dumps(attempt_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    get_room_events = getattr(graph_memory, "get_room_events", None)
    if callable(get_room_events):
        room_rows = list(get_room_events() or [])
        (episode_dir / ROOM_EVENTS_FILENAME).write_text(
            json.dumps(room_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def enrich_episode_metrics(
    metrics: EpisodeMetrics,
    *,
    agent: Any,
    choices: list[str] | None,
    formatted_answer: str = "",
    eqa_action: str = "",
    eqa_confidence_reasoning: str = "",
    vl_family: str | None = None,
    vl_hf_model_id: str | None = None,
    vl_endpoint: str | None = None,
    error: str = "",
    debug_bundle_dir: str = "",
) -> EpisodeMetrics:
    """Attach graph / VLM debug fields to ``EpisodeMetrics``."""
    gm = getattr(agent, "graph_memory", None)
    metrics.choices = list(choices or [])
    metrics.formatted_answer = formatted_answer
    metrics.eqa_action = eqa_action
    metrics.eqa_confidence_reasoning = eqa_confidence_reasoning
    metrics.vl_family = vl_family or ""
    metrics.vl_hf_model_id = vl_hf_model_id or ""
    metrics.vl_endpoint = vl_endpoint or ""
    metrics.error = error
    metrics.debug_bundle_dir = debug_bundle_dir
    if gm is not None:
        from emet.memory.graph_eqa.graph_stats import graph_health_metrics

        metrics.eqa_iterations = len(getattr(gm, "_history_outputs", []) or [])
        metrics.frontier_nodes = sum(1 for n in gm.get_nodes() if getattr(n, "is_frontier", False))
        metrics.graph_nodes = len(gm.get_nodes())
        metrics.observations = len(getattr(gm, "_observations", []) or [])
        health = graph_health_metrics(gm)
        metrics.graph_health = health
        # Keep legacy scalars aligned with health breakdown.
        metrics.frontier_nodes = int(health.get("n_frontier", metrics.frontier_nodes))
        metrics.graph_nodes = int(health.get("n_total", metrics.graph_nodes))
        metrics.observations = int(health.get("n_obs", metrics.observations))
    params = getattr(agent, "parameters", None)
    if params is None and gm is not None:
        params = getattr(gm, "parameters", None)
    fp = harness_fingerprint_from_parameters(params)
    if vl_family:
        fp["vl_family"] = str(vl_family)
    if vl_hf_model_id:
        fp["vl_hf_model_id"] = str(vl_hf_model_id)
    if vl_endpoint:
        fp["vl_endpoint"] = str(vl_endpoint)
    metrics.harness = fp
    return metrics


def save_episode_debug_bundle(
    *,
    run_tag: str,
    metrics: EpisodeMetrics,
    agent: Any,
    raw_eqa_full: str = "",
    recorder: Any | None = None,
    diagnostics_cfg: Any | None = None,
) -> Path:
    """
    Save per-question artifacts under ``~/.cache/habitat_eqa/episodes/<run_tag>/q<id>_<method>/``.

    Always writes: ``metrics.json``, ``raw_eqa.txt``, ``eqa_history.json``, ``scene_graph_report.txt``,
    ``frontier_nodes.json``, room events, and the attempt ledger. Enabled shadow/agent stores also write
    ``world_evidence.json`` plus their evidence views. Set ``HABITAT_EQA_EXPORT_GRAPH=1`` for the full
    ``export_graph_eqa_dir`` checkpoint, or ``EMET_EVAL_EXPORT_COMPACT_MEMORY=1`` for a reloadable
    graph/runtime checkpoint without voxel frames or view pixels.
    When ``recorder`` is set (or env ``EMET_EVAL_EXPORT_MAP``), also writes maps / video via
    :mod:`emet.eval.episode_diagnostics`.
    """
    episode_dir = default_episodes_root() / run_tag / f"q{metrics.question_id:04d}_{metrics.method}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    from emet.eval.episode_diagnostics import EpisodeDiagnosticsConfig

    cfg = diagnostics_cfg or EpisodeDiagnosticsConfig.from_env()

    gm = getattr(agent, "graph_memory", None)
    for stale_name in (
        "eqa_history.json",
        "raw_eqa.txt",
        "metrics.json",
        "error.txt",
        "scene_graph_report.txt",
        "frontier_nodes.json",
        ATTEMPT_LEDGER_FILENAME,
        ROOM_EVENTS_FILENAME,
        "world_evidence.json",
    ):
        (episode_dir / stale_name).unlink(missing_ok=True)
    for stale_dir_name in ("frontier_picks", "world_evidence_views"):
        stale_dir = episode_dir / stale_dir_name
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)
    if not (cfg.export_compact_memory and gm is not None):
        stale_compact = episode_dir / COMPACT_MEMORY_DIRNAME
        if stale_compact.is_dir():
            shutil.rmtree(stale_compact)

    history = list(getattr(gm, "_history_outputs", []) or []) if gm is not None else []
    (episode_dir / "eqa_history.json").write_text(
        json.dumps({"iterations": history}, indent=2) + "\n",
        encoding="utf-8",
    )

    raw = raw_eqa_full or metrics.raw_eqa_output
    (episode_dir / "raw_eqa.txt").write_text(raw, encoding="utf-8")

    # Agentic tool-loop traces (EMET_EQA_TRACE=1 → AgenticEQAExecutor._flush_trace_to_agent).
    trace_rows = getattr(agent, "_agentic_trace_rows", None) or []
    if trace_rows:
        (episode_dir / "agentic_trace.jsonl").write_text(
            "".join(json.dumps(r, default=str) + "\n" for r in trace_rows),
            encoding="utf-8",
        )
    summary = getattr(agent, "_agentic_eqa_summary", None)
    if isinstance(summary, dict) and summary:
        (episode_dir / "agentic_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    # Numbered frontier-pick panels written during explore_frontier.
    picks_dst = episode_dir / "frontier_picks"
    picks_dst.mkdir(parents=True, exist_ok=True)
    src_dirs: list[Path] = []
    for key in ("_frontier_pick_dir", "_episode_debug_dir"):
        raw = getattr(agent, key, None)
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        if key == "_episode_debug_dir":
            p = p / "frontier_picks"
        if p.is_dir() and p.resolve() != picks_dst.resolve():
            src_dirs.append(p)
    for src in src_dirs:
        for png in sorted(src.glob("iter_*.png")):
            shutil.copy2(png, picks_dst / png.name)
    for png_s in getattr(agent, "_frontier_pick_panels", None) or []:
        png = Path(str(png_s))
        # Panels may already live in picks_dst (agentic executor writes there
        # directly); copying a file onto itself raises SameFileError and would
        # abort the rest of the bundle export (maps, trajectory, floor metrics).
        if png.is_file() and png.resolve() != (picks_dst / png.name).resolve():
            shutil.copy2(png, picks_dst / png.name)

    if gm is not None:
        from emet.memory.graph_eqa.pretty_print import format_scene_graph_pretty

        report = format_scene_graph_pretty(gm, title=f"HM-EQA q{metrics.question_id}")
        (episode_dir / "scene_graph_report.txt").write_text(report, encoding="utf-8")
        (episode_dir / "frontier_nodes.json").write_text(
            json.dumps(_frontier_node_summaries(gm), indent=2) + "\n",
            encoding="utf-8",
        )
        _save_agentic_evidence_artifacts(
            episode_dir,
            gm,
            include_world_evidence_rgb=cfg.export_world_evidence_rgb,
        )

    compact_memory_dir: Path | None = None
    if cfg.export_compact_memory and gm is not None:
        from emet.memory.adapters import GraphEQABackend

        compact_memory_dir = episode_dir / COMPACT_MEMORY_DIRNAME
        GraphEQABackend(gm).save(
            str(compact_memory_dir),
            final_step=int(
                getattr(
                    agent,
                    "obs_count",
                    getattr(gm, "_graph_timestep", 0),
                )
                or 0
            ),
            include_frames=False,
            include_world_evidence_rgb=False,
        )

    export_graph = os.environ.get("HABITAT_EQA_EXPORT_GRAPH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if export_graph and gm is not None and getattr(agent, "voxel_map", None) is not None:
        from emet.memory.headless_export import export_graph_eqa_dir

        export_graph_eqa_dir(
            gm,
            agent.voxel_map,
            str(episode_dir / "graph_checkpoint"),
            title=f"HM-EQA q{metrics.question_id} checkpoint",
        )

    if metrics.error:
        (episode_dir / "error.txt").write_text(metrics.error, encoding="utf-8")

    from emet.eval.episode_diagnostics import (
        DIAGNOSTICS_MANIFEST,
        EpisodeDiagnosticsRecorder,
        flush_episode_diagnostics,
    )

    diag_rec = recorder
    if diag_rec is None and any(
        (
            cfg.export_map,
            cfg.export_map_video,
            cfg.export_video,
            cfg.export_rgb_frames,
            cfg.export_trajectory,
            cfg.export_obstacle_grids,
            cfg.export_object_crops,
            cfg.export_full_graph,
            cfg.export_compact_memory,
            cfg.export_voxel_history,
            cfg.export_voxel_pickle,
        )
    ):
        diag_rec = EpisodeDiagnosticsRecorder(cfg=cfg)
    if diag_rec is not None:
        manifest = flush_episode_diagnostics(episode_dir, agent, diag_rec)
        if compact_memory_dir is not None:
            manifest["compact_memory"] = str(compact_memory_dir)
            manifest_path = episode_dir / DIAGNOSTICS_MANIFEST
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
        if manifest.get("topdown_map"):
            metrics.topdown_map_path = str(manifest["topdown_map"])
        if manifest.get("diagnostics_manifest"):
            metrics.diagnostics_manifest_path = str(manifest["diagnostics_manifest"])

    gm = getattr(agent, "graph_memory", None)
    trace_root = getattr(gm, "_eqa_decision_trace_dir", None) if gm is not None else None
    if trace_root:
        from emet.eval.eqa_decision_trace import finalize_eqa_decision_trace

        n_iter = int(getattr(metrics, "eqa_iterations", 0) or 0)
        if n_iter <= 0 and gm is not None:
            n_iter = len(getattr(gm, "_history_outputs", []) or [])
        finalize_eqa_decision_trace(trace_root, n_iterations=n_iter)
        manifest_path = episode_dir / "diagnostics_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["eqa_decisions"] = str(trace_root)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    metrics.debug_bundle_dir = str(episode_dir)
    (episode_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_dir


def save_error_episode_bundle(*, run_tag: str, metrics: EpisodeMetrics) -> Path:
    """Minimal bundle when the episode crashes before an agent exists."""
    episode_dir = default_episodes_root() / run_tag / f"q{metrics.question_id:04d}_{metrics.method}"
    if episode_dir.is_dir():
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    if metrics.error:
        (episode_dir / "error.txt").write_text(metrics.error, encoding="utf-8")
    metrics.debug_bundle_dir = str(episode_dir)
    (episode_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_dir
