# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Per-episode debug bundles for Habitat HM-EQA (logs, graph snapshots, EQA history)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emet.habitat.config import default_habitat_eqa_data_dir
from emet.habitat.metrics import EpisodeMetrics


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


def run_tag_from_output_jsonl(output_jsonl: Path | None) -> str:
    if output_jsonl is None:
        return "habitat_episode"
    return output_jsonl.stem


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
    device: str | None,
    resume: bool,
    parameters: Any = None,
) -> Path:
    """Write or update ``run_manifest.json`` beside the results JSONL."""
    manifest_path = output_jsonl.parent / f"{output_jsonl.stem}_manifest.json"
    params = coerce_parameters_dict(parameters)
    eqa_cfg = {}
    raw = params.get("eqa", {}) or {}
    if isinstance(raw, dict):
        eqa_cfg = dict(raw)
    payload = {
        "run_tag": output_jsonl.stem,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": str(output_jsonl.expanduser()),
        "episodes_root": str(default_episodes_root() / output_jsonl.stem),
        "method": method,
        "question_ids": question_ids,
        "mock_llm": mock_llm,
        "max_planning_steps": max_planning_steps,
        "max_movement_step": max_movement_step,
        "eqa_vl_family": eqa_vl_family,
        "eqa_hf_model_id": eqa_hf_model_id or eqa_cfg.get("vl_hf_model_id"),
        "device": device,
        "resume": resume,
        "git_commit": _git_head(),
        "graph_eqa_frontier_nodes": params.get("graph_eqa_frontier_nodes"),
        "export_full_graph": os.environ.get("HABITAT_EQA_EXPORT_GRAPH", "").strip().lower()
        in ("1", "true", "yes", "on"),
        "export_diagnostics_map": os.environ.get("EMET_EVAL_EXPORT_MAP", os.environ.get("HABITAT_EQA_EXPORT_MAP", "1"))
        .strip()
        .lower()
        in ("1", "true", "yes", "on", ""),
    }
    if manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and prev.get("started_at"):
                payload["started_at"] = prev["started_at"]
        except json.JSONDecodeError:
            pass
    payload.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
    metrics.error = error
    metrics.debug_bundle_dir = debug_bundle_dir
    if gm is not None:
        metrics.eqa_iterations = len(getattr(gm, "_history_outputs", []) or [])
        metrics.frontier_nodes = sum(1 for n in gm.get_nodes() if getattr(n, "is_frontier", False))
        metrics.graph_nodes = len(gm.get_nodes())
        metrics.observations = len(getattr(gm, "_observations", []) or [])
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
    ``frontier_nodes.json``. Set ``HABITAT_EQA_EXPORT_GRAPH=1`` for full ``export_graph_eqa_dir`` checkpoint.
    When ``recorder`` is set (or env ``EMET_EVAL_EXPORT_MAP``), also writes maps / video via
    :mod:`emet.eval.episode_diagnostics`.
    """
    episode_dir = default_episodes_root() / run_tag / f"q{metrics.question_id:04d}_{metrics.method}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    gm = getattr(agent, "graph_memory", None)
    history = list(getattr(gm, "_history_outputs", []) or []) if gm is not None else []
    (episode_dir / "eqa_history.json").write_text(
        json.dumps({"iterations": history}, indent=2) + "\n",
        encoding="utf-8",
    )

    raw = raw_eqa_full or metrics.raw_eqa_output
    (episode_dir / "raw_eqa.txt").write_text(raw, encoding="utf-8")

    if gm is not None:
        from emet.memory.graph_eqa.pretty_print import format_scene_graph_pretty

        report = format_scene_graph_pretty(gm, title=f"HM-EQA q{metrics.question_id}")
        (episode_dir / "scene_graph_report.txt").write_text(report, encoding="utf-8")
        (episode_dir / "frontier_nodes.json").write_text(
            json.dumps(_frontier_node_summaries(gm), indent=2) + "\n",
            encoding="utf-8",
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
        EpisodeDiagnosticsConfig,
        EpisodeDiagnosticsRecorder,
        flush_episode_diagnostics,
    )

    cfg = diagnostics_cfg or EpisodeDiagnosticsConfig.from_env()
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
            cfg.export_voxel_history,
            cfg.export_voxel_pickle,
        )
    ):
        diag_rec = EpisodeDiagnosticsRecorder(cfg=cfg)
    if diag_rec is not None:
        manifest = flush_episode_diagnostics(episode_dir, agent, diag_rec)
        if manifest.get("topdown_map"):
            metrics.topdown_map_path = str(manifest["topdown_map"])
        if manifest.get("diagnostics_manifest"):
            metrics.diagnostics_manifest_path = str(manifest["diagnostics_manifest"])

    metrics.debug_bundle_dir = str(episode_dir)
    (episode_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_dir


def save_error_episode_bundle(*, run_tag: str, metrics: EpisodeMetrics) -> Path:
    """Minimal bundle when the episode crashes before an agent exists."""
    episode_dir = default_episodes_root() / run_tag / f"q{metrics.question_id:04d}_{metrics.method}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    if metrics.error:
        (episode_dir / "error.txt").write_text(metrics.error, encoding="utf-8")
    metrics.debug_bundle_dir = str(episode_dir)
    (episode_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_dir
