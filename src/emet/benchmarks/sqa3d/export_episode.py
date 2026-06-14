# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Persist SQA3D episode artifacts for analysis and model improvement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emet.benchmarks.sqa3d.datasets import SQA3DQuestion


def episode_export_dir(export_root: Path, question_id: int, method: str = "") -> Path:
    suffix = f"sqa3d_{int(question_id)}_{method}" if method else str(int(question_id))
    return Path(export_root) / suffix


def     export_sqa3d_episode_artifacts(
    agent: Any,
    q: SQA3DQuestion,
    *,
    method: str,
    profile: str,
    replay_mode: str,
    replay_meta: dict[str, Any],
    predicted: str,
    raw_eqa: str,
    model_confident: bool,
    planning_steps: int,
    export_root: Path,
    split: str = "",
    infra_failure: bool = False,
    recorder: Any | None = None,
) -> Path:
    """Write per-question export: metadata JSON + memory backend snapshot."""
    ep_dir = episode_export_dir(export_root, q.question_id, method=method)
    ep_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": "sqa3d",
        "split": split,
        "question_id": q.question_id,
        "scene_id": q.scene_id,
        "method": method,
        "profile": profile,
        "replay_mode": replay_mode,
        "question": q.question,
        "situation": q.situation,
        "formatted_prompt": q.formatted_prompt(),
        "gold_answers": list(q.answers),
        "predicted_answer": predicted,
        "raw_eqa_output": raw_eqa,
        "model_confident": model_confident,
        "planning_steps": planning_steps,
        "infra_failure": infra_failure,
        "question_type": q.question_type,
        "answer_type": q.answer_type,
        "position": list(q.position),
        "rotation_xyzw": list(q.rotation_xyzw),
        **replay_meta,
    }
    (ep_dir / "episode_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    memory_dir = ep_dir / "memory"
    try:
        if method == "dynagraph":
            graph_memory = getattr(agent, "graph_memory", None)
            voxel_map = getattr(agent, "voxel_map", None)
            if graph_memory is not None and voxel_map is not None:
                from emet.memory.headless_export import export_graph_eqa_dir

                export_graph_eqa_dir(
                    graph_memory,
                    voxel_map,
                    str(memory_dir),
                    title=f"SQA3D {q.question_id}",
                )
        elif method == "dynamem":
            voxel_map = getattr(agent, "voxel_map", None)
            if voxel_map is not None:
                from emet.memory.backend import get_memory_backend

                get_memory_backend("dynamem", voxel_map=voxel_map).save(str(memory_dir))
    except Exception as exc:
        (ep_dir / "export_error.txt").write_text(str(exc), encoding="utf-8")

    try:
        from emet.eval.episode_diagnostics import EpisodeDiagnosticsConfig, flush_episode_diagnostics

        manifest = flush_episode_diagnostics(ep_dir, agent, recorder, cfg=EpisodeDiagnosticsConfig.from_env())
        if manifest:
            meta["diagnostics_manifest"] = manifest.get("diagnostics_manifest", "")
            meta["topdown_map_path"] = manifest.get("topdown_map", "")
            (ep_dir / "episode_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:
        (ep_dir / "diagnostics_error.txt").write_text(str(exc), encoding="utf-8")

    return ep_dir
