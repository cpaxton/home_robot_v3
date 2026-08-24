# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emet.eval import hmeqa_completion as completion
from emet.eval.hmeqa_launch import (
    DEFAULT_HMEQA_ARTIFACT_PROFILE,
    build_hmeqa_run_config,
    prepare_hmeqa_run_manifest,
)


def _minimal_artifact_profile() -> dict[str, bool | int | float]:
    profile = dict(DEFAULT_HMEQA_ARTIFACT_PROFILE)
    for key, value in tuple(profile.items()):
        if isinstance(value, bool):
            profile[key] = False
    profile["snapshot_rgb_frames"] = 0
    return profile


def _prepare_run(
    tmp_path: Path,
    *,
    arms: str = "classic",
    ids: str = "7",
    artifact_profile: dict[str, bool | int | float] | None = None,
    decision_policy: str = "legacy",
    action_progress_mode: str = "off",
) -> tuple[Path, dict]:
    out = tmp_path / "out"
    data_dir = tmp_path / "data"
    hm3d = tmp_path / "hm3d"
    data_dir.mkdir()
    hm3d.mkdir()
    config = build_hmeqa_run_config(
        arms=arms,
        ids=ids,
        agentic_verifier="none",
        require_verified=False,
        agentic_router=False,
        decision_policy=decision_policy,
        action_progress_mode=action_progress_mode,
        data_dir=data_dir,
        hm3d_root=hm3d,
        artifact_profile=artifact_profile or _minimal_artifact_profile(),
    )
    manifest = prepare_hmeqa_run_manifest(
        out,
        project_root=tmp_path,
        config=config,
        resume=False,
        git_state={
            "commit": "a" * 40,
            "dirty": False,
            "dirty_digest": None,
            "status": [],
        },
        external_inputs={
            "data_dir": str(data_dir.resolve()),
            "questions": {
                "path": str(data_dir / "questions.csv"),
                "sha256": "sha256:questions",
            },
            "scene_init_poses": {
                "path": str(data_dir / "scene_init_poses.csv"),
                "sha256": "sha256:poses",
            },
            "hm3d_root": str(hm3d.resolve()),
        },
    )
    return out, manifest


def _episode_row(source: Path, *, qid: int = 7, correct: bool = False) -> dict:
    return {
        "dataset": "hmeqa",
        "method": "dynagraph",
        "question_id": qid,
        "scene": "scene",
        "floor": 0,
        "question": "question",
        "gold_answer_letter": "D",
        "predicted_answer": "A",
        "correct": correct,
        "confident": False,
        "planning_steps": 3,
        "success": correct,
        "parsed_answer_letter": "A",
        "raw_eqa_output": "answer: A",
        "error": "",
        "debug_bundle_dir": str(source),
    }


def _write_source_bundle(source: Path, row: dict) -> None:
    source.mkdir(parents=True)
    (source / "metrics.json").write_text(json.dumps(row), encoding="utf-8")
    (source / "eqa_history.json").write_text(
        json.dumps({"iterations": ["answer: A"]}),
        encoding="utf-8",
    )
    (source / "raw_eqa.txt").write_text("answer: A\n", encoding="utf-8")
    (source / "scene_graph_report.txt").write_text("scene graph\n", encoding="utf-8")
    (source / "frontier_nodes.json").write_text("[]\n", encoding="utf-8")


def _pending(out: Path, row: dict, *, name: str = "candidate.jsonl") -> Path:
    path = out / ".pending" / name
    path.parent.mkdir()
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _valid_candidate(tmp_path: Path) -> tuple[Path, dict, Path]:
    out, manifest = _prepare_run(tmp_path)
    source = completion.expected_debug_bundle_dir(out, "classic", 7, manifest=manifest)
    row = _episode_row(source)
    _write_source_bundle(source, row)
    return out, row, _pending(out, row)


def test_wrong_answer_commits_and_marker_is_authoritative(tmp_path: Path) -> None:
    out, _row, pending = _valid_candidate(tmp_path)

    committed = completion.commit_pending_episode(
        out,
        arm="classic",
        qid=7,
        pending_path=pending,
        exit_code=0,
    )

    assert committed["row"]["correct"] is False
    assert committed["row"]["success"] is False
    assert committed["row"]["h2h_arm"] == "classic"
    assert completion.completed_unit_count(out) == 1
    assert completion.unit_is_complete(out, "classic", 7)
    assert not pending.exists()
    assert json.loads((out / "classic_q7.jsonl").read_text())["correct"] is False


def test_action_progress_run_requires_matching_summary_and_gate_trace(tmp_path: Path) -> None:
    out, manifest = _prepare_run(
        tmp_path,
        arms="agentic",
        decision_policy="grounded_v2",
        action_progress_mode="shadow",
    )
    source = completion.expected_debug_bundle_dir(out, "agentic", 7, manifest=manifest)
    row = _episode_row(source)
    _write_source_bundle(source, row)
    pending = _pending(out, row)

    with pytest.raises(completion.HmeqaCompletionError, match="agentic_summary.json"):
        completion.commit_pending_episode(
            out,
            arm="agentic",
            qid=7,
            pending_path=pending,
            exit_code=0,
        )

    (source / "agentic_summary.json").write_text(
        json.dumps(
            {
                "effective_state_contract": {
                    "decision_policy": "grounded_v2",
                    "action_progress_mode": "off",
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "agentic_trace.jsonl").write_text(
        json.dumps({"event": "router_call", "action_gate_decisions": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(completion.HmeqaCompletionError, match="does not match frozen"):
        completion.commit_pending_episode(
            out,
            arm="agentic",
            qid=7,
            pending_path=pending,
            exit_code=0,
        )

    (source / "agentic_summary.json").write_text(
        json.dumps(
            {
                "effective_state_contract": {
                    "decision_policy": "grounded_v2",
                    "action_progress_mode": "shadow",
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "agentic_trace.jsonl").write_text(
        json.dumps({"event": "router_call", "action_gate_decisions": [{}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(completion.HmeqaCompletionError, match="boolean allowed"):
        completion.commit_pending_episode(
            out,
            arm="agentic",
            qid=7,
            pending_path=pending,
            exit_code=0,
        )

    (source / "agentic_trace.jsonl").write_text(
        json.dumps({"event": "router_call", "action_gate_decisions": []}) + "\n",
        encoding="utf-8",
    )
    committed = completion.commit_pending_episode(
        out,
        arm="agentic",
        qid=7,
        pending_path=pending,
        exit_code=0,
    )
    assert committed["row"]["question_id"] == 7
    assert (out / "bundles" / "agentic_q7" / "agentic_trace.jsonl").is_file()


def test_existing_valid_marker_is_idempotent_and_never_replaced(tmp_path: Path) -> None:
    out, row, pending = _valid_candidate(tmp_path)
    first = completion.commit_pending_episode(
        out,
        arm="classic",
        qid=7,
        pending_path=pending,
        exit_code=0,
    )
    marker_path = out / "bundles" / "classic_q7" / "COMPLETE.json"
    marker_before = marker_path.read_bytes()
    retry = out / ".pending" / "retry.jsonl"
    retry.write_text(json.dumps(row) + "\n", encoding="utf-8")

    second = completion.commit_pending_episode(
        out,
        arm="classic",
        qid=7,
        pending_path=retry,
        exit_code=0,
    )

    assert second == first
    assert marker_path.read_bytes() == marker_before
    assert not retry.exists()


def test_frozen_object_crop_artifact_is_validated_and_copied(tmp_path: Path) -> None:
    profile = _minimal_artifact_profile()
    profile["export_object_crops"] = True
    out, manifest = _prepare_run(tmp_path, artifact_profile=profile)
    source = completion.expected_debug_bundle_dir(out, "classic", 7, manifest=manifest)
    row = _episode_row(source)
    _write_source_bundle(source, row)
    crops = source / "dynagraph" / "crops_mosaic.png"
    crops.parent.mkdir()
    crops.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    (source / "diagnostics_manifest.json").write_text(
        json.dumps({"object_crops_mosaic": str(crops)}),
        encoding="utf-8",
    )
    pending = _pending(out, row)

    completion.commit_pending_episode(
        out,
        arm="classic",
        qid=7,
        pending_path=pending,
        exit_code=0,
    )

    assert (out / "bundles" / "classic_q7" / "dynagraph" / "crops_mosaic.png").is_file()


def test_best_effort_object_crop_absence_does_not_block_completion(tmp_path: Path) -> None:
    profile = _minimal_artifact_profile()
    profile["export_object_crops"] = True
    out, manifest = _prepare_run(tmp_path, artifact_profile=profile)
    source = completion.expected_debug_bundle_dir(out, "classic", 7, manifest=manifest)
    row = _episode_row(source)
    _write_source_bundle(source, row)
    (source / "diagnostics_manifest.json").write_text(
        json.dumps({"episode_dir": str(source)}),
        encoding="utf-8",
    )
    pending = _pending(out, row)

    marker = completion.commit_pending_episode(
        out,
        arm="classic",
        qid=7,
        pending_path=pending,
        exit_code=0,
    )

    assert marker["row"]["question_id"] == 7
    assert completion.unit_is_complete(out, "classic", 7)


def test_partial_or_multiple_json_objects_never_commit(tmp_path: Path) -> None:
    out, row, pending = _valid_candidate(tmp_path)
    pending.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(completion.HmeqaCompletionError, match="one valid JSON value"):
        completion.commit_pending_episode(
            out,
            arm="classic",
            qid=7,
            pending_path=pending,
            exit_code=0,
        )

    assert completion.completed_unit_count(out) == 0
    assert not (out / "bundles" / "classic_q7" / "COMPLETE.json").exists()


def test_nonzero_child_with_valid_row_never_commits(tmp_path: Path) -> None:
    out, _row, pending = _valid_candidate(tmp_path)
    with pytest.raises(completion.HmeqaCompletionError, match="exited nonzero"):
        completion.commit_pending_episode(
            out,
            arm="classic",
            qid=7,
            pending_path=pending,
            exit_code=139,
        )
    assert pending.exists()
    assert completion.completed_unit_count(out) == 0


def test_missing_or_malformed_snapshot_fails_closed(tmp_path: Path) -> None:
    out, _row, pending = _valid_candidate(tmp_path)
    source = completion.expected_debug_bundle_dir(out, "classic", 7)
    (source / "frontier_nodes.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(completion.HmeqaCompletionError, match="must contain a JSON list"):
        completion.commit_pending_episode(
            out,
            arm="classic",
            qid=7,
            pending_path=pending,
            exit_code=0,
        )
    assert pending.exists()
    assert completion.completed_unit_count(out) == 0


def test_snapshot_copy_failure_leaves_pending_and_no_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _row, pending = _valid_candidate(tmp_path)

    def fail_copy(_source: Path, _stage: Path, _profile: dict) -> None:
        raise OSError("injected snapshot failure")

    monkeypatch.setattr(completion, "_copy_snapshot", fail_copy)
    with pytest.raises(OSError, match="injected snapshot failure"):
        completion.commit_pending_episode(
            out,
            arm="classic",
            qid=7,
            pending_path=pending,
            exit_code=0,
        )
    assert pending.exists()
    assert not (out / "bundles" / "classic_q7").exists()


def test_debug_bundle_symlink_escape_is_rejected(tmp_path: Path) -> None:
    out, manifest = _prepare_run(tmp_path)
    expected = completion.expected_debug_bundle_dir(out, "classic", 7, manifest=manifest)
    outside = tmp_path / "outside"
    row = _episode_row(expected)
    _write_source_bundle(outside, row)
    expected.parent.mkdir(parents=True)
    expected.symlink_to(outside, target_is_directory=True)
    pending = _pending(out, row)

    with pytest.raises(completion.HmeqaCompletionError, match="symlink"):
        completion.commit_pending_episode(
            out,
            arm="classic",
            qid=7,
            pending_path=pending,
            exit_code=0,
        )


def test_done_is_json_hash_bound_and_corruption_does_not_skip(tmp_path: Path) -> None:
    out, _row, pending = _valid_candidate(tmp_path)
    completion.commit_pending_episode(
        out,
        arm="classic",
        qid=7,
        pending_path=pending,
        exit_code=0,
    )
    done = completion.finalize_run(out)

    assert done["unit_count"] == 1
    assert completion.validate_done(out)
    (out / "DONE").write_text("{broken", encoding="utf-8")
    assert completion.validate_done(out) is False


def test_cpu_only_reconciliation_rebuilds_rows_aggregates_and_done(tmp_path: Path) -> None:
    out, _row, pending = _valid_candidate(tmp_path)
    completion.commit_pending_episode(
        out,
        arm="classic",
        qid=7,
        pending_path=pending,
        exit_code=0,
    )
    (out / "classic_q7.jsonl").unlink()
    (out / "classic.jsonl").unlink(missing_ok=True)
    (out / "DONE").write_text("", encoding="utf-8")

    reconciled = completion.reconcile_run(out)

    assert reconciled["completed"] == reconciled["total"] == 1
    assert reconciled["done"] is True
    assert (out / "classic_q7.jsonl").is_file()
    assert (out / "classic.jsonl").is_file()
    assert completion.validate_done(out)


def test_nonempty_legacy_row_without_marker_is_not_counted(tmp_path: Path) -> None:
    out, _manifest = _prepare_run(tmp_path)
    (out / "classic_q7.jsonl").write_text('{"question_id":7}\n', encoding="utf-8")
    assert completion.completed_unit_count(out) == 0
    assert completion.has_resume_state(out)
