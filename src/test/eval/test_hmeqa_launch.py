# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for ``emet hmeqa h2h`` job env construction (remote VL)."""

from __future__ import annotations

from copy import deepcopy

import pytest

from emet.eval.hmeqa_launch import (
    HMEQA_RUN_MANIFEST_VERSION,
    HmeqaRunManifestError,
    build_hmeqa_run_config,
    hmeqa_h2h_env_parts,
    hmeqa_h2h_vl_endpoint_from_env_parts,
    hmeqa_run_config_digest,
    hmeqa_run_config_from_env,
    normalize_hmeqa_vl_endpoint,
    prepare_hmeqa_run_manifest,
    prepare_hmeqa_run_manifest_from_env,
)


def test_normalize_hmeqa_vl_endpoint_variants():
    assert normalize_hmeqa_vl_endpoint("openai@http://caliban:8000/v1") == ("openai@http://caliban:8000/v1")
    assert normalize_hmeqa_vl_endpoint("http://caliban:8000/v1") == "openai@http://caliban:8000/v1"
    assert normalize_hmeqa_vl_endpoint("caliban") == "openai@http://caliban:8000/v1"
    assert normalize_hmeqa_vl_endpoint("caliban:8001") == "openai@http://caliban:8001/v1"


def test_hmeqa_h2h_env_parts_host_caliban_injects_vl_endpoint():
    parts = hmeqa_h2h_env_parts(
        arms="classic",
        ids="15,56,65,68",
        coverage_qids="15,28,47",
        cooldown=30,
        crash_policy="skip",
        streak_abort=2,
        agentic_verifier="none",
        require_verified=False,
        agentic_router=False,
        host="caliban",
        eqa_hf_model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    joined = " ".join(parts)
    assert "EMET_LLM_HOST=caliban" in joined or "EMET_LLM_HOST='caliban'" in joined
    assert "EMET_OPENAI_BASE_URL=" in joined
    assert hmeqa_h2h_vl_endpoint_from_env_parts(parts) == "openai@http://caliban:8000/v1"
    # Remote VL: do not force local HF weights into the Habitat child.
    assert "EQA_HF_MODEL_ID=" not in joined


def test_hmeqa_h2h_env_parts_vl_endpoint_wins_over_host():
    parts = hmeqa_h2h_env_parts(
        arms="classic",
        ids="15",
        coverage_qids="15",
        cooldown=30,
        crash_policy="skip",
        streak_abort=2,
        agentic_verifier="none",
        require_verified=False,
        agentic_router=False,
        host="caliban",
        vl_endpoint="openai@http://caliban:8001/v1",
    )
    assert hmeqa_h2h_vl_endpoint_from_env_parts(parts) == "openai@http://caliban:8001/v1"


def test_hmeqa_h2h_env_parts_local_keeps_hf_model_id():
    parts = hmeqa_h2h_env_parts(
        arms="classic",
        ids="15",
        coverage_qids="15",
        cooldown=30,
        crash_policy="skip",
        streak_abort=2,
        agentic_verifier="none",
        require_verified=False,
        agentic_router=False,
        eqa_hf_model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    joined = " ".join(parts)
    assert "EQA_HF_MODEL_ID=" in joined
    assert hmeqa_h2h_vl_endpoint_from_env_parts(parts) is None
    assert "EMET_EQA_ROOM_STAMP_INVESTIGATE=1" not in joined
    assert "EMET_EQA_ATTEMPT_LEDGER=1" not in joined


def test_hmeqa_h2h_env_parts_agentic_router_does_not_force_room_stamp():
    """Paper-router must not auto-enable investigate stamps (known letter regression)."""
    parts = hmeqa_h2h_env_parts(
        arms="agentic",
        ids="2,104",
        coverage_qids="15,28,47",
        cooldown=20,
        crash_policy="skip",
        streak_abort=2,
        agentic_verifier="none",
        require_verified=True,
        agentic_router=True,
    )
    joined = " ".join(parts)
    assert "EMET_EQA_AGENTIC_ROUTER=1" in joined
    assert "EMET_EQA_ROOM_STAMP_INVESTIGATE=1" not in joined
    assert "EMET_EQA_ATTEMPT_LEDGER=1" not in joined
    assert "EMET_EQA_AGENTIC_DECISION_POLICY=legacy" in joined
    assert "EMET_EQA_GRAPH_EVIDENCE_MODE=off" in joined
    assert "EMET_EQA_ROOM_HISTORY_MODE=off" in joined
    assert "EMET_EQA_ATTEMPT_LEDGER_MODE=off" in joined


def test_hmeqa_h2h_env_parts_translates_explicit_variant_axes():
    parts = hmeqa_h2h_env_parts(
        arms="agentic",
        ids="2,104",
        coverage_qids="2,104",
        cooldown=20,
        crash_policy="skip",
        streak_abort=2,
        agentic_verifier="none",
        require_verified=False,
        agentic_router=True,
        decision_policy="grounded_v2",
        graph_evidence_mode="shadow",
        room_history_mode="agent",
        room_policy="llm",
        room_target_hints=False,
        investigate_stamp=True,
        attempt_ledger_mode="shadow",
        variant_id="grounded-shadow-r1",
    )
    joined = " ".join(parts)
    assert "EMET_EQA_AGENTIC_DECISION_POLICY=grounded_v2" in joined
    assert "EMET_EQA_GRAPH_EVIDENCE_MODE=shadow" in joined
    assert "EMET_EQA_ROOM_HISTORY_MODE=agent" in joined
    assert "EMET_EQA_ROOM_POLICY=llm" in joined
    assert "EMET_EQA_ROOM_TARGET_HINTS=0" in joined
    assert "EMET_EQA_ROOM_STAMP_INVESTIGATE=1" in joined
    assert "EMET_EQA_ATTEMPT_LEDGER_MODE=shadow" in joined
    assert "EMET_EQA_ATTEMPT_LEDGER=1" in joined
    assert "EMET_HMEQA_VARIANT_ID=grounded-shadow-r1" in joined


def test_direct_script_defaults_preserve_legacy_policy():
    config = hmeqa_run_config_from_env({})
    assert config["evaluation"]["require_verified"] is False
    assert config["evaluation"]["agentic_router"] is False
    assert config["variant"] == {
        "id": "legacy",
        "agentic_decision_policy": "legacy",
        "graph_evidence_mode": "off",
        "room_history_mode": "off",
        "room_policy": "canonical",
        "room_target_hints": True,
        "investigate_stamp": False,
        "attempt_ledger_mode": "off",
    }


def _treatment_config():
    return build_hmeqa_run_config(
        arms="agentic",
        ids="2,104",
        agentic_verifier="none",
        require_verified=False,
        agentic_router=True,
        decision_policy="grounded_v2",
        graph_evidence_mode="shadow",
        room_history_mode="agent",
        room_policy="llm",
        room_target_hints=False,
        investigate_stamp=True,
        attempt_ledger_mode="shadow",
        variant_id="grounded-shadow-r1",
        eqa_hf_model_id="Qwen/Qwen3-VL-8B-Instruct",
        eqa_answer_max_new_tokens=512,
        episode_timeout_seconds=3600,
        max_planning_steps=12,
        max_movement_step=6,
    )


def _git_state(commit: str = "a" * 40):
    return {
        "commit": commit,
        "dirty": False,
        "dirty_digest": None,
        "status": [],
    }


def test_hmeqa_run_config_digest_is_deterministic():
    config = _treatment_config()
    reordered = {key: deepcopy(config[key]) for key in reversed(config)}
    assert hmeqa_run_config_digest(config) == hmeqa_run_config_digest(reordered)


def test_hmeqa_run_manifest_freezes_config_and_refuses_mismatch(tmp_path):
    config = _treatment_config()
    manifest = prepare_hmeqa_run_manifest(
        tmp_path,
        project_root=tmp_path,
        config=config,
        sources={"variant.id": "command_line"},
        resume=False,
        git_state=_git_state(),
    )
    assert manifest["schema_version"] == HMEQA_RUN_MANIFEST_VERSION
    assert manifest["git"]["commit"] == "a" * 40
    assert manifest["variant"]["id"] == "grounded-shadow-r1"
    assert manifest["budgets"]["max_planning_steps"] == 12
    assert manifest["ids"]["question_ids"] == [2, 104]
    assert manifest["config_digest"] == hmeqa_run_config_digest(config)

    resumed = prepare_hmeqa_run_manifest(
        tmp_path,
        project_root=tmp_path,
        config=config,
        resume=True,
        git_state=_git_state(),
    )
    assert resumed == manifest

    changed = deepcopy(config)
    changed["variant"]["room_history_mode"] = "off"
    with pytest.raises(HmeqaRunManifestError, match="config digest"):
        prepare_hmeqa_run_manifest(
            tmp_path,
            project_root=tmp_path,
            config=changed,
            resume=True,
            git_state=_git_state(),
        )


def test_hmeqa_run_manifest_refuses_commit_or_dirty_state_mismatch(tmp_path):
    config = _treatment_config()
    prepare_hmeqa_run_manifest(
        tmp_path,
        project_root=tmp_path,
        config=config,
        resume=False,
        git_state=_git_state(),
    )
    with pytest.raises(HmeqaRunManifestError, match="git commit"):
        prepare_hmeqa_run_manifest(
            tmp_path,
            project_root=tmp_path,
            config=config,
            resume=True,
            git_state=_git_state("b" * 40),
        )
    with pytest.raises(HmeqaRunManifestError, match="git dirty"):
        prepare_hmeqa_run_manifest(
            tmp_path,
            project_root=tmp_path,
            config=config,
            resume=True,
            git_state={
                "commit": "a" * 40,
                "dirty": True,
                "dirty_digest": "sha256:different",
                "status": [" M src/example.py"],
            },
        )


def test_direct_script_resume_reuses_frozen_variant(monkeypatch, tmp_path):
    config = _treatment_config()
    git_state = _git_state()
    prepare_hmeqa_run_manifest(
        tmp_path,
        project_root=tmp_path,
        config=config,
        resume=False,
        git_state=git_state,
    )
    monkeypatch.setattr(
        "emet.eval.hmeqa_launch.hmeqa_git_state",
        lambda _project_root: git_state,
    )
    manifest = prepare_hmeqa_run_manifest_from_env(
        tmp_path,
        project_root=tmp_path,
        env={},
        resume=True,
    )
    assert manifest["config"]["variant"]["agentic_decision_policy"] == "grounded_v2"
    assert manifest["config"]["variant"]["room_history_mode"] == "agent"
    assert manifest["config"]["ids"]["question_ids"] == [2, 104]


def test_direct_script_resume_allows_empty_new_overnight_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "emet.eval.hmeqa_launch.hmeqa_git_state",
        lambda _project_root: _git_state(),
    )
    manifest = prepare_hmeqa_run_manifest_from_env(
        tmp_path,
        project_root=tmp_path,
        env={"HOLDOUT_IDS": "15,56", "ARMS": "classic,agentic"},
        resume=True,
    )
    assert manifest["ids"]["question_ids"] == [15, 56]
    assert manifest["sources"]["ids.question_ids"] == "environment:HOLDOUT_IDS"
    assert (tmp_path / "run_manifest.json").is_file()


def test_direct_script_resume_refuses_unfrozen_historical_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "emet.eval.hmeqa_launch.hmeqa_git_state",
        lambda _project_root: _git_state(),
    )
    (tmp_path / "classic_q15.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(HmeqaRunManifestError, match="run artifacts exist"):
        prepare_hmeqa_run_manifest_from_env(
            tmp_path,
            project_root=tmp_path,
            env={},
            resume=True,
        )
