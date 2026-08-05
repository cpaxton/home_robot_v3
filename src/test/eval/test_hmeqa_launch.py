# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for ``emet hmeqa h2h`` job env construction (remote VL)."""

from __future__ import annotations

from emet.eval.hmeqa_launch import (
    hmeqa_h2h_env_parts,
    hmeqa_h2h_vl_endpoint_from_env_parts,
    normalize_hmeqa_vl_endpoint,
)


def test_normalize_hmeqa_vl_endpoint_variants():
    assert normalize_hmeqa_vl_endpoint("openai@http://caliban:8000/v1") == (
        "openai@http://caliban:8000/v1"
    )
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
