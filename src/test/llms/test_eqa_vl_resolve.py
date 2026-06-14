# Copyright (c) Hello Robot, Inc. All rights reserved.

"""Unit tests for VRAM-tier Gemma checkpoint resolution (no model load)."""

from __future__ import annotations

from unittest.mock import patch

from emet.llms.eqa_vl_settings import resolve_vl_hf_model_id


def test_resolve_explicit_hf_id_wins():
    mid = resolve_vl_hf_model_id(
        "gemma4",
        {},
        device="cuda",
        explicit_hf_id="google/gemma-3-4b-it",
    )
    assert mid == "google/gemma-3-4b-it"


def test_resolve_gemma_e4b_when_vram_high():
    with patch("emet.llms.eqa_vl_settings.get_nvidia_gpu_free_mib", return_value=20000.0):
        mid = resolve_vl_hf_model_id("gemma4", {}, device="cuda")
    assert mid == "google/gemma-4-e2b-it"


def test_resolve_gemma_e4b_opt_in():
    with patch("emet.llms.eqa_vl_settings.get_nvidia_gpu_free_mib", return_value=22000.0):
        import os

        os.environ["EMET_EQA_GEMMA_E4B"] = "1"
        try:
            mid = resolve_vl_hf_model_id("gemma4", {}, device="cuda")
        finally:
            os.environ.pop("EMET_EQA_GEMMA_E4B", None)
    assert mid == "google/gemma-4-E4B-it"


def test_resolve_gemma_e2b_mid_tier():
    with patch("emet.llms.eqa_vl_settings.get_nvidia_gpu_free_mib", return_value=9000.0):
        mid = resolve_vl_hf_model_id("gemma4", {}, device="cuda")
    assert mid == "google/gemma-4-e2b-it"


def test_resolve_gemma_3_4b_low_tier():
    with patch("emet.llms.eqa_vl_settings.get_nvidia_gpu_free_mib", return_value=5000.0):
        mid = resolve_vl_hf_model_id("gemma4", {}, device="cuda")
    assert mid == "google/gemma-3-4b-it"


def test_resolve_qwen3_vl_registry_default_8b():
    mid = resolve_vl_hf_model_id("qwen3_vl", {}, device="cuda")
    assert mid == "Qwen/Qwen3-VL-8B-Instruct"


def test_resolve_qwen25_ignores_qwen3_config_id():
    mid = resolve_vl_hf_model_id(
        "qwen2_5_vl",
        {
            "eqa": {
                "vl_family": "qwen2_5_vl",
                "vl_hf_model_id": "Qwen/Qwen3-VL-4B-Instruct",
            }
        },
        device="cuda",
    )
    assert mid == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_resolve_config_hf_id():
    mid = resolve_vl_hf_model_id(
        "gemma4",
        {"eqa": {"vl_family": "gemma4", "vl_hf_model_id": "google/gemma-4-e2b-it"}},
        device="cuda",
    )
    assert mid == "google/gemma-4-e2b-it"


def test_resolve_ignores_mismatched_config_hf_id():
    with patch("emet.llms.eqa_vl_settings.get_nvidia_gpu_free_mib", return_value=20000.0):
        mid = resolve_vl_hf_model_id(
            "gemma4",
            {
                "eqa": {
                    "vl_family": "gemma4",
                    "vl_hf_model_id": "Qwen/Qwen3-VL-4B-Instruct",
                }
            },
            device="cuda",
        )
    assert mid == "google/gemma-4-e2b-it"
