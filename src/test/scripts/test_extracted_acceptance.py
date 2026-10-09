# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Offline acceptance-driver guards; no simulator or model startup."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "scripts/run_extracted_acceptance.sh"


def test_driver_syntax_and_invalid_phase(tmp_path):
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    result = subprocess.run(["bash", str(DRIVER), "invalid"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Usage:" in result.stderr


@pytest.mark.parametrize("robot", ["stretch", "rby1"])
def test_acceptance_config_survives_runtime_profile(monkeypatch, robot):
    from emet.core.parameters import get_parameters
    from emet.eval.benchmark_dynagraph import enable_query_driven_memory
    from emet.eval.ovmm_find_phase import apply_backend_parameters

    monkeypatch.setenv("EMET_CONFIG", str(ROOT / "configs/emet/shared_find_acceptance.yaml"))
    params = apply_backend_parameters(get_parameters("dynav_config.yaml", robot=robot), "lazy_graph")
    enable_query_driven_memory(params, "lazy_graph")
    assert params.get("query_driven_memory") is True
    assert params.get("query_memory")["grounding_backend"] == "vlm"
    assert params.get("query_memory")["mask_backend"] == "yoloe_sam2"
    assert params.get("eqa")["collect_agentic_trace"] is True
    assert params.get("eqa")["vl_hf_model_id"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_every_inventory_file_has_a_disposition():
    inventory = json.loads((ROOT / "docs/plans/shared_stack_inventory.json").read_text())
    allowed = {
        "already_main",
        "superseded_by_main",
        "represented_in_stack",
        "deferred",
        "partial_extraction_residual_deferred",
    }
    for branch in inventory["branches"].values():
        assert len(branch["source_sha"]) == 40
        paths = [row["path"] for row in branch["files"]]
        assert len(paths) == len(set(paths))
        for row in branch["files"]:
            assert row["status"] in allowed
            if "deferred" in row["status"] or row["status"] == "superseded_by_main":
                assert row["residual"]
            if row["status"] == "represented_in_stack":
                assert row["source_blob"] == row["stack_blob"]
