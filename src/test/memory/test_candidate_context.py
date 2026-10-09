# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from audit_candidate_context import selected_id, verify

from emet.memory.surface_candidates import surface_candidates


def test_blind_identification_does_not_receive_target_and_rejects_mixed_support():
    rgb = np.full((30, 30, 3), 120, dtype=np.uint8)
    regions = surface_candidates(np.ones((30, 30)), [0, 0, 1000, 1000], min_depth=0.25, max_depth=4)
    client = Mock(
        side_effect=[
            json.dumps({"candidates": [{"id": 0, "identity": "mug", "unambiguous": True, "mixed": True}]}),
            '{"selected_id":0,"target_unambiguous":true}',
        ]
    )
    chosen, requests, panels = verify(client, rgb, regions, "paper towel", "blind_context")
    assert chosen is None
    assert "paper towel" not in requests[0]["prompt"]
    assert "paper towel" in requests[1]["prompt"]
    assert requests[1]["image_count"] == 0
    assert len(panels) == 2


def test_selection_is_strict_and_ambiguous_identity_cannot_be_overridden():
    regions = [{"id": 0}]
    assert selected_id({"selected_id": True, "target_unambiguous": True}, regions) is None
    assert selected_id({"selected_id": 2, "target_unambiguous": True}, regions) is None
    assert selected_id({"selected_id": 0, "target_unambiguous": True}, regions, []) is None
    assert selected_id({"selected_id": 0, "target_unambiguous": True}, regions) == 0


def test_missing_candidates_do_not_call_model():
    client = Mock()
    assert verify(client, np.zeros((20, 20, 3), dtype=np.uint8), [], "cup", "context")[0] is None
    client.assert_not_called()


def test_support_only_verifier_never_receives_original_scene():
    rgb = np.full((30, 30, 3), 120, dtype=np.uint8)
    regions = surface_candidates(np.ones((30, 30)), [0, 0, 1000, 1000], min_depth=0.25, max_depth=4)
    client = Mock(return_value='{"selected_id":null,"target_unambiguous":false}')
    chosen, requests, panels = verify(client, rgb, regions, "cup", "support_only")
    assert chosen is None
    assert requests[0]["image_count"] == len(regions)
    assert client.call_args.args[0][1:] == panels
    assert all(panel.size != (30, 30) for panel in panels)


def test_best_local_runner_freezes_config_and_pairs_only_requested_verifiers(tmp_path, monkeypatch):
    import run_grounding_ablation

    dataset = tmp_path / "cache"
    dataset.mkdir()
    (dataset / "manifest.yaml").write_text("[]")
    (dataset / "truth.json").write_text("[]")
    output = tmp_path / "results"
    preset = Path(__file__).resolve().parents[3] / "configs/eval/grounding_best_local.yaml"
    calls = Mock()
    monkeypatch.setattr(run_grounding_ablation.subprocess, "run", calls)
    monkeypatch.setattr(
        sys, "argv", ["run", "--datasets", str(dataset), "--output-dir", str(output), "--preset", str(preset)]
    )
    run_grounding_ablation.main()
    assert calls.call_count == 4
    proposal = calls.call_args_list[0].args[0]
    verification = calls.call_args_list[1].args[0]
    assert proposal[proposal.index("--box-ablation") + 1] == "whole_object"
    assert verification[verification.index("--variants") + 1 :] == ["isolated", "context"]
    assert "blind_context" not in verification
    params = json.loads((output / "resolved_parameters.json").read_text())
    assert params["eqa"]["vl_quantization"] == "int4"
    assert params["eqa"]["vl_hf_model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
