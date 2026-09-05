# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
from types import SimpleNamespace

import pytest

from emet.mapping.voxel.dynamem_eqa import DynamemVoxelEQAMixin


@pytest.mark.parametrize("action", ["", "bad", "0", "2", None])
def test_json_answer_survives_invalid_navigation_action(tmp_path, monkeypatch, action):
    """A valid uncertain answer must not become a swallowed int(action) error."""
    vm = DynamemVoxelEQAMixin()
    vm.log = str(tmp_path)
    vm.image_descriptions = []
    vm.relevant_objects = []
    vm.history_outputs = []
    vm.parameters = SimpleNamespace(get=lambda *args: None)
    vm.eqa_client = object()
    vm._eqa_max_tokens = 128
    vm.extract_relevant_objects = lambda question: None
    vm.get_image_descriptions_str = lambda *args: ([1], "One description")
    reply = json.dumps(
        {
            "reasoning": "No grill visible",
            "answer": "Unknown",
            "confidence": False,
            "action": action,
            "confidence_reasoning": "Need another view",
        }
    )
    monkeypatch.setattr("emet.mapping.voxel.dynamem_eqa.dynamem_vllm_call", lambda *a, **kw: reply)
    monkeypatch.setattr("emet.llms.graph_eqa_vlm._eqa_system_prompt", lambda params: "")
    result = vm.query_answer("Is the grill covered?", None, None)
    assert result == ("No grill visible", "Unknown", False, "Need another view", None, [])
    assert vm._last_eqa_raw == reply


def test_json_confident_answer_and_legacy_text(tmp_path):
    vm = DynamemVoxelEQAMixin()
    vm.log = str(tmp_path)
    vm.image_descriptions = []
    assert vm.parse_answer('{"answer":"Yes","confidence":true}') == ("", "Yes", True, "", "")
    assert vm.parse_answer("Reasoning: visible Answer: Yes Confidence: true Action: Confidence_reasoning: clear") == (
        "visible",
        "yes",
        True,
        "",
        "clear",
    )
