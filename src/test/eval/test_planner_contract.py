# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from emet.eval.planner_contract import answer_before_action_schema, clarify_planner_schema


def schema():
    return {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "string", "enum": ["object_1", "frontier_1"]}},
            "answer": {"$ref": "#/$defs/Answer"},
            "scene_graph_description": {"type": "string"},
            "image_description": {"type": "string"},
        },
        "$defs": {
            "Answer": {
                "type": "object",
                "properties": {
                    "explanation_ans": {"type": "string"},
                    "explanation_conf": {"type": "string"},
                    "is_confident": {"type": "boolean"},
                    "answer": {"enum": ["red", "blue"]},
                },
            }
        },
    }


def without_descriptions(value):
    if isinstance(value, dict):
        return {key: without_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [without_descriptions(item) for item in value]
    return value


def test_annotations_change_descriptions_only_without_mutating_original():
    original = schema()
    before = copy.deepcopy(original)
    clarified = clarify_planner_schema(original)
    assert original == before and clarified != original
    assert without_descriptions(clarified) == without_descriptions(original)
    data = {"steps": [], "answer": {"answer": "blue", "is_confident": False}}
    assert Draft202012Validator(original).is_valid(data)
    assert Draft202012Validator(clarified).is_valid(data)


def test_answer_first_reorders_fields_without_adding_instructions_or_constraints():
    original = schema()
    original["required"] = list(original["properties"])
    before = copy.deepcopy(original)
    changed = answer_before_action_schema(original)
    assert original == before
    assert list(changed["properties"])[:2] == ["answer", "steps"]
    assert changed["required"][:2] == ["answer", "steps"]
    changed["required"] = original["required"]
    assert changed == original  # Dictionary key ordering is the only other delta.


def driver(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("planner_probe", root / "scripts/probe_grapheqa_planner_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_keeps_images_history_model_budget_and_original_request_exact(monkeypatch):
    probe = driver(monkeypatch)
    request = {
        "model": "qwen",
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": "Return the ROOT schema\n" + json.dumps(schema())},
            {"role": "assistant", "content": "earlier action"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        ],
    }
    original, _ = probe.request_variant(request, "original")
    assert original == request
    clarified, _ = probe.request_variant(request, "clarified")
    assert clarified["messages"][1:] == request["messages"][1:]
    assert clarified["model"] == request["model"] and clarified["max_tokens"] == 2048
    assert original == request


def test_failed_prior_model_reply_is_not_excluded_from_replay(tmp_path, monkeypatch):
    probe = driver(monkeypatch)
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"case_id": "failed-case", "condition": "reference_graph", "complete": False}) + "\n"
    )
    request = tmp_path / "request-0000"
    request.mkdir()
    (request / "model_calls.jsonl").write_text(json.dumps({"request": {"model": "qwen"}, "validated": False}) + "\n")
    assert probe.read_requests(tmp_path)[0][:2] == ("failed-case", {"model": "qwen"})


def test_wrong_response_shape_does_not_get_partially_annotated():
    with pytest.raises(ValueError, match="reference planner"):
        clarify_planner_schema({"type": "object", "properties": {"answer": {}}})
