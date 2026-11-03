# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import OpenAI
from PIL import Image

from emet.eval.frozen_eqa import FrozenServeState, PinnedPNGClient, run_emet_packet, semantic_choice
from emet.llms.openai_server import LLMServeState


def packet(tmp_path):
    images = []
    for index, color in enumerate(("red", "blue", "green", "yellow")):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), color).save(path)
        images.append({"path": path, "label": "CURRENT" if index == 0 else f"HISTORY {index}"})
    return {"question": "What color is the rug?", "choices": ["White", "Black"], "images": images}


def backend(tmp_path, monkeypatch, case, *, finish="stop"):
    state = FrozenServeState(llm_key="qwen3-vl-eqa", device="cpu", max_tokens=2048)
    state.begin_arm(tmp_path, case["images"], 2048)

    def complete(self, body):
        if body["max_tokens"] == 64:
            content = "rug"
        else:
            assert body["messages"][-1] == {"role": "assistant", "content": '{"reasoning":'}
            # Actual generation after the production assistant prefill is a suffix.
            content = '"Visible rug", "answer":"Black", "confidence":true, "action":"", "confidence_reasoning":"RGB"}'
        return {
            "id": "fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen3-vl-eqa",
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        }

    monkeypatch.setattr(LLMServeState, "complete", complete)

    def handle(request):
        return httpx.Response(200, json=state.complete(json.loads(request.content)))

    monkeypatch.setattr(
        "emet.eval.frozen_eqa.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle))),
    )
    return state


def test_production_answer_path_preserves_pngs_prefill_budget_and_final_answer(tmp_path, monkeypatch):
    case = packet(tmp_path)
    state = backend(tmp_path, monkeypatch, case)
    result = run_emet_packet(case, base_url="http://fixture/v1", model="qwen3-vl-eqa", api_key="private-test-key")
    state.validate_arm()
    assert result["answer"] == "Black" and result["is_confident"]
    assert [record["request"]["max_tokens"] for record in state.records] == [64, 1984]
    assert state.requested_tokens == 2048
    assert state.records[0]["image_sha256"] == []
    assert state.records[1]["image_sha256"] == state.expected_images
    assert "private-test-key" not in (tmp_path / "transport_calls.jsonl").read_text()


def test_truncated_reply_is_failure_even_when_production_parser_finds_an_answer(tmp_path, monkeypatch):
    case = packet(tmp_path)
    state = backend(tmp_path, monkeypatch, case, finish="length")
    result = run_emet_packet(case, base_url="http://fixture/v1", model="qwen3-vl-eqa", api_key="test")
    assert result["answer"] == "Black"
    with pytest.raises(ValueError, match="Incomplete"):
        state.validate_arm()


def test_unlisted_crop_and_excess_hidden_inference_fail_before_inference(tmp_path, monkeypatch):
    case = packet(tmp_path)
    state = backend(tmp_path, monkeypatch, case)
    client = PinnedPNGClient(case["images"], base_url="http://fixture/v1", model="qwen3-vl-eqa", api_key="test")
    with pytest.raises(ValueError, match="outside"):
        client._rgb_to_jpeg_data_url(np.zeros((8, 8, 3), dtype=np.uint8))
    state.requested_tokens = 2048
    with pytest.raises(ValueError, match="allowance"):
        state.complete({"max_tokens": 1, "messages": []})
    assert "response" not in state.records[-1]


def test_exact_semantic_scoring_does_not_extract_or_replace_answers():
    choices = ["Dark blue", "Black", "Gray", "White"]
    assert semantic_choice("Dark blue", choices) == "A"
    assert semantic_choice("B", choices) == "B"
    assert semantic_choice("B. Black", choices) is None
    assert semantic_choice("Unknown", choices) is None
    assert semantic_choice("red", ["Red", "red"]) is None


def driver(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("frozen_comparison", root / "scripts/run_frozen_eqa_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterbalanced_order_and_failed_scoring_denominator(tmp_path, monkeypatch):
    module = driver(monkeypatch)
    assert {module.trial_order(0, seed)[0] for seed in range(3)} == set(module.ARMS)
    answers = tmp_path / "scorer-only.json"
    answers.write_text('{"case": "B"}')
    rows = [
        {"case_id": "case", "answer": "Black", "complete": True},
        {"case_id": "case", "complete": False, "error": "truncated"},
    ]
    module.score(rows, [{"id": "case", "choices": ["White", "Black"]}], answers)
    assert len(rows) == 2 and rows[0]["correct_if_stopped_here"] is True
    assert rows[1]["correct_if_stopped_here"] is None


def test_cached_projection_migration_never_drops_connected_evidence(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("frozen_builder", root / "scripts/build_frozen_eqa_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = {
        "actions": {"frontiers": ["frontier_0"]},
        "scene_graph": {
            "nodes": [{"id": "frontier_0", "name": "frontier"}, {"id": "frontier_1800", "name": "frontier"}],
            "links": [],
        },
    }
    revised, removed = module.callable_frontier_projection(context)
    assert removed == ["frontier_1800"] and len(revised["scene_graph"]["nodes"]) == 1
    assert len(context["scene_graph"]["nodes"]) == 2
    context["scene_graph"]["links"] = [{"source": "frontier_1800", "target": "frontier_0"}]
    with pytest.raises(ValueError, match="connected"):
        module.callable_frontier_projection(context)


def test_driver_reads_gold_only_after_all_three_arms_and_server_cleanup(tmp_path, monkeypatch):
    module = driver(monkeypatch)
    case = {**packet(tmp_path), "id": "case"}
    cases_file, answers = tmp_path / "cases.json", tmp_path / "answers.json"
    cases_file.write_text("[]")
    answers.write_text('{"case":"B"}')
    calls, states, closed = [], [], []

    def load(state):
        state.client = SimpleNamespace(
            _resolved_hf_model_id="Qwen/Qwen3-VL-8B-Instruct",
            _quantization="int4",
            image_max_side=512,
            image_max_pixels=0,
            model=SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda: {"do_sample": True})),
        )
        states.append(state)

    def complete(state, body):
        calls.append(body)
        content = (
            "rug"
            if body["max_tokens"] == 64
            else '{"reasoning":"RGB", "answer":"Black", "confidence":true, "action":"", "confidence_reasoning":"RGB"}'
        )
        return {
            "id": "fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen3-vl-eqa",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }

    monkeypatch.setattr(FrozenServeState, "load", load)
    monkeypatch.setattr(LLMServeState, "complete", complete)
    monkeypatch.setattr(
        "emet.eval.frozen_eqa.OpenAI",
        lambda **kwargs: OpenAI(
            **kwargs,
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=states[0].complete(json.loads(request.content)))
                )
            ),
        ),
    )

    def run_reference(case, cfg, condition, output):
        client = PinnedPNGClient(
            case["images"], base_url=cfg.vlm.base_url, model=cfg.vlm.name, api_key=os.environ[cfg.vlm.api_key_env]
        )
        client.generate_multimodal(
            ["Question", *[np.asarray(Image.open(item["path"])) for item in case["images"]]], max_new_tokens=2048
        )
        return {"answer": "B"}

    monkeypatch.setattr(
        module, "load_reference", lambda _: SimpleNamespace(read_cases=lambda _: [case], run_one=run_reference)
    )
    monkeypatch.setattr(module, "validated_gpu_lock_fd", lambda: 9)
    monkeypatch.setattr(module, "assert_frozen_source", lambda *args: None)
    monkeypatch.setattr(module, "git", lambda *args: "test-revision")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(
        module,
        "ThreadingHTTPServer",
        lambda *args: SimpleNamespace(
            server_address=("127.0.0.1", 12345),
            serve_forever=lambda: None,
            shutdown=lambda: closed.append(True),
            server_close=lambda: None,
        ),
    )
    read_text = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == answers:
            assert len(calls) == 4 and closed, "Gold read before every arm and cleanup completed"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    assert (
        module.run(
            SimpleNamespace(fork=tmp_path, cases=cases_file, answers=answers, output=tmp_path / "run", seeds=[0])
        )
        == 0
    )
    rows = json.loads((tmp_path / "run/scored_results.json").read_text())
    assert len(rows) == 3 and all(row["correct_if_stopped_here"] for row in rows)
