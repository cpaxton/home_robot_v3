# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from emet.llms.conversation import VLMConversation
from emet.llms.image_context import ImageContext


def rgb(value):
    return np.full((4, 6, 3), value, dtype=np.uint8)


def attached(messages):
    return [
        part
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if isinstance(part, (np.ndarray, Image.Image))
    ]


def test_bounded_packet_keeps_current_and_distinct_recent_pixels_without_erasing_text():
    history = [{"role": "system", "content": "system"}]
    for index in range(10):
        history.extend(
            [
                {"role": "user", "content": [f"view turn {index}", rgb(index)]},
                {"role": "assistant", "content": f"choose {index}"},
                {"role": "user", "content": f"TOOL_RESULT {index} failed_move"},
            ]
        )
    history.append({"role": "user", "content": ["current image 1 then context image 2", rgb(10), rgb(0)]})
    original = copy.deepcopy(history)
    context = ImageContext(max_images=4)
    packet = context.pack(history)
    assert len(attached(packet.messages)) == 4
    # Current attachments stay in place, including a repeated historical image.
    assert [int(np.asarray(p)[0, 0, 0]) for p in attached(packet.messages)] == [8, 9, 10, 0]
    assert packet.audit["input_images"] == 12
    assert packet.audit["unique_pixel_images"] == 11
    assert packet.audit["attached_images"] == 4
    assert packet.messages[0] == original[0]
    for before, after in zip(original, packet.messages, strict=True):
        if isinstance(before["content"], str):
            assert before == after  # All actions, results and system text are exact.
        else:
            assert before["content"][0] in after["content"][0]
    for before, after in zip(original, history, strict=True):
        if isinstance(before["content"], list):
            for a, b in zip(before["content"], after["content"], strict=True):
                np.testing.assert_array_equal(a, b) if isinstance(a, np.ndarray) else None
        else:
            assert before == after
    omitted = packet.messages[1]["content"][1]
    assert "pixels omitted" in omitted and "recall_view(view_id)" in omitted
    assert packet.audit["occurrences"][0]["decision"] == "repeated_pixels"


def test_shape_is_part_of_pixel_identity_and_current_duplicates_are_not_silently_removed():
    a = np.zeros((4, 6, 3), dtype=np.uint8)
    b = a.reshape((6, 4, 3))
    context = ImageContext(3)
    packet = context.pack([{"role": "user", "content": [a, b, a.copy()]}])
    assert packet.audit["unique_pixel_images"] == 2
    assert packet.audit["attached_images"] == 3
    assert len(attached(packet.messages)) == 3


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_bad_image_limits_fail_at_configuration(limit):
    with pytest.raises(ValueError, match="positive integer"):
        ImageContext(limit)


def test_current_evidence_over_budget_fails_before_inference_instead_of_dropping_it():
    context = ImageContext(1)
    with pytest.raises(ValueError, match="no current evidence was silently dropped"):
        context.pack([{"role": "user", "content": [rgb(1), rgb(2)]}])


def test_full_context_ablation_keeps_every_original_image_occurrence():
    packet = ImageContext(None).pack(
        [{"role": "user", "content": ["old", rgb(0)]}, {"role": "user", "content": ["new", rgb(0)]}]
    )
    assert len(attached(packet.messages)) == 2


class Client:
    def __init__(self):
        self.calls = []
        self.fail = False

    def generate_messages(self, messages, **kwargs):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("inference failed")
        return "model decision"


def test_recall_old_image_is_pinned_historical_and_survives_failed_inference():
    conversation, client = VLMConversation(max_images=2), Client()
    for n in range(4):
        conversation.generate(client, [f"frame {n}", rgb(n)], system_prompt="system")
    oldest = next(iter(conversation.image_context.images))
    assert "pixels omitted" in str(client.calls[-1][1]["content"])
    assert conversation.recall_view(oldest)["status"] == "HISTORICAL_VIEW_QUEUED"
    conversation.record_tool_result("recall_view", {"view_id": oldest}, {"ok": True})
    before = len(conversation.messages)
    client.fail = True
    with pytest.raises(RuntimeError, match="inference failed"):
        conversation.generate(client, ["current", rgb(5)], system_prompt="system")
    assert len(conversation.messages) == before
    client.fail = False
    preview = conversation.preview(["current", rgb(5)], system_prompt="system")
    conversation.generate(client, ["current", rgb(5)], system_prompt="system")
    assert preview.audit == conversation.last_packet.audit
    last = client.calls[-1][-1]["content"]
    assert [int(np.asarray(p)[0, 0, 0]) for p in attached([{"content": last}])] == [5, 0]
    assert any("Requested historical view" in p and "not a new camera capture" in p for p in last if isinstance(p, str))
    assert "TOOL_RESULT" in str(client.calls[-1])
    assert len(conversation.messages) == before + 2
    conversation.reset()
    assert not conversation.messages
    assert conversation.recall_view(oldest)["status"] == "UNKNOWN_VIEW"


def test_recall_tool_uses_episode_registry_without_motion_verification_or_nav_budget():
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = SimpleNamespace(
        parameters={"eqa": {"agentic_decision_policy": "model_final"}}, graph_memory=None, robot=None
    )
    ex = AgenticEQAExecutor(agent, "Where is the bowl?", router=False)
    ex._grounded_obs_id = 7
    ex._verified = True
    ex._conversation.generate(Client(), ["earlier pixels", rgb(1)], system_prompt="system")
    view_id = next(iter(ex._conversation.image_context.images))
    before = (ex._n_nav, ex._n_explore, ex._verified, ex._grounded_obs_id)
    result = ex.handle_tool("recall_view", {"view_id": view_id})
    assert result["ok"] and result["status"] == "HISTORICAL_VIEW_QUEUED"
    assert before == (ex._n_nav, ex._n_explore, ex._verified, ex._grounded_obs_id)
    assert not ex.handle_tool("recall_view", {"view_id": "unknown"})["ok"]


def test_recorded_request_contains_actual_prepared_packet_and_identical_selection(tmp_path):
    import json

    from emet.eval.eqa_decision_trace import record_vlm_request

    conversation, client = VLMConversation(max_images=2), Client()
    for n in range(4):
        conversation.generate(client, [f"frame {n}", Image.fromarray(rgb(n))], system_prompt="system")
    content = ["next", Image.fromarray(rgb(8))]
    preview = conversation.preview(content, system_prompt="system")
    path = record_vlm_request(
        tmp_path / "request",
        system_prompt="system",
        user_text="next",
        images=[content[1]],
        metadata={},
        history=conversation.messages,
        packet=preview,
    )
    conversation.generate(client, content, system_prompt="system")
    recorded = json.loads(Path(path).read_text())
    historical_images = sum(
        part.get("type") == "image"
        for m in recorded["conversation_history"]
        if isinstance(m["content"], list)
        for part in m["content"]
    )
    assert historical_images + len(recorded["images"]) == 2
    assert recorded["metadata"]["image_context"] == conversation.last_packet.audit
    assert "pixels omitted" in str(recorded["conversation_history"])
    assert len(conversation.messages) == 10  # Complete original dialogue still retained.
