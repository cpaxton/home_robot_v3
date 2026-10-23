# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.llms.base import AbstractVLLMClient
from emet.llms.conversation import VLMConversation
from emet.llms.graph_eqa_vlm import GraphEQAVLMClient
from emet.llms.openai_server import generate_chat
from emet.llms.openai_vllm_client import OpenaiVLLMClient


class FakeVision(AbstractVLLMClient):
    def __init__(self):
        super().__init__("")
        self.seen = []
        self.fail = False

    def __call__(self, content, **kwargs):
        return self.generate_multimodal(content, **kwargs)

    def _conversation_user_content(self, content):
        return content

    def generate_multimodal(self, content, *, reset_context=True, system_prompt=None, **kwargs):
        if reset_context:
            self.reset()
        self.add_history({"role": "user", "content": content})
        self.seen.append(copy.deepcopy(self.conversation_history))
        if self.fail:
            raise RuntimeError("inference failed")
        result = '{"answer":"Unknown","confidence":false,"reasoning":"fixture"}'
        self.add_history({"role": "assistant", "content": result})
        return result


def wrapper(model):
    return GraphEQAVLMClient(model, system_prompt="unchanged routing instructions", max_tokens=128)


def test_session_preserves_images_assistant_and_all_tool_results_across_perception():
    model = FakeVision()
    conversation = VLMConversation()
    client = wrapper(model)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    client(["task", image], conversation=conversation)
    conversation.record_tool_result("inspect_graph", {}, {"n_detections": 1})
    conversation.record_tool_result("inspect_graph", {}, {"n_detections": 1})
    image[:] = 255
    model.generate_multimodal("caption only", reset_context=True)
    perception_history = copy.deepcopy(model.get_history())
    client("next routing turn", conversation=conversation)
    seen = model.seen[-1]
    assert [row["role"] for row in seen] == ["system", "user", "assistant", "user", "user", "user"]
    assert np.max(seen[1]["content"][1]) == 0
    assert sum("TOOL_RESULT" in str(row["content"]) for row in seen) == 2
    assert "caption only" not in str(seen)
    assert model.get_history() == perception_history


def test_episode_isolation_failure_rollback_and_reset():
    model = FakeVision()
    one, two = VLMConversation(), VLMConversation()
    wrapper(model)("episode one", conversation=one)
    wrapper(model)("episode two", conversation=two)
    assert "episode one" not in str(model.seen[-1])
    previous = list(two.messages)
    model.fail = True
    with pytest.raises(RuntimeError, match="inference failed"):
        wrapper(model)("failed turn", conversation=two)
    assert two.messages == previous
    one.reset()
    assert one.messages == []


def test_remote_client_server_preserve_assistant_and_tool_result_turns(monkeypatch):
    backend = FakeVision()
    remote = OpenaiVLLMClient(base_url="http://fixture/v1", api_key="fixture")
    requests = []

    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        result = generate_chat(backend, kwargs["messages"], max_tokens=kwargs["max_tokens"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])

    remote._openai.chat.completions.create = complete
    conversation = VLMConversation()
    client = wrapper(remote)
    client(["first", np.zeros((4, 4, 3), dtype=np.uint8)], conversation=conversation)
    conversation.record_tool_result("inspect_graph", {"query": "bowl"}, {"status": "invalid_tool_arguments"})
    client("second", conversation=conversation)
    assert [row["role"] for row in requests[-1]] == ["system", "user", "assistant", "user", "user"]
    assert "invalid_tool_arguments" in str(backend.seen[-1])
    assert backend.seen[-1][2]["role"] == "assistant"
    assert requests[-1][1]["content"][1]["type"] == "image_url"
    assert remote.get_history() == []
    assert backend.get_history() == []


def test_final_graph_answer_receives_the_episode_dialogue():
    from emet.memory.graph_eqa import GraphEQAMemory

    model = FakeVision()
    conversation = VLMConversation()
    client = wrapper(model)
    client("earlier routing request", conversation=conversation)
    conversation.record_tool_result("investigate", {"obs_id": 3}, {"ok": False, "status": "no_target"})
    memory = GraphEQAMemory(eqa_client=client, image_description_client=lambda _: "")
    memory.parameters = {"eqa": {"answer_format": "json"}}
    memory.extract_relevant_objects = lambda _: None
    memory.memory_summary_enabled = False
    memory.query_answer(
        "Where? A) Here B) There",
        model_final=True,
        conversation=conversation,
        evidence_images=[np.zeros((4, 4, 3), dtype=np.uint8)],
    )
    assert "earlier routing request" in str(model.seen[-1])
    assert "no_target" in str(model.seen[-1])


def test_unknown_arguments_are_rejected_recorded_and_never_executed():
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = Mock()
    agent.robot = None
    executor = AgenticEQAExecutor(agent, "Where is the bowl?", router=False)
    executor._tool_inspect_graph = Mock(return_value={"ok": True})
    out = executor.handle_tool("inspect_graph", {"query": "bowl"})
    assert (
        next(tool for tool in executor._tools if tool.name == "inspect_graph").parameters["additionalProperties"]
        is False
    )
    assert out["status"] == "invalid_tool_arguments"
    assert out["unknown_arguments"] == ["query"]
    assert out["allowed_arguments"] == []
    executor._tool_inspect_graph.assert_not_called()
    assert "invalid_tool_arguments" in executor._conversation.messages[-1]["content"]
    assert executor.handle_tool("inspect_graph", {}) == {"ok": True}
    executor._tool_inspect_graph.assert_called_once()


def test_ovmm_phases_share_dialogue_but_new_world_or_episode_does_not():
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = Mock()
    agent.robot = None
    agent.voxel_map.observation_scope_id = "world-one"
    first = AgenticEQAExecutor(agent, "Find bowl", trace_meta={"episode_id": "episode-one"})
    first._conversation.record_tool_result("inspect_graph", {}, {"n_detections": 1})
    second = AgenticEQAExecutor(agent, "Find microwave", trace_meta={"episode_id": "episode-one"})
    assert first._conversation is second._conversation
    agent.voxel_map.observation_scope_id = "world-two"
    third = AgenticEQAExecutor(agent, "Find bowl", trace_meta={"episode_id": "episode-one"})
    assert third._conversation is not first._conversation
    assert third._conversation.messages == []
    fourth = AgenticEQAExecutor(agent, "Find bowl", trace_meta={"episode_id": "episode-two"})
    assert fourth._conversation is not third._conversation
