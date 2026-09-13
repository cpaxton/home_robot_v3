"""Task mode must continue after actions and keep budget/finish outcomes distinct."""

import json
from types import SimpleNamespace

import pytest

from emet.agent.task import repair_missing_delimiters, run_agent_task
from emet.agent.tools import Tool


def run_responses(monkeypatch, responses, **kwargs):
    from emet.agent import loop

    replies = iter(responses)
    inputs, events, executed = [], [], []

    def call(_client, prompt, *_args, **_kwargs):
        inputs.append(prompt)
        assert _kwargs["reset_context"] == (len(inputs) == 1)
        return next(replies), 0.01

    monkeypatch.setattr(loop, "_call_llm", call)
    tool = Tool(
        "act",
        "Perform action",
        {"type": "object", "properties": {}},
        lambda: executed.append(1) or "action performed",
        returns_info=False,
    )
    result = run_agent_task(
        "Complete several actions",
        llm_client=SimpleNamespace(),
        tools=[tool],
        on_event=lambda kind, payload: events.append((kind, payload)),
        **kwargs,
    )
    return result, inputs, events, executed


ACTION = json.dumps({"tool_calls": [{"name": "act", "arguments": {}}], "message": ""})
FINISH = json.dumps({"tool_calls": [], "message": "Finished"})


def test_continues_beyond_three_action_rounds(monkeypatch):
    result, inputs, events, executed = run_responses(monkeypatch, [ACTION] * 5 + [FINISH])
    assert len(executed) == 5
    assert result["status"] == "model_finished" and result["rounds"] == 6
    assert "action performed" in inputs[-1]
    assert "Do not call any more tools" not in inputs[-1]
    assert len([k for k, _ in events if k == "model_input"]) == 6


def test_round_exhaustion_does_not_claim_success(monkeypatch):
    result, _, _, executed = run_responses(monkeypatch, [ACTION] * 4, max_rounds=4)
    assert result["status"] == "round_budget_exhausted" and len(executed) == 4


def test_malformed_output_is_recoverable(monkeypatch):
    result, inputs, _, executed = run_responses(monkeypatch, ["I am done", ACTION, FINISH])
    assert "Invalid response" in inputs[1]
    assert result["status"] == "model_finished" and len(executed) == 1


def test_explicit_json_finish_with_prefix_uses_shared_parser(monkeypatch):
    result, _, _, _ = run_responses(monkeypatch, ["[] > " + FINISH])
    assert result["status"] == "model_finished"


def test_action_budget_applies_inside_model_batch(monkeypatch):
    batch = json.dumps({"tool_calls": [{"name": "act"}] * 10})
    result, _, _, executed = run_responses(monkeypatch, [batch], max_actions=2, max_tools_per_round=10)
    assert result["status"] == "action_budget_exhausted" and len(executed) == 2


def test_cancellation_prevents_model_dispatch(monkeypatch):
    result, inputs, _, executed = run_responses(monkeypatch, [], cancelled=lambda: True)
    assert result["status"] == "cancelled" and not inputs and not executed


def test_observation_passed_exactly_without_evaluator(monkeypatch):
    observation = {"visible_objects": ["red cylinder"], "observation_id": "obs:4"}
    result, inputs, events, _ = run_responses(monkeypatch, [FINISH], observe=lambda: (observation, None))
    event = next(p for kind, p in events if kind == "model_input")
    assert event["observation"] == observation
    assert "red cylinder" in inputs[0] and "goals" not in event["observation"]


def test_invalid_budget_rejected():
    with pytest.raises(ValueError, match="budget"):
        run_agent_task("test", llm_client=None, tools=[], max_rounds=0)


def test_repeated_invalid_output_stops_with_protocol_failure(monkeypatch):
    result, inputs, _, executed = run_responses(monkeypatch, ["broken"] * 3)
    assert result["status"] == "model_protocol_failed" and len(inputs) == 3 and not executed


@pytest.mark.parametrize("task_mode", [False, True])
def test_existing_robot_entry_runs_local_client_in_both_modes(monkeypatch, task_mode):
    from unittest.mock import MagicMock

    from emet.agent import loop

    robot = MagicMock()
    robot.get_base_pose.return_value = [0, 0, 0]
    executor = MagicMock()
    executor.robot = executor.agent.robot = robot
    executor.agent.graph_memory = None
    executor.agent.log = "."
    client = MagicMock(return_value=FINISH)
    monkeypatch.setattr(loop, "StretchZmqClient", lambda **_: robot)
    monkeypatch.setattr(loop, "DynamemTaskExecutor", lambda *_a, **_k: executor)
    monkeypatch.setattr(loop, "get_memory_backend", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(loop, "get_llm_client", lambda *_a, **_k: client)
    monkeypatch.setattr(loop, "ChatLog", MagicMock)
    monkeypatch.setattr(loop, "print_memory_view_help_on_quit", lambda *_a, **_k: None)
    loop.run_agent_with_robot(
        robot="stretch", discord=False, use_llm=True, commands=["Complete this task"], device="cpu", task_mode=task_mode
    )
    assert client.call_count == 1
    robot.stop.assert_called_once()


def test_missing_closing_brace_is_repaired_and_recorded(monkeypatch):
    broken = '{"tool_calls":[{"name":"act","arguments":{}],"message":""}'
    result, _, events, executed = run_responses(monkeypatch, [broken, FINISH])
    assert result["status"] == "model_finished" and len(executed) == 1
    repair = next(payload for kind, payload in events if kind == "response_repair")
    assert repair["original"] == broken
    assert json.loads(repair["repaired"])["tool_calls"][0]["arguments"] == {}


def test_only_complete_first_action_from_truncated_batch_runs(monkeypatch):
    raw = '{"tool_calls":[{"name":"act","arguments":{}}, {"name":"act","arguments":'
    result, _, events, executed = run_responses(monkeypatch, [raw, FINISH])
    assert len(executed) == 1 and result["status"] == "model_finished"
    assert any(p.get("kind_of_repair") == "first_complete_tool_only" for _, p in events)


def test_complete_actions_survive_extra_batch_punctuation(monkeypatch):
    raw = '{"tool_calls":[{"name":"act","arguments":{}}}, {"name":"act","arguments":{}}}]}'
    result, _, events, executed = run_responses(monkeypatch, [raw, FINISH])
    assert len(executed) == 2 and result["status"] == "model_finished"
    assert any(p.get("kind_of_repair") == "complete_tools_from_broken_batch" for _, p in events)


@pytest.mark.parametrize("text", ['{"name":', '{"name":"unterminated', "please act now"])
def test_punctuation_repair_never_invents_values(text):
    assert repair_missing_delimiters(text) == text
