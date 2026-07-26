# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from datetime import datetime
from unittest.mock import MagicMock

from emet.agent.loop import terminal_timestamp
from emet.agent.thinking_status import (
    env_agent_thinking_status,
    format_action_running_status,
    format_llm_thinking_status,
    format_tool_running_status,
    short_llm_label,
)
from emet.llms.base import AbstractVLLMClient


def test_terminal_timestamp_format():
    assert terminal_timestamp(datetime(2026, 7, 10, 22, 45, 3)) == "22:45:03"


def test_short_llm_label_from_hf_id():
    client = MagicMock()
    client.hf_model_id = "Qwen/Qwen3-VL-8B-Instruct"
    assert short_llm_label("qwen3-vl-eqa", client) == "Qwen3-VL-8B-Instruct"


def test_short_llm_label_from_vl_key():
    client = MagicMock(spec=AbstractVLLMClient)
    client.hf_model_id = None
    client.canonical_model_key = "qwen3-vl-eqa"
    assert short_llm_label("qwen3-vl-eqa", client) == "qwen3-vl"


def test_format_llm_thinking_status_first_turn_with_camera():
    msg = format_llm_thinking_status(
        llm_label="Qwen3-VL-8B-Instruct",
        round_idx=1,
        max_rounds=3,
        has_image=True,
        followup=False,
    )
    assert msg.startswith("*Thinking…*")
    assert "head camera" in msg
    assert "step 1/3" in msg


def test_format_llm_thinking_status_followup():
    msg = format_llm_thinking_status(
        llm_label="Qwen3-VL-8B-Instruct",
        round_idx=2,
        max_rounds=3,
        has_image=False,
        followup=True,
    )
    assert "summarizing tool results" in msg


def test_format_tool_running_status():
    assert format_tool_running_status(["describe_scene", "send_image"]) == (
        "*Running tools…* describe_scene, send_image"
    )


def test_format_action_running_status_explore():
    assert format_action_running_status(["explore"]) == "*Exploring…*"
    assert format_action_running_status(["explore"], detail="sweeping head") == (
        "*Exploring…* sweeping head"
    )
    assert format_action_running_status(["scan_environment"]) == "*Looking around…*"
    assert format_action_running_status(["describe_scene"]) == "*Looking…*"
    assert format_action_running_status(["query_memory"]) is None
    assert format_action_running_status(["rotate_base", "describe_scene"]) == (
        "*Turning, then looking…*"
    )
    assert format_action_running_status(["face_toward", "describe_scene"]) == (
        "*Turning toward, then looking…*"
    )


def test_format_single_action_status():
    from emet.agent.thinking_status import format_single_action_status

    assert format_single_action_status("describe_scene") == "*Looking…*"
    assert format_single_action_status("face_toward") == "*Turning toward…*"
    assert format_single_action_status("query_memory") is None


def test_env_agent_thinking_status_default_on(monkeypatch):
    monkeypatch.delenv("EMET_AGENT_THINKING_STATUS", raising=False)
    assert env_agent_thinking_status() is True


def test_env_agent_thinking_status_off(monkeypatch):
    monkeypatch.setenv("EMET_AGENT_THINKING_STATUS", "0")
    assert env_agent_thinking_status() is False
