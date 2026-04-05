# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Tests for agent prompt building, tool registry, and response parsing.
# Run with:  pytest src/test/agent/test_agent_prompt_and_tools.py -v


def test_parse_strips_think_block():
    """Parser must strip <think>...</think> reasoning blocks before extracting JSON."""
    from emet.agent.prompt import parse_tool_calls_response

    raw = (
        "<think>\nThe user wants a picture. The JSON would be "
        '{"tool_calls": [{"name": "bad"}]}\n</think>\n\n'
        '{"tool_calls": [{"name": "take_picture", "arguments": {}}], "message": "Taking a picture."}'
    )
    result = parse_tool_calls_response(raw)
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "take_picture"
    assert result["message"] == "Taking a picture."


def test_parse_partial_think_block():
    """Parser handles partial think block where opening <think> was stripped by tokenizer."""
    from emet.agent.prompt import parse_tool_calls_response

    raw = (
        ", I should greet them. The tool for that is wave().\n"
        "</think>\n\n"
        '{"tool_calls": [{"name": "wave", "arguments": {}}], "message": "Hello!"}'
    )
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == [{"name": "wave", "arguments": {}}]
    assert result["message"] == "Hello!"


def test_parse_plain_json():
    from emet.agent.prompt import parse_tool_calls_response

    raw = '{"tool_calls": [{"name": "wave", "arguments": {}}], "message": "Hi!"}'
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == [{"name": "wave", "arguments": {}}]


def test_parse_plain_text_fallback():
    from emet.agent.prompt import parse_tool_calls_response

    result = parse_tool_calls_response("I cannot do that, sorry.")
    assert result["tool_calls"] == []
    assert "cannot" in result["message"]


def test_parse_markdown_fenced_json():
    from emet.agent.prompt import parse_tool_calls_response

    raw = 'Here you go:\n```json\n{"tool_calls": [{"name": "explore", "arguments": {}}], "message": "ok"}\n```'
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == [{"name": "explore", "arguments": {}}]


def test_tools_registry_nonempty():
    """get_tools() should return a non-empty list of Tool objects."""
    from emet.agent.tools import get_tools

    tools = get_tools({})
    assert len(tools) > 5
    names = {t.name for t in tools}
    for expected in ("query_memory", "explore", "pick_place", "wave", "quit", "take_picture", "send_image"):
        assert expected in names, f"Missing expected tool: {expected}"


def test_tool_schema_format():
    """Each tool schema must follow OpenAI function-calling format."""
    from emet.agent.tools import get_tools

    for tool in get_tools({}):
        schema = tool.schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_prompt_includes_all_tool_names():
    """The system prompt must mention every registered tool name."""
    from emet.agent.prompt import build_agent_system_prompt
    from emet.agent.tools import get_tools

    tools = get_tools({})
    prompt = build_agent_system_prompt(tools)
    for t in tools:
        assert t.name in prompt, f"Tool '{t.name}' not found in system prompt"


def test_prompt_has_no_double_braces():
    """Prompt examples must use single braces so the LLM doesn't copy {{ in output."""
    from emet.agent.prompt import build_agent_system_prompt

    prompt = build_agent_system_prompt()
    assert "{{" not in prompt, "Prompt contains literal {{ — model will copy double braces"


def test_prompt_builder_configurable_name():
    from emet.agent.prompt import AgentPromptBuilder

    builder = AgentPromptBuilder(name="Stretch")
    assert "Stretch" in str(builder)
    builder.configure(name="Bender")
    assert "Bender" in str(builder)


def test_query_memory_tool_with_mock_backend():
    """query_memory tool should call the memory backend and return an answer."""
    from emet.agent.tools import get_tools

    class MockBackend:
        def query_answer(self, question, xyt, planner):
            return ("reasoning", f"answer about {question}", True, "confident", None, [])

    context = {"memory_backend": MockBackend(), "xyt_for_query": None, "planner": None}
    tools = get_tools(context)
    qm = next(t for t in tools if t.name == "query_memory")
    result = qm.func(question="What objects are here?")
    assert "answer about" in result
