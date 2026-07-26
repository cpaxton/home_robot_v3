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
    assert result["message"] == "Hi!"


def test_parse_stringified_tool_arguments():
    """Models sometimes emit arguments as a JSON string instead of an object."""
    from emet.agent.prompt import parse_tool_calls_response

    raw = '{"tool_calls": [{"name": "rotate_base", "arguments": "{\\"degrees\\": 90}"}], "message": ""}'
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == [{"name": "rotate_base", "arguments": {"degrees": 90}}]


def test_parse_invalid_stringified_arguments_becomes_empty_dict():
    from emet.agent.prompt import parse_tool_calls_response

    raw = '{"tool_calls": [{"name": "rotate_base", "arguments": "not-json"}], "message": ""}'
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == [{"name": "rotate_base", "arguments": {}}]


def test_parse_json_with_trailing_text():
    """Trailing prose after a balanced JSON object must not break parsing or leak into message."""
    from emet.agent.prompt import parse_tool_calls_response

    raw = '{"tool_calls": [{"name": "wave", "arguments": {}}], "message": ""}\n(internal)'
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"][0]["name"] == "wave"
    assert result["message"] == ""


def test_parse_prefix_prose_before_json():
    from emet.agent.prompt import parse_tool_calls_response

    raw = 'Certainly.\n{"tool_calls": [], "message": "Here you go."}'
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == []
    assert result["message"] == "Here you go."


def test_parse_broken_json_blob_not_user_message():
    """Invalid JSON that mentions tool_calls must not become the assistant message verbatim."""
    from emet.agent.prompt import parse_tool_calls_response

    raw = '{"tool_calls": [{"name": "wave", "arguments": {}'
    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == []
    assert result["message"] == ""


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
    for expected in (
        "query_memory",
        "explore",
        "scan_environment",
        "rotate_base",
        "face_toward",
        "move_forward",
        "describe_scene",
        "pick_place",
        "wave",
        "quit",
        "take_picture",
        "send_image",
    ):
        assert expected in names, f"Missing expected tool: {expected}"


def test_prompt_routes_rotate_back_not_around():
    from emet.agent.prompt import build_agent_system_prompt
    from emet.agent.tools import get_tools

    prompt = build_agent_system_prompt(tools=get_tools({}), name="Virgil")
    assert "Rotate back" in prompt
    assert "Do NOT use 180" in prompt
    assert '"degrees": -45' in prompt or '"degrees":-45' in prompt


def test_prompt_routes_look_at_to_face_toward():
    from emet.agent.prompt import build_agent_system_prompt
    from emet.agent.tools import get_tools

    prompt = build_agent_system_prompt(tools=get_tools({}), name="Virgil")
    assert "face_toward" in prompt
    assert "Look at the aquarium" in prompt
    assert '"object_label": "aquarium"' in prompt or '"object_label":"aquarium"' in prompt


def test_prompt_routes_turn_around_to_rotate_base():
    from emet.agent.prompt import build_agent_system_prompt
    from emet.agent.tools import get_tools

    prompt = build_agent_system_prompt(tools=get_tools({}), name="Virgil")
    assert "rotate_base" in prompt
    assert 'degrees": 180' in prompt or '"degrees": 180' in prompt
    assert "move_forward" in prompt
    assert 'meters": 0.1' in prompt or '"meters": 0.1' in prompt
    assert "ask how far" in prompt.lower()
    assert "Can you move forward" in prompt


def test_prompt_routes_look_around_to_scan():
    from emet.agent.prompt import build_agent_system_prompt
    from emet.agent.tools import get_tools

    prompt = build_agent_system_prompt(tools=get_tools({}), name="Virgil")
    assert "look around" in prompt.lower()
    assert "scan_environment" in prompt
    # Example should actually call scan, not only describe/send_image.
    assert '"name": "scan_environment"' in prompt or '"name":"scan_environment"' in prompt


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


def test_prompt_distinguishes_info_vs_action_tools():
    from emet.agent.prompt import build_agent_system_prompt

    prompt = build_agent_system_prompt()
    assert "Info tools return text" in prompt
    assert "Action-only tools do not feed a tool-results summary" in prompt
    assert "take_picture alone" in prompt
    assert "send_image" in prompt
    assert "{{" not in prompt, "Prompt contains literal {{ — model will copy double braces"


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


def test_describe_scene_delegates_to_agent():
    """describe_scene should call executor.agent.describe_head_camera_scene_text when present."""
    import numpy as np

    from emet.agent.tools import get_tools
    from emet.core.interfaces import Observations

    class MockRobot:
        def get_observation(self):
            return Observations(
                gps=np.zeros(2, dtype=np.float32),
                compass=np.zeros(1, dtype=np.float32),
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                depth=np.zeros((4, 4), dtype=np.float32),
            )

    class MockAgent:
        def describe_head_camera_scene_text(self, **_kwargs):
            return "From my head camera: a table and a chair."

        def pick_interesting_scene_image(self, **_kwargs):
            # Must not be used by describe_scene (crops are for send_object_image).
            return np.ones((8, 8, 3), dtype=np.uint8) * 40, "mug"

    class MockExecutor:
        robot = None
        agent = MockAgent()

    class MockDiscord:
        def __init__(self):
            self.pushed = []

        def push_task_to_all_channels(self, message=None, content=None):
            self.pushed.append((message, content))

    discord = MockDiscord()
    context = {"robot": MockRobot(), "executor": MockExecutor(), "discord_bot": discord}
    tools = get_tools(context)
    ds = next(t for t in tools if t.name == "describe_scene")
    out = ds.func()
    assert "table and a chair" in out
    assert "current view" in out.lower()
    assert "mug" not in out
    from emet.agent.tools import PENDING_DISCORD_IMAGE_KEY

    assert PENDING_DISCORD_IMAGE_KEY in context
    assert context[PENDING_DISCORD_IMAGE_KEY].shape == (4, 4, 3)
    assert discord.pushed == []  # image is attached by the agent loop with the reply
