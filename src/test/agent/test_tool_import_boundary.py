# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Shared metadata and response parsing must work without CHAT robot backends."""

import os
import subprocess
import sys
from pathlib import Path


def test_metadata_and_parser_do_not_import_chat_robot_backends():
    script = """
import sys
from emet.agent import Tool, parse_tool_calls_response
from emet.agent.skills import AgentMode, skill_names_for_mode
from emet.agent.tool import Tool as ToolType

assert Tool is ToolType
assert 'ground_object' in skill_names_for_mode(AgentMode.CHAT)
assert parse_tool_calls_response('{"tool_calls": [], "message": "done"}')["message"] == "done"
assert 'emet.agent.tools' not in sys.modules
assert 'emet.agent.loop' not in sys.modules
assert not any(name.startswith(('emet.mapping.', 'emet.controller.', 'emet.perception.'))
               for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )


def test_original_tool_imports_retain_public_identity():
    from emet import agent
    from emet.agent import tool, tools

    assert agent.Tool is tools.Tool is tool.Tool
    assert agent.get_tool_descriptions_for_prompt is tools.get_tool_descriptions_for_prompt
    assert tools.get_tool_descriptions_for_prompt is tool.get_tool_descriptions_for_prompt
