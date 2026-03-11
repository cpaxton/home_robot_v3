# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Agent system prompt builder: identity + tool list (from registry) + JSON tool_calls response format.

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from emet.agent.tools import get_tool_schemas_for_llm


AGENT_IDENTITY = """You are Stretch, a friendly, helpful robot. You have a body (arm, gripper, head, base) and cameras. You can move, pick up and place objects, explore and build a map, answer questions about what you've seen, and use social gestures (wave, nod, etc.).

Rules:
- You will never harm a person or suggest harm.
- You cannot go up or down stairs.
- Use tools to accomplish the user's request. Respond with a JSON object containing your tool calls.
- If the user's request is ambiguous (e.g. "put that away" without a clear object or location), ask for clarification or say you need more information.
- Do not make up object or location names; use what the user said or ask."""


RESPONSE_FORMAT = """
You must respond with a valid JSON object. Use this exact format (no other text before or after the JSON):
{"tool_calls": [{"name": "<tool_name>", "arguments": {<key-value pairs>}}, ...], "message": "<optional short reply to the user>"}

Examples:
- User: "Explore the room."
  {"tool_calls": [{"name": "explore", "arguments": {}}], "message": "I'll explore and build a map."}

- User: "Where is the red cup?"
  {"tool_calls": [{"name": "query_memory", "arguments": {"question": "Where is the red cup?"}}], "message": ""}

- User: "Pick up the apple and put it on the table."
  {"tool_calls": [{"name": "pick_place", "arguments": {"object_name": "apple", "receptacle_name": "table"}}], "message": "Picking up the apple and placing it on the table."}

- User: "Wave at me."
  {"tool_calls": [{"name": "wave", "arguments": {}}], "message": "Hello!"}

- User: "Take a picture."
  {"tool_calls": [{"name": "take_picture", "arguments": {}}], "message": "Taking a picture now."}

If you cannot fulfill the request, respond with a JSON object that has an empty tool_calls list and a message explaining why:
{"tool_calls": [], "message": "I need to know which object you mean."}
"""


def build_tools_block(schemas: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build the tools section of the system prompt from tool schemas."""
    if schemas is None:
        schemas = get_tool_schemas_for_llm()
    lines = ["Available tools (use these exact names and pass the required arguments):"]
    for s in schemas:
        name = s.get("name", "")
        desc = s.get("description", "")
        params = s.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            args_desc = ", ".join(
                f"{k}" + (" (required)" if k in required else " (optional)")
                for k in props
            )
            lines.append(f"  - {name}: {desc} Arguments: {args_desc}")
        else:
            lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def build_agent_system_prompt(schemas: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build the full system prompt for the agent (identity + tools + response format)."""
    tools_block = build_tools_block(schemas)
    return f"{AGENT_IDENTITY}\n\n{tools_block}\n{RESPONSE_FORMAT}"


def parse_tool_calls_response(response: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a list of tool_calls (name, arguments). Returns [] on parse failure."""
    response = response.strip()
    # Try to extract JSON (allow surrounding markdown or text)
    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        return []
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []
    raw = data.get("tool_calls", [])
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict) and "name" in item:
            name = item["name"]
            args = item.get("arguments")
            if not isinstance(args, dict):
                args = {}
            out.append({"name": name, "arguments": args})
    return out


def tool_calls_to_executor_format(
    tool_calls: List[Dict[str, Any]],
    tool_call_to_commands,  # tools.tool_call_to_executor_commands
) -> List[tuple]:
    """Convert parsed tool_calls to list of (command, args) for executor."""
    result = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments") or {}
        result.extend(tool_call_to_commands(name, args))
    return result


class AgentPromptBuilder:
    """Prompt builder for the agent: system prompt with tools + JSON tool_calls response format.

    Compatible with LLMChatWrapper and get_llm_client (prompt can be str or has __str__, parse_response).
    """

    def __init__(self, schemas: Optional[List[Dict[str, Any]]] = None):
        self.schemas = schemas
        self.prompt_str = build_agent_system_prompt(schemas)

    def __str__(self) -> str:
        return self.prompt_str

    def configure(self, **kwargs: Any) -> str:
        self.schemas = kwargs.get("schemas", self.schemas)
        self.prompt_str = build_agent_system_prompt(self.schemas)
        return self.prompt_str

    def __call__(self, kwargs: Optional[Dict[str, Any]] = None, **kw: Any) -> str:
        """Return the system prompt string (for LLM client)."""
        if kwargs is not None:
            self.configure(**kwargs)
        if kw:
            self.configure(**kw)
        return self.prompt_str

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to {tool_calls: [...], message: str}."""
        tool_calls = parse_tool_calls_response(response)
        # Try to get optional message from JSON
        response = response.strip()
        json_match = re.search(r"\{[\s\S]*\}", response)
        message = ""
        if json_match:
            try:
                data = json.loads(json_match.group())
                message = data.get("message", "") or ""
            except json.JSONDecodeError:
                pass
        return {"tool_calls": tool_calls, "message": message}
