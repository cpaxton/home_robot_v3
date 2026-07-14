# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Agent system prompt builder: identity + tool list (derived from registry) + response format.
# Tools registered in tools.py are automatically included in the prompt.

from __future__ import annotations

import json
import re
from typing import Any

from emet.agent.tools import Tool, get_tool_descriptions_for_prompt, get_tools
from emet.utils.logger import Logger

_logger = Logger(__name__)

# Default persona for embodied agent + ``emet run agent`` / web chat (override with ``--name``).
DEFAULT_AGENT_NAME = "Virgil"

# Identity block: kept short so small models focus on the tool contract.
_IDENTITY_TEMPLATE = """\
You are {name}, a helpful mobile robot assistant with a wheeled base, arm, gripper, and cameras.
You help people by navigating, picking up objects, answering questions, and more.
If a request is ambiguous, ask the user to clarify before acting.
Keep replies brief and friendly. Do NOT output your internal reasoning."""

# Response format: kept short so local VL tool-routing stays fast (~token budget).
# NOTE: This is a plain string, NOT an f-string. Use single braces for JSON.
_FORMAT_BLOCK = """\
# Response format
Respond with ONLY a JSON object (no other text):
{"tool_calls": [{"name": "<tool>", "arguments": {...}}, ...], "message": "<short reply>"}
Use "tool_calls": [] when no action is needed.

Info tools return text (and sometimes photos) for a follow-up reply. When calling any of
query_*, describe_scene, explore, scan_environment, rotate_base, move_forward,
navigation_diagnostics, send_map_snapshot, list_scene_relations, send_image, send_object_image,
set "message" to "" on that first turn. After [Tool results], reply with tool_calls [] and a
message based only on those results (do not invent objects from this prompt).

Action-only tools do not feed a tool-results summary (wave, nod_head, shake_head, avert_gaze,
take_picture, take_ee_picture, go_home, hand_over, quit). Prefer "message": "" — the turn ends
after the action. Use send_image after take_picture if the user should receive a photo.

Routing hints:
- "what can you see" / "tell me what you see" / "describe the scene" (no motion asked)
  → describe_scene only: caption the image in front of you; ground with scene graph/map if useful.
  Do NOT scan, explore, or turn unless the user asked to look around / move.
- "turn around" → rotate_base with degrees=180
- "turn right" / "turn to the right" → rotate_base with degrees=-90
- "turn left" / "turn to the left" → rotate_base with degrees=90
- "move forward a bit" / "go forward a little" → move_forward with meters=0.5
  (controller shortens if obstacles; use ~1.0 for "a meter")
- "look around" / "scan the room" → scan_environment (full in-place 360° map update)
  Optionally follow with describe_scene after the scan.
- "explore" / "go explore" / "map the room" → explore (navigate to build the map)
- where/what/is-there → query_scene_graph (or query_memory if graph EQA is on)
- map stuck / explore failed → navigation_diagnostics + send_map_snapshot
- close-up of a known object → send_object_image (not describe_scene)
- "take a picture" / "send a photo" → send_image (live head camera to Discord); take_picture alone
  only captures locally and does not return a caption

# Examples
User: "Where is the red cup?"
{"tool_calls": [{"name": "query_scene_graph", "arguments": {"question": "Where is the red cup?"}}], "message": ""}
User: "What can you see?"
{"tool_calls": [{"name": "describe_scene", "arguments": {}}], "message": ""}
User: "Turn around"
{"tool_calls": [{"name": "rotate_base", "arguments": {"degrees": 180}}], "message": ""}
User: "Turn to the right"
{"tool_calls": [{"name": "rotate_base", "arguments": {"degrees": -90}}], "message": ""}
User: "Move forward a bit"
{"tool_calls": [{"name": "move_forward", "arguments": {"meters": 0.5}}], "message": ""}
User: "Look around"
{"tool_calls": [{"name": "scan_environment", "arguments": {}}, {"name": "describe_scene", "arguments": {}}], "message": ""}
User: "Wave"
{"tool_calls": [{"name": "wave", "arguments": {}}], "message": ""}
User: "Goodbye"
{"tool_calls": [{"name": "quit", "arguments": {}}], "message": "Bye!"}"""


def build_agent_system_prompt(
    tools: list[Tool] | None = None,
    name: str = DEFAULT_AGENT_NAME,
    context: dict[str, Any] | None = None,
) -> str:
    """Build the full system prompt: identity + tools + response format.

    If tools is None, derives from get_tools(context or {}).
    """
    if tools is None:
        tools = get_tools(context or {})
    identity = _IDENTITY_TEMPLATE.format(name=name)
    tools_block = get_tool_descriptions_for_prompt(tools)
    return f"{identity}\n\n{tools_block}\n\n{_FORMAT_BLOCK}"


def _fence_inner_json(text: str) -> str | None:
    """Return inner JSON string from first ```json ... ``` or ``` ... ``` fence, or None."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return m.group(1).strip() if m else None


def _first_json_dict(text: str) -> dict[str, Any] | None:
    """Parse the first balanced JSON object from *text* using :meth:`json.JSONDecoder.raw_decode`.

    Avoids greedy ``\\{[\\s\\S]*\\}`` bugs when there is trailing prose or multiple ``{`` tokens.
    """
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = j + 1
    return None


def _normalize_message_field(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return str(raw)


def parse_tool_calls_response(response: str) -> dict[str, Any]:
    """Parse LLM response into {tool_calls: [{name, arguments}], message: str}.

    Handles <think> blocks, markdown code fences, prefix/suffix text around JSON,
    and balanced-brace JSON via :meth:`json.JSONDecoder.raw_decode`.
    Returns {"tool_calls": [], "message": <raw_text>} on total parse failure (never leaks JSON blobs as message).
    """
    response = response.strip()
    response = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    candidate_sources: list[str] = []
    fenced = _fence_inner_json(response)
    if fenced:
        candidate_sources.append(fenced)
    candidate_sources.append(response)

    data: dict[str, Any] | None = None
    for blob in candidate_sources:
        data = _first_json_dict(blob)
        if data is not None:
            break

    tool_calls: list[dict[str, Any]] = []
    message = ""
    if data is not None:
        raw_tc = data.get("tool_calls", [])
        if isinstance(raw_tc, list):
            for item in raw_tc:
                if isinstance(item, dict) and "name" in item:
                    args = item.get("arguments")
                    if isinstance(args, str):
                        raw_args = args.strip()
                        if not raw_args:
                            args = {}
                        else:
                            try:
                                parsed_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                _logger.warning(f"Tool {item.get('name')!r}: arguments string is not JSON; using {{}}")
                                parsed_args = {}
                            if isinstance(parsed_args, dict):
                                args = parsed_args
                            else:
                                _logger.warning(
                                    f"Tool {item.get('name')!r}: arguments JSON was not an object; using {{}}"
                                )
                                args = {}
                    elif not isinstance(args, dict):
                        args = {}
                    tool_calls.append({"name": item["name"], "arguments": args})
        message = _normalize_message_field(data.get("message", ""))

    # Total failure: no dict — treat whole reply as natural language unless it looks like broken JSON.
    if data is None:
        if response.startswith("{") and '"tool_calls"' in response:
            _logger.warning(
                "Assistant returned JSON-shaped text that did not parse; suppressing raw JSON from user message."
            )
            return {"tool_calls": [], "message": ""}
        return {"tool_calls": [], "message": response}

    return {"tool_calls": tool_calls, "message": message}


class AgentPromptBuilder:
    """Prompt builder for the agent: system prompt with tools + JSON tool_calls response format.

    Compatible with LLMChatWrapper, get_llm_client, and AbstractPromptBuilder interface.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        name: str = DEFAULT_AGENT_NAME,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self._tools = tools
        self._context = context
        self.prompt_str = build_agent_system_prompt(tools, name=name, context=context)

    def __str__(self) -> str:
        return self.prompt_str

    def __call__(self, kwargs: dict[str, Any] | None = None, **kw: Any) -> str:
        if kwargs:
            self.configure(**kwargs)
        if kw:
            self.configure(**kw)
        return self.prompt_str

    def configure(self, **kwargs: Any) -> str:
        self._tools = kwargs.get("tools", self._tools)
        self.name = kwargs.get("name", self.name)
        self._context = kwargs.get("context", self._context)
        self.prompt_str = build_agent_system_prompt(self._tools, name=self.name, context=self._context)
        return self.prompt_str

    def parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response to {tool_calls: [...], message: str}."""
        return parse_tool_calls_response(response)
