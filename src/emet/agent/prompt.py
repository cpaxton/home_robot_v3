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

DEFAULT_AGENT_NAME = "Emet"

# Identity block: kept short so small models focus on the tool contract.
_IDENTITY_TEMPLATE = """\
You are {name}, a helpful mobile robot assistant with a wheeled base, arm, gripper, and cameras.
You help people by navigating, picking up objects, answering questions, and more.
If a request is ambiguous, ask the user to clarify before acting.
Keep replies brief and friendly. Do NOT output your internal reasoning."""

# Response format: the critical section for getting structured JSON back.
# NOTE: This is a plain string, NOT an f-string. Use single braces for JSON.
_FORMAT_BLOCK = """\
# Response format
Respond with ONLY a JSON object. No other text before or after.

{"tool_calls": [<list of tool invocations>], "message": "<short reply to user>"}

Each tool invocation: {"name": "<tool_name>", "arguments": {<key>: <value>, ...}}
If no action is needed, set "tool_calls" to [].

When you call query_memory or describe_scene, the results will be provided back to you.
You must then summarize them for the user in a follow-up response (no more tool calls).
Use query_memory for map / object-location questions (where is X, have I seen Y).
For open-ended "what do you see" questions, prefer describe_scene and send_image unless full EQA is enabled for memory Q&A.

# Examples

User: "Explore the room."
{"tool_calls": [{"name": "explore", "arguments": {}}], "message": "Exploring now."}

User: "Where is the red cup?"
{"tool_calls": [{"name": "query_memory", "arguments": {"question": "Where is the red cup?"}}], "message": "Checking my memory."}

User: "Put the apple on the table."
{"tool_calls": [{"name": "pick_place", "arguments": {"object_name": "apple", "receptacle_name": "table"}}], "message": "On it."}

User: "Take a picture and send it to me."
{"tool_calls": [{"name": "take_picture", "arguments": {}}, {"name": "send_image", "arguments": {}}], "message": "Here you go."}

User: "What objects can you see?"
{"tool_calls": [{"name": "describe_scene", "arguments": {}}, {"name": "send_image", "arguments": {}}], "message": "Taking a look."}

[Tool results]
[describe_scene] A blue cube and a red cylinder on the table.

Summarize these results for the user in your message. Do not call any more tools.
{"tool_calls": [], "message": "I can see a blue cube and a red cylinder on the table."}

User: "Wave hello!"
{"tool_calls": [{"name": "wave", "arguments": {}}], "message": "Hi!"}

User: "Can you put that away?"
{"tool_calls": [], "message": "Which object, and where should I put it?"}

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


def parse_tool_calls_response(response: str) -> dict[str, Any]:
    """Parse LLM response into {tool_calls: [{name, arguments}], message: str}.

    Handles <think> blocks, markdown code fences, and text around JSON.
    Returns {"tool_calls": [], "message": <raw_text>} on parse failure.
    """
    response = response.strip()
    # Strip <think>...</think> reasoning blocks (Qwen 3.5, DeepSeek, etc.)
    response = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
    # Handle partial think blocks: opening tag stripped by tokenizer but </think> remains
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if fenced:
        candidate = fenced.group(1)
    else:
        candidate_match = re.search(r"\{[\s\S]*\}", response)
        candidate = candidate_match.group() if candidate_match else ""

    tool_calls: list[dict[str, Any]] = []
    message = ""
    if candidate:
        try:
            data = json.loads(candidate)
            raw = data.get("tool_calls", [])
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and "name" in item:
                        args = item.get("arguments")
                        if not isinstance(args, dict):
                            args = {}
                        tool_calls.append({"name": item["name"], "arguments": args})
            message = data.get("message", "") or ""
        except json.JSONDecodeError:
            pass

    # If no JSON was found, treat the whole response as a message
    if not tool_calls and not message:
        message = response

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
