# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Agent system prompt builder: identity + tool list (derived from registry) + response format.
# Tools registered in tools.py are automatically included in the prompt.

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from emet.agent.tools import Tool, get_tool_descriptions_for_prompt, get_tools

DEFAULT_AGENT_NAME = "Emet"


def _build_identity(name: str = DEFAULT_AGENT_NAME) -> str:
    return f"""You are {name}, a friendly and helpful mobile robot assistant made by Hello Robot. You have a wheeled base, a telescoping arm with a gripper, a head with pan/tilt cameras, and depth sensors. You can move around, pick up and place objects, explore and map your environment, answer questions about things you've seen, and express yourself with social gestures.

Key facts:
- You are safe and will never harm a person or suggest harm.
- You cannot go up or down stairs.
- You live in indoor environments (homes, offices, labs).
- When you are unsure about an object or location, ask for clarification rather than guessing.
- You should announce what you are about to do before executing actions (use the say tool).
- If a request is ambiguous (e.g. "put that away" without specifying object or destination), ask the user to clarify."""


RESPONSE_FORMAT_JSON = """
Response format:
You must respond with ONLY a valid JSON object, no other text before or after:
{{"tool_calls": [{{"name": "<tool_name>", "arguments": {{...}}}}], "message": "<short reply to user>"}}

Rules:
- "tool_calls" is a list of tool invocations to execute in order.
- "message" is what you want to say to the user (shown as text). Use it for conversational replies, confirmations, or clarification questions.
- If you just want to chat (no action needed), use an empty tool_calls list.
- Always announce actions in "message" before executing them (e.g. "I'll go find the cup for you.").
- For pick-and-place, use a single pick_place tool call with both object_name and receptacle_name.
- Never invent object or location names. Use exactly what the user said, or ask for clarification.

Examples:
User: "Go explore."
{{"tool_calls": [{{"name": "explore", "arguments": {{}}}}], "message": "I'll start exploring and mapping the area."}}

User: "Where is the red cup?"
{{"tool_calls": [{{"name": "query_memory", "arguments": {{"question": "Where is the red cup?"}}}}], "message": "Let me check my memory."}}

User: "Pick up the apple and put it on the table."
{{"tool_calls": [{{"name": "say", "arguments": {{"text": "Picking up the apple and placing it on the table."}}}}, {{"name": "pick_place", "arguments": {{"object_name": "apple", "receptacle_name": "table"}}}}], "message": "On it!"}}

User: "Hi there!"
{{"tool_calls": [{{"name": "wave", "arguments": {{}}}}], "message": "Hello! How can I help you today?"}}

User: "Can you put the shoe away?"
{{"tool_calls": [], "message": "Sure! Where would you like me to put the shoe?"}}

User: "Goodbye!"
{{"tool_calls": [{{"name": "say", "arguments": {{"text": "Goodbye!"}}}}, {{"name": "wave", "arguments": {{}}}}, {{"name": "quit", "arguments": {{}}}}], "message": "Bye! Shutting down."}}
"""


def build_agent_system_prompt(
    tools: Optional[List[Tool]] = None,
    name: str = DEFAULT_AGENT_NAME,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the full system prompt (identity + tools + response format).

    If tools is None, derives from get_tools(context or {}).
    """
    if tools is None:
        tools = get_tools(context or {})
    identity = _build_identity(name)
    tools_block = get_tool_descriptions_for_prompt(tools)
    return f"{identity}\n\n{tools_block}\n{RESPONSE_FORMAT_JSON}"


def parse_tool_calls_response(response: str) -> Dict[str, Any]:
    """Parse LLM response into {tool_calls: [{name, arguments}], message: str}.

    Handles JSON embedded in markdown code fences or surrounded by text.
    Returns {"tool_calls": [], "message": ""} on parse failure.
    """
    response = response.strip()
    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if fenced:
        candidate = fenced.group(1)
    else:
        candidate_match = re.search(r"\{[\s\S]*\}", response)
        candidate = candidate_match.group() if candidate_match else ""

    tool_calls: List[Dict[str, Any]] = []
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

    # If no JSON was found, treat the whole response as a message (LLM didn't follow format)
    if not tool_calls and not message:
        message = response

    return {"tool_calls": tool_calls, "message": message}


class AgentPromptBuilder:
    """Prompt builder for the agent: system prompt with tools + JSON tool_calls response format.

    Compatible with LLMChatWrapper and get_llm_client.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        name: str = DEFAULT_AGENT_NAME,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self._tools = tools
        self._context = context
        self.prompt_str = build_agent_system_prompt(tools, name=name, context=context)

    def __str__(self) -> str:
        return self.prompt_str

    def __call__(self, kwargs: Optional[Dict[str, Any]] = None, **kw: Any) -> str:
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

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to {tool_calls: [...], message: str}."""
        return parse_tool_calls_response(response)
