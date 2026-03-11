# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent tools: single registry of tools with name, description, parameters (JSON Schema), func,
# and executor command mapping. The prompt and tool-calling interface derive from this registry.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class Tool:
    """A single agent tool: name, description, JSON Schema parameters, callable, and executor mapping."""

    __slots__ = ("name", "description", "parameters", "func", "executor_commands")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
        executor_commands: Optional[Callable[[Dict[str, Any]], List[Tuple[str, str]]]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.executor_commands = executor_commands

    def schema(self) -> Dict[str, Any]:
        """Return OpenAI-style tool schema (for tools param or prompt generation)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_executor(self, arguments: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Map tool call arguments to executor (command, args) list. Empty list means use func instead."""
        if self.executor_commands is not None:
            return self.executor_commands(arguments)
        return []


_NO_PARAMS: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}


def _simple_exec_mapping(cmd: str) -> Callable[[Dict[str, Any]], List[Tuple[str, str]]]:
    """Return a mapping function for a no-arg executor command."""
    return lambda args: [(cmd, "")]


def get_tools(context: Dict[str, Any]) -> List[Tool]:
    """Return list of tools available given context (executor, robot, discord_bot, memory_backend).

    This is the single source of truth for agent capabilities. The prompt builder and tool-calling
    interface derive their tool lists from this function.
    """
    tools: List[Tool] = []

    # -- query_memory --------------------------------------------------------
    def query_memory(question: str) -> str:
        executor = context.get("executor")
        memory_backend = context.get("memory_backend")
        if executor is not None and hasattr(executor, "agent"):
            agent = executor.agent
            if hasattr(agent, "run_eqa"):
                discord_text, relevant_images = agent.run_eqa(question)
                return discord_text or "No answer."
        if memory_backend is not None and hasattr(memory_backend, "query_answer"):
            try:
                xyt = context.get("xyt_for_query")
                planner = context.get("planner")
                out = memory_backend.query_answer(question, xyt, planner)
                reasoning, answer, confidence, _, _, relevant_images = out[:6]
                return f"{answer}. (Confidence: {confidence})"
            except NotImplementedError:
                pass
        return "Memory or EQA not available."

    tools.append(Tool(
        name="query_memory",
        description="Answer a question about what the robot has seen (memory log). E.g. How far is it to the sink? Have I seen a red cylinder?",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "Natural language question about the environment or memory."}},
            "required": ["question"],
        },
        func=query_memory,
    ))

    # -- send_image ----------------------------------------------------------
    def send_image() -> str:
        discord_bot = context.get("discord_bot")
        robot = context.get("robot")
        image = None
        if robot is not None and hasattr(robot, "get_observation"):
            obs = robot.get_observation()
            if obs is not None and getattr(obs, "rgb", None) is not None:
                image = np.asarray(obs.rgb)
        if discord_bot is not None and image is not None:
            if hasattr(discord_bot, "push_task_to_all_channels"):
                discord_bot.push_task_to_all_channels(
                    message="Here's what I see:", content=image,
                )
                return "Image sent to Discord."
        if image is not None:
            return "Image captured (no Discord to send to)."
        return "No image available."

    tools.append(Tool(
        name="send_image",
        description="Send a picture to the user (e.g. what the robot sees). Use when asked to show a photo.",
        parameters=_NO_PARAMS,
        func=send_image,
    ))

    # -- explore -------------------------------------------------------------
    def _exec(cmd: str, args: Any = "") -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([(cmd, args)])
        return "Done." if ok else "Command was interrupted or failed."

    tools.append(Tool(
        name="explore",
        description="Explore and build a map of the environment.",
        parameters=_NO_PARAMS,
        func=lambda: _exec("explore", ""),
        executor_commands=_simple_exec_mapping("explore"),
    ))

    # -- pick_place ----------------------------------------------------------
    def pick_place(object_name: str, receptacle_name: str) -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([("pickup", object_name), ("place", receptacle_name)])
        return f"Pick and place ({object_name} -> {receptacle_name}) done." if ok else "Pick/place failed or interrupted."

    tools.append(Tool(
        name="pick_place",
        description="Pick up an object and place it on a receptacle.",
        parameters={
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "Name of the object to pick up."},
                "receptacle_name": {"type": "string", "description": "Name of the receptacle or surface to place it on."},
            },
            "required": ["object_name", "receptacle_name"],
        },
        func=pick_place,
        executor_commands=lambda args: [("pickup", args.get("object_name", "")), ("place", args.get("receptacle_name", ""))],
    ))

    # -- find_objects --------------------------------------------------------
    tools.append(Tool(
        name="find_objects",
        description="Find and navigate to an object or location in the environment by name.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Object or location name to find."}},
            "required": ["text"],
        },
        func=lambda text: _exec("find", text),
        executor_commands=lambda args: [("find", args.get("text", ""))],
    ))

    # -- describe_scene ------------------------------------------------------
    def describe_scene() -> str:
        robot = context.get("robot")
        if robot is None or not hasattr(robot, "get_observation"):
            return "No robot view available."
        obs = robot.get_observation()
        if obs is None or getattr(obs, "rgb", None) is None:
            return "No current image."
        return "I can see the environment through my camera. Use send_image to share the picture."

    tools.append(Tool(
        name="describe_scene",
        description="Describe what you see from the robot's cameras.",
        parameters=_NO_PARAMS,
        func=describe_scene,
    ))

    # -- say -----------------------------------------------------------------
    tools.append(Tool(
        name="say",
        description="Speak text to the user via TTS (text-to-speech). Use to announce what you are doing.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The message to speak aloud."}},
            "required": ["text"],
        },
        func=lambda text: _exec("say", text),
        executor_commands=lambda args: [("say", args.get("text", ""))],
    ))

    # -- emotes and gestures -------------------------------------------------
    for cmd, desc in [
        ("wave", "Wave at a person (e.g. when greeting or saying goodbye)."),
        ("nod_head", "Nod the robot's head (e.g. to indicate yes or agreement)."),
        ("shake_head", "Shake the robot's head (e.g. to indicate no or disagreement)."),
        ("avert_gaze", "Avert the robot's gaze (look away)."),
    ]:
        tools.append(Tool(
            name=cmd,
            description=desc,
            parameters=_NO_PARAMS,
            func=lambda _cmd=cmd: _exec(_cmd, ""),
            executor_commands=_simple_exec_mapping(cmd),
        ))

    # -- navigation ----------------------------------------------------------
    tools.append(Tool(
        name="go_home",
        description="Navigate the robot back to its starting position. Requires a map from prior exploration.",
        parameters=_NO_PARAMS,
        func=lambda: _exec("go_home", ""),
        executor_commands=_simple_exec_mapping("go_home"),
    ))

    tools.append(Tool(
        name="scan_environment",
        description="Rotate in place to scan the environment and update the map (360-degree scan). Saves memory.",
        parameters=_NO_PARAMS,
        func=lambda: _exec("rotate_in_place", ""),
        executor_commands=_simple_exec_mapping("rotate_in_place"),
    ))

    # -- camera --------------------------------------------------------------
    tools.append(Tool(
        name="take_picture",
        description="Take a picture with the main camera. Does NOT send it — use send_image after to send.",
        parameters=_NO_PARAMS,
        func=lambda: _exec("take_picture", ""),
        executor_commands=_simple_exec_mapping("take_picture"),
    ))

    tools.append(Tool(
        name="take_ee_picture",
        description="Take a picture with the end-effector (wrist) camera.",
        parameters=_NO_PARAMS,
        func=lambda: _exec("take_ee_picture", ""),
        executor_commands=_simple_exec_mapping("take_ee_picture"),
    ))

    # -- manipulation --------------------------------------------------------
    tools.append(Tool(
        name="hand_over",
        description="Hand the held object to a person (find person, navigate, extend arm).",
        parameters=_NO_PARAMS,
        func=lambda: _exec("hand_over", ""),
        executor_commands=_simple_exec_mapping("hand_over"),
    ))

    # -- quit ----------------------------------------------------------------
    tools.append(Tool(
        name="quit",
        description="End the conversation and stop the robot. Use when the user says goodbye or asks to stop.",
        parameters=_NO_PARAMS,
        func=lambda: None,
        executor_commands=_simple_exec_mapping("quit"),
    ))

    return tools


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_tool_schemas_for_llm(context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return OpenAI-style tool definitions derived from the tool registry.

    These are suitable for passing as the `tools` parameter to the OpenAI API,
    or for building the system prompt for local models.
    """
    if context is None:
        context = {}
    return [t.schema() for t in get_tools(context)]


def get_tool_descriptions_for_prompt(tools: List[Tool]) -> str:
    """Build a compact tools block for the system prompt from Tool objects."""
    lines = ["TOOLS (use these names in tool_calls):"]
    for t in tools:
        props = t.parameters.get("properties", {})
        if props:
            arg_names = ", ".join(props.keys())
            lines.append(f"- {t.name}({arg_names}): {t.description}")
        else:
            lines.append(f"- {t.name}(): {t.description}")
    return "\n".join(lines)
