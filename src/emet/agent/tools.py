# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent tools: query_memory, send_image, explore, pick_place, find_objects, describe_scene, etc.
# Tools delegate to executor, memory backend, and Discord bot when available.
# Tool registry provides name, description, and JSON Schema parameters for LLM tool-calling.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# JSON Schema for tool parameters (OpenAI-style). Used by get_tool_schemas_for_llm() and agent prompt.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "query_memory",
        "description": "Answer a question about what the robot has seen (memory log). E.g. How far is it to the sink? Have I seen a red cylinder?",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "Natural language question about the environment or memory."}},
            "required": ["question"],
        },
    },
    {
        "name": "send_image",
        "description": "Send a picture to the user (e.g. what the robot sees, or an image from memory). Use when asked to show a photo.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "explore",
        "description": "Explore and build a map of the environment.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pick_place",
        "description": "Pick up an object and place it on a receptacle. Provide object and receptacle names.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "Name of the object to pick up."},
                "receptacle_name": {"type": "string", "description": "Name of the receptacle or surface to place it on."},
            },
            "required": ["object_name", "receptacle_name"],
        },
    },
    {
        "name": "find_objects",
        "description": "Find and localize an object in the environment by name.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Object or location name to find."}},
            "required": ["text"],
        },
    },
    {
        "name": "describe_scene",
        "description": "Describe what you see from the robot's cameras.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "say",
        "description": "Speak text to the user via TTS (text-to-speech). Use to announce what you are doing or to reply verbally.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The message to speak aloud."}},
            "required": ["text"],
        },
    },
    {
        "name": "wave",
        "description": "Wave at a person (e.g. when greeting or saying goodbye).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "nod_head",
        "description": "Nod the robot's head (e.g. to indicate yes or agreement).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "shake_head",
        "description": "Shake the robot's head (e.g. to indicate no or disagreement).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "avert_gaze",
        "description": "Avert the robot's gaze (look away).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "go_home",
        "description": "Navigate the robot back to its starting position (home). Requires a map from prior exploration.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "scan_environment",
        "description": "Rotate in place to scan the environment and update the map (360-degree scan). Saves memory.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "take_picture",
        "description": "Take a picture with the robot's main camera and optionally send it to the user.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "take_ee_picture",
        "description": "Take a picture with the end-effector (wrist) camera.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "hand_over",
        "description": "Hand the object currently held by the robot to a person (find person, extend arm toward them).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "quit",
        "description": "End the conversation and stop the robot. Use when the user says goodbye or asks to stop.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


def get_tool_schemas_for_llm(context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return tool definitions (name, description, parameters) for the LLM system prompt or API tools payload.

    If context is provided, can filter or adapt (e.g. only tools that have executor). For now returns all TOOL_SCHEMAS.
    """
    return list(TOOL_SCHEMAS)


def get_tools(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of tools available given context (executor, robot, discord_bot, memory_backend).

    Each tool is a dict: {"name": str, "description": str, "func": callable}.
    """
    tools = []

    # query_memory: ask a question about loaded memory (EQA); returns (answer_text, relevant_images).
    def query_memory(question: str) -> Tuple[str, List[Any]]:
        executor = context.get("executor")
        memory_backend = context.get("memory_backend")
        if executor is not None and hasattr(executor, "agent"):
            agent = executor.agent
            if hasattr(agent, "run_eqa"):
                discord_text, relevant_images = agent.run_eqa(question)
                return discord_text or "No answer.", relevant_images if relevant_images is not None else []
        if memory_backend is not None and hasattr(memory_backend, "query_answer"):
            try:
                xyt = context.get("xyt_for_query")
                planner = context.get("planner")
                out = memory_backend.query_answer(question, xyt, planner)
                reasoning, answer, confidence, _, _, relevant_images = out[:6]
                text = f"{answer}. (Confident: {confidence})"
                imgs = list(relevant_images) if relevant_images is not None else []
                return text, imgs
            except NotImplementedError:
                pass
        return "Memory or EQA not available.", []

    tools.append({
        "name": "query_memory",
        "description": "Answer a question about what the robot has seen (memory log). E.g. How far is it to the sink? Have I seen a red cylinder?",
        "func": query_memory,
    })

    # send_image: send an image to Discord (or no-op if no bot).
    def send_image(image: Optional[np.ndarray] = None, channel: Any = None) -> str:
        discord_bot = context.get("discord_bot")
        if discord_bot is None:
            return "Discord not connected; cannot send image."
        if image is None:
            robot = context.get("robot")
            if robot is not None and hasattr(robot, "get_observation"):
                obs = robot.get_observation()
                if obs is not None and getattr(obs, "rgb", None) is not None:
                    image = np.asarray(obs.rgb)
        if image is None:
            return "No image available to send."
        if hasattr(discord_bot, "push_task_to_all_channels"):
            discord_bot.push_task_to_all_channels(content=image)
            return "Image sent to Discord."
        if channel is not None and hasattr(discord_bot, "push_task"):
            discord_bot.push_task(channel, content=image)
            return "Image sent."
        return "Could not send image."

    tools.append({
        "name": "send_image",
        "description": "Send a picture to the user (e.g. what the robot sees, or an image from memory). Use when asked to show a photo.",
        "func": send_image,
    })

    # explore: run exploration and update map.
    def explore() -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected; cannot explore."
        ok = executor([("explore", None)])
        return "Exploration finished." if ok else "Exploration was interrupted."

    tools.append({
        "name": "explore",
        "description": "Explore and build a map of the environment.",
        "func": explore,
    })

    # pick_place: pick object and place on receptacle.
    def pick_place(object_name: str, receptacle_name: str) -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected; cannot pick or place."
        ok = executor([("pickup", object_name), ("place", receptacle_name)])
        return f"Pick and place ({object_name} -> {receptacle_name}) done." if ok else "Pick/place failed or interrupted."

    tools.append({
        "name": "pick_place",
        "description": "Pick up an object and place it on a receptacle. Provide object and receptacle names.",
        "func": pick_place,
    })

    # find_objects: localize object by name.
    def find_objects(text: str) -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected; cannot find objects."
        ok = executor([("find", text)])
        return f"Find '{text}' completed." if ok else "Find failed or interrupted."

    tools.append({
        "name": "find_objects",
        "description": "Find and localize an object in the environment by name.",
        "func": find_objects,
    })

    # describe_scene: describe what the robot sees (optional; could use vision model).
    def describe_scene() -> str:
        robot = context.get("robot")
        if robot is None or not hasattr(robot, "get_observation"):
            return "No robot view available."
        obs = robot.get_observation()
        if obs is None or getattr(obs, "rgb", None) is None:
            return "No current image."
        return "I see the current camera view. (Use send_image to show a picture.)"

    tools.append({
        "name": "describe_scene",
        "description": "Describe what you see from the robot's cameras.",
        "func": describe_scene,
    })

    def _exec(cmd: str, args: Any = "") -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([(cmd, args)])
        return "Done." if ok else "Command was interrupted or failed."

    tools.append({
        "name": "say",
        "description": "Speak text to the user via TTS (text-to-speech). Use to announce what you are doing or to reply verbally.",
        "func": lambda text: _exec("say", text),
    })
    tools.append({
        "name": "wave",
        "description": "Wave at a person (e.g. when greeting or saying goodbye).",
        "func": lambda: _exec("wave", ""),
    })
    tools.append({
        "name": "nod_head",
        "description": "Nod the robot's head (e.g. to indicate yes or agreement).",
        "func": lambda: _exec("nod_head", ""),
    })
    tools.append({
        "name": "shake_head",
        "description": "Shake the robot's head (e.g. to indicate no or disagreement).",
        "func": lambda: _exec("shake_head", ""),
    })
    tools.append({
        "name": "avert_gaze",
        "description": "Avert the robot's gaze (look away).",
        "func": lambda: _exec("avert_gaze", ""),
    })
    tools.append({
        "name": "go_home",
        "description": "Navigate the robot back to its starting position (home). Requires a map from prior exploration.",
        "func": lambda: _exec("go_home", ""),
    })
    tools.append({
        "name": "scan_environment",
        "description": "Rotate in place to scan the environment and update the map (360-degree scan). Saves memory.",
        "func": lambda: _exec("rotate_in_place", ""),
    })
    tools.append({
        "name": "take_picture",
        "description": "Take a picture with the robot's main camera and optionally send it to the user.",
        "func": lambda: _exec("take_picture", ""),
    })
    tools.append({
        "name": "take_ee_picture",
        "description": "Take a picture with the end-effector (wrist) camera.",
        "func": lambda: _exec("take_ee_picture", ""),
    })
    tools.append({
        "name": "hand_over",
        "description": "Hand the object currently held by the robot to a person (find person, extend arm toward them).",
        "func": lambda: _exec("hand_over", ""),
    })
    tools.append({
        "name": "quit",
        "description": "End the conversation and stop the robot. Use when the user says goodbye or asks to stop.",
        "func": lambda: None,
    })

    return tools


def tool_call_to_executor_commands(name: str, arguments: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Map a tool call (name + arguments) to a list of (command, args) for DynamemTaskExecutor.__call__."""
    args = arguments or {}
    if name == "query_memory":
        return []  # Handled by tool func, not executor
    if name == "send_image":
        return []
    if name == "explore":
        return [("explore", "")]
    if name == "pick_place":
        return [
            ("pickup", args.get("object_name", "")),
            ("place", args.get("receptacle_name", "")),
        ]
    if name == "find_objects":
        return [("find", args.get("text", ""))]
    if name == "describe_scene":
        return []
    if name == "say":
        return [("say", args.get("text", ""))]
    if name == "wave":
        return [("wave", "")]
    if name == "nod_head":
        return [("nod_head", "")]
    if name == "shake_head":
        return [("shake_head", "")]
    if name == "avert_gaze":
        return [("avert_gaze", "")]
    if name == "go_home":
        return [("go_home", "")]
    if name == "scan_environment":
        return [("rotate_in_place", "")]
    if name == "take_picture":
        return [("take_picture", "")]
    if name == "take_ee_picture":
        return [("take_ee_picture", "")]
    if name == "hand_over":
        return [("hand_over", "")]
    if name == "quit":
        return [("quit", "")]
    return []


def get_tool_descriptions_for_prompt(context: Dict[str, Any]) -> str:
    """Return a newline-separated list of tool names and descriptions for the system prompt."""
    tools = get_tools(context)
    return "\n".join(f"  - {t['name']}: {t['description']}" for t in tools)
