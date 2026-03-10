# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent tools: query_memory, send_image, explore, pick_place, find_objects, describe_scene.
# Tools delegate to executor, memory backend, and Discord bot when available.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


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

    return tools


def get_tool_descriptions_for_prompt(context: Dict[str, Any]) -> str:
    """Return a newline-separated list of tool names and descriptions for the system prompt."""
    tools = get_tools(context)
    return "\n".join(f"  - {t['name']}: {t['description']}" for t in tools)
