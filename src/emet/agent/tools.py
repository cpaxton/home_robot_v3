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
# Agent tools: single registry of tools with name, description, parameters (JSON Schema), func,
# and executor command mapping. The prompt and tool-calling interface derive from this registry.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from emet.agent.camera_debug import print_camera_frame_diagnostics
from emet.utils.logger import Logger
from emet.visualization.map_snapshot import format_navigation_report, snapshot_from_voxel_map

_logger = Logger(__name__)


class Tool:
    """A single agent tool: name, description, JSON Schema parameters, callable, and executor mapping.

    If returns_info is True, the tool returns text that the LLM should see and
    summarize for the user (e.g. query_memory, describe_scene).  The agent loop
    will feed the result back to the LLM for a follow-up response.
    """

    __slots__ = ("name", "description", "parameters", "func", "executor_commands", "returns_info")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
        executor_commands: Callable[[dict[str, Any]], list[tuple[str, str]]] | None = None,
        returns_info: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.executor_commands = executor_commands
        self.returns_info = returns_info

    def schema(self) -> dict[str, Any]:
        """Return OpenAI-style tool schema (for tools param or prompt generation)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_executor(self, arguments: dict[str, Any]) -> list[tuple[str, str]]:
        """Map tool call arguments to executor (command, args) list. Empty list means use func instead."""
        if self.executor_commands is not None:
            return self.executor_commands(arguments)
        return []


_NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}


def _robot_base_xy(robot: Any) -> tuple[float, float] | None:
    if robot is None or not hasattr(robot, "get_base_pose"):
        return None
    try:
        bp = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
        if bp.size >= 2:
            return float(bp[0]), float(bp[1])
    except Exception:
        return None
    return None


def _voxel_map_from_executor(executor: Any) -> Any:
    if executor is None or not hasattr(executor, "agent"):
        return None
    agent = executor.agent
    if hasattr(agent, "get_voxel_map"):
        return agent.get_voxel_map()
    return None


def _simple_exec_mapping(cmd: str) -> Callable[[dict[str, Any]], list[tuple[str, str]]]:
    """Return a mapping function for a no-arg executor command."""
    return lambda args: [(cmd, "")]


def get_tools(context: dict[str, Any]) -> list[Tool]:
    """Return list of tools available given context (executor, robot, discord_bot, memory_backend).

    This is the single source of truth for agent capabilities. The prompt builder and tool-calling
    interface derive their tool lists from this function.
    """
    tools: list[Tool] = []

    # -- query_memory --------------------------------------------------------
    def query_memory(question: str) -> str:
        memory_backend = context.get("memory_backend")
        if memory_backend is not None and hasattr(memory_backend, "query_answer"):
            try:
                xyt = context.get("xyt_for_query")
                planner = context.get("planner")
                out = memory_backend.query_answer(question, xyt, planner)
                reasoning, answer, confidence, _, _, relevant_images = out[:6]
                return f"{answer} (Confidence: {confidence})"
            except NotImplementedError:
                pass
            except AttributeError as e:
                _logger.warning("query_memory backend call failed (%s); using localize_text fallback.", e)
        # Fallback: check if the object is in the voxel map via localize_text
        executor = context.get("executor")
        if executor is not None and hasattr(executor, "agent"):
            voxel_map = executor.agent.get_voxel_map()
            if voxel_map is not None and hasattr(voxel_map, "localize_text"):
                result = voxel_map.localize_text(question, return_debug=True)
                point = result[0] if isinstance(result, (list, tuple)) else result
                if point is not None:
                    coords = point.squeeze()
                    return f"Yes, found at approximately ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})."
                return "I haven't seen that in my memory."
        return "Memory not available."

    tools.append(
        Tool(
            name="query_memory",
            description=(
                "Questions about DynaMem voxel map / semantic memory (where is X, have I seen Y) when voxel EQA is enabled. "
                "For graph-style questions (relations, what connects to what) use list_scene_relations or query_scene_graph. "
                "For open-ended 'what do you see', prefer describe_scene and send_image unless full voxel EQA is enabled."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the environment or memory.",
                    }
                },
                "required": ["question"],
            },
            func=query_memory,
            returns_info=True,
        )
    )

    # -- send_image ----------------------------------------------------------
    def send_image() -> str:
        discord_bot = context.get("discord_bot")
        robot = context.get("robot")
        image = None
        if robot is not None and hasattr(robot, "get_observation"):
            obs = robot.get_observation()
            if obs is not None and getattr(obs, "rgb", None) is not None:
                # Copy so ZMQ recv cannot mutate the buffer while Discord async-upload runs.
                image = np.asarray(obs.rgb).copy()
        if image is not None:
            print_camera_frame_diagnostics(
                "send_image (head obs rgb → Discord)",
                image,
                force=bool(context.get("verbose_tools")) or bool(context.get("camera_debug")),
            )
        if discord_bot is not None and image is not None:
            if hasattr(discord_bot, "push_task_to_all_channels"):
                # Image only: the assistant message (e.g. "Here's a picture…") is already sent by the agent loop.
                discord_bot.push_task_to_all_channels(message=None, content=image)
                return "Image sent to Discord."
        if image is not None:
            return "Image captured (no Discord to send to)."
        return "No image available."

    tools.append(
        Tool(
            name="send_image",
            description="Send a picture to the user (e.g. what the robot sees). Use when asked to show a photo.",
            parameters=_NO_PARAMS,
            func=send_image,
        )
    )

    def _exec(cmd: str, args: Any = "") -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([(cmd, args)])
        return "Done." if ok else "Command was interrupted or failed."

    # -- explore -------------------------------------------------------------
    def explore() -> str:
        executor = context.get("executor")
        robot = context.get("robot")
        if executor is None:
            return "Robot not connected."
        ok = executor([("explore", "")])
        robot_xy = _robot_base_xy(robot)
        vm = _voxel_map_from_executor(executor)
        _img, stats, _ = snapshot_from_voxel_map(vm, robot_xy)
        summary = format_navigation_report(stats, explore_ok=ok)
        head = "Explore finished." if ok else "Explore failed or interrupted."
        return f"{head} {summary}"

    tools.append(
        Tool(
            name="explore",
            description=(
                "Explore and build a map of the environment. Returns a short map diagnostic "
                "(coverage, base cell) after the run — not a camera stream; pair with send_map_snapshot or describe_scene if stuck."
            ),
            parameters=_NO_PARAMS,
            func=explore,
            returns_info=True,
        )
    )

    # -- navigation_diagnostics / send_map_snapshot -------------------------
    def navigation_diagnostics() -> str:
        executor = context.get("executor")
        robot = context.get("robot")
        if executor is None:
            return "Robot not connected."
        robot_xy = _robot_base_xy(robot)
        vm = _voxel_map_from_executor(executor)
        _img, stats, _ = snapshot_from_voxel_map(vm, robot_xy)
        return format_navigation_report(stats, explore_ok=None)

    def send_map_snapshot() -> str:
        executor = context.get("executor")
        robot = context.get("robot")
        discord_bot = context.get("discord_bot")
        if executor is None:
            return "Robot not connected."
        robot_xy = _robot_base_xy(robot)
        vm = _voxel_map_from_executor(executor)
        img, stats, img_discord = snapshot_from_voxel_map(vm, robot_xy)
        summary = format_navigation_report(stats, explore_ok=None)
        if img is None:
            return f"No map image available. {summary}"
        viz = None
        if hasattr(executor, "agent"):
            viz = getattr(executor.agent, "rerun_visualizer", None)
        rerun_logged = False
        if viz is not None and getattr(viz, "enabled", True) and hasattr(viz, "log_custom_2d_image"):
            try:
                viz.log_custom_2d_image("world/map_snapshot/topdown", img)
                rerun_logged = True
            except Exception as e:
                _logger.warning(f"Rerun map snapshot log failed: {e}")
        discord_sent = False
        if discord_bot is not None and hasattr(discord_bot, "push_task_to_all_channels"):
            try:
                to_send = img_discord if img_discord is not None else img
                discord_bot.push_task_to_all_channels(message=None, content=np.asarray(to_send).copy())
                discord_sent = True
            except Exception as e:
                _logger.warning(f"Discord map snapshot failed: {e}")
        parts = [summary]
        if discord_sent:
            parts.append("Top-down map image sent to Discord (cropped to explored region).")
        if rerun_logged:
            parts.append("Top-down map logged to Rerun at world/map_snapshot/topdown (cropped to explored region).")
        return " ".join(parts)

    tools.append(
        Tool(
            name="navigation_diagnostics",
            description=(
                "Text summary of the current 2D voxel map: explored vs obstacle cell counts, base pose in grid, "
                "and hints if the map is empty or the base sits on an obstacle cell. Use after failed explore/find or when the user asks why navigation failed."
            ),
            parameters=_NO_PARAMS,
            func=navigation_diagnostics,
            returns_info=True,
        )
    )

    tools.append(
        Tool(
            name="send_map_snapshot",
            description=(
                "Render a top-down RGB view of obstacles vs explored space (cropped to the explored region plus margin) "
                "and send to Discord if configured; also logs the same crop to Rerun at world/map_snapshot/topdown "
                "when the live Rerun visualizer is enabled."
            ),
            parameters=_NO_PARAMS,
            func=send_map_snapshot,
            returns_info=True,
        )
    )

    # -- pick_place ----------------------------------------------------------
    def pick_place(object_name: str, receptacle_name: str) -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([("pickup", object_name), ("place", receptacle_name)])
        return (
            f"Pick and place ({object_name} -> {receptacle_name}) done." if ok else "Pick/place failed or interrupted."
        )

    tools.append(
        Tool(
            name="pick_place",
            description="Pick up an object and place it on a receptacle.",
            parameters={
                "type": "object",
                "properties": {
                    "object_name": {"type": "string", "description": "Name of the object to pick up."},
                    "receptacle_name": {
                        "type": "string",
                        "description": "Name of the receptacle or surface to place it on.",
                    },
                },
                "required": ["object_name", "receptacle_name"],
            },
            func=pick_place,
            executor_commands=lambda args: [
                ("pickup", args.get("object_name", "")),
                ("place", args.get("receptacle_name", "")),
            ],
        )
    )

    # -- find_objects --------------------------------------------------------
    tools.append(
        Tool(
            name="find_objects",
            description="Find and navigate to an object or location in the environment by name.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Object or location name to find."}},
                "required": ["text"],
            },
            func=lambda text: _exec("find", text),
            executor_commands=lambda args: [("find", args.get("text", ""))],
        )
    )

    # -- describe_scene ------------------------------------------------------
    def describe_scene() -> str:
        robot = context.get("robot")
        executor = context.get("executor")
        if robot is None or not hasattr(robot, "get_observation"):
            return "No robot view available."
        obs = robot.get_observation()
        if obs is None or getattr(obs, "rgb", None) is None:
            return "No current image."
        agent = getattr(executor, "agent", None) if executor is not None else None
        if agent is not None and hasattr(agent, "describe_head_camera_scene_text"):
            return agent.describe_head_camera_scene_text()
        return (
            "I have a camera frame but this session's controller does not expose scene description; "
            "use send_image to show the view."
        )

    tools.append(
        Tool(
            name="describe_scene",
            description=(
                "Brief text about the camera view; pair with send_image to show the user a photo. "
                "Use for 'what can you see' style questions instead of query_memory when not using full EQA. "
                'With send_image, use an empty JSON "message" on the tool-call turn so chat/Discord only show your answer after [Tool results].'
            ),
            parameters=_NO_PARAMS,
            func=describe_scene,
            returns_info=True,
        )
    )

    # -- say -----------------------------------------------------------------
    tools.append(
        Tool(
            name="say",
            description="Speak text to the user via TTS (text-to-speech). Use to announce what you are doing.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The message to speak aloud."}},
                "required": ["text"],
            },
            func=lambda text: _exec("say", text),
            executor_commands=lambda args: [("say", args.get("text", ""))],
        )
    )

    # -- emotes and gestures -------------------------------------------------
    for cmd, desc in [
        ("wave", "Wave at a person (e.g. when greeting or saying goodbye)."),
        ("nod_head", "Nod the robot's head (e.g. to indicate yes or agreement)."),
        ("shake_head", "Shake the robot's head (e.g. to indicate no or disagreement)."),
        ("avert_gaze", "Avert the robot's gaze (look away)."),
    ]:
        tools.append(
            Tool(
                name=cmd,
                description=desc,
                parameters=_NO_PARAMS,
                func=lambda _cmd=cmd: _exec(_cmd, ""),
                executor_commands=_simple_exec_mapping(cmd),
            )
        )

    # -- navigation ----------------------------------------------------------
    tools.append(
        Tool(
            name="go_home",
            description="Navigate the robot back to its starting position. Requires a map from prior exploration.",
            parameters=_NO_PARAMS,
            func=lambda: _exec("go_home", ""),
            executor_commands=_simple_exec_mapping("go_home"),
        )
    )

    tools.append(
        Tool(
            name="scan_environment",
            description="Rotate in place to scan the environment and update the map (360-degree scan). Saves memory.",
            parameters=_NO_PARAMS,
            func=lambda: _exec("rotate_in_place", ""),
            executor_commands=_simple_exec_mapping("rotate_in_place"),
        )
    )

    # -- camera --------------------------------------------------------------
    tools.append(
        Tool(
            name="take_picture",
            description="Take a picture with the main camera. Does NOT send it — use send_image after to send.",
            parameters=_NO_PARAMS,
            func=lambda: _exec("take_picture", ""),
            executor_commands=_simple_exec_mapping("take_picture"),
        )
    )

    tools.append(
        Tool(
            name="take_ee_picture",
            description="Take a picture with the end-effector (wrist) camera.",
            parameters=_NO_PARAMS,
            func=lambda: _exec("take_ee_picture", ""),
            executor_commands=_simple_exec_mapping("take_ee_picture"),
        )
    )

    # -- manipulation --------------------------------------------------------
    tools.append(
        Tool(
            name="hand_over",
            description="Hand the held object to a person (find person, navigate, extend arm).",
            parameters=_NO_PARAMS,
            func=lambda: _exec("hand_over", ""),
            executor_commands=_simple_exec_mapping("hand_over"),
        )
    )

    # -- query_scene_graph (GraphEQA + open-vocab fallback) -------------------
    def query_scene_graph(question: str) -> str:
        gmb = context.get("graph_memory_backend")
        if gmb is not None and hasattr(gmb, "query_answer"):
            try:
                xyt = context.get("xyt_for_query")
                planner = context.get("planner")
                out = gmb.query_answer(question, xyt, planner)
                reasoning, answer, confidence, cr, _, _ = out[:6]
                tail = f" ({cr})" if cr else ""
                return f"{answer} (confidence={confidence}){tail}\nReasoning: {reasoning}"
            except Exception as e:
                _logger.warning("query_scene_graph graph backend failed: %s", e)
        executor = context.get("executor")
        if executor is not None and hasattr(executor, "agent"):
            sg = executor.agent.get_voxel_map().get_scene_graph()
            if sg is not None and sg.num_objects > 0:
                return f"[Open-vocab scene graph snapshot]\n{sg.to_string()}\n(User question was: {question})"
        return "No graph memory or open-vocab scene graph available yet; explore or scan first."

    tools.append(
        Tool(
            name="query_scene_graph",
            description=(
                "Embodied questions using GraphEQA memory (objects, navigation context) when enabled; "
                "otherwise dumps the open-vocab spatial scene graph as text. Use for 'why', relational, or multi-step scene questions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the scene graph / objects.",
                    },
                },
                "required": ["question"],
            },
            func=query_scene_graph,
            returns_info=True,
        )
    )

    def list_scene_relations() -> str:
        executor = context.get("executor")
        if executor is None or not hasattr(executor, "agent"):
            return "Robot not connected."
        sg = executor.agent.get_voxel_map().get_scene_graph()
        if sg is None or sg.num_objects == 0:
            return "No open-vocab scene graph data yet."
        return sg.to_string()

    tools.append(
        Tool(
            name="list_scene_relations",
            description=(
                "List objects and spatial relations (near, on, on_floor) from the open-vocabulary 3D scene graph. "
                "Use for 'what is connected to what' and structured connectivity questions."
            ),
            parameters=_NO_PARAMS,
            func=list_scene_relations,
            returns_info=True,
        )
    )

    def send_object_image(object_label: str) -> str:
        discord_bot = context.get("discord_bot")
        executor = context.get("executor")
        if executor is None or not hasattr(executor, "agent"):
            return "Robot not connected."
        sg = executor.agent.get_voxel_map().get_scene_graph()
        if sg is None:
            return "Open-vocab scene graph is not available."
        node = sg.get_node_by_label(object_label)
        if node is None:
            return f"No object matching label {object_label!r} in the scene graph."
        crop = getattr(node, "best_crop", None)
        if crop is None:
            return f"Object {object_label!r} has no stored crop image yet."
        image = np.asarray(crop).copy()
        if discord_bot is not None and image is not None and hasattr(discord_bot, "push_task_to_all_channels"):
            discord_bot.push_task_to_all_channels(message=None, content=image)
            return f"Sent last crop image for {object_label!r} to Discord."
        return "Crop available but Discord is not connected."

    tools.append(
        Tool(
            name="send_object_image",
            description=(
                "Send the robot's last stored crop image for a named object from the open-vocab scene graph "
                "(not the live camera). Use after the object has been observed while mapping."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "object_label": {"type": "string", "description": "Object name or label (e.g. red cylinder)."},
                },
                "required": ["object_label"],
            },
            func=send_object_image,
        )
    )

    # -- quit ----------------------------------------------------------------
    tools.append(
        Tool(
            name="quit",
            description="End the conversation and stop the robot. Use when the user says goodbye or asks to stop.",
            parameters=_NO_PARAMS,
            func=lambda: None,
            executor_commands=_simple_exec_mapping("quit"),
        )
    )

    return tools


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_tool_schemas_for_llm(context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return OpenAI-style tool definitions derived from the tool registry.

    These are suitable for passing as the `tools` parameter to the OpenAI API,
    or for building the system prompt for local models.
    """
    if context is None:
        context = {}
    return [t.schema() for t in get_tools(context)]


def get_tool_descriptions_for_prompt(tools: list[Tool]) -> str:
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
