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
from emet.mapping.close_map import close_map_from_agent, format_close_map_hint
from emet.mapping.voxel_localize import localize_text_xyz, voxel_map_from_agent
from emet.memory.graph_eqa.graph_stats import format_graph_size_report
from emet.memory.graph_eqa.human_answer import HumanEQAResult, format_eqa_tool_response
from emet.utils.logger import Logger
from emet.visualization.map_snapshot import format_navigation_report, snapshot_from_voxel_map

_logger = Logger(__name__)

# Agent loop reads this and attaches the ndarray to the next Discord text reply (one message).
PENDING_DISCORD_IMAGE_KEY = "pending_discord_image"


def _graph_size_line(context: dict[str, Any], *, verbose: bool = True) -> str:
    """Compact graph growth line from chat context (empty if no graph)."""
    gm = context.get("graph_memory")
    if gm is None:
        executor = context.get("executor")
        agent = getattr(executor, "agent", None) if executor is not None else None
        gm = getattr(agent, "graph_memory", None)
    if gm is None:
        return ""
    return format_graph_size_report(gm, verbose=verbose)


def stash_discord_image(context: dict[str, Any], image: np.ndarray | None) -> bool:
    """Copy *image* into ``context`` for the agent loop to send with the user-facing reply."""
    if image is None:
        return False
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4) or arr.size == 0:
        return False
    context[PENDING_DISCORD_IMAGE_KEY] = arr.copy()
    return True


def _agent_from_context(context: dict[str, Any]) -> Any | None:
    executor = context.get("executor")
    if executor is not None and hasattr(executor, "agent"):
        return executor.agent
    return context.get("agent")


def format_last_nav_plan_summary(agent: Any | None) -> str:
    """Compact last-plan line for explore/find/diagnostics tool returns."""
    if agent is None:
        return ""
    meta = getattr(agent, "_last_nav_plan", None)
    if not isinstance(meta, dict) or not meta:
        return ""
    parts: list[str] = []
    src = meta.get("localize_source") or meta.get("mode")
    if src:
        parts.append(f"localize={src}")
    n = meta.get("n_planned")
    if n is None:
        n = meta.get("n_waypoints")
    if n is not None:
        parts.append(f"planned_wps={n}")
    path_m = meta.get("path_m") or meta.get("full_path_m")
    if path_m:
        try:
            parts.append(f"path≈{float(path_m):.2f}m")
        except (TypeError, ValueError):
            pass
    min_c = meta.get("min_clearance_m")
    if min_c is not None:
        try:
            parts.append(f"min_clearance={float(min_c):.2f}m")
        except (TypeError, ValueError):
            pass
    outcome = meta.get("outcome")
    if outcome:
        parts.append(f"outcome={outcome}")
    status_code = meta.get("status_code")
    if status_code and status_code != outcome:
        parts.append(f"status={status_code}")
    if meta.get("confirmed") is True:
        parts.append("confirmed=yes")
    elif meta.get("confirmed") is False or outcome == "user_cancelled":
        parts.append("user_cancelled")
    if not parts:
        return ""
    return "Last plan: " + ", ".join(parts) + "."


def format_nav_outcome_head(outcome: str | None, *, ok: bool | None, verb: str) -> str:
    """Human head line for explore/find tool returns from ``_last_nav_plan`` outcome."""
    if outcome == "user_cancelled":
        return f"{verb} cancelled by user (confirm-nav)."
    if outcome == "aborted_waypoint_timeout":
        return f"{verb} aborted: waypoint timeout (near wall or unreachable heading)."
    if outcome in {"rejected_low_clearance", "rejected_unexplored"}:
        return f"{verb} plan rejected ({outcome})."
    if ok is True:
        return f"{verb} finished."
    if ok is False:
        return f"{verb} failed or interrupted."
    return f"{verb} did not complete."


def format_base_clearance_hint(agent: Any | None) -> str:
    """Whether the base currently sits under the configured min clearance."""
    if agent is None:
        return ""
    planner = getattr(agent, "planner", None)
    if planner is None or not hasattr(planner, "clearance_at_xy"):
        return ""
    robot = getattr(agent, "robot", None)
    if robot is None or not hasattr(robot, "get_base_pose"):
        return ""
    try:
        pose = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
        if pose.size < 2:
            return ""
        if getattr(planner, "_clearance_m", None) is None:
            planner.reset()
        c = float(planner.clearance_at_xy(pose[:2]))
        req = float(getattr(agent, "_min_clearance_m", getattr(planner, "min_clearance_m", 0.0)) or 0.0)
        if req > 0 and c < req:
            return f"Base clearance {c:.2f}m is below min_clearance_m={req:.2f}m (near obstacle)."
        return f"Base clearance ≈ {c:.2f}m (min required {req:.2f}m)."
    except Exception:
        return ""


def take_pending_discord_image(context: dict[str, Any] | None) -> np.ndarray | None:
    """Pop a stashed RGB image from *context*, or None."""
    if not context:
        return None
    img = context.pop(PENDING_DISCORD_IMAGE_KEY, None)
    if img is None:
        return None
    return np.asarray(img)


def pending_discord_image_for_send(
    context: dict[str, Any] | None,
    *,
    attach_pending_image: bool = True,
) -> np.ndarray | None:
    """Return a stashed image only when *attach_pending_image* is True.

    Status lines (*Thinking…*) pass ``attach_pending_image=False`` so the crop/photo
    stays queued for the user-facing reply.
    """
    if not attach_pending_image:
        return None
    return take_pending_discord_image(context)


def _graph_eqa_tool_string(query_out: tuple) -> str:
    """Format ``query_answer`` tuple for agent tools (human answer, not image ids)."""
    reasoning, answer, confidence, confidence_reasoning, _, _ = query_out[:6]
    human = HumanEQAResult(
        user_answer=str(answer).strip(),
        location_hint=None,
        confidence_summary="confident" if confidence else "not confident",
        debug_reasoning=(reasoning or confidence_reasoning or "").strip(),
    )
    return format_eqa_tool_response(human)


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


def _robot_base_xy(robot: Any, executor: Any | None = None) -> tuple[float, float] | None:
    """Base XY in the voxel-map world frame (matches visited / explored stamps).

    Prefer the controller's ``world_base_xy`` (gps → world via ``navigation_origin_xyt``).
    Raw ``get_base_pose`` is episode-relative and misplaces the Discord map marker.
    """
    if executor is not None:
        agent = getattr(executor, "agent", None)
        if agent is not None and hasattr(agent, "world_base_xy"):
            try:
                xy = agent.world_base_xy()
                if xy is not None:
                    return float(xy[0]), float(xy[1])
            except Exception:
                pass
        if agent is not None and hasattr(agent, "_planning_base_xyt") and robot is not None:
            try:
                bp = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
                if bp.size >= 2:
                    wxyt = agent._planning_base_xyt(bp)
                    return float(wxyt[0]), float(wxyt[1])
            except Exception:
                pass
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


def build_chat_tools(context: dict[str, Any]) -> list[Tool]:
    """Build the Discord/terminal CHAT tool pack for *context*.

    Prefer :func:`get_tools` (routes through :func:`emet.agent.skills.build_skill_pack`).
    """
    from emet.agent.env_flags import env_base_rotate_only

    tools: list[Tool] = []

    # -- query_memory --------------------------------------------------------
    def query_memory(question: str) -> str:
        xyt = context.get("xyt_for_query")
        planner = context.get("planner")
        graph_backend = context.get("graph_memory_backend")
        if graph_backend is not None and hasattr(graph_backend, "query_answer"):
            try:
                return _graph_eqa_tool_string(graph_backend.query_answer(question, xyt, planner))
            except Exception as e:
                _logger.warning(f"query_memory graph backend failed ({e}); trying voxel memory.")
        memory_backend = context.get("memory_backend")
        if memory_backend is not None and hasattr(memory_backend, "query_answer"):
            try:
                out = memory_backend.query_answer(question, xyt, planner)
                return _graph_eqa_tool_string(out)
            except NotImplementedError:
                pass
            except AttributeError as e:
                _logger.warning(f"query_memory backend call failed ({e}); using localize_text fallback.")
        # Fallback: same gated localize_text helper as agentic investigate cards.
        executor = context.get("executor")
        if executor is not None and hasattr(executor, "agent"):
            xyz, _stats = localize_text_xyz(voxel_map_from_agent(executor.agent), question)
            if xyz is not None:
                return f"Yes, found at approximately ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})."
            return "I haven't seen that in my memory."
        return "Memory not available."

    tools.append(
        Tool(
            name="query_memory",
            description=(
                "Questions about the scene (where is X, have I seen Y). Uses graph EQA when enabled, else voxel memory. "
                "Returns a final user-facing Answer line (not image numbers). "
                "For relations use list_scene_relations; for open-ended views use describe_scene and send_image."
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
        if stash_discord_image(context, image):
            if context.get("discord_bot") is not None:
                return "Image queued for Discord (attached to the reply)."
            return "Image captured (no Discord to send to)."
        return "No image available."

    tools.append(
        Tool(
            name="send_image",
            description=(
                "Capture the head-camera view and attach it to the reply (Discord when connected). "
                "Use with describe_scene for 'what do you see', or alone when asked for a photo."
            ),
            parameters=_NO_PARAMS,
            func=send_image,
            returns_info=True,
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
        agent = _agent_from_context(context)
        robot_xy = _robot_base_xy(robot)
        vm = _voxel_map_from_executor(executor)
        _img, stats, _ = snapshot_from_voxel_map(vm, robot_xy)
        summary = format_navigation_report(stats, explore_ok=ok)
        plan_line = format_last_nav_plan_summary(agent)
        outcome = (getattr(agent, "_last_nav_plan", None) or {}).get("outcome") if agent else None
        head = format_nav_outcome_head(outcome, ok=ok, verb="Explore")
        parts = [head, summary]
        if plan_line:
            parts.append(plan_line)
        gsize = _graph_size_line(context)
        if gsize:
            _logger.info(gsize)
            parts.append(f"[{gsize}]")
        return " ".join(parts)

    tools.append(
        Tool(
            name="explore",
            description=(
                "Navigate to explore and build a map (moves through the space — longer than scan_environment). "
                "Use for 'explore', 'map the room', 'go look around the house'. "
                "For a quick in-place look, prefer scan_environment. "
                "Returns map diagnostics plus last-plan summary (localize source, waypoint count, min clearance, "
                "and outcomes such as user_cancelled / aborted_waypoint_timeout / rejected_low_clearance when "
                "--confirm-nav or safety filters fire). Pair with send_map_snapshot or describe_scene if stuck."
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
        agent = _agent_from_context(context)
        robot_xy = _robot_base_xy(robot, executor)
        vm = _voxel_map_from_executor(executor)
        _img, stats, _ = snapshot_from_voxel_map(vm, robot_xy)
        parts = [format_navigation_report(stats, explore_ok=None)]
        plan_line = format_last_nav_plan_summary(agent)
        if plan_line:
            parts.append(plan_line)
        clr = format_base_clearance_hint(agent)
        if clr:
            parts.append(clr)
        # Action-outcome ledger summary (opt-in; empty when ledger off).
        gm = getattr(agent, "graph_memory", None) if agent is not None else None
        if gm is None:
            gm = context.get("graph_memory")
        if gm is not None and hasattr(gm, "get_attempt_records"):
            try:
                recent = gm.get_attempt_records(limit=5)
            except Exception:
                recent = []
            if recent:
                bits = "; ".join(r.summary_bit() for r in recent)
                parts.append(f"Recent attempts: {bits}.")
        return " ".join(parts)

    def send_map_snapshot() -> str:
        executor = context.get("executor")
        robot = context.get("robot")
        discord_bot = context.get("discord_bot")
        if executor is None:
            return "Robot not connected."
        agent = _agent_from_context(context)
        robot_xy = _robot_base_xy(robot, executor)
        vm = _voxel_map_from_executor(executor)
        traj = None
        meta = getattr(agent, "_last_nav_plan", None) if agent is not None else None
        if isinstance(meta, dict):
            traj = meta.get("traj") or meta.get("waypoints") or meta.get("path")
        if traj is None and agent is not None:
            # Prefer last logged plan from Rerun helper fields if present.
            traj = getattr(agent, "_last_nav_traj", None)
        img = None
        img_discord = None
        if traj is not None:
            try:
                from emet.controller.nav_confirm import render_nav_plan_map_rgb

                img = render_nav_plan_map_rgb(vm, robot_xy, traj)
                img_discord = img
            except Exception as e:
                _logger.debug("nav plan overlay failed: %s", e)
        if img is None:
            img, stats, img_discord = snapshot_from_voxel_map(vm, robot_xy)
        else:
            _img2, stats, _ = snapshot_from_voxel_map(vm, robot_xy)
        summary = format_navigation_report(stats, explore_ok=None)
        plan_line = format_last_nav_plan_summary(agent)
        if plan_line:
            summary = f"{summary} {plan_line}"
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
        if traj is not None:
            parts.append("Last motion plan overlaid on map.")
        return " ".join(parts)

    tools.append(
        Tool(
            name="navigation_diagnostics",
            description=(
                "Text summary of the current 2D voxel map: explored vs obstacle cell counts, base pose in grid, "
                "last motion-plan summary (localize / waypoints / min clearance / confirm-nav or abort outcomes), "
                "and whether the base sits under min_clearance_m of obstacles. "
                "Use after failed explore/find or when the user asks why navigation failed."
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
                "Render a top-down RGB view of obstacles vs explored space (cropped to the explored region plus margin), "
                "optionally overlaying the last motion plan, and send to Discord if configured; also logs the same crop "
                "to Rerun at world/map_snapshot/topdown when the live Rerun visualizer is enabled."
            ),
            parameters=_NO_PARAMS,
            func=send_map_snapshot,
            returns_info=True,
        )
    )

    # -- pick_place / plan_pick_place ----------------------------------------
    def _tamp_robot() -> Any | None:
        robot = context.get("robot")
        if robot is not None:
            return robot
        executor = context.get("executor")
        return getattr(executor, "robot", None) if executor is not None else None

    def _tamp_manip_mode() -> str:
        executor = context.get("executor")
        return str(getattr(executor, "_manip_mode", None) or context.get("manip_mode") or "auto")

    def _fallback_pick_place(executor: Any, object_name: str, receptacle_name: str) -> str:
        keep_going = executor([("pickup", object_name), ("place", receptacle_name)])
        task_ok = keep_going and bool(getattr(executor, "_last_exec_ok", True))
        return (
            f"Pick and place ({object_name} -> {receptacle_name}) done."
            if task_ok
            else "Pick/place failed or interrupted."
        )

    def _build_tamp_plan(
        *,
        task_ref: str = "",
        object_name: str = "",
        receptacle_name: str = "",
    ):
        from emet.controller.task.tamp.agent_bridge import (
            AgentPlanBuild,
            AgentTaskRef,
            build_agent_pick_place_plan,
        )

        requested_ref = str(task_ref).strip()
        selected = (context.get("_tamp_task_refs") or {}).get(requested_ref)
        if selected is not None and not isinstance(selected, AgentTaskRef):
            selected = None
        if requested_ref and selected is None:
            return AgentPlanBuild(
                task=None,
                plan=None,
                mode=None,
                live_sim=False,
                reason="unknown_task_ref",
            )
        if selected is not None:
            return build_agent_pick_place_plan(
                _tamp_robot(),
                selected.object_query,
                selected.receptacle_query,
                object_body=selected.object_body,
                receptacle_body=selected.receptacle_body,
                manip_mode=_tamp_manip_mode(),
            )
        if not str(object_name).strip() or not str(receptacle_name).strip():
            return AgentPlanBuild(
                task=None,
                plan=None,
                mode=None,
                live_sim=False,
                reason="provide_task_ref_or_object_and_receptacle",
            )
        return build_agent_pick_place_plan(
            _tamp_robot(),
            str(object_name),
            str(receptacle_name),
            manip_mode=_tamp_manip_mode(),
        )

    def _format_tamp_plan(plan_ref: str, build: Any) -> str:
        plan = build.plan
        if plan is None or not plan.success or build.task is None:
            return f"TAMP plan failed: {build.reason or getattr(plan, 'message', 'unknown_error')}."
        ops = " -> ".join(step.op for step in plan.steps)
        return (
            f"TAMP plan {plan_ref}: {build.task.object_query} -> {build.task.receptacle_query}; "
            f"mode={build.mode}; steps={ops}; chosen_grasp={plan.chosen_grasp_index}. "
            "No motion executed; call execute_pick_place_plan with this plan_ref."
        )

    def plan_pick_place(
        task_ref: str = "",
        object_name: str = "",
        receptacle_name: str = "",
    ) -> str:
        from emet.controller.task.tamp.agent_bridge import store_agent_plan

        build = _build_tamp_plan(
            task_ref=task_ref,
            object_name=object_name,
            receptacle_name=receptacle_name,
        )
        if build.plan is None or not build.plan.success:
            return _format_tamp_plan("", build)
        plan_ref = store_agent_plan(context, _tamp_robot(), build)
        return _format_tamp_plan(plan_ref, build)

    def execute_pick_place_plan(plan_ref: str) -> str:
        from emet.controller.task.tamp.agent_bridge import execute_stored_agent_plan

        robot = _tamp_robot()
        if robot is None:
            return "TAMP execution failed: robot not connected."
        ok, message = execute_stored_agent_plan(robot, context, str(plan_ref))
        return f"TAMP execution {'succeeded' if ok else 'failed'}: {message}."

    def pick_place(object_name: str, receptacle_name: str) -> str:
        executor = context.get("executor")
        if executor is None and _tamp_robot() is None:
            return "Robot not connected."
        if executor is not None and (
            bool(getattr(executor, "visual_servo", False))
            or _tamp_manip_mode().strip().lower() not in {"auto", "teleport", "kinematic"}
        ):
            return _fallback_pick_place(executor, object_name, receptacle_name)
        build = _build_tamp_plan(object_name=object_name, receptacle_name=receptacle_name)
        if build.plan is not None and build.plan.success:
            from emet.controller.task.tamp.agent_bridge import (
                execute_stored_agent_plan,
                store_agent_plan,
            )

            plan_ref = store_agent_plan(context, _tamp_robot(), build)
            ok, message = execute_stored_agent_plan(_tamp_robot(), context, plan_ref)
            return (
                f"Pick and place ({object_name} -> {receptacle_name}) done." if ok else f"Pick/place failed: {message}."
            )
        if build.live_sim:
            return f"Pick/place not run: {build.reason or 'TAMP planning failed'}."
        if executor is None:
            return "Pick/place not run: no configured manipulation controller."
        return _fallback_pick_place(executor, object_name, receptacle_name)

    tools.append(
        Tool(
            name="pick_place",
            description="Pick up an object and place it on a receptacle. "
            "In simulation, plan and validate the task before guarded kinematic or teleport execution; "
            "on real hardware use the configured manipulation controller.",
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
            returns_info=True,
        )
    )
    tools.append(
        Tool(
            name="plan_pick_place",
            description=(
                "Plan, but do not execute, a pick-and-place task. Prefer a semantic task_ref from "
                "scene_tasks; otherwise provide object_name and receptacle_name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_ref": {"type": "string", "description": "Semantic task handle from scene_tasks."},
                    "object_name": {"type": "string", "description": "Object name if no task_ref is supplied."},
                    "receptacle_name": {
                        "type": "string",
                        "description": "Receptacle name if no task_ref is supplied.",
                    },
                },
                "required": [],
            },
            func=plan_pick_place,
            returns_info=True,
        )
    )
    tools.append(
        Tool(
            name="execute_pick_place_plan",
            description=(
                "Execute a plan_ref returned by plan_pick_place after rechecking the live scene. "
                "Plans are one-shot and stale plans are refused."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_ref": {"type": "string", "description": "Opaque plan handle from plan_pick_place."}
                },
                "required": ["plan_ref"],
            },
            func=execute_pick_place_plan,
            returns_info=True,
        )
    )

    # -- scene_tasks ----------------------------------------------------------
    def scene_tasks(object_filter: str = "", robot: str = "") -> str:
        """Enumerate pick-and-place options from the current scene as a compact digest."""
        from collections import Counter

        from emet.controller.task.tamp.agent_bridge import stable_scene_task_refs
        from emet.eval.scene_task_extractor import (
            default_molmospaces_scenes_dir,
            emit_tasks,
            load_scene_metadata,
            pickable_objects,
            receptacle_objects,
            resolve_scene_metadata_for_session,
            scene_objects,
        )

        scene_dir = default_molmospaces_scenes_dir()
        ithor = scene_dir / "ithor"
        executor = context.get("executor")
        robot_obj = context.get("robot") or (getattr(executor, "robot", None) if executor is not None else None)
        session = (
            robot_obj.get_emet_session() if robot_obj is not None and hasattr(robot_obj, "get_emet_session") else None
        )
        live_sim = isinstance(session, dict) and bool(session.get("is_simulation"))
        if live_sim:
            from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

            placements = read_sim_object_placements(session) or {}
        else:
            placements = {}
        environment = session.get("environment") if isinstance(session, dict) else {}
        if not isinstance(environment, dict):
            environment = {}
        session_key = (
            str(environment.get("kind") or ""),
            str(environment.get("scene") or ""),
            str(environment.get("index") or ""),
            str(session.get("scene_source_basename") or "") if isinstance(session, dict) else "",
        )
        active_metadata = resolve_scene_metadata_for_session(session, scenes_dir=scene_dir)
        candidates = (
            [active_metadata]
            if active_metadata is not None
            else ([] if live_sim else (sorted(ithor.glob("*_physics_metadata.json")) if ithor.is_dir() else []))
        )
        if not candidates:
            stable_scene_task_refs(context, [], {}, session_key=session_key)
            if live_sim:
                return "No MolmoSpaces metadata matches the active simulation scene."
            return (
                "No MolmoSpaces scene metadata installed "
                "(expected *_physics_metadata.json under ~/.cache/molmospaces/assets/scenes/ithor)."
            )
        metadata = load_scene_metadata(candidates[0])
        objs = scene_objects(metadata)
        picks = pickable_objects(objs)
        recepts = receptacle_objects(objs)
        tasks = emit_tasks(objs, sim="configs/sim/molmospaces_ithor_train_0.yaml")
        filt = (object_filter or "").strip().lower()
        if filt:
            tasks = [t for t in tasks if filt in t.object]
        task_refs = stable_scene_task_refs(
            context,
            tasks,
            placements,
            session_key=session_key,
        )

        by_obj: Counter = Counter(t.object for t in tasks)
        lines = [
            f"Scene: {candidates[0].parent.name}/{candidates[0].stem}",
            f"Pickable objects ({len(picks)}): {', '.join(sorted({p.category for p in picks}))}",
            f"Receptacles ({len(recepts)}): {', '.join(sorted({r.category for r in recepts}))}",
        ]
        if filt:
            lines.append(f"Tasks matching {filt!r} ({len(tasks)}):")
        else:
            lines.append(f"Task options by object ({len(tasks)} unique object->receptacle pairs):")
        shown = 0
        for obj, count in by_obj.most_common(10):
            lines.append(f"  - {obj}: {count} placement options")
            shown += 1
        if len(by_obj) > shown:
            lines.append(f"  ... and {len(by_obj) - shown} more object categories")
        if task_refs:
            lines.append("Semantic task handles (use task_ref with plan_pick_place):")
            for ref in task_refs[:12]:
                lines.append(
                    f"  - {ref.ref}: pick {ref.object_query} from {ref.start_receptacle or 'unknown'} "
                    f"to {ref.receptacle_query}"
                )
        elif tasks:
            ex = tasks[0]
            lines.append(f"Example: pick {ex.object} from {ex.start_recep} to {ex.goal_recep}")
        rid = (robot or "").strip().lower()
        if rid:
            try:
                from emet.eval.scene_task_extractor import compute_reachability_priors

                # Prefer real object poses from a connected sim server; fall back to a
                # documented zero-pose proxy so reachability still exercises the IK path.
                if placements:

                    def _pose(body: str) -> np.ndarray:
                        info = placements.get(body)
                        if info is None and body.endswith("_1_1_0"):
                            info = placements.get(body[: -len("_1_1_0")] + "_1_0_0")
                        pos = (info or {}).get("pos")
                        return np.asarray(pos, dtype=np.float64).reshape(3) if pos is not None else np.zeros(3)

                    priors = compute_reachability_priors(objs, robot_id=rid, arm="left", object_pose_fn=_pose)
                else:
                    priors = compute_reachability_priors(objs, robot_id=rid, arm="left")
                reachable = [r.category for r in priors.values() if r.reachable]
                src = "sim placements" if placements else "zero-pose proxy (no sim connected)"
                lines.append(
                    f"Reachable by {rid} from {src} ({len(reachable)}): {', '.join(sorted(set(reachable))) or 'none'}"
                )
            except Exception as e:
                lines.append(f"(reachability for {rid} unavailable: {e})")
        return "\n".join(lines)

    tools.append(
        Tool(
            name="scene_tasks",
            description=(
                "Enumerate candidate pick-and-place tasks from the current sim scene: pickable objects "
                "with grasp assets, receptacle sites, and proposed tasks (object -> start_recep -> goal_recep)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "object_filter": {
                        "type": "string",
                        "description": "Optional category substring to limit proposed tasks (e.g. 'bowl').",
                    },
                    "robot": {
                        "type": "string",
                        "description": "Optional robot id (e.g. 'sourccey', 'rby1') to also report which objects its arm can reach.",
                    },
                },
                "required": [],
            },
            func=scene_tasks,
            returns_info=True,
        )
    )

    # -- find_objects --------------------------------------------------------
    def find_objects(text: str) -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([("find", text)])
        agent = _agent_from_context(context)
        plan_line = format_last_nav_plan_summary(agent)
        outcome = (getattr(agent, "_last_nav_plan", None) or {}).get("outcome") if agent else None
        head = format_nav_outcome_head(outcome, ok=ok, verb="Find")
        parts = [head]
        if text:
            parts.append(f"Query={text!r}.")
        if plan_line:
            parts.append(plan_line)
        clr = format_base_clearance_hint(agent)
        if clr:
            parts.append(clr)
        if outcome in {
            "user_cancelled",
            "aborted_waypoint_timeout",
            "rejected_low_clearance",
            "rejected_unexplored",
        }:
            parts.append(
                "Do not immediately re-call find_objects; ask the user or use "
                "navigation_diagnostics / send_map_snapshot / scan_environment first."
            )
        return " ".join(parts)

    tools.append(
        Tool(
            name="find_objects",
            description=(
                "Find and navigate to an object or location by name. Returns last-plan summary "
                "(localize source, waypoints, min clearance) and outcomes such as user_cancelled / "
                "aborted_waypoint_timeout / rejected_low_clearance when --confirm-nav or safety filters fire. "
                "On cancel/abort/reject, do not blindly retry — use navigation_diagnostics or ask the user."
            ),
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Object or location name to find."}},
                "required": ["text"],
            },
            func=find_objects,
            executor_commands=lambda args: [("find", args.get("text", ""))],
            returns_info=True,
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
        live_rgb = np.asarray(obs.rgb).copy()
        agent = getattr(executor, "agent", None) if executor is not None else None
        if agent is None or not hasattr(agent, "describe_head_camera_scene_text"):
            stash_discord_image(context, live_rgb)
            return (
                "I have a camera frame but this session's controller does not expose scene description; "
                "a photo is attached when Discord is connected."
            )
        text = agent.describe_head_camera_scene_text(
            graph_memory=context.get("graph_memory"),
            memory_backend=context.get("memory_backend"),
            graph_memory_backend=context.get("graph_memory_backend"),
        )
        # Always attach the *live* head camera for "what can you see" — graph crops are for
        # send_object_image, not scene description (crops of walls/wrong nodes confuse users).
        obs2 = robot.get_observation()
        if obs2 is not None and getattr(obs2, "rgb", None) is not None:
            live_rgb = np.asarray(obs2.rgb).copy()
        image = live_rgb
        if image is not None:
            print_camera_frame_diagnostics(
                "describe_scene (live head RGB → Discord)",
                image,
                force=bool(context.get("verbose_tools")) or bool(context.get("camera_debug")),
            )
        if stash_discord_image(context, image):
            text = f"{text}\n(Attaching a photo of my current view.)"
        gsize = _graph_size_line(context)
        if gsize:
            _logger.info(gsize)
            text = f"{text}\n[{gsize}]"
        return text

    tools.append(
        Tool(
            name="describe_scene",
            description=(
                "Caption what is in front of the robot *right now* (live head camera) and optionally "
                "ground with known scene-graph / map labels. Does NOT move or reorient the robot — "
                "this alone is not a 'closer look'. For 'are you sure' / 'look closer' / confirm, "
                "first call rotate_base / move_forward / scan_environment (as allowed), then this tool. "
                "Queues the live head-camera photo for Discord (not an object crop; use send_object_image "
                'for that). Use an empty JSON "message" on the tool-call turn so chat/Discord only show '
                "your answer after [Tool results]."
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

    def scan_environment() -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([("rotate_in_place", "")])
        if not ok:
            return "Scan interrupted or failed."
        gsize = _graph_size_line(context)
        if gsize:
            _logger.info(gsize)
            return f"Completed in-place ≈360° scan; map/memory updated. [{gsize}]"
        return "Completed in-place ≈360° scan; map/memory updated."

    tools.append(
        Tool(
            name="scan_environment",
            description=(
                "Rotate in place through a full ≈360° scan to update the map and save memory. "
                "Use for 'look around', 'scan the room', or a full in-place survey — not for a single "
                "turn (use rotate_base) or a short drive (use move_forward). "
                "After scanning, you may call describe_scene to report the new view."
            ),
            parameters=_NO_PARAMS,
            func=scan_environment,
            returns_info=True,
        )
    )

    def rotate_base(degrees: float = 90.0) -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        try:
            deg = float(degrees)
        except (TypeError, ValueError):
            return f"Invalid degrees: {degrees!r}."
        deg = float(np.clip(deg, -360.0, 360.0))
        ok = executor([("rotate_base", str(deg))])
        if not ok:
            return "Rotate failed or interrupted."
        context["last_rotate_degrees"] = deg
        return f"Rotated about {deg:.0f}° in place."

    tools.append(
        Tool(
            name="rotate_base",
            description=(
                "Rotate the wheeled base in place by a relative yaw in degrees (positive = left/CCW, "
                "negative = right/CW). Pass an explicit angle: turn around → 180, turn right → -90, "
                "turn left → 90, slight turn → ±30–45. "
                "'rotate back' / 'turn back' means NEGATE the previous rotate (e.g. after +45 use -45) "
                "— NOT 180 (that is 'turn around'). "
                "For 'look at / face / go toward X' when you have a named object, prefer face_toward."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "degrees": {
                        "type": "number",
                        "description": "Relative yaw in degrees (e.g. 180, -90, 45, or -last for 'back').",
                    }
                },
                "required": ["degrees"],
            },
            func=rotate_base,
            returns_info=True,
        )
    )

    def face_toward(object_label: str) -> str:
        """Yaw in place to face a scene-graph / voxel object (no XY drive)."""
        import math

        from emet.agent.face_toward import resolve_object_xy, yaw_to_face_xy

        executor = context.get("executor")
        robot = context.get("robot")
        label = (object_label or "").strip()
        if not label:
            return "Need an object label to face (e.g. 'aquarium', 'shelf')."
        if executor is None or robot is None or not hasattr(robot, "get_base_pose"):
            return "Robot not connected."
        agent = getattr(executor, "agent", None)
        xy, source = resolve_object_xy(agent, label)
        if xy is None:
            return (
                f"I don't have a map location for {label!r} yet — "
                "I can rotate_base blindly or describe_scene from here."
            )
        try:
            pose = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
        except Exception as e:
            return f"Could not read base pose: {e}"
        if pose.size < 3:
            return "Base pose incomplete."
        delta_rad, _bearing = yaw_to_face_xy(pose[:3], xy)
        deg = float(math.degrees(delta_rad))
        if abs(deg) < 3.0:
            msg = f"Already roughly facing {label!r} ({source})."
        else:
            deg = float(np.clip(deg, -180.0, 180.0))
            ok = executor([("rotate_base", str(deg))])
            if not ok:
                return f"Tried to face {label!r} ({source}) but rotate failed."
            context["last_rotate_degrees"] = deg
            msg = f"Turned about {deg:+.0f}° to face {label!r} ({source})."
        hint = format_close_map_hint(close_map_from_agent(agent), float(xy[0]), float(xy[1]), is_chat=True)
        if hint:
            msg = f"{msg} {hint}"
        return msg

    tools.append(
        Tool(
            name="face_toward",
            description=(
                "Rotate in place to face a named object from the scene graph / map "
                "(computes yaw toward its remembered XY). Use for 'look at the aquarium', "
                "'face the shelf', 'turn toward the TV'. Does NOT drive closer — follow with "
                "describe_scene. Prefer this over a blind rotate_base(±45) when the object is known. "
                "If the object is unknown, fall back to rotate_base or ask."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "object_label": {
                        "type": "string",
                        "description": "Object name to face (e.g. aquarium, shelf, cardboard box).",
                    }
                },
                "required": ["object_label"],
            },
            func=face_toward,
            returns_info=True,
        )
    )

    def move_forward(meters: float = 0.1) -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        try:
            dist = float(meters)
        except (TypeError, ValueError):
            return f"Invalid meters: {meters!r}."
        dist = float(np.clip(dist, 0.0, 1.5))
        # Prefer controller path so every nudge (including 0.1 m) is map-clipped.
        agent = getattr(executor, "agent", None)
        if agent is not None and hasattr(agent, "move_forward_meters"):
            commanded = float(agent.move_forward_meters(dist))
            if commanded < 0.02:
                return (
                    "I don't have enough explored map to drive that way yet "
                    "(empty map, local free disk too small, or obstacle too close). "
                    "Want me to scan_environment or rotate_base in place first so I can see "
                    "what's ahead?"
                )
            if commanded + 1e-3 < dist:
                return f"Moved forward {commanded:.2f} m (map-clipped from {dist:.2f} m so I don't hit anything)."
            return f"Moved forward {commanded:.2f} m (map clear along path)."
        ok = executor([("move_forward", str(dist))])
        if not ok:
            return "Move forward failed or interrupted."
        return f"Moved forward (requested {dist:.2f} m; map-clipped by the controller if needed)."

    tools.append(
        Tool(
            name="move_forward",
            description=(
                "Drive the base forward along its current heading by approximately *meters*. "
                "Uses the 2D map (including a local_radius explored disk around the base when "
                "depth is still empty): shortens or refuses if blocked or the path leaves "
                "explored space. If refused, ask the user whether to scan_environment / "
                "rotate_base — do NOT auto-scan. If the user did not say how far, do NOT call "
                "this tool — ask how far first. Use meters=0.1 for 'a bit' / 'a little' / "
                "'nudge'; ~0.5 for half a meter; ~1.0 for a meter. Cap near 1.5 m. "
                "Do not use for turning (use rotate_base)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "meters": {
                        "type": "number",
                        "description": (
                            "Forward distance in meters (required). "
                            "0.1 for a small nudge / 'a bit'; 0.5 for half a meter; 1.0 for a meter."
                        ),
                    }
                },
                "required": ["meters"],
            },
            func=move_forward,
            returns_info=True,
        )
    )

    # -- camera --------------------------------------------------------------
    def take_picture() -> str:
        executor = context.get("executor")
        robot = context.get("robot")
        if executor is not None:
            executor([("take_picture", "")])
        image = None
        if robot is not None and hasattr(robot, "get_observation"):
            obs = robot.get_observation()
            if obs is not None and getattr(obs, "rgb", None) is not None:
                image = np.asarray(obs.rgb).copy()
        if stash_discord_image(context, image):
            return "Head-camera photo queued for Discord."
        return "Took a head-camera picture locally (no frame to attach)."

    tools.append(
        Tool(
            name="take_picture",
            description=(
                "Capture the head camera locally. Prefer send_image or describe_scene when the user "
                "should see or hear a description — those attach a photo / caption to Discord."
            ),
            parameters=_NO_PARAMS,
            func=take_picture,
            returns_info=True,
        )
    )

    def take_ee_picture():
        from emet.agent.tool_outcome import ToolOutcome
        from emet.controller.manipulation.closer_look import consume_closer_look_aim_for_ee_picture

        agent = _agent_from_context(context)
        allowed, gate_note, aim = consume_closer_look_aim_for_ee_picture(agent=agent, context=context)
        if not allowed:
            return ToolOutcome(
                ok=False,
                status="aim_required",
                note=gate_note,
                tool="take_ee_picture",
                payload={"action_kind": "closer_look"},
            )

        robot = context.get("robot")
        image = None
        if robot is not None and hasattr(robot, "get_servo_observation"):
            try:
                obs = robot.get_servo_observation()
                ee = getattr(obs, "ee_rgb", None) if obs is not None else None
                if ee is not None:
                    image = np.asarray(ee).copy()
            except Exception as e:
                _logger.warning(f"take_ee_picture: servo obs failed ({e})")
        phrase = str((aim or {}).get("phrase") or "")
        if stash_discord_image(context, image):
            return ToolOutcome(
                ok=True,
                status="ok",
                note=f"{gate_note} Wrist-camera photo queued for Discord.",
                tool="take_ee_picture",
                payload={"action_kind": "closer_look", "phrase": phrase, "queued": True},
            )
        return ToolOutcome(
            ok=False,
            status="ee_frame_missing",
            note=(
                f"{gate_note} Wrist / EE camera frame not available "
                "(common on Innate Mars). Use describe_scene or send_image (head camera)."
            ),
            tool="take_ee_picture",
            payload={"action_kind": "closer_look", "phrase": phrase, "queued": False},
        )

    tools.append(
        Tool(
            name="take_ee_picture",
            description=(
                "Capture the wrist/end-effector camera after aim_arm_at in this session "
                "(no further arm motion). Requires a prior aim: successful aim grants one capture; "
                "if aim returns not_implemented (no kinematic stack), one soft-allow capture is ok. "
                "Prefer face_toward + describe_scene (head camera) when the wrist stream is dark."
            ),
            parameters=_NO_PARAMS,
            func=take_ee_picture,
            returns_info=True,
        )
    )

    # -- arm aim / closer look -----------------------------------------------
    def aim_arm_at(object_label: str):
        # See docs/attempt_ledger.md and plans/2026-08-08_embodied_agent_planning.md Phase 3.
        from emet.controller.manipulation.closer_look import aim_wrist_at_phrase, record_closer_look_aim

        agent = _agent_from_context(context)
        robot = context.get("robot")
        executor = context.get("executor")
        manip_mode = str(getattr(executor, "_manip_mode", None) or context.get("manip_mode") or "teleport")
        visual_servo = bool(getattr(executor, "visual_servo", False))
        manip_collision = str(getattr(executor, "_manip_collision", None) or "none")
        manip_planner = str(getattr(executor, "_manip_planner", None) or "rrt_connect")
        result = aim_wrist_at_phrase(
            agent=agent,
            robot=robot,
            phrase=str(object_label or ""),
            manip_mode=manip_mode,
            visual_servo=visual_servo,
            manip_collision=manip_collision,
            manip_planner=manip_planner,
        )
        record_closer_look_aim(result, agent=agent, context=context)
        return result.to_tool_outcome()

    tools.append(
        Tool(
            name="aim_arm_at",
            description=(
                "Point the arm / wrist camera at a named object (localize + kinematic EE aim when "
                "available). On success you may call take_ee_picture once. If aim fails or is "
                "unavailable, prefer face_toward + describe_scene (head camera) instead of claiming "
                "a closer look."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "object_label": {
                        "type": "string",
                        "description": "Object or region to aim at (e.g. cables, red cup).",
                    }
                },
                "required": ["object_label"],
            },
            func=aim_arm_at,
            returns_info=True,
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
                return _graph_eqa_tool_string(gmb.query_answer(question, xyt, planner))
            except Exception as e:
                _logger.warning(f"query_scene_graph graph backend failed: {e}")
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
                "Embodied where/what questions using GraphEQA (objects, 3D positions) when enabled. "
                "Returns Answer / Location / Confidence for the user (not internal image ids). "
                "Otherwise dumps the open-vocab spatial scene graph. Prefer for 'where is', 'what color', 'is there'."
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
        """Prefer Dynagraph / GraphEQAMemory; fall back to open-vocab only if that is empty."""
        executor = context.get("executor")
        gm = context.get("graph_memory")
        if gm is None and executor is not None and hasattr(executor, "agent"):
            gm = getattr(executor.agent, "graph_memory", None)
        if gm is not None and hasattr(gm, "get_nodes"):
            nodes = [
                n
                for n in gm.get_nodes()
                if not getattr(n, "is_viewpoint", False) and not getattr(n, "is_frontier", False)
            ]
            if nodes:
                try:
                    from emet.memory.graph_eqa.pretty_print import format_scene_graph_pretty

                    return format_scene_graph_pretty(gm, title="Dynagraph / GraphEQA scene graph")
                except Exception:
                    if hasattr(gm, "to_string"):
                        return gm.to_string()
                    if hasattr(gm, "print_memory"):
                        return gm.print_memory()
                    # Minimal dump when FakeGM / partial graph lacks pretty helpers.
                    lines = ["Dynagraph / GraphEQA scene graph"]
                    for n in nodes:
                        labels = getattr(n, "labels", None) or []
                        lab = ", ".join(str(x) for x in labels[:3]) or "(no labels)"
                        xyz = getattr(n, "xyz", None)
                        if xyz is not None:
                            arr = np.asarray(xyz, dtype=float).reshape(-1)
                            if arr.size >= 3:
                                lines.append(
                                    f"  [{getattr(n, 'node_id', '?')}] {lab} "
                                    f"xyz=({arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f})"
                                )
                                continue
                        lines.append(f"  [{getattr(n, 'node_id', '?')}] {lab}")
                    return "\n".join(lines)
        if executor is None or not hasattr(executor, "agent"):
            return "Robot not connected."
        agent = executor.agent
        sg = None
        if hasattr(agent, "get_voxel_map"):
            sg = agent.get_voxel_map().get_scene_graph()
        if sg is not None and getattr(sg, "num_objects", 0) > 0:
            return f"[Open-vocab scene graph]\n{sg.to_string()}"
        return (
            "No scene graph data yet. Explore or scan first, or reload a checkpoint "
            "that includes graph.json (dynagraph) or open_vocab_scene_graph/."
        )

    tools.append(
        Tool(
            name="list_scene_relations",
            description=(
                "List objects and spatial relations (near, on, on_floor, supports). "
                "Uses the Dynagraph / GraphEQA memory plug-in when present; otherwise the "
                "open-vocab scene graph. Use for 'what objects are in the room' and "
                "structured connectivity questions."
            ),
            parameters=_NO_PARAMS,
            func=list_scene_relations,
            returns_info=True,
        )
    )

    def send_object_image(object_label: str) -> str:
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
        if stash_discord_image(context, image):
            if context.get("discord_bot") is not None:
                return f"Queued crop image for {object_label!r} (attached to the reply)."
            return f"Crop available for {object_label!r} (no Discord to send to)."
        return f"Object {object_label!r} has no stored crop image yet."

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
            returns_info=True,
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

    from emet.agent.skills.specs import CHAT_EXCLUSIVE_TOOL_NAMES, CHAT_SKILL_SPECS

    by_name = {s.name: s for s in CHAT_SKILL_SPECS}
    registered = {t.name for t in tools}
    expected = set(CHAT_EXCLUSIVE_TOOL_NAMES)
    if registered != expected:
        missing = sorted(expected - registered)
        extra = sorted(registered - expected)
        raise RuntimeError(f"CHAT tool pack drift vs CHAT_SKILL_SPECS: missing={missing} extra={extra}")
    for t in tools:
        spec = by_name[t.name]
        t.description = spec.description
        t.parameters = spec.parameters
        t.returns_info = spec.returns_info

    if env_base_rotate_only():
        # Keep tool names in the pack so the LLM does not invent calls; stubs return a clear refusal.
        blocked = frozenset(
            {
                "explore",
                "find_objects",
                "move_forward",
                "go_home",
                "pick_place",
                "execute_pick_place_plan",
                "hand_over",
            }
        )
        stub_msg = (
            "I can't drive or translate right now (EMET_BASE_ROTATE_ONLY — tethered / plugged in). "
            "I can rotate in place, scan the room, or describe what I see. "
            "Unset EMET_BASE_ROTATE_ONLY when the robot is free to move."
        )

        def _make_stub(orig: Tool) -> Tool:
            def _fn(**_kwargs: Any) -> str:
                return stub_msg

            return Tool(
                name=orig.name,
                description=(
                    f"{orig.description} [DISABLED: EMET_BASE_ROTATE_ONLY — do not call; "
                    "ask the user or use rotate_base / scan_environment / describe_scene.]"
                ),
                parameters=orig.parameters,
                func=_fn,
                returns_info=True,
            )

        tools = [_make_stub(t) if t.name in blocked else t for t in tools]
        _logger.warning(
            f"EMET_BASE_ROTATE_ONLY=1: {len(blocked)} drive/manip tools stubbed "
            "(rotate_base / scan_environment / describe still available)."
        )

    return tools


def get_tools(context: dict[str, Any]) -> list[Tool]:
    """Return the CHAT-mode tool pack (Discord / terminal embodied agent).

    Assembles via :func:`emet.agent.skills.build_skill_pack` (:class:`~emet.agent.skills.AgentMode.CHAT`).
    EQA episode tools are a separate pack — see
    :func:`emet.memory.graph_eqa.agentic_tools.build_agentic_eqa_tools`.
    """
    from emet.agent.skills import AgentMode, build_skill_pack

    return build_skill_pack(AgentMode.CHAT, context)


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
