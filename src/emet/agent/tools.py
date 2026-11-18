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
"""Context-bound CHAT handlers for terminal and Discord sessions.

``skills.specs.CHAT_SKILL_SPECS`` owns all names, descriptions, argument schemas
and result-visibility flags. ``build_chat_tools`` binds those specs to closures
over a mutable session context, delegating task handlers to ``chat_tamp``; construction does not probe robot capabilities
or load perception models. The session loop applies provider advertisement gates.

Handlers either return information directly or carry legacy executor command
mappings. The dispatcher preserves call order and TAMP stop boundaries; it must
inspect actual outcomes rather than treating executor continuation as success.
Images are queued separately for the user-facing reply. EQA orchestration lives
in ``memory.graph_eqa.agentic``, not in these CHAT handlers. See
``docs/agent_architecture.md`` and ``docs/agent_tool_contracts.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from emet.agent.camera_debug import print_camera_frame_diagnostics
from emet.agent.tool import Tool, get_tool_descriptions_for_prompt  # noqa: F401 — compatibility exports
from emet.agent.tool_outcome import ToolOutcome
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


def navigation_feedback(agent: Any | None) -> dict:
    """Bounded navigation evidence; never dump full waypoint arrays into prompts."""
    meta = getattr(agent, "_last_nav_plan", None) or {}
    keys = (
        "mode",
        "outcome",
        "status_code",
        "localize_source",
        "goal_xyt",
        "requested_goal_xyt",
        "goal_resolution",
        "motion_outcome",
        "new_sensor_cells",
        "new_sensor_area_m2",
        "observation_outcome",
        "n_planned",
        "min_clearance_m",
        "min_clearance_required_m",
        "footprint",
        "approach_sampling",
    )
    result = {key: meta[key] for key in keys if key in meta}
    candidates = meta.get("view_candidates")
    if candidates:
        eligible = [candidate for candidate in candidates if candidate.get("reason") == "eligible"]
        compact_keys = ("index", "resolved_goal", "estimated_gain_m2", "path_m", "turn_rad", "score")
        result["view_selection"] = {
            "candidate_count": len(candidates),
            "reason_counts": dict(Counter(candidate.get("reason", "unknown") for candidate in candidates)),
            "top_estimated_views": [
                {key: candidate[key] for key in compact_keys if key in candidate}
                for candidate in sorted(eligible, key=lambda c: -c.get("score", 0))[:3]
            ],
        }
    return result


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
        planning_pose = getattr(agent, "_planning_base_xyt", None)
        if callable(planning_pose):
            pose = np.asarray(planning_pose(pose), dtype=np.float64).reshape(-1)
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

    Bind canonical specs once to lazy closures over executor, robot, memory and
    reply-image state. Explicit ``head_view_capability`` specializes gaze schemas;
    an absent key preserves generic registry schemas for introspection. The CHAT
    loop subsequently applies task-provider advertisement gates.

    Prefer :func:`get_tools` (routes through :func:`emet.agent.skills.build_skill_pack`).
    """
    from emet.agent.env_flags import env_base_rotate_only
    from emet.agent.skills.specs import CHAT_EXCLUSIVE_TOOL_NAMES, CHAT_SKILL_SPECS, head_view_skill

    specs = {spec.name: spec for spec in CHAT_SKILL_SPECS}

    def _bind(
        name: str,
        *,
        func: Callable[..., Any],
        executor_commands: Callable[[dict[str, Any]], list[tuple[str, str]]] | None = None,
    ) -> Tool:
        """Attach a handler to canonical metadata without inspecting live runtime objects."""
        spec = specs[name]
        if "head_view_capability" in context:
            spec = head_view_skill(spec, context["head_view_capability"])
        return Tool(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            func=func,
            executor_commands=executor_commands,
            returns_info=spec.returns_info,
        )

    tools: list[Tool] = []

    def _query_graph_backend(question: str, *, caller: str) -> str | None:
        """Shared graph QA; callers retain their own fallback order."""
        backend = context.get("graph_memory_backend")
        if backend is None or not hasattr(backend, "query_answer"):
            return None
        try:
            return _graph_eqa_tool_string(
                backend.query_answer(question, context.get("xyt_for_query"), context.get("planner"))
            )
        except Exception as error:
            _logger.warning(f"{caller} graph backend failed: {error}")
            return None

    # -- query_memory --------------------------------------------------------
    def query_memory(question: str) -> str:
        """Read graph EQA, then memory-backend QA, then voxel retrieval as available.

        The fallback XYZ is a search hypothesis, not admitted object geometry.
        """
        xyt = context.get("xyt_for_query")
        planner = context.get("planner")
        graph_answer = _query_graph_backend(question, caller="query_memory")
        if graph_answer is not None:
            return graph_answer
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

    tools.append(_bind("query_memory", func=query_memory))

    # -- send_image ----------------------------------------------------------
    def send_image() -> ToolOutcome:
        """Queue an owned head RGB frame for the next user reply; report missing camera data."""
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
                return ToolOutcome(
                    True, status="captured", note="Image queued for Discord (attached to the reply).", tool="send_image"
                )
            return ToolOutcome(
                True, status="captured", note="Image captured (no Discord to send to).", tool="send_image"
            )
        return ToolOutcome(False, status="observation_unavailable", note="No image available.", tool="send_image")

    tools.append(_bind("send_image", func=send_image))

    def _exec(cmd: str, args: Any = "") -> str:
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        ok = executor([(cmd, args)])
        return "Done." if ok else "Command was interrupted or failed."

    # -- explore -------------------------------------------------------------
    def explore() -> ToolOutcome:
        """Run coverage exploration and return navigation status plus bounded plan evidence."""
        executor = context.get("executor")
        robot = context.get("robot")
        if executor is None:
            return ToolOutcome(False, status="unavailable", note="Robot not connected.", tool="explore")
        keep_running = executor([("explore", "")])
        ok = keep_running and bool(getattr(executor, "_last_exec_ok", True))
        agent = _agent_from_context(context)
        robot_xy = _robot_base_xy(robot)
        vm = _voxel_map_from_executor(executor)
        _img, stats, _ = snapshot_from_voxel_map(vm, robot_xy)
        # The explicit outcome below owns completion status; map statistics
        # must not turn executor liveness into an additional success claim.
        summary = format_navigation_report(stats)
        plan_line = format_last_nav_plan_summary(agent)
        outcome = (getattr(agent, "_last_nav_plan", None) or {}).get("outcome") if agent else None
        if str(outcome or "").startswith(("rejected_", "aborted_", "failed")) or outcome == "user_cancelled":
            ok = False
        head = format_nav_outcome_head(outcome, ok=ok, verb="Explore")
        parts = [head, summary]
        if plan_line:
            parts.append(plan_line)
        gsize = _graph_size_line(context)
        if gsize:
            _logger.info(gsize)
            parts.append(f"[{gsize}]")
        return ToolOutcome(
            bool(ok),
            status=outcome or ("ok" if ok else "failed"),
            note=" ".join(parts),
            tool="explore",
            payload={"navigation": navigation_feedback(agent)},
        )

    tools.append(_bind("explore", func=explore))

    # -- navigation_diagnostics / send_map_snapshot -------------------------
    def navigation_diagnostics() -> str:
        """Summarize map coverage, base clearance and the last plan without moving."""
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
        """Render the voxel map and last path, queueing an image without executing the path."""
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

    tools.append(_bind("navigation_diagnostics", func=navigation_diagnostics))

    tools.append(_bind("send_map_snapshot", func=send_map_snapshot))

    from emet.agent.chat_tamp import build_tamp_handlers

    for name, handler in build_tamp_handlers(context).items():
        tools.append(_bind(name, func=handler))

    # -- current-view object geometry ----------------------------------------
    def ground_object(query: str) -> dict:
        """Ground the decision RGB-D snapshot without moving; return geometry or explicit abstention."""
        agent = _agent_from_context(context)
        ground = getattr(agent, "ground_current_view", None)
        if not callable(ground):
            return {"ok": False, "status": "unavailable", "note": "Current-view RGB-D grounding is unavailable."}
        if "decision_observation" in context:
            observation = context["decision_observation"]
            if observation is None:
                agent._grounded_query_target = None
                return {
                    "ok": False,
                    "status": "observation_unavailable",
                    "note": "No camera view attached for this decision.",
                }
            result = ground(query, observation=observation, decision_view=context["decision_view"])
        else:
            result = ground(query)
        return {**result, "note": result.get("reason") or "Object geometry admitted from the captured RGB-D view."}

    tools.append(_bind("ground_object", func=ground_object))

    # -- find_objects --------------------------------------------------------
    def find_objects(text: str) -> dict:
        """Run object search/approach and return task success separately from executor continuation."""
        executor = context.get("executor")
        if executor is None:
            return {"ok": False, "status": "unavailable", "note": "Robot not connected."}
        keep_going = executor([("find", text)])
        task_ok = keep_going and bool(getattr(executor, "_last_exec_ok", True))
        agent = _agent_from_context(context)
        plan_line = format_last_nav_plan_summary(agent)
        outcome = (getattr(agent, "_last_nav_plan", None) or {}).get("outcome") if agent else None
        head = format_nav_outcome_head(outcome, ok=task_ok, verb="Find")
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
        return {
            "ok": task_ok,
            "status": outcome or ("ok" if task_ok else "failed"),
            "note": " ".join(parts),
            "navigation": navigation_feedback(agent),
        }

    tools.append(_bind("find_objects", func=find_objects))

    # -- describe_scene ------------------------------------------------------
    def describe_scene(pan_rad=None, tilt_rad=None):
        """Caption without motion, or request measured head gaze and return its capture receipt."""
        robot = context.get("robot")
        executor = context.get("executor")
        if pan_rad is not None or tilt_rad is not None:
            from emet.agent.tool_outcome import ToolOutcome
            from emet.controller.dynamem.head_observation import capture_head_view

            agent = _agent_from_context(context)
            if agent is None:
                return ToolOutcome(False, status="unavailable", note="Robot not connected.")
            receipt, obs = capture_head_view(agent, pan_rad=pan_rad, tilt_rad=tilt_rad)
            if receipt["ok"]:
                stash_discord_image(context, np.asarray(obs.rgb).copy())
            return ToolOutcome.from_eqa_dict("describe_scene", receipt)
        if robot is None or not hasattr(robot, "get_observation"):
            return "No robot view available."
        obs = robot.get_observation()
        if obs is None or getattr(obs, "rgb", None) is None:
            return "No current image."
        from emet.agent.camera_snapshot import camera_snapshot

        obs, _identity = camera_snapshot(obs)
        live_rgb = np.asarray(obs.rgb)
        agent = getattr(executor, "agent", None) if executor is not None else None
        if agent is None or not hasattr(agent, "describe_head_camera_scene_text"):
            stash_discord_image(context, live_rgb)
            return (
                "I have a camera frame but this session's controller does not expose scene description; "
                "a photo is attached when Discord is connected."
            )
        text = agent.describe_head_camera_scene_text(
            observation=obs,
            graph_memory=context.get("graph_memory"),
            memory_backend=context.get("memory_backend"),
            graph_memory_backend=context.get("graph_memory_backend"),
        )
        # Caption and delivery refer to the same owned observation. A later live
        # frame cannot silently replace the image described by the model.
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

    tools.append(_bind("describe_scene", func=describe_scene))

    # -- say -----------------------------------------------------------------
    tools.append(
        _bind(
            "say", func=lambda text: _exec("say", text), executor_commands=lambda args: [("say", args.get("text", ""))]
        )
    )

    # -- emotes and gestures -------------------------------------------------
    for cmd in ("wave", "nod_head", "shake_head", "avert_gaze"):
        tools.append(_bind(cmd, func=lambda _cmd=cmd: _exec(_cmd, ""), executor_commands=_simple_exec_mapping(cmd)))

    # -- navigation ----------------------------------------------------------
    tools.append(_bind("go_home", func=lambda: _exec("go_home", ""), executor_commands=_simple_exec_mapping("go_home")))

    def observe_floor(pan_rad=None, tilt_rad=-1.0, target_blocker=False):
        """Inspect downward RGB-D and recheck the rejected footprint, restoring the prior task view."""
        from emet.agent.tool_outcome import ToolOutcome
        from emet.controller.dynamem.look import inspect_floor as capture_floor

        agent = _agent_from_context(context)
        if agent is None:
            return ToolOutcome(False, status="unavailable", note="Robot not connected.")
        return ToolOutcome.from_eqa_dict(
            "observe_floor", capture_floor(agent, pan_rad=pan_rad, tilt_rad=tilt_rad, target_blocker=target_blocker)
        )

    tools.append(_bind("observe_floor", func=observe_floor))

    def scan_environment() -> ToolOutcome:
        """Run an in-place survey and return map/plan feedback; base rotation is possible."""
        executor = context.get("executor")
        if executor is None:
            return ToolOutcome(False, status="unavailable", note="Robot not connected.", tool="scan_environment")
        keep_running = executor([("rotate_in_place", "")])
        task_ok = keep_running and bool(getattr(executor, "_last_exec_ok", True))
        if not task_ok:
            return ToolOutcome(False, status="failed", note="Scan interrupted or failed.", tool="scan_environment")
        gsize = _graph_size_line(context)
        if gsize:
            _logger.info(gsize)
            return ToolOutcome(
                True,
                status="completed",
                note=f"Completed in-place ≈360° scan; map/memory updated. [{gsize}]",
                tool="scan_environment",
            )
        return ToolOutcome(
            True, status="completed", note="Completed in-place ≈360° scan; map/memory updated.", tool="scan_environment"
        )

    tools.append(_bind("scan_environment", func=scan_environment))

    def rotate_base(degrees: float = 90.0) -> str | dict:
        """Request relative yaw in degrees; prefer validated motion and retain measured feedback."""
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        try:
            deg = float(degrees)
        except (TypeError, ValueError):
            return f"Invalid degrees: {degrees!r}."
        if not np.isfinite(deg):
            return "Invalid degrees: must be finite."
        deg = float(np.clip(deg, -180.0, 180.0))
        agent = getattr(executor, "agent", None)
        from emet.controller.dynamem.relative_motion import execute_relative_motion, supports_relative_motion

        if supports_relative_motion(agent):
            receipt = execute_relative_motion(agent, "rotate_base", deg)
            if receipt.get("code") == "ok":
                context["last_rotate_degrees"] = deg
            return receipt
        if agent is not None and callable(getattr(agent, "rotate_base_degrees", None)):
            agent.rotate_base_degrees(deg)
            receipt = getattr(agent, "_last_relative_motion_result", None)
            if isinstance(receipt, dict):
                if receipt.get("code") == "ok":
                    context["last_rotate_degrees"] = deg
                return receipt
            return "No rotation receipt available; no additional command issued."
        ok = executor([("rotate_base", str(deg))])
        if not ok:
            return "Rotate failed or interrupted."
        context["last_rotate_degrees"] = deg
        return f"Rotated about {deg:.0f}° in place."

    tools.append(_bind("rotate_base", func=rotate_base))

    def face_toward(object_label: str) -> str | dict:
        """Yaw in place to face a scene-graph / voxel object (no XY drive)."""
        import math

        from emet.agent.face_toward import resolve_object_xy, yaw_to_face_xy
        from emet.agent.tool_outcome import ToolOutcome
        from emet.utils.geometry import nav_xyt_to_world_xyt

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
            planning_pose = getattr(agent, "_planning_base_xyt", None)
            if callable(planning_pose):
                pose = np.asarray(planning_pose(pose), dtype=np.float64).reshape(-1)
            else:
                session_reader = getattr(robot, "get_emet_session", None)
                pose = nav_xyt_to_world_xyt(pose, session_reader() if callable(session_reader) else None)
        except Exception as e:
            return {"ok": False, "status": "base_pose_unavailable", "note": str(e)}
        if pose.size < 3 or not np.isfinite(pose[:3]).all():
            return {"ok": False, "status": "base_pose_unavailable", "note": "Base pose incomplete or nonfinite."}
        delta_rad, _bearing = yaw_to_face_xy(pose[:3], xy)
        deg = float(math.degrees(delta_rad))
        if abs(deg) < 3.0:
            msg = f"Already roughly facing the search location for {label!r} ({source})."
            hint = format_close_map_hint(close_map_from_agent(agent), float(xy[0]), float(xy[1]), is_chat=True)
            return f"{msg} {hint}".strip()
        deg = float(np.clip(deg, -180.0, 180.0))
        turn = ToolOutcome.coerce("rotate_base", rotate_base(deg))
        msg = (
            f"Measured turn reached {deg:+.0f}° toward the search location for {label!r} ({source})."
            if turn.ok
            else f"Turn toward the search location for {label!r} ({source}) did not complete."
        )
        hint = format_close_map_hint(close_map_from_agent(agent), float(xy[0]), float(xy[1]), is_chat=True)
        if hint:
            msg = f"{msg} {hint}"
        return {"ok": turn.ok, "status": turn.status, "note": msg, "navigation": turn.to_dict()}

    tools.append(_bind("face_toward", func=face_toward))

    def move_forward(meters: float = 0.1) -> str | dict:
        """Request bounded forward travel in meters, retaining clearance and measured-progress failures."""
        executor = context.get("executor")
        if executor is None:
            return "Robot not connected."
        try:
            dist = float(meters)
        except (TypeError, ValueError):
            return f"Invalid meters: {meters!r}."
        if not np.isfinite(dist):
            return "Invalid meters: must be finite."
        dist = float(np.clip(dist, 0.0, 1.5))
        # Prefer controller path so every nudge (including 0.1 m) is map-clipped.
        agent = getattr(executor, "agent", None)
        from emet.controller.dynamem.relative_motion import execute_relative_motion, supports_relative_motion

        if supports_relative_motion(agent):
            return execute_relative_motion(agent, "move_forward", dist)
        if agent is not None and hasattr(agent, "move_forward_meters"):
            commanded = float(agent.move_forward_meters(dist))
            receipt = getattr(agent, "_last_relative_motion_result", None)
            if isinstance(receipt, dict):
                return receipt
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

    tools.append(_bind("move_forward", func=move_forward))

    # -- camera --------------------------------------------------------------
    def take_picture() -> ToolOutcome:
        """Capture the head view locally through the executor; no remote delivery is promised."""
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
            return ToolOutcome(
                True, status="captured", note="Head-camera photo queued for Discord.", tool="take_picture"
            )
        return ToolOutcome(
            False,
            status="observation_unavailable",
            note="No head-camera frame is available; local capture is unverified.",
            tool="take_picture",
        )

    tools.append(_bind("take_picture", func=take_picture))

    def take_ee_picture():
        """Capture wrist RGB only after a successful arm aim in this session."""
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

    tools.append(_bind("take_ee_picture", func=take_ee_picture))

    # -- arm aim / closer look -----------------------------------------------
    def aim_arm_at(object_label: str):
        # See docs/attempt_ledger.md and plans/2026-08-08_embodied_agent_planning.md Phase 3.
        """Aim the wrist at a localized target when supported and retain success for wrist capture."""
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

    tools.append(_bind("aim_arm_at", func=aim_arm_at))

    # -- manipulation --------------------------------------------------------
    tools.append(
        _bind("hand_over", func=lambda: _exec("hand_over", ""), executor_commands=_simple_exec_mapping("hand_over"))
    )

    # -- query_scene_graph (GraphEQA + open-vocab fallback) -------------------
    def query_scene_graph(question: str) -> str:
        """Ask configured graph EQA or return the open-vocabulary graph fallback."""
        graph_answer = _query_graph_backend(question, caller="query_scene_graph")
        if graph_answer is not None:
            return graph_answer
        executor = context.get("executor")
        if executor is not None and hasattr(executor, "agent"):
            sg = executor.agent.get_voxel_map().get_scene_graph()
            if sg is not None and sg.num_objects > 0:
                return f"[Open-vocab scene graph snapshot]\n{sg.to_string()}\n(User question was: {question})"
        return "No graph memory or open-vocab scene graph available yet; explore or scan first."

    tools.append(_bind("query_scene_graph", func=query_scene_graph))

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

    tools.append(_bind("list_scene_relations", func=list_scene_relations))

    def send_object_image(object_label: str) -> str:
        """Queue a stored object crop by label; this does not capture a current camera view."""
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

    tools.append(_bind("send_object_image", func=send_object_image))

    # -- quit ----------------------------------------------------------------
    tools.append(_bind("quit", func=lambda: None, executor_commands=_simple_exec_mapping("quit")))

    registered = {t.name for t in tools}
    expected = set(CHAT_EXCLUSIVE_TOOL_NAMES)
    if registered != expected:
        missing = sorted(expected - registered)
        extra = sorted(registered - expected)
        raise RuntimeError(f"CHAT tool pack drift vs CHAT_SKILL_SPECS: missing={missing} extra={extra}")
    if env_base_rotate_only():
        # Keep tool names in the pack so the LLM does not invent calls; stubs return a clear refusal.
        from emet.agent.tool_policy import ROTATE_ONLY_BLOCKED_TOOLS
        from emet.controller.task.tamp.api import TAMP_TOOLS, response

        blocked = ROTATE_ONLY_BLOCKED_TOOLS
        stub_msg = (
            "I can't drive or translate right now (EMET_BASE_ROTATE_ONLY — tethered / plugged in). "
            "I can rotate in place, scan the room, or describe what I see. "
            "Unset EMET_BASE_ROTATE_ONLY when the robot is free to move."
        )

        def _make_stub(orig: Tool) -> Tool:
            def _fn(**_kwargs: Any) -> ToolOutcome:
                if orig.name in TAMP_TOOLS or orig.name in {"rotate_base", "move_forward"}:
                    envelope = response(orig.name, code="base_rotate_only", recovery="none")
                    envelope["message"] = stub_msg
                    return ToolOutcome.from_versioned_result(orig.name, envelope)
                return ToolOutcome(False, status="base_rotate_only", tool=orig.name, note=stub_msg)

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
