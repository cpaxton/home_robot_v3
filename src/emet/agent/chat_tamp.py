# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Context-bound CHAT manipulation handlers, separate from tool metadata.

Runtime input/adapter checks remain inside calls: registration never discovers
simulator metadata, initializes perception or qualifies an observed provider.
The existing TAMP APIs own plan handles, revalidation and physical execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def build_tamp_handlers(context: dict[str, Any]) -> dict[str, Callable[..., Any]]:
    """Bind lazy task handlers over the mutable CHAT session context."""
    from emet.controller.task.tamp.api import failure_code, json_tool, plan_data, response

    def _tamp_robot() -> Any | None:
        robot = context.get("robot")
        if robot is not None:
            return robot
        executor = context.get("executor")
        return getattr(executor, "robot", None) if executor is not None else None

    def _tamp_manip_mode() -> str:
        from emet.simulation.env_flags import env_manip_mode

        executor = context.get("executor")
        return (
            str(env_manip_mode() or getattr(executor, "_manip_mode", None) or context.get("manip_mode") or "auto")
            .strip()
            .lower()
        )

    def _tamp_inputs() -> str:
        executor = context.get("executor")
        return str(getattr(executor, "_tamp_inputs", None) or context.get("tamp_inputs") or "observed")

    def _observed_native_adapter():
        from emet.controller.task.tamp.observed_adapter import ObservedNativeAdapter

        adapter = context.get("observed_native_adapter")
        return adapter if isinstance(adapter, ObservedNativeAdapter) and adapter.robot is _tamp_robot() else None

    def _native_input_gate() -> str | None:
        if _tamp_manip_mode() != "physical":
            return None
        inputs = _tamp_inputs()
        if inputs not in {"observed", "privileged"}:
            return "invalid_tamp_inputs"
        if inputs == "observed":
            # Do not read simulator task metadata before an observed-scene
            # adapter is installed. Input isolation includes task discovery.
            return None if _observed_native_adapter() is not None else "observed_scene_unavailable"
        robot = _tamp_robot()
        session_getter = getattr(robot, "get_emet_session", None)
        session = session_getter() if callable(session_getter) else None
        if not isinstance(session, dict) or session.get("is_simulation") is not True:
            return "privileged_inputs_require_simulation"
        from emet.controller.manipulation.native_inputs import ObservationUnavailable
        from emet.controller.task.tamp.native_adapter import adapter_for

        try:
            return None if adapter_for(context, robot) is not None else "native_execution_unavailable"
        except (ObservationUnavailable, ValueError) as exc:
            return failure_code(str(exc))

    def _native_adapter():
        observed = _observed_native_adapter()
        if observed is not None:
            inputs = _tamp_inputs()
            if inputs == "observed":
                return observed
        from emet.controller.task.tamp.native_adapter import adapter_for

        return adapter_for(context, _tamp_robot())

    def _fallback_pick_place(executor: Any, object_name: str, receptacle_name: str) -> dict:
        keep_going = executor([("pickup", object_name), ("place", receptacle_name)])
        task_ok = keep_going and bool(getattr(executor, "_last_exec_ok", True))
        return response(
            "pick_place",
            code="ok" if task_ok else "controller_failed",
            data={"mode": "configured_controller", "assistance": None, "completed_ops": None, "measurements": None},
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
            robot_session_key,
        )

        requested_ref = str(task_ref).strip()
        native_error = _native_input_gate()
        if native_error:
            return AgentPlanBuild(None, None, "physical", True, native_error)
        selected = (context.get("_tamp_task_refs") or {}).get(requested_ref)
        if selected is not None and not isinstance(selected, AgentTaskRef):
            selected = None
        if requested_ref and context.get("_tamp_scene_key") != robot_session_key(_tamp_robot()):
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

    def _format_tamp_plan(plan_ref: str, build: Any) -> dict:
        plan = build.plan
        if plan is None or not plan.success or build.task is None:
            return response("plan_pick_place", code=failure_code(build.reason or "planner_error"))
        return response("plan_pick_place", data=plan_data(plan, build.mode, plan_ref))

    @json_tool
    def plan_pick_place(
        task_ref: str = "",
        object_name: str = "",
        receptacle_name: str = "",
        placement_region: list[list[float]] | None = None,
        execution_scope: str = "full_task",
    ) -> dict:
        """Cache a plan without actuation, optionally a native prefix before closure."""
        from emet.controller.task.tamp.agent_bridge import store_agent_plan

        if _tamp_manip_mode() == "physical":
            error = _native_input_gate()
            return (
                response("plan_pick_place", code=error)
                if error
                else _native_adapter().plan(
                    task_ref,
                    object_name,
                    receptacle_name,
                    placement_region,
                    **({} if execution_scope == "full_task" else {"execution_scope": execution_scope}),
                )
            )
        if execution_scope != "full_task":
            return response("plan_pick_place", code="execution_scope_unsupported")
        if placement_region is not None:
            return response("plan_pick_place", code="placement_region_unsupported")
        build = _build_tamp_plan(
            task_ref=task_ref,
            object_name=object_name,
            receptacle_name=receptacle_name,
        )
        if build.plan is None or not build.plan.success:
            return _format_tamp_plan("", build)
        plan_ref = store_agent_plan(context, _tamp_robot(), build)
        return _format_tamp_plan(plan_ref, build)

    @json_tool
    def execute_pick_place_plan(plan_ref: str) -> dict:
        """Consume and revalidate a primitive plan; return measured or refused execution JSON."""
        native_error = _native_input_gate()
        if native_error:
            return response("execute_pick_place_plan", code=native_error)
        if _tamp_manip_mode() == "physical":
            return _native_adapter().execute(str(plan_ref))
        from emet.controller.task.tamp.agent_bridge import execute_stored_agent_plan_result

        robot = _tamp_robot()
        if robot is None:
            return response("execute_pick_place_plan", code="robot_not_connected")
        return execute_stored_agent_plan_result(robot, context, str(plan_ref))

    @json_tool
    def pick_place(object_name: str, receptacle_name: str) -> dict:
        """Plan and execute via the configured manipulation adapter, retaining partial outcomes."""
        if _tamp_manip_mode() == "physical":
            error = _native_input_gate()
            if error:
                return response("pick_place", code=error)
            adapter = _native_adapter()
            planned = adapter.plan(object_name=object_name, receptacle_name=receptacle_name)
            result = adapter.execute(planned["data"]["plan_ref"]) if planned["status"] == "ok" else planned
            return {**result, "tool": "pick_place"}
        executor = context.get("executor")
        if executor is None and _tamp_robot() is None:
            return response("pick_place", code="robot_not_connected")
        if (
            executor is not None
            and _tamp_manip_mode() != "physical"
            and (
                bool(getattr(executor, "visual_servo", False))
                or _tamp_manip_mode() not in {"auto", "teleport", "kinematic"}
            )
        ):
            return _fallback_pick_place(executor, object_name, receptacle_name)
        build = _build_tamp_plan(object_name=object_name, receptacle_name=receptacle_name)
        if build.plan is not None and build.plan.success:
            from emet.controller.task.tamp.agent_bridge import execute_stored_agent_plan_result, store_agent_plan

            plan_ref = store_agent_plan(context, _tamp_robot(), build)
            result = execute_stored_agent_plan_result(_tamp_robot(), context, plan_ref)
            result["tool"] = "pick_place"
            return result
        if build.live_sim or _tamp_manip_mode() == "physical":
            return response("pick_place", code=failure_code(build.reason or "planner_error"))
        if executor is None:
            return response("pick_place", code="controller_unavailable")
        return _fallback_pick_place(executor, object_name, receptacle_name)

    @json_tool
    def set_receptacle_state(task_ref: str, state: str) -> dict:
        """Request assisted articulation using a task handle; physical mode can refuse it."""
        native_error = _native_input_gate()
        if native_error:
            return response("set_receptacle_state", code=native_error)
        if _tamp_manip_mode() == "physical":
            return response("set_receptacle_state", code="native_articulation_unavailable")
        from emet.controller.task.tamp.articulation import set_receptacle_state as apply_state

        return apply_state(_tamp_robot(), context, task_ref, state)

    @json_tool
    def scene_tasks(object_filter: str = "", robot: str = "") -> dict:
        """Enumerate pick-and-place options from the current scene as a compact digest."""
        session = context.get("tamp_task_session")
        if session is not None:
            return session.discover(object_filter, robot)
        if _tamp_manip_mode() == "physical":
            inputs = _tamp_inputs()
            if inputs == "observed":
                observed = _observed_native_adapter()
                if observed is not None:
                    return observed.discover(object_filter, robot)
                from emet.controller.task.tamp.observed_discovery import discover_observed_tasks

                return discover_observed_tasks(context.get("observed_tamp_provider"), object_filter, robot)
        native_error = _native_input_gate()
        if native_error:
            return response("scene_tasks", code=native_error)
        if _tamp_manip_mode() == "physical":
            return _native_adapter().discover(object_filter, robot)
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
        from emet.controller.task.tamp.agent_bridge import robot_session_key

        session_key = robot_session_key(robot_obj)
        active_metadata = resolve_scene_metadata_for_session(session, scenes_dir=scene_dir)
        candidates = (
            [active_metadata]
            if active_metadata is not None
            else ([] if live_sim else (sorted(ithor.glob("*_physics_metadata.json")) if ithor.is_dir() else []))
        )
        if not candidates:
            stable_scene_task_refs(context, [], {}, session_key=session_key)
            return response("scene_tasks", code="metadata_unavailable")
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

        reachability = None
        reachability_status = "not_requested"
        rid = (robot or "").strip().lower()
        if rid:
            try:
                from emet.eval.scene_task_extractor import compute_reachability_priors

                def _pose(body: str) -> np.ndarray:
                    info = placements.get(body)
                    if info is None and body.endswith("_1_1_0"):
                        info = placements.get(body[: -len("_1_1_0")] + "_1_0_0")
                    pos = (info or {}).get("pos")
                    return np.asarray(pos, dtype=float).reshape(3) if pos is not None else np.zeros(3)

                priors = compute_reachability_priors(
                    objs, robot_id=rid, arm="left", **({"object_pose_fn": _pose} if placements else {})
                )
                reachability = {
                    "robot": rid,
                    "source": "sim_placements" if placements else "zero_pose_proxy",
                    "reachable_categories": sorted({r.category for r in priors.values() if r.reachable}),
                }
                reachability_status = "evaluated"
            except Exception:
                reachability_status = "unavailable"
        from emet.simulation.articulation import articulation_access

        return response(
            "scene_tasks",
            data={
                "tasks": [
                    {
                        "task_ref": ref.ref,
                        "object_name": ref.object_query,
                        "receptacle_name": ref.receptacle_query,
                        "start_receptacle": ref.start_receptacle,
                        "access": articulation_access(session if live_sim else {}, ref.receptacle_body),
                    }
                    for ref in task_refs
                ],
                "pickable_categories": sorted({p.category for p in picks}),
                "receptacle_categories": sorted({r.category for r in recepts}),
                "reachability": reachability,
                "reachability_status": reachability_status,
            },
        )

    @json_tool
    def plan_task(instruction: str, strategy: str = "beam") -> dict:
        """Ask the bound multi-step provider to plan; return unavailable if no provider is bound."""
        from emet.controller.task.tamp.api import response

        session = context.get("tamp_task_session")
        if session is None:
            return response("plan_task", code="task_provider_unavailable")
        return session.plan(instruction, strategy)

    @json_tool
    def execute_task_plan(plan_ref: str) -> dict:
        """Consume a bound multi-step plan and report verified effects or the stopping failure."""
        from emet.controller.task.tamp.api import response

        session = context.get("tamp_task_session")
        if session is None:
            return response("execute_task_plan", code="task_provider_unavailable")
        return session.execute(plan_ref)

    return {
        "pick_place": pick_place,
        "plan_pick_place": plan_pick_place,
        "execute_pick_place_plan": execute_pick_place_plan,
        "set_receptacle_state": set_receptacle_state,
        "scene_tasks": scene_tasks,
        "plan_task": plan_task,
        "execute_task_plan": execute_task_plan,
    }
