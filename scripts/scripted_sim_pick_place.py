#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""No-neural-nets sim pick/place — GT + MuJoCo only (no LLM, VLM, SigLIP, YOLO, …).

Modes:
  - ``teleport`` (default): ``sim_set_body_pose`` GT snap (same as OVMM manip_mode=sim)
  - ``kinematic``: MuJoCo IK + RRT-Connect + joint streaming + kinematic attach (rby1)

When ``--tool-calls-json`` is supplied, calls run through the live CHAT tool
registry (including semantic handles, guarded plan storage, and TAMP execution).
Without it, kinematic mode directly invokes the executor for interactive
motion/video debugging only.

Verified (default table + rby1)::

  # Teleport — fastest smoke
  uv run python scripts/scripted_sim_pick_place.py --start-sim \\
    --sim configs/sim/default_table_rby1.yaml --manip-mode teleport \\
    --object "red cylinder" --receptacle "blue cube"

  # Kinematic — IK + RRT-Connect (set nav teleport so the base can approach)
  EMET_SIM_NAV_TELEPORT=1 uv run python scripts/scripted_sim_pick_place.py --start-sim \\
    --sim configs/sim/default_table_rby1.yaml --manip-mode kinematic \\
    --object "red cylinder" --receptacle "blue cube"

MolmoSpaces (needs ``.venv-molmospaces``)::

  EMET_SIM_NAV_TELEPORT=1 uv run python scripts/scripted_sim_pick_place.py --start-sim \\
    --sim configs/sim/molmospaces_ithor_train_0.yaml --manip-mode kinematic \\
    --object bowl --receptacle microwave

See ``docs/motion_planning.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _placement_pos(robot: Any, body: str) -> np.ndarray | None:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    pl = read_sim_object_placements(robot.get_emet_session())
    if not pl or body not in pl:
        return None
    return np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)


def _resolve_body(robot: Any, query: str) -> str | None:
    from emet.simulation.sim_manipulation import resolve_sim_object_body

    return resolve_sim_object_body(robot, query)


def run_scripted_tool_calls(
    robot: Any,
    tool_calls: list[dict[str, Any]],
    *,
    manip_mode: str,
    context_out: dict[str, Any] | None = None,
    video_recorder: Any = None,
) -> bool:
    """Run agent-shaped tool calls through the live CHAT tool implementation."""
    from emet.agent.tools import get_tools

    # Keep the execution path faithful to CHAT: plan/execute must resolve the
    # live session, select the requested manipulation capability, and validate
    # the one-shot plan. A direct KinematicPickPlaceExecutor call skips all of
    # those agent-facing contracts.
    context: dict[str, Any] = {"robot": robot, "manip_mode": manip_mode}
    tools_by_name = {t.name: t for t in get_tools(context)}

    print("Scripted tool_calls:")
    print(json.dumps(tool_calls, indent=2))
    ok_all = True
    handles: dict[str, str] = {}
    resolved_calls = []
    media_events = []
    for i, call in enumerate(tool_calls):
        name = str(call.get("name") or "")
        args = dict(call.get("arguments") or {})
        for key in ("task_ref", "plan_ref"):
            if args.get(key) == f"${key}":
                if key not in handles:
                    print(f"[{i}] missing prior successful {key}", file=sys.stderr)
                    return False
                args[key] = handles[key]
        resolved_calls.append({"name": name, "arguments": args})
        tool = tools_by_name.get(name)
        if tool is None:
            print(f"[{i}] unknown tool {name!r}", file=sys.stderr)
            ok_all = False
            continue
        if video_recorder is not None:
            video_recorder.set_status(name, detail=f"requested state: {args.get('state', '')}")
            time.sleep(1.0)  # Recording-only dwell; do not interpolate joint teleports.
            video_recorder.dump_paper_stills(f"{i:02d}_before_{name}")
        print(f"[{i}] {name}({args})")
        if tool.func is not None:
            result = tool.func(**args) if args else tool.func()
            print(f"    -> {result}")
            if video_recorder is not None:
                time.sleep(1.0)  # Allow a rendered observation after the measured receipt.
                stills = video_recorder.dump_paper_stills(f"{i:02d}_after_{name}_{args.get('state', '')}")
                try:
                    public_result = json.loads(result)
                except (TypeError, ValueError):
                    public_result = None
                media_events.append({"tool": name, "requested_state": args.get("state"),
                                     "result": public_result,
                                     "stills": {k: str(v) for k, v in stills.items()}})
                video_recorder.out_path.with_suffix(".json").write_text(
                    json.dumps({"schema_version": 1, "recording_dwell_s": 1.0,
                                "events": media_events}, indent=2) + "\n")
            from emet.controller.task.tamp.api import TAMP_TOOLS
            if name in TAMP_TOOLS:
                try:
                    outcome = json.loads(result)
                    if outcome['schema_version'] != 1 or outcome['status'] != 'ok':
                        ok_all = False
                        break
                    if name == 'scene_tasks':
                        handles.pop('task_ref', None)
                        tasks = outcome['data'].get('tasks') or []
                        if tasks:
                            handles['task_ref'] = tasks[0]['task_ref']
                    elif name == 'plan_pick_place':
                        handles['plan_ref'] = outcome['data']['plan_ref']
                except (TypeError, ValueError, KeyError):
                    ok_all = False
                    break
            elif isinstance(result, str) and "fail" in result.lower():
                ok_all = False

        else:
            print(f"[{i}] tool {name!r} has no callable implementation", file=sys.stderr)
            ok_all = False
    context["_scripted_resolved_calls"] = resolved_calls
    if context_out is not None:
        context_out.update(context)
    return ok_all


def _planned_receptacle_body(tool_calls: list[dict[str, Any]], context: dict[str, Any]) -> str | None:
    """Find the private GT receptacle selected by a semantic task handle."""
    refs = context.get("_tamp_task_refs")
    if not isinstance(refs, dict):
        return None
    for call in context.get("_scripted_resolved_calls", tool_calls):
        if str(call.get("name")) != "plan_pick_place":
            continue
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        task = refs.get(str(args.get("task_ref") or ""))
        body = getattr(task, "receptacle_body", None)
        if body:
            return str(body)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sim",
        default=None,
        help="Sim YAML when using --start-sim. Default: default_table_rby1 (kinematic) or default_table_stretch (teleport).",
    )
    parser.add_argument("--start-sim", action="store_true", help="Spawn headless mujoco_server from --sim.")
    parser.add_argument("--robot", default=None, help="Override robot id (default: from --sim or stretch).")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--port-offset", type=int, default=None)
    parser.add_argument("--object", default="red cylinder", help="Object query for pick_place.")
    parser.add_argument("--receptacle", default="blue cube", help="Receptacle query for pick_place.")
    parser.add_argument(
        "--manip-mode",
        choices=("teleport", "kinematic"),
        default=None,
        help="teleport (GT snap) or kinematic (IK+RRT+attach). Default: EMET_MANIP_MODE or teleport.",
    )
    parser.add_argument(
        "--tool-calls-json",
        default=None,
        help='JSON list of tool calls, e.g. \'[{"name":"pick_place","arguments":{...}}]\'.',
    )
    parser.add_argument("--articulation-cycle", action="store_true",
                        help="Separate assisted open/close smoke on the first discovered task; no placement score.")
    parser.add_argument("--cpu-only", action="store_true", help="Hide GPUs in sim subprocess env.")
    parser.add_argument("--verbose-sim", action="store_true", help="Print sim server stderr.")
    parser.add_argument(
        "--record-mp4",
        action="store_true",
        help="Record third-person MuJoCo view to MP4 (sets EMET_SIM_THIRD_PERSON=1).",
    )
    parser.add_argument("--video-fps", type=float, default=12.0, help="MP4 sample rate when --record-mp4.")
    parser.add_argument(
        "--video-out",
        type=str,
        default=None,
        help="MP4 path (default ~/runs/emet/manip_smoke/<stamp>/third_person.mp4).",
    )
    args = parser.parse_args()

    os.chdir(REPO)

    if args.record_mp4:
        os.environ["EMET_SIM_THIRD_PERSON"] = "1"
    from emet.simulation.env_flags import env_manip_mode
    from emet.simulation.sim_manipulation import resolve_agent_manip_mode

    manip_mode_early = resolve_agent_manip_mode(config_mode=args.manip_mode or env_manip_mode() or "teleport")
    if args.sim is None:
        args.sim = (
            "configs/sim/default_table_rby1.yaml"
            if manip_mode_early == "kinematic"
            else "configs/sim/default_table_stretch.yaml"
        )

    print(
        "=== no-neural-nets pick/place ===\n"
        f"  manip_mode={manip_mode_early!r}  sim={args.sim!r}\n"
        "  (GT placements + MuJoCo only — no LLM/VLM/SigLIP/YOLO)",
        flush=True,
    )

    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.eval.sim_eval_session import (
        connect_benchmark_robot,
        launch_benchmark_sim_server,
        terminate_benchmark_sim_server,
    )

    has_explicit_tool_calls = args.tool_calls_json is not None
    if has_explicit_tool_calls:
        tool_calls = json.loads(args.tool_calls_json)
        if not isinstance(tool_calls, list):
            raise SystemExit("--tool-calls-json must be a JSON list")
    else:
        tool_calls = [
            {
                "name": "pick_place",
                "arguments": {
                    "object_name": args.object,
                    "receptacle_name": args.receptacle,
                },
            }
        ]

    if args.articulation_cycle:
        if has_explicit_tool_calls:
            parser.error("--articulation-cycle cannot be combined with --tool-calls-json")
        has_explicit_tool_calls = True
        tool_calls = [
            {"name": "scene_tasks", "arguments": {"object_filter": args.object}},
            {"name": "set_receptacle_state", "arguments": {"task_ref": "$task_ref", "state": "open"}},
            {"name": "set_receptacle_state", "arguments": {"task_ref": "$task_ref", "state": "closed"}},
        ]

    sim_handle = None
    robot = None
    video = None
    try:
        if args.start_sim:
            sim_cfg = load_sim_launch_config_from_path(args.sim)
            if args.robot:
                sim_cfg = replace(sim_cfg, robot=str(args.robot))
            if args.port_offset is not None:
                sim_cfg = replace(sim_cfg, port_offset=int(args.port_offset))
            stderr = sys.stderr if args.verbose_sim else None
            print(f"Starting sim from {args.sim} robot={getattr(sim_cfg, 'robot', None)!r} …", flush=True)
            sim_handle = launch_benchmark_sim_server(
                sim_cfg,
                repo=REPO,
                cpu_only=bool(args.cpu_only),
                cwd=REPO,
                server_stderr=stderr,
            )
            robot = connect_benchmark_robot(sim_cfg, sim_handle.port_offset)
        else:
            robot_id = args.robot or "stretch"
            port_offset = int(args.port_offset or 0)
            print(f"Connecting to existing sim robot={robot_id!r} port_offset={port_offset} …", flush=True)
            robot = create_robot_client_from_cli(
                robot_id,
                args.robot_ip,
                port_offset=port_offset,
                enable_rerun_server=False,
                start_immediately=True,
                allow_missing_depth=True,
            )

        for _ in range(60):
            sess = robot.get_emet_session()
            if isinstance(sess, dict) and sess.get("is_simulation"):
                break
            time.sleep(0.25)
        sess = robot.get_emet_session() or {}
        caps = sess.get("capabilities") or {}
        manip_mode = manip_mode_early
        print(
            f"session runtime={sess.get('runtime_kind')!r} "
            f"sim_set_body_pose={caps.get('sim_set_body_pose')} "
            f"kinematic_manip={caps.get('kinematic_manip')} "
            f"manip_mode={manip_mode!r} "
            f"env={sess.get('environment')}",
            flush=True,
        )

        obj_body = _resolve_body(robot, args.object)
        before = _placement_pos(robot, obj_body) if obj_body else None
        print(f"GT object body={obj_body!r} pos_before={None if before is None else before.tolist()}")
        tool_context: dict[str, Any] = {}

        # Supplying calls selects the agent contract test even in kinematic
        # mode. The direct branch below is retained for interactive motion/video
        # debugging, where it is useful to call the executor by itself.
        if manip_mode == "kinematic" and not has_explicit_tool_calls:
            from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

            # Approach object on table (freejoint robots need EMET_SIM_NAV_TELEPORT for reliable snap).
            # Stand off farther than the base footprint so teleport does not embed the chassis in the table.
            if obj_body and before is not None:
                os.environ.setdefault("EMET_SIM_NAV_TELEPORT", "1")
                approach = np.array(
                    [float(before[0]), float(before[1]) + 0.55, -np.pi / 2],
                    dtype=np.float64,
                )
                print(f"Approaching object: move_base_to {approach.tolist()} (nav_teleport)", flush=True)
                robot.move_base_to(approach, blocking=True, world_frame=True)
                time.sleep(0.5)
                print(f"base_after_approach={np.asarray(robot.get_base_pose()).tolist()}", flush=True)

            video = None
            if args.record_mp4:
                from datetime import datetime

                from emet.visualization.manip_video import ManipVideoRecorder

                out = (
                    Path(args.video_out)
                    if args.video_out
                    else Path.home()
                    / "runs/emet/manip_smoke"
                    / datetime.now().strftime("%Y%m%d_%H%M%S")
                    / "third_person.mp4"
                )
                video = ManipVideoRecorder(
                    robot,
                    out,
                    fps=float(args.video_fps),
                    title="kinematic pick-place",
                )
                video.set_status(
                    "pick_and_place",
                    goal=f"{args.object} → {args.receptacle}",
                    detail=f"body={obj_body!r}",
                )
                video.start()

            exe = KinematicPickPlaceExecutor(robot, manip_collision="none")
            result = exe.pick_and_place(args.object, args.receptacle, object_gt_body=obj_body)
            if video is not None:
                video.set_status("done", detail=result.message)
                video.capture_once()
                mp4 = video.stop()
                video = None
                if mp4 is not None:
                    print(f"mp4 -> {mp4}", flush=True)
            print(
                f"kinematic result success={result.success} msg={result.message!r} "
                f"grasp_err={result.grasp_err_m} place_err={result.place_err_m}"
            )
            ok = bool(result.success)
        else:
            video = None
            if args.record_mp4:
                from datetime import datetime

                from emet.visualization.manip_video import ManipVideoRecorder

                out = (
                    Path(args.video_out)
                    if args.video_out
                    else Path.home()
                    / "runs/emet/manip_smoke"
                    / datetime.now().strftime("%Y%m%d_%H%M%S")
                    / "third_person.mp4"
                )
                video = ManipVideoRecorder(
                    robot,
                    out,
                    fps=float(args.video_fps),
                    title="ASSISTED joint teleport (no door-sweep check)" if args.articulation_cycle else f"{manip_mode} CHAT pick-place",
                )
                video.set_status("pick_place", goal=f"{args.object} → {args.receptacle}")
                video.start()
            print(
                f"Executing {len(tool_calls)} supplied call(s) through live CHAT tools (manip_mode={manip_mode})",
                flush=True,
            )
            ok = run_scripted_tool_calls(
                robot,
                tool_calls,
                manip_mode=manip_mode,
                context_out=tool_context,
                video_recorder=video,
            )
            if video is not None:
                video.set_status("done")
                video.capture_once()
                mp4 = video.stop()
                video = None
                if mp4 is not None:
                    print(f"mp4 -> {mp4}", flush=True)
        if args.articulation_cycle:
            print(json.dumps({"schema_version": 1, "track": "assisted_articulation_cycle",
                              "success": bool(ok), "placement_tested": False}))
            return 0 if ok else 1
        # Re-resolve after manip (session placements patched in place)
        after = _placement_pos(robot, obj_body) if obj_body else None
        print(f"pos_after={None if after is None else after.tolist()}")
        receptacle_body = _planned_receptacle_body(tool_calls, tool_context) or _resolve_body(robot, args.receptacle)
        receptacle_pos = _placement_pos(robot, receptacle_body) if receptacle_body else None
        print(
            f"GT receptacle body={receptacle_body!r} pos={None if receptacle_pos is None else receptacle_pos.tolist()}"
        )
        if before is not None and after is not None:
            delta = float(np.linalg.norm(after - before))
            print(f"displacement_m={delta:.4f}")
            if delta < 0.04:
                print("FAIL: object barely moved; check object/receptacle queries.", file=sys.stderr)
                ok = False
            if receptacle_pos is not None:
                placement_error = float(np.linalg.norm(after - receptacle_pos))
                print(f"placement_error_m={placement_error:.4f}")
                if placement_error > 0.25:
                    print("FAIL: object is not near the selected receptacle.", file=sys.stderr)
                    ok = False
            if ok:
                print(f"OK: measured object move after scripted {manip_mode} pick_place")
        elif obj_body is None:
            print("WARN: could not resolve GT body for displacement check", file=sys.stderr)
        return 0 if ok else 1
    finally:
        if video is not None:
            video.stop()
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if sim_handle is not None:
            terminate_benchmark_sim_server(sim_handle)


if __name__ == "__main__":
    raise SystemExit(main())
