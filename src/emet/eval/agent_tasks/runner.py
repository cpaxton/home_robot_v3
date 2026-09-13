"""Bounded assisted witnesses through the real shared CHAT tool registry."""

from __future__ import annotations

import hashlib
import math
import subprocess
import time
from pathlib import Path

from .recording import Recorder, write_json
from .spec import fingerprint, policy_task, score


def preflight(suite: dict) -> dict:
    from importlib.util import find_spec

    from .fixture import check_route, route_to

    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for module in ("numpy", "mujoco", "PIL", "matplotlib", "rerun"):
        check(f"dependency:{module}", find_spec(module) is not None, "availability only; no native initialization")
    scene = suite["scene"]
    check(
        "initial_not_solved",
        all(
            not score(e, {k: v["position"] for k, v in scene["objects"].items()})["success"] for e in suite["episodes"]
        ),
        "independent placement predicates",
    )
    try:
        check_route(scene, [scene["start_xyt"][:2]] * 2)
        for obj in scene["objects"].values():
            x, y, _ = obj["position"]
            route_to(scene, scene["start_xyt"], [x, y + 0.55, -math.pi / 2])
        for ep in suite["episodes"]:
            for change in ep.get("changes", []):
                x, y, _ = change["position"]
                route_to(scene, scene["start_xyt"], [x, y + 0.55, -math.pi / 2])
        check("footprint_routes", True, "2 cm chord samples; fixture obstacles inflated by robot radius")
    except ValueError as exc:
        check("footprint_routes", False, str(exc))
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "suite_fingerprint": fingerprint(suite),
        "paid_requests_enabled": False,
        "certification": "not_run",
        "next": "assisted fixture, then live ZMQ and learned-agent gates",
    }


def run_fixture(suite: dict, episode: dict, output: Path, *, render=True, control="witness", timeout_s=600) -> dict:
    from .fixture import FixtureRobot

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "episode": episode,
        "source_revision": revision,
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "suite_fingerprint": fingerprint(suite),
        "episode_fingerprint": fingerprint(episode),
        "robot": suite["robot"],
        "seed": suite["seed"],
        "control": control,
        "mode": "assisted_fixture",
        "integration_level": "shared_CHAT_tool_registry",
        "assistance": {
            "geometry": "simulator_truth",
            "navigation": "footprint_checked_kinematic",
            "manipulation": "object_pose_teleport",
            "physics": "forward_kinematics_only",
            "fixture_head_pitch_degrees": suite["scene"]["head_pitch_degrees"],
        },
        "models": {},
        "paid_requests_enabled": False,
        "timeout_s": timeout_s,
        "render_requested": render,
    }
    recorder = Recorder(output, manifest)
    robot = None
    start = time.monotonic()
    errors = []
    statuses = []
    event_changes = []
    actions = 0
    result = {"success": False, "completed": 0, "total": len(episode["goals"]), "goals": []}
    status = "infrastructure_failed"
    try:
        robot = FixtureRobot(suite, output, render=render)
        from emet.agent.tools import get_tools

        context = {"robot": robot, "manip_mode": "teleport"}
        registry = {tool.name: tool for tool in get_tools(context)}

        def snapshot(kind, policy=None, command_id=None, with_images=True):
            if time.monotonic() - start > timeout_s:
                raise TimeoutError("episode wall budget exhausted")
            evaluator = {
                "positions": robot.positions(),
                "robot_xyt": robot.xyt[:],
                "score": score(episode, robot.positions(), held=robot.held),
                "held": robot.held[:],
                "applied_changes": event_changes[:],
                "cameras": robot.camera_metadata(),
            }
            recorder.append(
                kind,
                policy=policy,
                evaluator=evaluator,
                images=robot.images() if with_images else {},
                command_id=command_id,
            )

        robot.on_motion = lambda: snapshot("motion", with_images=len(robot.trajectory) % 4 == 0)
        snapshot("start", policy_task(episode))
        goals = {g["object"]: g for g in episode["goals"]}
        order = episode["witness_order"] if control != "noop" else []
        if control == "partial":
            order = order[:1]
        for name in order:
            destination = goals[name]["destination"]
            if control == "wrong_destination":
                destination = "green_tray" if destination != "green_tray" else "red_tray"
            snapshot("decision", {"proposer": "scripted_oracle", "tool": "plan_pick_place"})
            arguments = {
                "object_name": suite["scene"]["objects"][name]["label"],
                "receptacle_name": suite["scene"]["objects"][destination]["label"],
            }
            before = int(context.get("_tamp_plan_counter", 0))
            response = registry["plan_pick_place"].func(**arguments)
            actions += 1
            plan_ref = f"plan:{before + 1}"
            planned = plan_ref in context.get("_tamp_plans", {})
            snapshot(
                "tool_result",
                {"tool": "plan_pick_place", "arguments": arguments, "result": response, "ok": planned},
                f"command:{actions}",
            )
            if not planned:
                statuses.append(False)
                break
            if control == "stale_plan":
                pos = robot.positions()[name]
                robot.relocate(name, [pos[0] + 0.45, pos[1], pos[2]])
                snapshot("world_change", {"note": "diagnostic change; evaluator only"})
            record = context["_tamp_plans"][plan_ref]
            response = registry["execute_pick_place_plan"].func(plan_ref=plan_ref)
            actions += 1
            # The shared tool currently returns text; match its explicit success
            # prefix and the stored structured plan, never 'does not contain fail'.
            ok = response.startswith("TAMP execution succeeded:") and bool(record["plan"].success)
            statuses.append(ok)
            snapshot(
                "tool_result",
                {"tool": "execute_pick_place_plan", "arguments": {"plan_ref": plan_ref}, "result": response, "ok": ok},
                f"command:{actions}",
            )
            if not ok:
                break
            for change in episode.get("changes", []):
                if change["after_goal"] == name and change not in event_changes:
                    if not next(g["passed"] for g in score(episode, robot.positions())["goals"] if g["object"] == name):
                        raise ValueError("world change trigger goal did not actually pass")
                    robot.relocate(change["object"], change["position"])
                    event_changes.append(change)
                    snapshot("world_change")
        result = score(episode, robot.positions(), held=robot.held)
        snapshot("finish", {"claimed_success": control == "noop" or all(statuses)})
        status = "completed" if result["success"] and all(statuses) else "task_failed"
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        status = "timeout" if isinstance(exc, TimeoutError) else "infrastructure_failed"
        recorder.append("error", evaluator={"error": errors[-1]})
    finally:
        if robot is not None:
            robot.close()
        recorder.close()
    metrics = {
        **result,
        "status": status,
        "errors": errors,
        "actions": actions,
        "elapsed_s": time.monotonic() - start,
        "event_hash": recorder.last_hash,
        "rendered": render and robot is not None and status != "infrastructure_failed",
        "physical_success_verified": False,
        "learned_agent_validated": False,
        "certification": "assisted_witness" if status == "completed" else "not_certified",
        "evidence_complete": False,
        "paid_cost_usd": 0.0,
    }
    write_json(output / "metrics.json", metrics)
    return metrics
