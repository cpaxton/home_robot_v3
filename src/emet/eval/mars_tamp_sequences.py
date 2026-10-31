"""Agent-authored multi-transfer development cases in the physical Mars fixture.

The agent proposes declarative goals and selects public tools, never simulator
writes or executable code. State is continuous within each episode. This is
privileged, independently stepped development evidence, not observed-agent,
household-scene or hardware qualification. Reference and fixed-checkpoint
execution are explicitly distinguished, including actual actor call counts.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from emet.eval.mars_tamp_agent import AgentToolDriver
from emet.eval.mars_tamp_control import CANONICAL_SUITE, final_predicates, fixture, sources


def validate_program(value):
    if not isinstance(value, dict) or set(value) != {"id", "instruction", "stages"}:
        raise ValueError("program requires exactly id, instruction and stages")
    if not isinstance(value["id"], str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", value["id"]):
        raise ValueError("invalid program id")
    if not isinstance(value["instruction"], str) or not 1 <= len(value["instruction"]) <= 1500:
        raise ValueError("invalid program instruction")
    stages = value["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 4:
        raise ValueError("program requires one to four transfers")
    previous = "source"
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"object", "support"} or stage["object"] != "blue":
            raise ValueError("this fixture supports the blue payload only")
        if stage["support"] not in {"source", "destination"} or stage["support"] == previous:
            raise ValueError("each stage must transfer to a different existing support")
        previous = stage["support"]
    return json.loads(json.dumps(value))


def stage_goals(program, admission):
    validate_program(program)
    source = np.asarray(admission["source_support"])[:2]
    regions = {
        "source": [(source - 0.03).tolist(), (source + 0.03).tolist()],
        "destination": admission["placement_region"],
    }
    return [
        {
            "object_filter": "blue",
            "support_body": "table" if s["support"] == "source" else "destination",
            "receptacle_name": s["support"] + " support",
            "placement_region": regions[s["support"]],
        }
        for s in program["stages"]
    ]


def manifest():
    result = sources()
    for path in (Path(__file__), Path(__file__).with_name("mars_tamp_agent.py")):
        result["eval/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_sequence(
    case,
    program,
    *,
    infer=None,
    max_turns=20,
    policy_directory=None,
    policy_evidence=None,
    record_observations=False,
    arm_execution="joint",
    arm_reference="legacy",
):
    """Reset only at episode start; never rearm/clear faults between stages."""
    from emet.agent.tools import get_tools
    from emet.controller.manipulation.native_scene import model_signature
    from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
    from emet.controller.task.tamp.native_adapter import adapter_for
    from emet.eval.mars_tamp_metrics import motion_metrics

    program = validate_program(program)
    if arm_execution not in {"joint", "cartesian"} or (arm_execution == "cartesian" and policy_directory is None):
        raise ValueError("Cartesian replay requires an explicit policy checkpoint")
    if arm_reference not in {"legacy", "jerk_limited"} or (arm_reference != "legacy" and arm_execution != "cartesian"):
        raise ValueError("jerk-limited replay requires Cartesian checkpoint execution")
    env, admission = fixture(case)
    row = {
        "case": case,
        "program": program,
        "fixture_admission": admission,
        "success": False,
        "selector": "scripted_reference" if infer is None else "llm_privileged",
        "completed_stages": 0,
        "physics_during_inference": False,
        "resets_between_stages": 0,
        "source_hashes": manifest(),
        "stages": [],
        "calls": [],
        "pose_references": {},
        "controller": "checkpoint_policy" if policy_directory is not None else "reference_teacher",
        "learned_policy_evaluated": False,
        "policy_calls": 0,
        "arm_execution": arm_execution,
        "arm_reference": arm_reference,
    }
    try:
        if not admission["accepted"]:
            return row
        expected = case.get("frozen_fixture")
        if (
            not expected
            or expected.get("accepted") is not True
            or case.get("model_signature") != model_signature(env.model)
            or not np.allclose(env.data.qpos, case["initial_qpos"], atol=1e-8, rtol=0)
            or any(not np.allclose(admission[k], expected[k], atol=1e-8, rtol=0) for k in expected if k != "accepted")
        ):
            row["fixture_admission"] = {"accepted": False, "reason": "frozen_fixture_changed"}
            return row
        goals = stage_goals(program, admission)
        row["goals"] = goals
        row["initial_state"] = {
            "qpos": env.data.qpos.tolist(),
            "qvel": env.data.qvel.tolist(),
            "ctrl": env.data.ctrl.tolist(),
            "time": float(env.data.time),
        }
        if policy_directory is not None:
            if policy_evidence is None:
                raise ValueError("checkpoint execution requires archived training evidence")
            try:
                robot = MarsSimRobot.from_checkpoint(
                    env,
                    policy_directory,
                    evidence=policy_evidence,
                    record_observations=record_observations,
                    arm_execution=arm_execution,
                    arm_reference=arm_reference,
                )
            except ValueError as exc:
                row["policy_admission"] = {"accepted": False, "reason": str(exc)}
                row["failure_at"] = {"stage": 0, "tool": None, "category": "policy_admission"}
                return row
            row["policy_admission"] = {"accepted": True, **robot.policy.admission}
        else:
            robot = MarsSimRobot(env, record_observations=record_observations)
        context = {
            "robot": robot,
            "manip_mode": "physical",
            "tamp_inputs": "privileged",
            "_native_tamp_model": env.model,
        }
        tools = {t.name: t for t in get_tools(context)}
        discovered, plans = {}, {}
        failed = False

        def rejected(name, code):
            from emet.controller.task.tamp.api import response

            result = response(name, status="error", code=code, data={"execution_started": False})
            row["calls"].append({"tool": name, "result": result, "dispatched": False})
            return result

        def call(name, **kwargs):
            nonlocal failed
            stage = row["completed_stages"]
            if failed or stage >= len(goals):
                return rejected(name, "episode_closed")
            goal = goals[stage]
            if name == "plan_pick_place":
                ref = kwargs.get("task_ref")
                if ref not in discovered or discovered[ref] != goal["receptacle_name"]:
                    return rejected(name, "rediscover_and_choose_current_support")
                region = np.asarray(kwargs.get("placement_region", []), dtype=float)
                if region.shape != (2, 2) or not np.allclose(region, goal["placement_region"], atol=1e-8, rtol=0):
                    return rejected(name, "frozen_stage_region_required")
            if name == "execute_pick_place_plan" and plans.get(kwargs.get("plan_ref")) != stage:
                return rejected(name, "current_stage_plan_required")
            before, commands = env.data.qpos.copy(), len(robot.commands)
            checkpoint_before = snapshot()
            robot.refresh()
            result = json.loads(tools[name].func(**kwargs))
            entry = {
                "tool": name,
                "arguments": kwargs,
                "raw_result": result,
                "dispatched": True,
                "stage": stage,
                "sim_time": float(env.data.time),
                "checkpoint_before": checkpoint_before,
                "checkpoint_after": snapshot(),
            }
            row["calls"].append(entry)
            if name in {"scene_tasks", "plan_pick_place"} and (
                not np.array_equal(before, env.data.qpos) or len(robot.commands) != commands
            ):
                raise RuntimeError("discovery/planning changed live physics")
            if name == "scene_tasks" and result["status"] == "ok":
                discovered.update(
                    {
                        x["task_ref"]: x["receptacle_name"]
                        for x in result["data"]["tasks"]
                        if "blue" in x["object_name"].lower()
                    }
                )
            if name == "plan_pick_place" and result["status"] == "ok":
                ref = result["data"]["plan_ref"]
                plans[ref] = stage
                row["pose_references"][ref] = adapter_for(context, robot).pose_references(ref)
            if name == "execute_pick_place_plan":
                predicates = final_predicates(robot, goal, support_body=goal["support_body"])
                verified = result["status"] == "ok" and all(predicates.values())
                if verified:
                    for _ in range(40):
                        robot.step(0.05)
                        predicates = final_predicates(robot, goal, support_body=goal["support_body"])
                        verified &= all(predicates.values())
                row["stages"].append(
                    {
                        "index": stage,
                        "goal": goal,
                        "tool_status": result["status"],
                        "verified": verified,
                        "predicates": predicates,
                        "sim_time": float(env.data.time),
                        "trace_end": len(robot.trace),
                    }
                )
                if verified:
                    row["completed_stages"] += 1
                    discovered.clear()
                    plans.clear()
                elif result["status"] in {"ok", "partial"} or result["data"].get("execution_started"):
                    failed = True
                    robot.stop_and_confirm(
                        result["code"] if result["status"] != "ok" else "independent_verification_failed"
                    )
                    if result["status"] == "ok":
                        result = {**result, "status": "partial", "code": "independent_verification_failed"}
                entry["independent_verified"] = verified
            entry["result"] = result
            return result

        def snapshot():
            # Diagnostic execution checkpoints, not resumable simulator saves.
            return {
                "qpos": env.data.qpos.tolist(),
                "qvel": env.data.qvel.tolist(),
                "ctrl": env.data.ctrl.tolist(),
                "time": float(env.data.time),
                "phase": robot.phase,
                "base_goal": robot.servo.base_goal.tolist(),
                "hand_goal": robot.servo.hand_goal.tolist(),
                "grip_reference": robot.servo.grip_reference,
                "policy_calls": robot.policy_calls,
                "failure": robot.failure,
            }

        with robot.stepped_clock():
            if infer is None:
                for goal in goals:
                    discovery = call("scene_tasks", object_filter="blue", robot="innate_mars")
                    candidates = [
                        x
                        for x in discovery.get("data", {}).get("tasks", [])
                        if x["receptacle_name"] == goal["receptacle_name"]
                    ]
                    if len(candidates) != 1:
                        break
                    plan = call(
                        "plan_pick_place", task_ref=candidates[0]["task_ref"], placement_region=goal["placement_region"]
                    )
                    if plan["status"] != "ok":
                        break
                    result = call("execute_pick_place_plan", plan_ref=plan["data"]["plan_ref"])
                    if result["status"] != "ok":
                        break
            else:
                instruction = (
                    "Execute this ordered multi-step task using the blue object. "
                    + program["instruction"]
                    + " The authoritative ordered stage contracts are "
                    + json.dumps(goals)
                    + ". Discover scene_tasks before EACH stage. Choose the matching task_ref and pass its exact "
                    "placement_region to plan_pick_place. Execute only its returned plan_ref. One tool call per turn. "
                    "Do not change goals. Stop after partial execution. Physics is continuous between transfers; "
                    "you cannot reset, attach, teleport, or recover a held-object failure. Inputs are simulator ground truth."
                )
                driver = AgentToolDriver(
                    infer,
                    input_mode="privileged",
                    max_turns=max_turns,
                    instruction=instruction,
                    executions_required=len(goals),
                )
                driver(call, {}, tools)
                row.update(agent_trace=driver.trace, agent_terminal=driver.terminal, agent_instruction=instruction)
        row.update(
            success=row["completed_stages"] == len(goals) and not failed,
            robot_failure=robot.failure,
            stop_confirmed=robot.stop_confirmed,
            commands=robot.commands,
            trace=robot.trace,
            events=adapter_for(context, robot).events,
            motion_metrics=motion_metrics(robot.trace),
            contact_metrics=env.monitor.metrics(),
            max_payload_translation_m=robot.max_translation,
            max_payload_rotation_rad=robot.max_rotation,
            learned_policy_evaluated=robot.policy_calls > 0,
            policy_calls=robot.policy_calls,
            policy_error=robot.policy_error,
            final_state={"qpos": env.data.qpos.tolist(), "qvel": env.data.qvel.tolist(), "time": float(env.data.time)},
        )
        if row["source_hashes"] != manifest():
            raise RuntimeError("sequence sources changed during execution")
        if robot.policy is not None:
            robot.policy.verify_unchanged()
        row["uncompleted_stages"] = len(goals) - row["completed_stages"]
        if not row["success"] and row["calls"]:
            last = row["calls"][-1]
            row["failure_at"] = {
                "stage": row["completed_stages"],
                "tool": last["tool"],
                "code": last["result"]["code"],
                "category": "planning" if last["tool"] == "plan_pick_place" else "execution_or_verification",
            }
        return row
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="left:20g:0")
    parser.add_argument("--base-url")
    parser.add_argument("--remote-model")
    parser.add_argument("--program", type=Path, help="Replay a frozen declarative program without model generation")
    parser.add_argument("--policy", type=Path, help="Compare this fixed checkpoint against the reference")
    parser.add_argument("--policy-evidence", type=Path, help="Committed training report summary binding the checkpoint")
    parser.add_argument("--agent", action="store_true", help="Also select tools with the remote agent during replay")
    parser.add_argument(
        "--record-observations",
        action="store_true",
        help="Retain actor inputs and pre/post physics for offline diagnosis",
    )
    parser.add_argument("--arm-execution", choices=["joint", "cartesian"], default="joint")
    parser.add_argument("--arm-reference", choices=["legacy", "jerk_limited"], default="legacy")
    parser.add_argument("--max-turns", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_turns <= 32:
        parser.error("turn budget must be in [1, 32]")
    if bool(args.policy) != bool(args.policy_evidence):
        parser.error("--policy and --policy-evidence must be supplied together")
    if args.arm_execution == "cartesian" and args.policy is None:
        parser.error("Cartesian arm execution requires --policy")
    if (args.program is None or args.agent) and not (args.base_url and args.remote_model):
        parser.error("generation and agent selection require --base-url and --remote-model")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(CANONICAL_SUITE.read_text())["cases"]
    case = next((x for x in cases if x["id"] == args.case), None)
    if case is None:
        parser.error("unknown frozen fixture")
    from emet.agent.loop import _call_llm
    from emet.llms.openai_client import OpenaiClient

    def infer(system, prompt):
        client = OpenaiClient(system, model=args.remote_model, base_url=args.base_url)
        client._openai = client._openai.with_options(timeout=120, max_retries=0)
        return _call_llm(client, prompt, None, False, reset_context=True)[0]

    system = "Generate one bounded robotic manipulation test case as strict JSON. No tools, code, or markdown."
    prompt = (
        "Existing environment: Innate Mars, one blue primitive payload initially on source support; "
        "one other empty destination support. Generate an ordered TWO-transfer task that exercises "
        "placement, release, regrasp and return without resetting. Only existing supports source/destination "
        'and object blue are allowed. Return exactly {"id":"lowercase_id","instruction":"task description",'
        '"stages":[{"object":"blue","support":"destination"},{"object":"blue","support":"source"}]}. '
        "Do not claim feasibility or invent additional objects, actions, obstacles or supports."
    )
    raw = args.program.read_text() if args.program else infer(system, prompt)
    (args.output / "generation.json").write_text(
        json.dumps(
            {
                "system": system,
                "prompt": prompt,
                "response": raw,
                "frozen_program": str(args.program) if args.program else None,
                "model_id": args.remote_model,
                "input_mode": "privileged",
                "source_hashes": manifest(),
            },
            indent=2,
        )
        + "\n"
    )
    program = validate_program(json.loads(raw))
    (args.output / "program.json").write_text(json.dumps(program, indent=2) + "\n")
    report = {
        "scope": __doc__,
        "model_id": args.remote_model,
        "program": program,
        "case": case,
        "source_hashes": manifest(),
        "training_started": False,
        "safety_promoted": False,
        "observed_agent_qualified": False,
        "household_scene_qualified": False,
        "episodes": [],
        "case_generation": "template-constrained language-model program; not household task diversity",
    }
    selectors = (
        ("scripted_reference", "llm_privileged") if args.program is None or args.agent else ("scripted_reference",)
    )
    combinations = [
        (selector, checkpoint)
        for selector in selectors
        for checkpoint in ([None, args.policy] if args.policy else [None])
    ]
    for selector, checkpoint in combinations:
        row = run_sequence(
            case,
            program,
            infer=None if selector == "scripted_reference" else infer,
            max_turns=args.max_turns,
            policy_directory=checkpoint,
            policy_evidence=args.policy_evidence,
            record_observations=args.record_observations,
            arm_execution=args.arm_execution if checkpoint is not None else "joint",
            arm_reference=args.arm_reference,
        )
        report["episodes"].append(row)
        report["complete"] = len(report["episodes"]) == len(combinations)
        path = args.output / "report.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)
        print(
            selector,
            row["controller"],
            row["completed_stages"],
            "/",
            len(program["stages"]),
            row["success"],
            flush=True,
        )


if __name__ == "__main__":
    main()
