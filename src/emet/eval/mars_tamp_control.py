"""Mars public-tool oracle benchmark; preserve admission and execution failures.

Two distinct supports, physical grasp/transport/release, no relocation after
fixture initialization. Coordinated-base transport is screened in private planning state; observed-agent
acceptance and hardware remain separate gates.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from emet.controller.manipulation.native_scene import model_signature
from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.tracking import TaskFrameFK

CANONICAL_SUITE = Path(__file__).resolve().parents[3] / "configs/benchmarks/mars_public_tamp_whole_body_v2.json"


def suite():
    return {
        "version": "mars-public-tamp-whole-body-v2",
        "scope": __doc__,
        "cases": [
            {
                "id": f"{layout}:{mass}g:{i}",
                "layout": layout,
                "seed": 1861000 + i,
                "mass_kg": mass / 1000,
                "yaw_offset_rad": direction * 0.6,
            }
            for layout, direction in [("left", 1), ("right", -1)]
            for i in range(5)
            for mass in [20, 50, 100]
        ],
        "contract": {
            "support_half_width_m": 0.04,
            "placement_region_half_width_m": 0.03,
            "retention_translation_m": 0.02,
            "retention_rotation_rad": 0.1,
            "minimum_verified_lift_m": 0.05,
            "lift_reference_m": 0.06,
            "unsafe_contact_N": 30.0,
            "wheel_lift_allowed": False,
        },
        "base_motion": "coordinated",
    }


def sources():
    from emet.rl.manipulation.carry_benchmark import source_hashes

    root = Path(__file__).resolve().parents[1]
    result = source_hashes()
    paths = [p for package in ("controller/task/tamp", "agent", "core") for p in (root / package).rglob("*.py")] + [
        Path(__file__),
        Path(__file__).with_name("mars_tamp_metrics.py"),
    ]
    if CANONICAL_SUITE.exists():
        result["@canonical_public_tamp_suite"] = hashlib.sha256(CANONICAL_SUITE.read_bytes()).hexdigest()
    for p in paths:
        result[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def fixture(case):
    env = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="pregrasp",
            action_mode="direct_velocity",
            randomization_scale=0,
            two_supports=True,
            payload_object="object1",
            payload_mass_kg=case["mass_kg"],
            grasp_lift_height=0.06,
            support_half_width=0.04,
            balanced_grasp_setup=True,
            contact_cone="elliptic",
            contact_impratio=3,
            base_xy_tolerance=0.005,
            hand_position_tolerance=0.005,
            hand_orientation_tolerance=0.05,
        )
    )
    try:
        _, info = env.reset(seed=case["seed"])
        if info["initialization_failed"]:
            return env, {"accepted": False, "reason": info.get("initialization_failure_reason")}
        c = env.control
        fk = TaskFrameFK(env.model, env.data, c)
        q = c.reference.copy()
        reference = fk(c.base_pose(), q)
        base_witness = c.base_pose().copy()
        q[0] += case["yaw_offset_rad"]
        if np.any(q < c.limits[:, 0]) or np.any(q > c.limits[:, 1]):
            q = c.reference.copy()
            base_witness[2] += case["yaw_offset_rad"]
        hand = fk(base_witness, q)

        def rotation(pose):
            columns = pose[3:].reshape(3, 2)
            return np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])

        local = rotation(reference).T @ (env.data.xpos[env.payload] - reference[:3])
        target = hand[:3] + rotation(hand) @ local
        source = env.model.body_pos[env.model.body("table").id].copy()
        if np.max(abs(target[:2] - source[:2])) < 0.09:
            return env, {"accepted": False, "reason": "supports_not_separate"}
        destination = env.model.body("destination").id
        env.model.body_pos[destination] = [*target[:2], source[2]]
        source_geom = env.model.body_geomadr[env.model.body("table").id]
        env.model.geom_size[env.model.body_geomadr[destination]] = env.model.geom_size[source_geom]
        mujoco.mj_forward(env.model, env.data)
        return env, {
            "accepted": True,
            "source_support": source.tolist(),
            "destination_support": env.model.body_pos[destination].tolist(),
            "witness_arm": q.tolist(),
            "witness_base": base_witness.tolist(),
            "witness_hand": hand.tolist(),
            "placement_region": [list(target[:2] - 0.03), list(target[:2] + 0.03)],
        }
    except BaseException:
        env.close()
        raise


def final_predicates(robot, admission):
    from emet.simulation.sim_object_placements import placements_from_mujoco_model

    env = robot.env
    entry = placements_from_mujoco_model(env.model, env.data)[env.payload_name]
    bounds = np.asarray(entry["bounds"])[:, :2]
    region = np.asarray(admission["placement_region"])
    dof = env.model.body(env.payload).dofadr[0]
    velocity = env.data.qvel[dof : dof + 6].copy()
    fingers = {env.model.body(name).id for name in env.profile.finger_bodies}
    finger_contact = any(
        env.payload in env.model.geom_bodyid[c.geom] and bool(fingers & set(env.model.geom_bodyid[c.geom]))
        for c in env.data.contact[: env.data.ncon]
    )
    base_velocity = env.data.qvel[robot.servo.base_dof : robot.servo.base_dof + 6].copy()
    base_error = robot.servo.base_goal - robot.servo.base_pose()
    heading_error = np.arctan2(np.sin(base_error[2]), np.cos(base_error[2]))
    predicates = {
        "inside_goal_region": bool(np.all(bounds[0] >= region[0] - 0.005) and np.all(bounds[1] <= region[1] + 0.005)),
        "destination_support": robot.support_load("destination") >= 0.5 * env.model.body_mass[env.payload] * 9.81,
        "gripper_open": robot.servo.gripper_is_open(),
        "fingers_clear": not finger_contact,
        "object_still": bool(np.linalg.norm(velocity[:3]) <= 0.01 and np.linalg.norm(velocity[3:]) <= 0.1),
        "safe": not robot.failure and not env.monitor.wheel_lift and env.monitor.peak_force <= env.profile.unsafe_force,
        "robot_still": bool(
            np.max(abs(robot.servo.velocities())) < 0.03
            and np.linalg.norm(base_velocity[:2]) < 0.005
            and abs(base_velocity[5]) < 0.02
        ),
        "base_goal_within_tolerance": bool(np.linalg.norm(base_error[:2]) < 0.005 and abs(heading_error) < 0.04),
    }
    return {k: bool(v) for k, v in predicates.items()}


def freeze_catalog():
    catalog = suite()
    for case in catalog["cases"]:
        env, admission = fixture(case)
        try:
            case["frozen_fixture"] = admission
            case["model_signature"] = model_signature(env.model)
            case["initial_qpos"] = env.data.qpos.tolist()
        finally:
            env.close()
    return catalog


def run_case(case, *, tool_driver=None, input_mode="privileged"):
    """Default oracle calls the real tool registry; a driver may select tools.

    A tool_driver is an integration seam, not evidence of observed grounding or
    an LLM result. Caller must label and archive its actual inputs and outputs.
    """
    from emet.agent.tools import get_tools
    from emet.controller.task.tamp.native_adapter import adapter_for
    from emet.eval.mars_tamp_metrics import motion_metrics

    if input_mode not in {"privileged", "observed"}:
        raise ValueError("invalid TAMP input mode")
    env, admission = fixture(case)
    try:
        row = {
            "case": case,
            "fixture_admission": admission,
            "success": False,
            "input_mode": input_mode,
            "selector": "oracle" if tool_driver is None else getattr(tool_driver, "selector_label", "external_driver"),
        }
        if not admission["accepted"]:
            return row
        if "frozen_fixture" in case:
            expected = case["frozen_fixture"]
            fields = (
                "source_support",
                "destination_support",
                "witness_arm",
                "witness_base",
                "witness_hand",
                "placement_region",
            )
            matching = expected["accepted"] == admission["accepted"] and all(
                np.allclose(admission[k], expected[k], atol=1e-8, rtol=0) for k in fields if k in expected
            )
            if (
                not matching
                or case["model_signature"] != model_signature(env.model)
                or not np.allclose(env.data.qpos, case["initial_qpos"], atol=1e-8, rtol=0)
            ):
                row["fixture_admission"] = {"accepted": False, "reason": "frozen_fixture_changed"}
                return row
        row["frozen_fixture_verified"] = "frozen_fixture" in case
        robot = MarsSimRobot(env)
        context = {
            "robot": robot,
            "manip_mode": "physical",
            "tamp_inputs": input_mode,
            "_native_tamp_model": env.model,
        }
        tools = {t.name: t for t in get_tools(context)}
        calls = []
        references = {}

        def call(name, **kwargs):
            robot.refresh()
            result = json.loads(tools[name].func(**kwargs))
            calls.append({"tool": name, "arguments": kwargs, "result": result})
            if name == "plan_pick_place" and result["status"] == "ok":
                references.update(adapter_for(context, robot).pose_references(result["data"]["plan_ref"]))
            return result

        before = env.data.qpos.copy()
        initial_base = robot.servo.base_pose().copy()
        with robot.stepped_clock():
            if tool_driver is None:
                discovery = call("scene_tasks", object_filter="blue", robot="innate_mars")
                matches = [
                    x
                    for x in discovery.get("data", {}).get("tasks", [])
                    if x["receptacle_name"] == "destination support"
                ]
                plan = (
                    call(
                        "plan_pick_place",
                        task_ref=matches[0]["task_ref"],
                        placement_region=admission["placement_region"],
                    )
                    if len(matches) == 1
                    else None
                )
                if not np.array_equal(before, env.data.qpos) or robot.commands:
                    raise RuntimeError("planning changed live physics or sent commands")
                execution = (
                    call("execute_pick_place_plan", plan_ref=plan["data"]["plan_ref"])
                    if plan and plan["status"] == "ok"
                    else None
                )
            else:
                execution = tool_driver(call, {"placement_region": admission["placement_region"]}, tools)
        executed = [
            x
            for x in calls
            if x["tool"] == "execute_pick_place_plan"
            and x["result"]["status"] == "ok"
            and {"grasp", "place"} <= set(x["result"]["data"].get("completed_ops", []))
        ]
        predicates = final_predicates(robot, admission)
        stable = bool(executed and all(predicates.values()))
        if stable:
            for _ in range(40):
                robot.step(0.05)
                predicates = final_predicates(robot, admission)
                stable &= all(predicates.values())
        adapter = adapter_for(context, robot)
        row.update(
            calls=calls,
            pose_references=references,
            task_pose_trajectory=references.get("transport"),
            success=bool(execution and execution["status"] == "ok" and stable),
            final_predicates=predicates,
            post_execution_stable_s=2.0 if stable else 0.0,
            robot_failure=robot.failure,
            model_signature=model_signature(env.model),
            planning_rejections=adapter.planning_rejections,
            events=adapter.events,
            commands=robot.commands,
            controller_options={
                "physical_wheel_slew": True,
                "measured_path_retiming": True,
                "transport_timeout": "3 * nominal_duration + 4 seconds",
            },
            trace=robot.trace,
            motion_metrics=motion_metrics(robot.trace),
            max_payload_translation_m=robot.max_translation,
            max_payload_rotation_rad=robot.max_rotation,
            contact_metrics=env.monitor.metrics(),
            final_object=env.data.xpos[env.payload].tolist(),
            peak_base_displacement_m=max(
                (float(np.linalg.norm(np.asarray(x["base"])[:2] - initial_base[:2])) for x in robot.trace), default=0.0
            ),
        )
        return row
    finally:
        env.close()


def require_public_readiness(report):
    """Bind public-tool acceptance to the frozen catalog, sources and all cases."""
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("public TAMP evidence contains nonfinite or invalid values") from exc
    canonical = json.loads(CANONICAL_SUITE.read_text())
    expected = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    if report.get("suite_sha256") != expected or report.get("suite") != canonical:
        raise ValueError("public TAMP suite differs from canonical catalog")
    if report.get("source_hashes") != sources():
        raise ValueError("public TAMP source manifest is stale")
    rows = report.get("cases", [])
    if (
        report.get("complete") is not True
        or len(rows) != 30
        or {x["case"]["id"] for x in rows} != {x["id"] for x in canonical["cases"]}
    ):
        raise ValueError("complete canonical 30-case public TAMP evidence required")
    expected_cases = {x["id"]: x for x in canonical["cases"]}
    for row in rows:
        if row["case"] != expected_cases[row["case"]["id"]] or row.get("frozen_fixture_verified") is not True:
            raise ValueError("public TAMP fixture binding failed")
        if (
            row.get("success") is not True
            or row["selector"] != "oracle"
            or row["input_mode"] != "privileged"
            or row["fixture_admission"]["accepted"] is not True
            or row.get("model_signature") != row["case"]["model_signature"]
            or row.get("post_execution_stable_s", 0) < 2
            or set(row.get("final_predicates", {}))
            != {
                "inside_goal_region",
                "destination_support",
                "gripper_open",
                "fingers_clear",
                "object_still",
                "safe",
                "robot_still",
                "base_goal_within_tolerance",
            }
            or any(v is not True for v in row["final_predicates"].values())
            or row.get("peak_base_displacement_m", 0) < 0.02
            or row.get("max_payload_rotation_rad", 1) > 0.1
            or row.get("max_payload_translation_m", 1) > 0.02
        ):
            raise ValueError(f"public TAMP task/retention contract failed: {row['case']['id']}")
        metrics = row["contact_metrics"]
        if (
            metrics["wheel_lift"] is not False
            or metrics.get("wheel_support_evaluated") is not True
            or not 0 <= metrics["peak_contact_force"] <= canonical["contract"]["unsafe_contact_N"]
            or row.get("robot_failure") is not None
        ):
            raise ValueError("public TAMP force/wheel contract failed")
        executions = [x for x in row.get("calls", []) if x["tool"] == "execute_pick_place_plan"]
        if (
            len(executions) != 1
            or executions[0]["result"]["status"] != "ok"
            or not {"grasp", "place"} <= set(executions[0]["result"]["data"].get("completed_ops", []))
        ):
            raise ValueError("public TAMP successful execution receipt required")
    return True


def evaluate(catalog, output):
    from emet.eval.mars_tamp_metrics import summarize

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("refusing to overwrite public-tool evidence")
    manifest = sources()
    rows = []
    for case in catalog["cases"]:
        try:
            row = run_case(case)
        except Exception as exc:
            row = {
                "case": case,
                "success": False,
                "fixture_admission": {"accepted": None, "reason": "exception"},
                "exception": f"{type(exc).__name__}: {exc}",
            }
        if sources() != manifest:
            raise RuntimeError("public-tool sources changed during evaluation")
        rows.append(row)
        report = {
            "suite": catalog,
            "suite_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
            "source_hashes": manifest,
            "complete": len(rows) == len(catalog["cases"]),
            "successes": sum(x["success"] for x in rows),
            "cases": rows,
            "summary": summarize(rows),
            "scope": "public-tool oracle, independently stepped simulator, privileged inputs; coordinated-base track",
            "observed_agent_qualified": False,
            "moving_base_qualified": False,
            "hardware_qualified": False,
        }
        report["public_oracle_ready"] = False
        if report["complete"]:
            try:
                require_public_readiness(report)
                report["public_oracle_ready"] = report["moving_base_qualified"] = True
            except ValueError as exc:
                report["readiness_failure"] = str(exc)
        tmp = output / "report.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(output / "report.json")
        print(case["id"], "success", row["success"], "admission", row["fixture_admission"]["accepted"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--report", type=Path, required=True)
    plot.add_argument("--case", required=True)
    plot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(freeze_catalog(), indent=2) + "\n")
    elif args.command == "plot":
        from emet.eval.mars_tamp_metrics import plot_case

        report = json.loads(args.report.read_text())
        selected = next((row for row in report["cases"] if row["case"]["id"] == args.case), None)
        if selected is None:
            raise ValueError("unknown benchmark case")
        plot_case(selected, args.output)
    else:
        evaluate(json.loads(args.suite.read_text()), args.output)


if __name__ == "__main__":
    main()
