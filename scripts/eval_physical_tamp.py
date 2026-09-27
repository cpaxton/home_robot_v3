#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""GT-only physical TAMP acceptance. Run physical trials serially via emet jobs.

--dry-run validates inputs and prints the exact trial identity without loading a simulator.
--tier static builds a sampled geometry/IK witness; it does not claim execution.
--tier physical uses the normal robot base/arm/gripper controllers, with an independent trace.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class TrialWallTimeout(BaseException):
    """Escape task/controller Exception handlers to terminate the whole trial."""


def arm_client_command(values):
    """Bind MJCF coordinates to the ZMQ client's base/lift/arm/roll/pitch/yaw API."""
    return [
        0,
        values["joint_lift"],
        sum(values[f"joint_arm_l{i}"] for i in range(4)),
        values["joint_wrist_roll"],
        values["joint_wrist_pitch"],
        values["joint_wrist_yaw"],
    ]


def latest_trace(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 262144))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if "qpos" in row:
                return row
        except (ValueError, UnicodeDecodeError):
            continue
    raise RuntimeError("missing_measured_state")


def run(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
    from emet.controller.task.tamp.task_search import execute_task_plan, plan_pick_place_mcts
    from emet.eval.physical_tamp import (
        SceneNavigationSpace,
        base_pose,
        check_arrival_ik_samples,
        freeze_fixture,
        kinematic_base_candidates,
        make_scene_checker,
        save_motion_overview,
        score_physical_acceptance,
        support_release_points,
        write_offline_base_pose,
    )
    from emet.motion.mujoco_collision import verify_collision_kernel
    from emet.motion.navigation_sweep import execute_measured_route

    output = Path(args.output_dir).expanduser().resolve()
    model, data, scorer, manifest = freeze_fixture(args.sim, args.scorer, output, seed=args.seed)
    manifest.update(
        tier=args.tier,
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    )
    manifest["budgets"]["wall_timeout_s"] = args.timeout
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    server = robot = server_fh = None
    trace = output / "physical_trace.jsonl"
    audit = output / "actuation_audit.jsonl"
    events = (output / "stages.jsonl").open("x")
    result = {"task_success": False, "tier": args.tier, "seed": args.seed, "status": "not_started"}
    started = time.monotonic()
    diagnostics, selected = [], {}
    initial_base = None
    capture_counts = {}

    def timed_out(_signum, _frame):
        raise TrialWallTimeout("trial_wall_timeout")

    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)

    def event(**kwargs):
        events.write(json.dumps({"wall_elapsed_s": time.monotonic() - started, **kwargs}, allow_nan=False) + "\n")
        events.flush()
        if time.monotonic() - started > args.timeout:
            raise TrialWallTimeout("trial_wall_timeout")
        if kwargs.get("phase") in ("pregrasp_complete", "grasp_complete", "lift_complete", "place_complete", "release"):
            capture(kwargs["phase"])

    def synchronize(target):
        if args.tier == "physical":
            row = latest_trace(trace)
            if time.time() - row["wall_time"] > 2:
                raise RuntimeError("stale_measured_state")
            target.qpos[:] = row["qpos"]
            target.qvel[:] = row["qvel"]
            target.ctrl[:] = row["ctrl"]
            target.time = row["sim_time"]
        mujoco.mj_kinematics(model, target)

    def capture(stage):
        if robot is None:
            return
        try:
            capture_counts[stage] = capture_counts.get(stage, 0) + 1
            stage = f"{stage}_{capture_counts[stage]}"
            observation = robot.get_observation()
            from PIL import Image

            Image.fromarray(observation.rgb).save(output / f"{stage}_head.png")
            np.save(output / f"{stage}_depth.npy", observation.depth)
            synchronize(data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (data.body("base_link").xpos + data.body(scorer["object_body"]).xpos) / 2
            camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -25
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(output / f"{stage}_side.png")
        except Exception as exc:
            event(phase="capture", stage=stage, error=str(exc))

    try:
        # These are robot interface bindings, not scene-specific grasp policies.
        from emet.robots import get_robot_spec

        spec = get_robot_spec(manifest["robot"])
        chain = spec.arm_chain
        required = (
            "joint_lift",
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
        )
        if chain is None or not set(required).issubset(chain.joint_names):
            raise RuntimeError("unsupported_physical_arm_adapter")
        joints = tuple(name for name in chain.joint_names if name in required)
        if args.tier == "physical":
            if not os.environ.get("EMET_JOB_ID"):
                raise RuntimeError("physical_trials_require_emet_jobs")
            os.environ.update(
                EMET_PHYSICAL_EXECUTION="1",
                EMET_PHYSICAL_AUDIT=str(audit),
                EMET_SIM_NAV_TELEPORT="0",
                EMET_MOLMOSPACES_NAV_TELEPORT="0",
                EMET_SIM_BASE_SPEED_SCALE="1",
                EMET_SIM_EVAL_TRACE=str(trace),
                EMET_PHYSICAL_START_MARKER=str(output / "execution_started.json"),
            )
            scorer = {
                **scorer,
                "require_robot_collision_audit": True,
                "release_actuator": "gripper",
                "release_open_fraction": 0.9,
            }
            (output / "runtime_scorer.json").write_text(json.dumps(scorer, indent=2) + "\n")
            os.environ["EMET_SIM_EVAL_CONFIG"] = str(output / "runtime_scorer.json")
            from eval_tamp_clutter import _launch_server

            from emet.app.robot_cli import create_robot_client_from_cli

            server, server_log, server_fh = _launch_server(str(output / "sim.yaml"), args.port_offset, cpu_only=False)
            (output / "server_log_path.txt").write_text(str(server_log) + "\n")
            robot = create_robot_client_from_cli(
                manifest["robot"],
                "127.0.0.1",
                port_offset=args.port_offset,
                start_immediately=True,
                allow_missing_depth=True,
            )
            with trace.open() as stream:
                runtime = json.loads(stream.readline())
            # The server may resolve startup model settings. Plan against its
            # exact compiled geometry/physics, preserving the source model too.
            model = mujoco.MjModel.from_binary_path(runtime["runtime_model_path"])
            data = mujoco.MjData(model)
            manifest["runtime_model"] = {k: v for k, v in runtime.items() if k not in ("config", "schema")}
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            synchronize(data)
            np.savez(
                output / "measured_initial_state.npz", qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, act=data.act
            )
            (output / "execution_started.json").write_text(
                json.dumps({"sim_time": data.time, "wall_time": time.time()})
            )
            time.sleep(2)  # Independent scorer needs a supported, settled baseline.
            synchronize(data)
            capture("initial")
        else:
            # Offline witness uses the declared initial state, never counts as live acceptance.
            if args.initial_state:
                state = np.load(args.initial_state)
                data.qpos[:] = state["qpos"]
                mujoco.mj_kinematics(model, data)

        kernel = verify_collision_kernel(model, data)
        (output / "collision_kernel.json").write_text(json.dumps(kernel, indent=2) + "\n")
        if not kernel["equivalent"]:
            raise RuntimeError("collision_kernel_validation_failed")
        checker = make_scene_checker(model, scorer)
        collision_links = [model.body(int(model.joint(name).bodyid[0])).name for name in joints]
        result["unsupported_collision_links"] = checker.unsupported_links(model, collision_links)
        if result["unsupported_collision_links"]:
            raise RuntimeError("unsupported_arm_collision_geometry")
        # Normal simulated base control allows reverse below 0.5 m. Our 0.2 m
        # segments leave room for measured arrival error within that threshold.
        space = SceneNavigationSpace(model, data, checker, seed=args.seed, allow_reverse=True)
        initial = data.qpos.copy()
        initial_base = base_pose(model, data)
        initial_rotation = data.body(scorer["ee_body"]).xmat.reshape(3, 3).copy()
        object_pos = data.body(scorer["object_body"]).xpos.copy()
        support_pos = data.body(scorer["support_body"]).xpos.copy()
        placements = {
            scorer["object_body"]: {"pos": object_pos.tolist(), "cat": "target"},
            scorer["support_body"]: {"pos": support_pos.tolist(), "cat": "support"},
        }
        oracle = SimpleNamespace(get_emet_session=lambda: {"is_simulation": True, "sim_object_placements": placements})

        def command(q):
            robot.switch_to_manipulation_mode()
            values = dict(zip(joints, q, strict=True))
            cmd = arm_client_command(values)
            # arm_to otherwise slews the head to a default EE view, outside the
            # validated arm path. Hold the freshly measured head configuration.
            head = [float(data.qpos[model.joint(name).qposadr[0]]) for name in ("joint_head_pan", "joint_head_tilt")]
            ok = robot.arm_to(cmd, head=head, blocking=True, timeout=15, min_time=0.1)
            time.sleep(0.12)
            return ok

        executor = PhysicalPickPlaceExecutor(
            robot,
            model=model,
            data=data,
            ee_body=scorer["ee_body"],
            joint_names=joints,
            collision=checker,
            synchronize=synchronize,
            command_joints=command,
            coupled_groups=(tuple(f"joint_arm_l{i}" for i in range(4)),),
            event=event,
            base_body="base_link",
            joint_limit_margins={f"joint_arm_l{i}": 0.005 for i in range(4)},
        )
        preparation_path = None
        preparation_state = initial.copy()
        preparation_error = None
        if chain.navigation_arm_q:
            targets = dict(zip(chain.joint_names, chain.navigation_arm_q, strict=True))
            preparation_path, preparation_error = executor.plan_joint_target(np.array([targets[n] for n in joints]))
            if preparation_path is not None:
                preparation_state = data.qpos.copy()
            data.qpos[:] = initial
        candidates = []
        # First preserve a stationary witness; then explore a fixed ring order.
        poses = [initial_base.copy()]
        poses.extend(
            kinematic_base_candidates(
                model,
                data,
                target_xy=object_pos[:2],
                ee_body=scorer["ee_body"],
                extension_joints=tuple(f"joint_arm_l{i}" for i in range(4)),
            )
        )
        for pose in poses:
            candidates.append(
                {
                    "object_query": "target",
                    "receptacle_query": "support",
                    "object_gt_body": scorer["object_body"],
                    "receptacle_gt_body": scorer["support_body"],
                    "approach_pose": pose.tolist(),
                }
            )

        def validate(plan):
            # Each candidate starts with the same measured scene and no assumed payload.
            data.qpos[:] = initial
            checker.set_payload(model, data, None)
            approach = np.asarray(plan.steps[0].args["xyt"])
            item = {"approach": approach.tolist(), "phase": "approach", "accepted": False}
            diagnostics.append(item)
            prepare = not np.allclose(approach, initial_base)
            if prepare and preparation_error:
                item.update(phase="navigation_posture", reason=preparation_error)
                plan.message = preparation_error
                event(**item)
                return False
            if prepare:
                data.qpos[:] = preparation_state
            route = space.plan_route(initial_base, approach)
            item["approach_route_stats"] = dict(space.route_stats)
            if not route:
                plan.message = "approach_route_invalid"
                item["contacts"] = list(checker.last_contacts)
                item["validity"] = dict(space.last_validity)
                event(**item)
                return False
            write_offline_base_pose(
                model, data, base_body_name="base_link", x=approach[0], y=approach[1], theta=approach[2]
            )
            mujoco.mj_kinematics(model, data)
            navigation_rotation = data.body(scorer["ee_body"]).xmat.reshape(3, 3).copy()
            yaw_delta = approach[2] - initial_base[2]
            c, sn = np.cos(yaw_delta), np.sin(yaw_delta)
            base_rotation = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
            rotations = [base_rotation @ initial_rotation, navigation_rotation]
            approach_state = data.qpos.copy()
            grasp_paths = []
            item["grasp_rejections"] = []
            # Transit posture need not be the grasp orientation. Try both the
            # nominal initial wrist frame and the tucked navigation frame.
            for rotation in rotations:
                data.qpos[:] = approach_state
                checker.set_payload(model, data, None)
                grasp_paths = []
                for phase, point in [
                    ("pregrasp", object_pos + [0, 0, 0.12]),
                    ("grasp", object_pos),
                    ("lift", object_pos + [0, 0, 0.12]),
                ]:
                    if phase == "lift":
                        checker.set_payload(model, data, scorer["object_body"], scorer["ee_body"])
                    path, error = executor.plan_pose(point, rotation)
                    if error:
                        item["grasp_rejections"].append(
                            {"phase": phase, "reason": error, "rotation": rotation.tolist()}
                        )
                        break
                    grasp_paths.append(path)
                if len(grasp_paths) == 3:
                    error = check_arrival_ik_samples(
                        executor, approach_state,
                        [(point, rotation) for point in
                         (object_pos + [0, 0, 0.12], object_pos, object_pos + [0, 0, 0.12])],
                        position_radius_m=0.01, yaw_radius_rad=0.015,
                    )
                    if error is None:
                        break
                    item["grasp_rejections"].append({"phase": "arrival_margin", **error})
                    grasp_paths = []
            if len(grasp_paths) != 3:
                plan.message = "no_grasp_witness_within_budget"
                item.update(phase="grasp", reason=plan.message)
                event(**item)
                return False
            releases = support_release_points(
                model,
                data,
                support_body=scorer["support_body"],
                payload_body=scorer["object_body"],
                ee_body=scorer["ee_body"],
            )
            if not releases:
                plan.message = "unsupported_support_surface"
                item.update(phase="place", reason=plan.message)
                event(**item)
                return False
            lift_state = data.qpos.copy()
            place_route = None
            place_pose = None
            place_paths = []
            place_targets = []
            payload_transform = checker.payload_transform.copy()
            place_rejections = []
            # Keep partial search evidence if the wall budget interrupts a
            # support/base candidate before the full placement loop finishes.
            item["place_rejections"] = place_rejections
            for release in releases:
                event(phase="release_candidate", approach=approach.tolist(), release=release.tolist())
                data.qpos[:] = lift_state
                place_goals = [approach.copy()]
                place_goals.extend(
                    kinematic_base_candidates(
                        model,
                        data,
                        target_xy=release[:2],
                        ee_body=scorer["ee_body"],
                        extension_joints=tuple(f"joint_arm_l{i}" for i in range(4)),
                    )
                )
                for goal in place_goals:
                    data.qpos[:] = lift_state
                    checker.payload_body, checker.payload_parent = scorer["object_body"], scorer["ee_body"]
                    checker.payload_transform = payload_transform.copy()
                    possible = space.plan_route(approach, goal)
                    if not possible:
                        place_rejections.append(
                            {"pose": goal.tolist(), "release": release.tolist(), "phase": "transport", "validity": dict(space.last_validity)}
                        )
                        continue
                    # Extra preplace height is a path candidate, not a task
                    # requirement. Try smaller clearances if the first exceeds reach.
                    for clearance in (0.12, 0.06, 0.03):
                        data.qpos[:] = lift_state
                        checker.payload_body, checker.payload_parent = scorer["object_body"], scorer["ee_body"]
                        checker.payload_transform = payload_transform.copy()
                        write_offline_base_pose(
                            model, data, base_body_name="base_link", x=goal[0], y=goal[1], theta=goal[2]
                        )
                        place_rotation = data.body(scorer["ee_body"]).xmat.reshape(3, 3).copy()
                        targets = [release + [0, 0, clearance], release, release + [0, 0, clearance]]
                        paths = []
                        for phase, point in zip(("preplace", "place", "retreat"), targets, strict=True):
                            if phase == "retreat":
                                checker.set_payload(model, data, None)
                            path, error = executor.plan_pose(point, place_rotation)
                            if error:
                                place_rejections.append(
                                    {"pose": goal.tolist(), "phase": phase, "reason": error, "clearance_m": clearance}
                                )
                                break
                            paths.append(path)
                        if len(paths) == 3:
                            place_pose, place_route, place_paths = goal, possible, paths
                            place_targets = [[point.tolist(), place_rotation.tolist()] for point in targets]
                            break
                    if place_route is not None:
                        break
                if place_route is not None:
                    break
            item["place_rejections"] = place_rejections
            if place_route is None:
                plan.message = "no_place_witness_within_budget"
                item.update(phase="place", reason=plan.message)
                event(**item)
                return False
            item.update(phase="witness", accepted=True)
            selected.update(
                approach_route=route,
                place_route=place_route,
                place_pose=place_pose.tolist(),
                grasp_paths=[[q.tolist() for q in p] for p in grasp_paths],
                place_paths=[[q.tolist() for q in p] for p in place_paths],
                grasp_targets=[
                    [point.tolist(), rotation.tolist()]
                    for point in (object_pos + [0, 0, 0.12], object_pos, object_pos + [0, 0, 0.12])
                ],
                place_targets=place_targets,
                preparation_path=[q.tolist() for q in preparation_path] if prepare and preparation_path else [],
            )
            event(**item)
            return True

        # Keep a main-thread profile of live planning: an offline replay does
        # not include observation-thread contention or the same settled state.
        import cProfile
        import pstats

        profile = cProfile.Profile()
        profile.enable()
        try:
            plan = plan_pick_place_mcts(
                oracle, candidates=candidates, executor=None, plan_validator=validate, seed=args.seed, max_candidates=48
            )
        finally:
            profile.disable()
            profile.dump_stats(str(output / "planning.prof"))
            with (output / "planning_profile.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(50)
        (output / "candidates.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        (output / "witness.json").write_text(json.dumps(selected, indent=2) + "\n")
        result.update(
            status="witness_found" if plan.success else "no_plan_within_budget",
            message=plan.message,
            static_witness=plan.success,
            checks=["scene_contacts", "pose_ik", "arm_segments", "payload_route"],
        )
        if args.tier == "physical" and plan.success:
            executor.grasp_paths = [list(map(np.asarray, path)) for path in selected["grasp_paths"]]
            executor.place_paths = [list(map(np.asarray, path)) for path in selected["place_paths"]]
            executor.grasp_targets = selected["grasp_targets"]
            executor.place_targets = selected["place_targets"]
            checker.set_payload(model, data, None)
            preparation_pending = bool(selected["preparation_path"])

            def measure():
                synchronize(data)
                if not executor.payload_retained():
                    raise RuntimeError("payload_not_retained")
                return base_pose(model, data)

            def navigate(goal):
                nonlocal preparation_pending
                if preparation_pending:
                    prepared = executor.prepare_for_navigation(list(map(np.asarray, selected["preparation_path"])))
                    if not prepared.success:
                        return prepared
                    preparation_pending = False
                outcome = execute_measured_route(
                    robot, goal=goal, measure=measure, plan_route=space.plan_route, space=space, event=event,
                    navigation_policy="manipulation", position_tolerance_m=0.01, yaw_tolerance_rad=0.015,
                )
                capture("navigation")
                return outcome

            executor.transport = lambda: navigate(selected["place_pose"])
            out = execute_task_plan(
                robot,
                plan,
                executor=executor,
                grasp_poses=plan.grasp_poses,
                manip_mode="physical",
                approach_executor=navigate,
            )
            result.update(controller_success=out.success, failed_op=out.failed_op, message=out.message)
            capture("final")
            time.sleep(3)
            result.update(score_physical_acceptance(trace, audit, execution_completed=out.success))
            result["status"] = "passed" if result["task_success"] else "failed"
        elif args.tier == "physical":
            result.update(score_physical_acceptance(trace, audit, execution_completed=False))
    except (TrialWallTimeout, Exception) as exc:
        result.update(
            status=(
                "timeout"
                if isinstance(exc, (TrialWallTimeout, TimeoutError))
                else "unsupported_capability"
                if str(exc).startswith("unsupported_")
                else "error"
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
        import traceback

        (output / "error.txt").write_text(traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if args.tier == "physical" and trace.exists() and "verified" not in result:
            try:
                result.update(score_physical_acceptance(trace, audit, execution_completed=False))
            except Exception as exc:
                result["scoring_error"] = f"{type(exc).__name__}: {exc}"
        (output / "candidates.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        (output / "witness.json").write_text(json.dumps(selected, indent=2) + "\n")
        if initial_base is not None:
            try:
                data.qpos[:] = initial
                save_motion_overview(
                    output / "motion_overview.png",
                    model,
                    data,
                    scorer=scorer,
                    footprint=spec.footprint,
                    initial_pose=initial_base,
                    candidates=diagnostics,
                    witness=selected,
                )
            except Exception as exc:
                result["visualization_error"] = str(exc)
        if robot is not None:
            robot.stop()
        if server is not None:
            from emet.utils.process_tree import terminate_process_tree

            terminate_process_tree(server, grace_s=10.0)
        if server_fh is not None:
            server_fh.close()
        result["wall_s"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        events.close()
        print(json.dumps(result), flush=True)
    return 0 if result.get("task_success") or (args.tier == "static" and result.get("static_witness")) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tier", choices=["static", "physical"], default="static")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port-offset", type=int, default=920)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--initial-state")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        import yaml

        config = yaml.safe_load(Path(args.sim).expanduser().read_text())
        scorer = json.loads(Path(args.scorer).expanduser().read_text())
        scene = Path(config["scene_path"]).expanduser().resolve(strict=True)
        print(
            json.dumps(
                {
                    "sim": str(Path(args.sim).expanduser()),
                    "scene": str(scene),
                    "task": scorer,
                    "tier": args.tier,
                    "seed": args.seed,
                    "output_dir": args.output_dir,
                    "timeout_s": args.timeout,
                },
                indent=2,
            )
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
