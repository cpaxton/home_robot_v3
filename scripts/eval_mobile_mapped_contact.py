#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""OVMM-shaped MuJoCo development episode with bounded local map lifetimes.

Native-state assistance and a provided external RGBD scanner remain explicit.
This tests physical navigation/pick/carry/place/departure, not Habitat/RoboCasa
qualification, observed-only perception, or LLM orchestration. No runtime poses
are written; base motion uses normal wheel velocity actuators.
"""

import argparse
import faulthandler
import json
import os
import resource
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from eval_mapped_native_contact import MappedContactRobot, MappedGuard, provided_scan
from tamp_mobile_cases import load_case
from tamp_release_diagnostics import install_release_diagnostics

from emet.agent.tools import get_tools
from emet.controller.manipulation import physical_pick_place
from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
from emet.controller.task.tamp import native_adapter, native_approach, native_placement, native_policy
from emet.eval.physical_tamp import base_pose, make_scene_checker, write_offline_base_pose
from emet.motion import release
from emet.motion.local_scene_store import LocalMapUnavailable, LocalScene, LocalSceneStore
from emet.motion.mujoco_arm_ik import joint_qpos_addrs
from emet.motion.mujoco_collision import body_subtree
from emet.motion.observed_solids import WorldSolid
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.utils.cpu_affinity import apply_affinity, safe_cpu_ids


class MobileRobot(MappedContactRobot):
    def __init__(self, model, obj):
        self.nav_goal, self.gate, self.motion_gate, self.velocity = None, None, None, 0.0
        super().__init__(model, obj)
        # Declared fixture initialization only, before discovery or any command.
        root = int(model.body("base_link").jntadr[0])
        self.data.qpos[int(model.jnt_qposadr[root])] -= 0.10
        self.step(1.0)
        self.refresh()

    def get_emet_session(self):
        return {**super().get_emet_session(), "capabilities": {"teleport_base": False}}

    def move_base_to(self, goal, **kwargs):
        if self.motion_gate is not None:
            self.motion_gate()
        self.commands.append({"base_goal": list(goal), "sim_time": float(self.data.time)})
        self.nav_goal = np.asarray(goal, float)
        return True

    def cancel_navigation(self):
        self.nav_goal, self.velocity = None, 0.0
        for name in ("left_wheel_vel", "right_wheel_vel"):
            self.data.ctrl[self.model.actuator(name).id] = 0.0
        self.commands.append({"base_stop": True, "sim_time": float(self.data.time)})
        return True

    def _servo(self):
        current = base_pose(self.model, self.data)
        goal = self.nav_goal
        if goal is None:
            return
        if self.motion_gate is not None:
            try:
                self.motion_gate()
            except LocalMapUnavailable:
                self.cancel_navigation()
                raise
        error = goal[:2] - current[:2]
        forward = np.array([np.cos(goal[2]), np.sin(goal[2])])
        along = float(error @ forward)
        target = float(np.clip(1.5 * along, -0.035, 0.035))
        if np.linalg.norm(error) < 0.0015:
            target = 0.0
        self.velocity += np.clip(target - self.velocity, -0.0005, 0.0005)
        turn = np.arctan2(np.sin(goal[2] - current[2]), np.cos(goal[2] - current[2]))
        cross = float(forward[0] * error[1] - forward[1] * error[0])
        omega = float(np.clip(2 * turn + 2 * cross, -0.1, 0.1))
        for name, speed in zip(
            ("left_wheel_vel", "right_wheel_vel"),
            ((self.velocity - 0.17035 * omega) / 0.05, (self.velocity + 0.17035 * omega) / 0.05),
            strict=True,
        ):
            a = self.model.actuator(name).id
            j = int(self.model.actuator_trnid[a, 0])
            gear = float(self.model.actuator_gear[a, 0])
            kv = float(self.model.actuator_gainprm[a, 0])
            friction = float(self.model.dof_frictionloss[self.model.jnt_dofadr[j]])
            # Model-specific position/velocity controller feed-forward, not a
            # force sensor or a change to robot/contact dynamics.
            command = gear * speed + np.sign(speed) * friction / (gear * kv)
            self.data.ctrl[a] = np.clip(command, *self.model.actuator_ctrlrange[a])

    def step(self, duration):
        if self.nav_goal is None:
            return super().step(duration)
        for _ in range(max(1, round(duration / 0.01))):
            self._servo()
            super().step(0.01)

    def arm_to(self, *args, **kwargs):
        self.gate()
        return super().arm_to(*args, **kwargs)

    def open_gripper(self, **kwargs):
        self.gate()
        return super().open_gripper(**kwargs)

    def close_gripper(self, **kwargs):
        self.gate()
        return super().close_gripper(**kwargs)

    def gripper_to(self, *args, **kwargs):
        self.gate()
        return super().gripper_to(*args, **kwargs)


class LeaseGuard(MappedGuard):
    """No retained background arrays: release clones retain an opaque lease."""

    def __init__(self, model, lease, *, target_in_map=False):
        self.model, self.lease = model, lease
        moving = body_subtree(model, "link_lift")
        self.geoms = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in moving and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        self.map_queries = self.native_only_queries = self.rejections = 0
        self.last_map_reason, self.rejection_witnesses = None, []
        self.free_witness_enabled, self.target_in_map = True, target_in_map

    def configuration_collides(self, checker, model, data):
        with self.lease.borrow() as scene:
            # Transient query object only; no closure or plan keeps scene alive.
            query = MappedGuard(
                checker,
                model,
                scene.admission.background.workspace,
                scene.admission,
                scene.support,
                scene.support.capture,
            )
            if self.target_in_map:
                query.geoms = self.geoms
                result = False
                for gid in query.geoms:
                    solid = WorldSolid.from_geom(
                        model, data, gid, convex_material_calibration_id="native-debug-convex-collision-model"
                    )
                    if query._query(solid):
                        checker.last_reason = "robot_scene_collision"
                        result = True
                        break
            else:
                result = query.configuration_collides(checker, model, data)
            self.map_queries += query.map_queries
            self.native_only_queries += query.native_only_queries
            self.rejections += query.rejections
            self.last_map_reason = query.last_map_reason
            self.rejection_witnesses.extend(query.rejection_witnesses[: max(0, 30 - len(self.rejection_witnesses))])
            return result


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cpus = sorted(set(os.sched_getaffinity(0)) & set(safe_cpu_ids()))
    if not cpus:
        raise RuntimeError("no_safe_allocated_cpu")
    apply_affinity(cpus)
    case = load_case(args.case_config)
    mobile = case.get("mobile", True)
    region = np.asarray(case.get("placement_region", [[0.06, -0.73], [0.22, -0.57]]), float)
    result = {
        "scope": __doc__,
        "passed": False,
        "phases": [],
        "responses": [],
        "map_builds": [],
        "assistance": ["native_state", "provided_external_rgbd_scanner", "native_navigation_geometry"],
        "object": args.object,
        "cpu_affinity": cpus,
        "case_config": case,
    }

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    @contextmanager
    def phase(name):
        start = time.perf_counter()
        row = {"phase": name}
        result["phases"].append(row)
        save()
        try:
            yield row
        finally:
            row.update(
                wall_seconds=time.perf_counter() - start,
                cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            )
            save()

    model = mujoco.MjModel.from_xml_path(str(args.fixture.resolve() / "scene.xml"))
    robot = MobileRobot(model, args.object)
    # Declared initial conditions only, before discovery or any command.
    if "initial_base_x" in case:
        j = int(model.body("base_link").jntadr[0])
        robot.data.qpos[int(model.jnt_qposadr[j])] = float(case["initial_base_x"])
    if "initial_object_xy" in case:
        xy = np.asarray(case["initial_object_xy"], float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("invalid_mobile_case_config")
        addr = int(model.jnt_qposadr[model.body(args.object).jntadr[0]])
        robot.data.qpos[addr : addr + 2] = xy
    if "initial_distractor_xy" in case:
        other = "object1" if args.object == "object2" else "object2"
        a = int(model.jnt_qposadr[model.body(other).jntadr[0]])
        robot.data.qpos[a : a + 2] = case["initial_distractor_xy"]
    if {"initial_base_x", "initial_object_xy", "initial_distractor_xy"} & set(case):
        robot.step(1.0)
    robot.refresh()
    adapter = native_adapter.NativeTableAdapter(robot, model=model, policy="bounded_release")
    raw_data = adapter._data
    adapter._data = lambda **kw: {
        **raw_data(**kw),
        "geometry_source": "rgbd_map_plus_native_scene",
        "base_motion": "checked_unloaded_and_loaded_straight" if mobile else "stationary_control",
        "collision_scope": "local_contained_solids_and_native_full_robot_payload_sweeps",
        "assistance": result["assistance"],
    }
    raw_sync = adapter.synchronize

    def sync(d):
        robot.refresh()
        raw_sync(d)

    adapter.synchronize = sync
    events = []

    def event(**row):
        events.append(row)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    adapter.events = events
    store = LocalSceneStore()
    lease, guards = None, []

    def gate():
        if lease is None:
            raise LocalMapUnavailable("retired")
        lease.require_current()

    robot.gate = gate

    def acquire(name, *, include_target=False):
        nonlocal lease
        old = lease
        store.invalidate()
        adapter.plans.clear()
        directory = out / name
        directory.mkdir(exist_ok=True)
        stats = {"phase": name}
        with phase("map_" + name):

            def build():
                _, admission, support, _ = provided_scan(
                    args.fixture,
                    robot,
                    args.object,
                    directory,
                    resolution=args.resolution,
                    include_target=include_target,
                    build_stats=stats,
                )
                return LocalScene(admission, support)

            lease = store.load(name, build, reserve_bytes=48 * 1024**2)
        result["map_builds"].append(stats)
        if old is not None:
            try:
                old.require_current()
            except LocalMapUnavailable:
                event(phase="old_map_lease_invalidated", previous=old.key)
            else:
                raise AssertionError("retired map lease remained valid")
        robot.refresh()
        guard = LeaseGuard(model, lease, target_in_map=include_target)
        guards.append(guard)
        return guard

    names = STRETCH_ARM_CHAIN.joint_names[:-1]
    nav = SimpleNamespace(
        model=model,
        data=adapter.data,
        synchronize=sync,
        qadr=joint_qpos_addrs(model, names),
        joint_names=names,
        joint_tolerance=0.03,
        coupled_groups=(tuple(f"joint_arm_l{i}" for i in range(4)),),
        event=event,
    )
    raw_executor = adapter._executor
    current_guard = None

    def executor(obj):
        ex = raw_executor(obj)
        result["endpoint_settle_s"] = ex.endpoint_settle_s
        ex.collision.additional_scene_checks = (current_guard.configuration_collides,)
        robot.gate = current_guard.lease.require_current
        ex.event = event
        install_release_diagnostics(ex, robot, out / "release-states")
        return ex

    adapter._executor = executor

    def prepare_transport(ex, obj, goal):
        start = base_pose(model, adapter.data)
        end = start.copy()
        if not mobile:
            return end.tolist(), None
        end[0] += float(goal[0] - adapter.data.body(obj).xpos[0])
        if not native_approach.sweep_clear(model, adapter.data, start, end, loaded_checker=ex.collision, event=event):
            return None, "transport_collision"
        write_offline_base_pose(model, adapter.data, base_body_name="base_link", x=end[0], y=end[1], theta=end[2])
        if ex.collision.configuration_collides(model, adapter.data):
            return None, "transport_collision"
        return end.tolist(), None

    adapter._prepare_transport = prepare_transport

    def transport(plan):
        nonlocal current_guard
        if not mobile:
            current_guard = acquire("destination")
            plan.executor.collision.additional_scene_checks = (current_guard.configuration_collides,)
            robot.gate = current_guard.lease.require_current
            return PhysicalMotionResult(True, "stationary_control", "transport")
        # A stopped, loaded handoff is a new acquisition boundary. Long grasp
        # search must not consume the carry scan's original evidence lifetime.
        current_guard = acquire("carry")
        plan.executor.collision.additional_scene_checks = (current_guard.configuration_collides,)
        robot.gate = robot.motion_gate = current_guard.lease.require_current
        with phase("loaded_carry") as row:
            sync(adapter.data)
            start = base_pose(model, adapter.data)
            outcome = native_approach.execute_approach(
                robot, plan.executor, native_approach.BaseApproach(tuple(start), tuple(plan.transport)), loaded=True
            )
            row.update(
                success=outcome.success, reason=outcome.reason, measured_base=base_pose(model, adapter.data).tolist()
            )
        if not outcome.success:
            return PhysicalMotionResult(False, outcome.reason, "transport")
        current_guard = acquire("destination")
        plan.executor.collision.additional_scene_checks = (current_guard.configuration_collides,)
        robot.gate = robot.motion_gate = current_guard.lease.require_current
        return PhysicalMotionResult(True, "ok", "transport")

    adapter._transport = transport
    tools = {
        t.name: t
        for t in get_tools(
            {"robot": robot, "manip_mode": "physical", "tamp_inputs": "privileged", "_native_tamp_adapter": adapter}
        )
    }

    def call(name, **kw):
        row = json.loads(tools[name].func(**kw))
        result["responses"].append(row)
        save()
        return row

    clock = SimpleNamespace(monotonic=lambda: float(robot.data.time), sleep=robot.step)
    modules = [physical_pick_place, native_adapter, native_policy, native_placement, native_approach, release]
    originals = [m.time for m in modules]
    try:
        for module in modules:
            module.time = clock
        discovery = call("scene_tasks", object_filter="red" if args.object == "object2" else "blue")
        task = next(t for t in discovery["data"]["tasks"] if t["receptacle_name"] == "table")
        with phase("navigate_to_source") as row:
            sync(adapter.data)
            start = base_pose(model, adapter.data)
            end = start + [float(case.get("approach_m", 0.10 if mobile else 0.0)), 0, 0]
            outcome = (
                native_approach.execute_approach(robot, nav, native_approach.BaseApproach(tuple(start), tuple(end)))
                if np.linalg.norm(end - start)
                else native_approach.ApproachResult(True, "stationary_control")
            )
            row.update(success=outcome.success, reason=outcome.reason)
            if not outcome.success:
                raise RuntimeError(outcome.reason)
        current_guard = acquire("source")
        with phase("plan_transfer"):
            planned = call("plan_pick_place", task_ref=task["task_ref"], placement_region=region.tolist())
            if planned["status"] != "ok":
                raise RuntimeError(planned["code"])
        with phase("pick_carry_place"):
            executed = call("execute_pick_place_plan", plan_ref=planned["data"]["plan_ref"])
            if executed["status"] != "ok":
                raise RuntimeError(executed["code"])
        released = robot.data.body(args.object).xpos.copy()
        current_guard = acquire("released", include_target=True)
        robot.gate = robot.motion_gate = current_guard.lease.require_current
        with lease.borrow() as scene:
            grid = scene.admission.background.workspace
            result["released_object_occupied_cells"] = int(
                np.sum(
                    grid.cells[grid.query_region([released - [0.02, 0.02, 0.04], released + [0.02, 0.02, 0.04]])] == 2
                )
            )
        checker = make_scene_checker(model, {"object_body": "base_link", "gripper_bodies": ()})
        checker.additional_scene_checks = (current_guard.configuration_collides,)
        with phase("navigate_away") as row:
            sync(adapter.data)
            start = base_pose(model, adapter.data)
            end = start + [float(case.get("departure_m", 0.10 if mobile else 0.0)), 0, 0]
            outcome = (
                native_approach.execute_approach(
                    robot, nav, native_approach.BaseApproach(tuple(start), tuple(end)), scene_checker=checker
                )
                if np.linalg.norm(end - start)
                else native_approach.ApproachResult(True, "stationary_control")
            )
            row.update(success=outcome.success, reason=outcome.reason)
            if not outcome.success:
                raise RuntimeError(outcome.reason)
        drift = float(np.linalg.norm(robot.data.body(args.object).xpos - released))
        result.update(
            released_object_drift_m=drift, passed=drift < 0.01 and result["released_object_occupied_cells"] > 0
        )
    except (RuntimeError, ValueError, TimeoutError) as error:
        result.update(error=type(error).__name__ + ":" + str(error))
    finally:
        robot.cancel_navigation()
        for module, original in zip(modules, originals, strict=True):
            module.time = original
        store.invalidate()
        result.update(
            map_store=store.stats,
            commands=robot.commands,
            map_queries=sum(g.map_queries for g in guards),
            native_only_queries=sum(g.native_only_queries for g in guards),
            cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        (out / "trace.json").write_text(json.dumps(robot.trace) + "\n")
        save()
    print(
        json.dumps({k: result.get(k) for k in ("passed", "error", "map_store", "cumulative_peak_rss_bytes")}),
        flush=True,
    )
    return 0 if result["passed"] else 1


def main():
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case-config", type=Path, help="JSON fixture/goal/transport overrides; initial conditions only."
    )
    parser.add_argument("--object", choices=["object1", "object2"], default="object2")
    parser.add_argument("--resolution", type=float, choices=[0.02, 0.005], default=0.005)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
