#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""RGBD-map-assisted native contact control, NOT observed-input acceptance.

The static map is a provided simulated overhead depth scan before the robot and
target enter the workspace. Native robot/object state remains privileged. The
ordinary public tools and actuators execute the transfer. Unknown cells inside
the declared map reject; geometry outside it retains the native scene checker.
This isolates map/path/execution integration from onboard map acquisition.
"""

import argparse
import faulthandler
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from eval_native_table_contact import ContactRobot
from scipy.ndimage import binary_erosion

from emet.agent.tools import get_tools
from emet.controller.manipulation import physical_pick_place
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.task.tamp import native_adapter, native_placement, native_policy
from emet.motion import release
from emet.motion.mujoco_collision import body_subtree
from emet.motion.observed_admission import ObservedEnvironmentAdmission
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_solids import WorldSolid
from emet.motion.observed_support import fit_observed_support
from emet.motion.observed_support_contact import ObservedSupportContact
from emet.motion.observed_workspace import ObservedWorkspace
from emet.simulation.depth_registration import rendered_camera_intrinsics
from emet.simulation.synchronized_capture import camera_opencv_pose
from emet.utils.cpu_affinity import apply_affinity, safe_cpu_ids


def provided_scan(fixture, robot, target, out, *, resolution=0.005, include_target=False, build_stats=None):
    """Actual pinhole depth returns from an explicitly assisted static prescan."""
    xml = ET.parse(fixture / "scene_environment.xml")
    world = xml.getroot().find("worldbody")
    for name in ("object1", "object2"):
        body = world.find(f"body[@name='{name}']")
        if name == target and not include_target:
            world.remove(body)
        else:
            body.set("pos", " ".join(map(str, robot.data.body(name).xpos)))
            body.set("quat", " ".join(map(str, robot.data.body(name).xquat)))
            body.remove(body.find("freejoint"))
    ET.SubElement(world, "camera", name="provided_scanner", pos="0 -.65 1.8", fovy="65")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    model.vis.global_.offheight = model.vis.global_.offwidth = 512
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = model.camera("provided_scanner").id
    started = time.monotonic()
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    k = rendered_camera_intrinsics(65, 512, 512)
    pose = camera_opencv_pose(model, data, camera, upright=False)
    now = time.monotonic()
    ticket = CaptureInterval(f"provided-static-scan:{time.monotonic_ns()}", started, now, now, "provided-simulation-scanner")
    observation = SimpleNamespace(rgb=rgb, depth=depth, camera_K=k, camera_pose=pose, received_monotonic=now)
    # Operator selects wood by appearance; the support plane is fitted from depth.
    c = rgb.astype(float)
    mask = binary_erosion((c[..., 0] > 1.15 * c[..., 1]) & (c[..., 1] > 1.25 * c[..., 2]), iterations=1)
    y, x = np.indices(depth.shape)
    points = np.stack(((x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth), axis=-1)
    points = points @ pose[:3, :3].T + pose[:3, 3]
    # Operator-selected local patch, retained as a finite footprint. Selecting
    # a patch does not crop occupancy or infer hidden tabletop continuity.
    support_mask = mask & (np.abs(points[..., 0]) < 0.24) & (points[..., 1] > -0.86) & (points[..., 1] < -0.501)
    np.savez_compressed(out / "raw-scan.npz", rgb=rgb, depth=depth, intrinsic=k, world_from_camera=pose, mask=mask)
    support = fit_observed_support(
        observation,
        support_mask,
        observation_id=ticket.observation_id,
        measurement_margin_m=0.0001,
        resolution=0.002,
        max_cells=100_000,
    )
    workspace = [[-0.4, -0.9, 0.44], [0.4, -0.3, 1.04]]
    grid = ObservedWorkspace.from_depth(
        depth, k, pose, workspace=workspace, resolution=resolution, depth_margin_m=0.0001, max_cells=3_000_000,
        build_stats=build_stats
    )
    labels = np.where(mask, "table", "background")
    background = ObservedBackground.from_capture(
        grid.partition(points.reshape(-1, 3), labels.ravel()),
        excluded_refs=(),
        scene_epoch="provided-static",
        capture=ticket,
        now=now,
    )
    assumption = StaticBackgroundAssumption(
        "provided-static", started - 0.001, started + 299.0, "isolated-static-debug-fixture"
    )
    contact = ObservedSupportContact(
        target,
        "table",
        "provided-static",
        "ideal-scanner-debug",
        support,
        ticket,
        depth,
        k,
        pose,
        support_mask,
        payload_motion_error_m=0.0,
        backing_depth_m=0.025,
        backing_prior_id="declared-solid-table-top-layer",
    )
    admission = ObservedEnvironmentAdmission(background, static_assumption=assumption)
    np.savez_compressed(
        out / "provided-scan.npz",
        rgb=rgb,
        depth=depth,
        intrinsic=k,
        world_from_camera=pose,
        cells=grid.cells,
        origin=grid.origin,
        resolution=grid.resolution,
        support_bounds=support.bounds,
    )
    return grid, admission, contact, ticket


class MappedGuard:
    """Additional strict map gate; native geometry remains an explicit control."""

    def __init__(self, native, model, grid, admission, support, ticket):
        self.native, self.model, self.grid = native, model, grid
        self.admission, self.support, self.ticket = admission, support, ticket
        moving = body_subtree(model, "link_lift")
        self.geoms = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in moving and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        self.map_queries = self.native_only_queries = self.rejections = 0
        self.last_map_reason = None
        self.rejection_witnesses = []
        self.free_witness_enabled = True

    def _query(self, solid, *, payload=False):
        upper = self.grid.origin + np.array(self.grid.cells.shape) * self.grid.resolution
        if np.any(solid.bounds[0] < self.grid.origin) or np.any(solid.bounds[1] > upper):
            self.native_only_queries += 1
            return False  # Already checked by the explicitly privileged native guard.
        self.map_queries += 1
        now = time.monotonic()
        if payload and self.free_witness_enabled:
            result = self.admission.query_payload(
                solid,
                solid,
                contact=self.support,
                object_ref=self.support.object_ref,
                phase="place",
                now=now,
                allow_captured_robot=False,
            )
        else:
            result = self.admission.query(
                solid,
                now=now,
                allow_captured_robot=False,
                free_witness=self.support if self.free_witness_enabled else None,
            )
        self.last_map_reason = result.reason
        if not result.allowed:
            self.rejections += 1
            if len(self.rejection_witnesses) < 30:
                witness = {"payload": payload, "reason": result.reason, "bounds": solid.bounds.tolist()}
                self.rejection_witnesses.append(witness)
                print(json.dumps({"map_rejection": witness}), flush=True)
        return not result.allowed

    def configuration_collides(self, checker, model, data):
        # The current native/release clone already checked contacts and
        # transformed its payload. Never consult the original checker's pose.
        for gid in self.geoms:
            solid = WorldSolid.from_geom(
                model, data, gid, convex_material_calibration_id="native-debug-convex-collision-model"
            )
            if self._query(solid):
                checker.last_reason = "robot_scene_collision"
                return True
        ids = body_subtree(model, self.support.object_ref)
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) in ids and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                if self._query(WorldSolid.from_geom(model, data, gid), payload=True):
                    checker.last_reason = "payload_scene_collision"
                    return True
        return False


class MappedContactRobot(ContactRobot):
    def __init__(self, model, object_body):
        self.trace, self.trace_object = [], object_body
        super().__init__(model, object_body)

    def step(self, duration):
        super().step(duration)
        joint = int(self.model.body(self.trace_object).jntadr[0])
        addr = int(self.model.jnt_qposadr[joint])
        self.trace.append(
            {
                "sim_time": float(self.data.time),
                "object_position": self.data.qpos[addr : addr + 3].tolist(),
                "ee_position": self.data.body("link_grasp_center").xpos.tolist(),
            }
        )

    def gripper_to(self, target, *, blocking):
        assert not blocking
        self.commands.append({"sim_time": float(self.data.time), "gripper_target": float(target)})
        self.data.ctrl[self.model.actuator("gripper").id] = (target + 0.376) / 0.936 * 0.06 - 0.02
        return True


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Intersect, never expand, the job's allocated CPU subset. A taskset after
    # the managed wrapper must not re-enable this host's unstable turbo cores.
    cpus = sorted(set(os.sched_getaffinity(0)) & set(safe_cpu_ids()))
    if not cpus:
        raise RuntimeError("no_safe_allocated_cpu")
    apply_affinity(cpus)
    repo = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__),
        repo / "scripts/eval_native_table_contact.py",
        repo / "src/emet/motion/mujoco_collision.py",
        repo / "src/emet/motion/observed_admission.py",
        repo / "src/emet/motion/observed_support_contact.py",
        repo / "src/emet/motion/release.py",
        repo / "src/emet/controller/task/tamp/native_adapter.py",
        repo / "src/emet/controller/task/tamp/native_policy.py",
        repo / "src/emet/controller/manipulation/physical_pick_place.py",
    ]
    implementation = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    model = mujoco.MjModel.from_xml_path(str((args.fixture / "scene.xml").resolve()))
    robot = MappedContactRobot(model, args.object)
    grid, admission, support, ticket = provided_scan(
        args.fixture, robot, args.object, out, resolution=args.map_resolution
    )
    robot.refresh()  # Acquire actual current native state after expensive map construction.
    adapter = native_adapter.NativeTableAdapter(robot, model=model, policy="bounded_release")

    class LiveEvents(list):
        def append(self, row):
            super().append(row)
            with (out / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    adapter.events = LiveEvents()
    raw_sync, raw_executor = adapter.synchronize, adapter._executor
    guards = []
    fault_receipts = []

    def inject_fault(guard, kind):
        pos = robot.data.body(args.object).xpos.copy()
        index = tuple(np.floor((pos - guard.grid.origin) / guard.grid.resolution).astype(int))
        cells = guard.grid.cells.copy()
        old = int(cells[index])
        cells[index] = 0 if kind == "unknown" else 2
        changed = ObservedWorkspace(guard.grid.origin, guard.grid.resolution, cells)
        stamps = guard.admission.background.observed_monotonic.copy()
        stamps[index] = np.nan if kind == "unknown" else guard.ticket.earliest
        background = replace(guard.admission.background, workspace=changed, observed_monotonic=stamps)
        guard.grid = changed
        guard.admission = ObservedEnvironmentAdmission(
            background,
            static_assumption=guard.admission.static_assumption,
        )
        if kind == "unknown":
            guard.free_witness_enabled = False
        receipt = {
            "kind": kind,
            "index": tuple(map(int, index)),
            "old_cell": old,
            "new_cell": int(cells[index]),
            "commands_at_fault": len(robot.commands),
            "sim_time": float(robot.data.time),
            "scope": "injected evidence fault; not a sensor measurement or task-success case",
        }
        fault_receipts.append(receipt)
        np.savez_compressed(out / "map-fault.npz", cells=cells, origin=changed.origin, resolution=changed.resolution)
        print(json.dumps({"fault": receipt}), flush=True)

    def sync(data):
        robot.refresh()
        raw_sync(data)

    adapter.synchronize = sync

    def executor(obj):
        ex = raw_executor(obj)
        if args.collision_mode == "native":
            return ex
        guard = MappedGuard(ex.collision, model, grid, admission, support, ticket)
        ex.collision.additional_scene_checks = (guard.configuration_collides,)
        guards.append(guard)
        if args.fault == "obstacle_before_plan":
            inject_fault(guard, "obstacle")
        return ex

    adapter._executor = executor
    raw_binding = adapter._bind_loaded_payload

    def bind_loaded(plan, **kwargs):
        ok = raw_binding(plan, **kwargs)
        if ok and args.fault == "unknown_after_grasp":
            inject_fault(guards[-1], "unknown")
        return ok

    adapter._bind_loaded_payload = bind_loaded
    raw_data = adapter._data
    adapter._data = lambda **kw: {
        **raw_data(**kw),
        "geometry_source": "rgbd_map_plus_native_scene" if args.collision_mode == "map" else "simulator_ground_truth",
        "assistance": (
            ["provided_environment_prescan", "native_robot_and_object_state", "native_unmapped_geometry"]
            if args.collision_mode == "map"
            else ["native_robot_and_object_state"]
        ),
        "collision_scope": "whole_solids_contained_in_map_plus_native_scene"
        if args.collision_mode == "map"
        else "sampled_arm_and_payload",
    }
    tools = {
        t.name: t
        for t in get_tools(
            {"robot": robot, "manip_mode": "physical", "tamp_inputs": "privileged", "_native_tamp_adapter": adapter}
        )
    }
    responses = []
    physics_clock = SimpleNamespace(monotonic=lambda: float(robot.data.time), sleep=robot.step)
    clock_modules = [physical_pick_place, native_adapter, native_policy, native_placement, release]
    old_clocks = [module.time for module in clock_modules]
    try:
        for module in clock_modules:
            module.time = physics_clock
        for name, kwargs in [
            (
                "plan_pick_place",
                {
                    "object_name": "red cylinder" if args.object == "object2" else "blue cube",
                    "receptacle_name": "table",
                },
            )
        ]:
            # Discovery binds opaque handles exactly as ordinary agent tooling.
            discovery = json.loads(
                tools["scene_tasks"].func(object_filter="red" if args.object == "object2" else "blue")
            )
            responses.append(discovery)
            row = next(r for r in discovery["data"]["tasks"] if r["receptacle_name"] == "table")
            kwargs = {"task_ref": row["task_ref"], "placement_region": args.placement_region}
            planned = json.loads(tools[name].func(**kwargs))
            responses.append(planned)
            if planned["status"] == "ok":
                responses.append(
                    json.loads(tools["execute_pick_place_plan"].func(plan_ref=planned["data"]["plan_ref"]))
                )
    finally:
        for module, old in zip(clock_modules, old_clocks, strict=True):
            module.time = old
    result = {
        "schema_version": 1,
        "scope": __doc__,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "object": args.object,
        "collision_mode": args.collision_mode,
        "map_resolution_m": args.map_resolution,
        "placement_region": args.placement_region,
        "model_signature": model_signature(model),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mujoco_version": mujoco.__version__,
        "execution_clock": "in_process_step_on_control_sleep; not a wall-time performance measurement",
        "implementation_sha256": implementation,
        "fault": args.fault,
        "fault_receipts": fault_receipts,
        "responses": responses,
        "events": adapter.events,
        "commands": robot.commands,
        "map_cells": {
            name: int(np.sum(grid.cells == code)) for name, code in [("unknown", 0), ("free", 1), ("occupied", 2)]
        },
        "map_queries": sum(g.map_queries for g in guards),
        "native_only_queries": sum(g.native_only_queries for g in guards),
        "map_rejections": sum(g.rejections for g in guards),
        "last_map_reasons": [g.last_map_reason for g in guards],
        "rejection_witnesses": [w for g in guards for w in g.rejection_witnesses],
        "final_object_position": robot.data.body(args.object).xpos.tolist(),
        "passed": responses[-1]["tool"] == "execute_pick_place_plan" and responses[-1]["status"] == "ok",
    }
    last = responses[-1]
    if args.fault == "none":
        expected = result["passed"]
    elif args.fault == "unknown_after_grasp":
        expected = (
            last["status"] == "partial"
            and last["data"].get("completed_ops") == ["grasp"]
            and last["data"].get("execution_started") is True
            and last["recovery"] == "inspect"
            and bool(fault_receipts)
            and len(robot.commands) == fault_receipts[-1]["commands_at_fault"]
        )
    else:
        expected = last["status"] == "error" and last["data"].get("execution_started") is False and not robot.commands
    result["expectation_passed"] = bool(expected)
    (out / "trace.json").write_text(json.dumps(robot.trace) + "\n")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: result[k] for k in ["passed", "map_queries", "map_rejections", "last_map_reasons"]}), flush=True
    )
    return 0 if result["expectation_passed"] else 1


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", choices=["object1", "object2"], default="object2")
    parser.add_argument("--collision-mode", choices=["map", "native"], default="map")
    parser.add_argument("--map-resolution", type=float, choices=[0.02, 0.01, 0.005], default=0.005)
    parser.add_argument(
        "--placement-region",
        type=json.loads,
        default=[[-0.18, -0.68], [0.18, -0.58]],
        help="JSON world-XY goal bounds. The region must fit the full payload and clearance.",
    )
    parser.add_argument("--fault", choices=["none", "unknown_after_grasp", "obstacle_before_plan"], default="none")
    args = parser.parse_args()
    if args.fault != "none" and args.collision_mode != "map":
        parser.error("fault injection requires map mode")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
