# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Shared private placement planning, including measured loaded refinement."""
import time
from contextlib import contextmanager

import mujoco
import numpy as np

from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
from emet.controller.task.tamp.native_revalidation import joint_proof, payload_copy
from emet.motion.mujoco_collision import body_subtree
from emet.motion.placement_geometry import HeldObject, PlacementScene
from emet.motion.placement_surfaces import free_surface_centers
from emet.motion.release import refine_release
from emet.simulation.sim_object_placements import placements_from_mujoco_model


def local_support_patches(surfaces, point):
    """Retain full support patches and the same local reach patches in both stages."""
    patches = list(surfaces)
    for radius in (.08, .16, .32):
        for surface in surfaces:
            patch = surface.copy()
            patch[0, :2] = np.maximum(patch[0, :2], point[:2]-radius)
            patch[1, :2] = np.minimum(patch[1, :2], point[:2]+radius)
            if np.all(patch[0, :2] < patch[1, :2]):
                patches.append(patch)
    return patches


def placement_paths(adapter, executor, targets):
    """Use one planning path for initial candidates and loaded refinements."""
    paths, proof = [], []
    record = getattr(adapter, "reacquire_scene", None) is not None
    executor._place_proof = ()
    for i, (target, rotation) in enumerate(targets):
        phase = ('preplace', 'place', 'retreat')[i]
        if i == 2:
            executor.collision.set_payload(adapter.model, adapter.data, None)
            opening_start = adapter.data.qpos.copy()
            opening_payload = payload_copy(executor.collision) if record else None
            opening, error = executor.plan_gripper_open()
            if error:
                adapter._record_rejection('release_open', error, executor)
                return None, error
            if record:
                proof.append(joint_proof(executor, 'place_open', opening_start, opening_payload, opening,
                    names=tuple(executor.gripper_open_configuration)))
        phase_start = adapter.data.qpos.copy()
        phase_payload = payload_copy(executor.collision) if record else None
        path, error = adapter._plan_candidate_pose(executor, target, rotation)
        if error:
            adapter._record_rejection(phase, error, executor)
            return None, error
        paths.append(path)
        if record:
            proof.append(joint_proof(executor, phase, phase_start, phase_payload, path, endpoint=(target, rotation)))
    executor._place_proof = tuple(proof)
    return paths, None


def refine_loaded_placement(adapter, plan):
    """Refresh free support centers after pickup, retaining the verified grasp.

    No command is sent and no reference is renewed. Original destination-region
    constraints remain mandatory. The initial primitive's unused IK/time budget
    limits this refinement; controller execution time is not planning time.
    """
    ex, model, data = plan.executor, adapter.model, adapter.data
    adapter.synchronize(data)
    if not ex.payload_retained():
        return PhysicalMotionResult(False, 'payload_not_retained', 'place_refinement')
    before, payload = data.qpos.copy(), ex.collision.capture_payload()
    old_limit, old_deadline = adapter.max_planning_ik_calls, adapter._planning_deadline
    adapter.planning_ik_calls = plan.planning_ik_calls
    adapter.max_planning_ik_calls = plan.planning_ik_limit
    planning_started = time.monotonic()
    adapter._planning_deadline = (planning_started+plan.planning_seconds_remaining
                                  if plan.planning_seconds_remaining is not None else None)
    try:
        entries = placements_from_mujoco_model(model, data)
        surfaces = np.asarray(entries[plan.support_body]['support_surfaces']).copy()
        if plan.placement_region is not None:
            surfaces[:, 0, :2] = np.maximum(surfaces[:, 0, :2], plan.placement_region[0])
            surfaces[:, 1, :2] = np.minimum(surfaces[:, 1, :2], plan.placement_region[1])
        robot_ids = body_subtree(model, 'base_link')
        scene = PlacementScene.from_placements({name: row for name, row in entries.items()
            if model.body(name).id not in robot_ids}, held_object=plan.object_body)
        ee = data.body(ex.ee_body)
        held = HeldObject.from_world_bounds(entries[plan.object_body]['bounds'],
            ee_position=ee.xpos, ee_rotation=ee.xmat.reshape(3, 3))
        patches = local_support_patches(surfaces, np.asarray(entries[plan.object_body]['pos']))
        centers = free_surface_centers(patches, scene=scene, payload=held,
            ee_rotation=plan.rotation, max_centers=16).centers
        centers.sort(key=lambda c: float(np.linalg.norm(c[:2]-plan.goal[:2])))
        # Preserve the original proposal if its newly measured full footprint
        # still fits the requested support/region. All updated arm/payload paths
        # must certify it again; nominal planning alone never authorizes motion.
        half = np.ptp(held.vertices_ee @ plan.rotation.T, axis=0)/2
        if any(np.all(plan.goal[:2]-half[:2] >= patch[0, :2])
               and np.all(plan.goal[:2]+half[:2] <= patch[1, :2]) for patch in surfaces):
            centers = [plan.goal.copy(), *[c for c in centers if not np.allclose(c, plan.goal, atol=1e-9)]]
        for goal in centers[:16]:
            data.qpos[:] = before
            ex.collision.restore_payload(payload)
            mujoco.mj_kinematics(model, data)
            targets = adapter._place_targets(ex, plan.object_body, goal, plan.rotation)
            paths, error = placement_paths(adapter, ex, targets)
            adapter.events.append({'phase': 'loaded_place_candidate', 'accepted': error is None,
                                   'goal': np.asarray(goal).tolist(), 'reason': error})
            if error is None:
                if adapter._planning_deadline is not None and time.monotonic() >= adapter._planning_deadline:
                    raise ValueError('no_plan_within_budget')
                plan.goal = np.asarray(goal).copy()
                ex.place_targets, ex.place_paths = targets, paths
                return PhysicalMotionResult(True, 'ok', 'place_refinement')
        return PhysicalMotionResult(False, 'no_collision_free_placement', 'place_refinement')
    except ValueError as error:
        if str(error) != 'no_plan_within_budget':
            raise
        return PhysicalMotionResult(False, str(error), 'place_refinement')
    finally:
        # Retain the consumed IK count: the task provider charges this delta to
        # its shared motion budget. Only temporary limits/deadlines are restored.
        data.qpos[:] = before
        ex.collision.restore_payload(payload)
        mujoco.mj_kinematics(model, data)
        adapter.max_planning_ik_calls, adapter._planning_deadline = old_limit, old_deadline
        plan.planning_ik_calls = adapter.planning_ik_calls
        if plan.planning_seconds_remaining is not None:
            plan.planning_seconds_remaining = max(0., plan.planning_seconds_remaining - (time.monotonic()-planning_started))


def refine_release_placement(adapter, plan):
    """Charge release IK to the retained primitive budget, including alternatives."""
    ex = plan.executor
    previous = getattr(ex, 'release_pose_planner', None)
    previous_budget = getattr(ex, 'release_planning_budget', None)

    @contextmanager
    def planning_pass():
        old_limit, old_deadline = adapter.max_planning_ik_calls, adapter._planning_deadline
        adapter.planning_ik_calls = plan.planning_ik_calls
        adapter.max_planning_ik_calls = plan.planning_ik_limit
        remaining = plan.planning_seconds_remaining
        started = time.monotonic()
        adapter._planning_deadline = None if remaining is None else started + remaining
        try:
            if remaining is not None and remaining <= 0:
                raise ValueError('no_plan_within_budget')
            yield
            if remaining is not None and time.monotonic() >= started + remaining:
                raise ValueError('no_plan_within_budget')
        finally:
            plan.planning_ik_calls = adapter.planning_ik_calls
            if remaining is not None:
                plan.planning_seconds_remaining = max(0., remaining - (time.monotonic() - started))
            adapter.max_planning_ik_calls, adapter._planning_deadline = old_limit, old_deadline

    def pose(probe, point, rotation):
        return adapter._plan_candidate_pose(probe, point, rotation)

    ex.release_pose_planner = pose
    ex.release_planning_budget = planning_pass
    try:
        result = refine_release(ex, support=plan.support_body, region=plan.placement_region, limits=adapter.release_limits)
        if result.success:
            plan.goal = np.asarray(plan.goal).copy()
            plan.goal[:2] += ex.release_lateral_shift
        return result
    except ValueError as error:
        if str(error) != 'no_plan_within_budget':
            raise
        return PhysicalMotionResult(False, str(error), 'release')
    finally:
        ex.release_pose_planner = previous
        ex.release_planning_budget = previous_budget
