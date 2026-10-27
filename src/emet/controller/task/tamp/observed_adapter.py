# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicitly injected observed native execution, separate from simulator controls.

A source supplies current tracked payloads, observed background geometry, support
bounds and temporally bracketed visual frames. A robot-specific refiner supplies
robot-only executors and calibrated grasp/place targets. Neither is synthesized
from read-only discovery proposals. There is deliberately no simulator fallback.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker, PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import (
    ObservedVisualVerifier,
    VisualEffectFrame,
    evaluate_place,
    rotation_distance,
)
from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult, PhysicalPickPlaceExecutor
from emet.motion.placement_geometry import PlacementScene, bounds_array, box_corners

from .api import failure_code, response


@dataclass(frozen=True)
class ObservedNativeTask:
    object_ref: str
    support_ref: str
    object_name: str
    receptacle_name: str


@dataclass(frozen=True)
class ObservedNativeSnapshot:
    frame: VisualEffectFrame
    scene: PlacementScene
    scene_revision: str
    support_bounds: np.ndarray

    def __post_init__(self):
        bounds = bounds_array(self.support_bounds)
        if (not self.scene_revision or self.scene.source != 'observed_voxels'
                or self.scene.workspace is None or not callable(self.scene.known_free)):
            raise ValueError('observed_scene_unavailable')
        bounds = np.frombuffer(bounds.tobytes(), dtype=bounds.dtype).reshape(2, 3)
        object.__setattr__(self, 'support_bounds', bounds)
        object.__setattr__(self, 'scene', PlacementScene(self.scene.boxes, source=self.scene.source,
            workspace=self.scene.workspace, known_free=self.scene.known_free))


@dataclass
class ObservedStoredPlan:
    task: ObservedNativeTask
    initial: ObservedNativeSnapshot
    joints: np.ndarray
    executor: PhysicalPickPlaceExecutor


class _PayloadEnvelope:
    input_mode = 'observed'

    def __init__(self, observe, initial):
        self.observe, self.initial = observe, initial
        frame = initial.frame
        vertices = frame.payload.vertices_reference
        radius = np.linalg.norm(vertices, axis=1).max()
        margin = frame.position_error_m + 2 * radius * np.sin(frame.rotation_error_rad / 2)
        self.vertices = box_corners([vertices.min(0)-margin, vertices.max(0)+margin])

    def observe_payload(self, object_ref):
        current, initial = self.observe().frame, self.initial.frame
        a, b = current.payload, initial.payload
        if ((a.object_ref, a.reference_id, a.revision, a.construction) != (b.object_ref, b.reference_id, b.revision, b.construction)
                or a.object_ref != object_ref or not np.array_equal(a.vertices_reference, b.vertices_reference)
                or current.position_error_m > initial.position_error_m
                or current.rotation_error_rad > initial.rotation_error_rad):
            raise ObservationUnavailable('payload_geometry_changed')
        return PayloadGeometryEvidence(a.object_ref, a.reference_id, a.revision, self.vertices, a.pose, a.construction)


class _CommandBoundary:
    """Only the gripper interfaces used by PhysicalPickPlaceExecutor are exposed."""
    def __init__(self, robot, mark):
        self.robot, self.mark = robot, mark

    def open_gripper(self, **kwargs):
        self.mark()
        return self.robot.open_gripper(**kwargs)

    def close_gripper(self, **kwargs):
        self.mark()
        return self.robot.close_gripper(**kwargs)

    def get_gripper_position(self):
        return self.robot.get_gripper_position()


class ObservedNativeAdapter:
    input_mode = 'observed'

    def __init__(self, robot, *, source, refine, tasks, clock=time.monotonic, max_ik_calls=128):
        if (source.input_mode != 'observed' or not callable(refine) or type(max_ik_calls) is not int
                or not 6 <= max_ik_calls <= 2048 or not all(callable(getattr(source, name, None))
                for name in ('snapshot', 'collect', 'current_revision', 'invalidate'))):
            raise ValueError('invalid_observed_native_provider')
        self.robot, self.source, self.refine, self.clock = robot, source, refine, clock
        self.namespace = uuid.uuid4().hex
        self.task_session = None
        self.tasks = {}
        namespace = uuid.uuid4().hex
        identities = set()
        for task in tasks:
            if (not isinstance(task, ObservedNativeTask) or not all((task.object_ref, task.support_ref,
                    task.object_name, task.receptacle_name)) or (task.object_ref, task.support_ref) in identities):
                raise ValueError('invalid_observed_native_task')
            identities.add((task.object_ref, task.support_ref))
            self.tasks[f'observed_task:{namespace}:{len(self.tasks)}'] = task
        if not self.tasks:
            raise ValueError('invalid_observed_native_task')
        self.lock, self.plans = threading.RLock(), {}
        self.max_ik_calls = max_ik_calls
        self.effects = []
        self._model_signature = None

    def bind_regions(self, regions, *, robot_profile_id, reset_id):
        """Explicitly bind the same observed primitives to multi-step task search."""
        from .observed_transfer import ObservedRegionTransferProvider
        from .task_session import TaskSession

        self.task_session = TaskSession(ObservedRegionTransferProvider(self, regions,
            robot_profile_id=robot_profile_id, reset_id=reset_id))
        return self.task_session

    def invalidate(self):
        # TaskSession acquires its own lock before the provider's adapter lock.
        if self.task_session is not None:
            self.task_session.invalidate()
        with self.lock:
            self.plans.clear()
            self.source.invalidate()

    @staticmethod
    def _data(**extra):
        return {'mode': 'physical', 'inputs': 'observed', 'assistance': [], 'base_motion': 'stationary',
                'geometry_source': 'observed_voxels', 'collision_scope': 'sampled_arm_and_payload', **extra}

    def _failure(self, tool, reason, *, started=False, completed=(), phase='preflight'):
        return response(tool, code=failure_code(reason), status='partial' if started else 'error',
            data=self._data(execution_started=started, completed_ops=list(completed), failed_stage=phase,
                            measurements=[effect.as_dict() for effect in self.effects]))

    def _observe(self, task):
        snapshot = self.source.snapshot(task.object_ref, task.support_ref)
        if not isinstance(snapshot, ObservedNativeSnapshot):
            raise ObservationUnavailable('observed_scene_unavailable')
        snapshot.frame.require(now=self.clock())
        if snapshot.frame.payload.object_ref != task.object_ref:
            raise ObservationUnavailable('payload_identity_mismatch')
        if self.source.current_revision() != snapshot.scene_revision:
            raise ObservationUnavailable('scene_revision_changed')
        return snapshot

    def discover(self, object_filter='', robot=''):
        with self.lock:
            try:
                if robot:
                    return self._failure('scene_tasks', 'native_robot_unsupported')
                rows = []
                for ref, task in self.tasks.items():
                    self._observe(task)
                    if object_filter.casefold() in task.object_name.casefold():
                        rows.append({'task_ref': ref, 'object_name': task.object_name,
                                     'receptacle_name': task.receptacle_name})
                return response('scene_tasks', data=self._data(tasks=rows, discovery_only=False,
                    planning_available=True, execution_available=True,
                    execution_scope='injected_observed_native_provider'))
            except (ObservationUnavailable, ValueError, RuntimeError) as exc:
                return self._failure('scene_tasks', str(exc))

    def _refresh_checker(self, ex, task, *, allow_scene_change=True):
        snapshot = self._observe(task)
        if not allow_scene_change and (snapshot.scene_revision != ex.collision.scene_revision
                or snapshot.scene.geometry_digest != ex.collision.scene.geometry_digest):
            raise ObservationUnavailable('scene_revision_changed')
        if (hasattr(ex, 'observed_support_bounds')
                and not np.array_equal(ex.observed_support_bounds, snapshot.support_bounds)):
            raise ObservationUnavailable('scene_changed_replan')
        ex.collision.scene = snapshot.scene
        ex.collision.scene_revision = snapshot.scene_revision
        # This is the measured capture's original receive time, never 'now'.
        ex.collision.received_monotonic = snapshot.frame.capture.received_monotonic
        return snapshot

    def _certify(self, ex, task, region=None):
        before, payload = ex.data.qpos.copy(), ex.collision.capture_payload()
        try:
            _, error = ex.plan_gripper_open()
            if error:
                return error
            for grasp, phases, targets in ((True, ('pregrasp', 'grasp', 'lift'), ex.grasp_targets),
                                            (False, ('preplace', 'place', 'retreat'), ex.place_targets)):
                paths = []
                for phase, (point, rotation) in zip(phases, targets, strict=True):
                    if phase == 'lift':
                        ex.collision.set_payload(ex.model, ex.data, task.object_ref, ex.ee_body)
                    if phase == 'retreat':
                        ex.collision.set_payload(ex.model, ex.data, None)
                        _, error = ex.plan_gripper_open()
                        if error:
                            return error
                    path, error = ex.plan_pose(point, rotation)
                    if error:
                        return error
                    if phase == 'place' and region is not None:
                        ee = ex.data.body(ex.ee_body)
                        vertices = ex.collision.payload.world_vertices(ee.xpos, ee.xmat)
                        if np.any(vertices[:, :2].min(0) < region[0]) or np.any(vertices[:, :2].max(0) > region[1]):
                            return 'placement_region_unavailable'
                    paths.append(path)
                if grasp:
                    ex.grasp_paths = paths
                else:
                    ex.place_paths = paths
            return None
        finally:
            ex.data.qpos[:] = before
            ex.collision.restore_payload(payload)
            mujoco.mj_kinematics(ex.model, ex.data)

    def plan(self, task_ref='', object_name='', receptacle_name='', placement_region=None):
        with self.lock:
            self.effects = []
            try:
                region = None
                if placement_region is not None:
                    region = np.asarray(placement_region, float)
                    if region.shape != (2, 2) or not np.isfinite(region).all() or np.any(region[0] >= region[1]):
                        return self._failure('plan_pick_place', 'invalid_placement_region')
                if task_ref:
                    task = self.tasks.get(task_ref)
                    if task is None:
                        return self._failure('plan_pick_place', 'unknown_task_ref')
                    if any(given and given.strip().casefold() != expected.casefold()
                           for given, expected in ((object_name, task.object_name), (receptacle_name, task.receptacle_name))):
                        return self._failure('plan_pick_place', 'task_selector_mismatch')
                else:
                    matches = [task for task in self.tasks.values() if task.object_name.casefold() == object_name.casefold()
                               and task.receptacle_name.casefold() == receptacle_name.casefold()]
                    if len(matches) != 1:
                        return self._failure('plan_pick_place', 'ambiguous_object' if matches else 'task_not_found')
                    task = matches[0]
                initial = self._observe(task)
                frames = tuple(self.source.collect(task.object_ref, task.support_ref, after=initial.frame.capture.latest))
                supported = evaluate_place(initial.frame, frames, support_bounds=initial.support_bounds,
                    after=initial.frame.capture.latest, now=self.clock())
                if not supported.satisfied:
                    return self._failure('plan_pick_place', 'task_object_not_settled_on_support')
                current = self._observe(task)
                self._require_same_initial(initial, current)
                initial = current
                if region is not None and (np.any(region[0] < initial.support_bounds[0, :2])
                                           or np.any(region[1] > initial.support_bounds[1, :2])):
                    return self._failure('plan_pick_place', 'placement_region_unavailable')
                ex = (self.refine(task, initial, placement_region=region) if region is not None
                      else self.refine(task, initial))
                if (not isinstance(ex, PhysicalPickPlaceExecutor) or ex.input_mode != 'observed'
                        or ex.robot is not self.robot or ex.use_sim_time
                        or not isinstance(ex.collision, ObservedNativeCollisionChecker)
                        or ex.object_pose_provider is not ex.collision
                        or len(ex.grasp_targets) != 3 or len(ex.place_targets) != 3
                        or ex.payload_body is not None or ex.collision.payload_body is not None):
                    raise ValueError('invalid_observed_native_provider')
                signature = model_signature(ex.model)
                if self._model_signature is not None and self._model_signature != signature:
                    raise ObservationUnavailable('scene_changed_replan')
                self._model_signature = signature
                ex.collision.provider = _PayloadEnvelope(lambda: self._observe(task), initial)
                ex.synchronize(ex.data)
                initial_joints = ex.data.qpos.copy()
                ex.observed_support_bounds = initial.support_bounds.copy()
                raw_plan = ex.plan_pose
                calls = [0]

                def plan_pose(*args, **kwargs):
                    if calls[0] >= self.max_ik_calls:
                        return None, 'no_plan_within_budget'
                    calls[0] += 1
                    self._refresh_checker(ex, task)
                    return raw_plan(*args, **kwargs)

                raw_sync = ex.synchronize

                def synchronize(data):
                    raw_sync(data)
                    # Refresh from a real current capture, not from a renewed
                    # timestamp on the old geometry. Mid-segment scene changes
                    # stop execution; a future attempt must replan explicitly.
                    if self.source.current_revision() != ex.collision.scene_revision:
                        raise ObservationUnavailable('scene_revision_changed')
                    self._refresh_checker(ex, task, allow_scene_change=False)

                ex.synchronize, ex.plan_pose = synchronize, plan_pose
                error = self._certify(ex, task, region)
                if error:
                    return self._failure('plan_pick_place', error)
                # Recheck after planning without renewing old evidence.
                current = self._observe(task)
                self._require_same_initial(initial, current)
                target_bounds = initial.support_bounds.copy()
                if region is not None:
                    target_bounds[:, :2] = region
                def observe_effect(_):
                    return self._refresh_checker(ex, task, allow_scene_change=False).frame

                def collect_effect(obj, *, after):
                    frames = tuple(self.source.collect(obj, task.support_ref, after=after))
                    current = observe_effect(obj)
                    if frames and frames[-1] is not None:
                        a, b = frames[-1].payload, current.payload
                        if ((a.reference_id, a.revision) != (b.reference_id, b.revision)
                                or np.linalg.norm(a.pose.world_from_object[:3, 3] - b.pose.world_from_object[:3, 3]) > .005
                                or rotation_distance(a.pose.world_from_object[:3, :3], b.pose.world_from_object[:3, :3]) > .02):
                            raise ObservationUnavailable('payload_geometry_changed')
                    return frames

                verifier = ObservedVisualVerifier(object_ref=task.object_ref, support_bounds=target_bounds,
                    observe=observe_effect, collect=collect_effect, clock=self.clock)
                ex.visual_verifier = verifier
                ex.transport = lambda: PhysicalMotionResult(True, 'stationary', 'transport')
                ref = f'observed_plan:{uuid.uuid4().hex}'
                if len(self.plans) >= 32:
                    self.plans.pop(next(iter(self.plans)))
                self.plans[ref] = ObservedStoredPlan(task, initial, initial_joints, ex)
                return response('plan_pick_place', data=self._data(plan_ref=ref, planned_ops=['grasp', 'place'],
                    completed_ops=[], execution_started=False, planning_ik_calls=calls[0]))
            except (ObservationUnavailable, ValueError, RuntimeError, TimeoutError) as exc:
                return self._failure('plan_pick_place', str(exc))

    @staticmethod
    def _require_same_initial(initial, current):
        a, b = initial.frame.payload, current.frame.payload
        if (initial.scene_revision != current.scene_revision
                or initial.frame.capture.calibration_id != current.frame.capture.calibration_id
                or initial.scene.geometry_digest != current.scene.geometry_digest
                or not np.array_equal(initial.support_bounds, current.support_bounds)
                or (a.object_ref, a.reference_id, a.revision) != (b.object_ref, b.reference_id, b.revision)
                or not np.array_equal(a.vertices_reference, b.vertices_reference)
                or np.linalg.norm(a.pose.world_from_object[:3, 3] - b.pose.world_from_object[:3, 3]) > .005
                or rotation_distance(a.pose.world_from_object[:3, :3], b.pose.world_from_object[:3, :3]) > .02):
            raise ObservationUnavailable('scene_changed_replan')

    def execute(self, plan_ref):
        with self.lock:
            self.effects = []
            plan = self.plans.pop(plan_ref, None)
            if plan is None:
                return self._failure('execute_pick_place_plan', 'unknown_plan')
            started, completed, phase = False, [], 'preflight'
            ex, task = plan.executor, plan.task
            original_robot, original_command = ex.robot, ex.command_joints

            def mark():
                nonlocal started
                started = True

            def command(q):
                mark()
                return original_command(q)

            try:
                self._require_same_initial(plan.initial, self._observe(task))
                if model_signature(ex.model) != self._model_signature:
                    raise ObservationUnavailable('scene_changed_replan')
                ex.synchronize(ex.data)
                if np.max(np.abs(ex.data.qpos - plan.joints)) > .01:
                    raise ObservationUnavailable('scene_changed_replan')
                ex.robot, ex.command_joints = _CommandBoundary(self.robot, mark), command
                for phase, action in (('grasp', lambda: ex.grasp_only('', object_gt_body=task.object_ref)),
                                      ('place', lambda: ex.place_only('', object_gt_body=task.object_ref))):
                    # Refresh foreground/background evidence before each skill;
                    # the checker still rejects revision changes during a path.
                    self._refresh_checker(ex, task)
                    result = action()
                    self.effects = ex.visual_verifier.results.copy()
                    if not result.success:
                        return self._failure('execute_pick_place_plan', result.message, started=started,
                                             completed=completed, phase=result.phase)
                    completed.append(phase)
                return response('execute_pick_place_plan', data=self._data(execution_started=started,
                    completed_ops=completed, failed_stage=None, measurements=[e.as_dict() for e in self.effects]))
            except TimeoutError:
                reason = 'command_outcome_unknown' if started else 'observation_unavailable'
                return self._failure('execute_pick_place_plan', reason, started=started,
                                     completed=completed, phase=phase)
            except Exception as exc:
                # Unknown exceptions after a command cannot assert no actuation.
                self.effects = ex.visual_verifier.results.copy()
                return self._failure('execute_pick_place_plan', str(exc), started=started, completed=completed, phase=phase)
            finally:
                ex.robot, ex.command_joints = original_robot, original_command
