# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicitly injected observed native execution, separate from simulator controls.

A source supplies current tracked payloads, observed background geometry, support
bounds and temporally bracketed visual frames. A robot-specific refiner supplies
robot-only executors and calibrated grasp/place targets. Neither is synthesized
from read-only discovery proposals. There is deliberately no simulator fallback.
"""
from __future__ import annotations

import logging
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
    pose_uncertainty_margin,
)
from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult, PhysicalPickPlaceExecutor
from emet.controller.manipulation.shape_pose import ShapePoseEvidence, evidence_rotation_distance, shape_identity
from emet.motion.observed_admission import ObservedMaterialAdmission
from emet.motion.observed_contact import ObservedTargetContact
from emet.motion.observed_support_contact import ObservedSupportContact
from emet.motion.placement_geometry import PlacementScene, bounds_array

from .api import failure_code, response

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservedNativeTask:
    object_ref: str
    support_ref: str
    object_name: str
    receptacle_name: str
    source_support_ref: str | None = None


@dataclass(frozen=True)
class ObservedNativeSnapshot:
    frame: VisualEffectFrame
    scene: PlacementScene
    scene_revision: str
    support_bounds: np.ndarray
    admission: ObservedMaterialAdmission | None = None
    target_contact: ObservedTargetContact | None = None
    support_contact: ObservedSupportContact | None = None
    source_support_bounds: np.ndarray | None = None
    source_support_contact: ObservedSupportContact | None = None
    local_scene: object | None = None
    # support_bounds describes destination placement; same-support transfers can omit the source bounds.

    def __post_init__(self):
        bounds = bounds_array(self.support_bounds)
        if (not self.scene_revision or self.scene.source != 'observed_voxels'
                or self.scene.workspace is None or not callable(self.scene.known_free)):
            raise ValueError('observed_scene_unavailable')
        bounds = np.frombuffer(bounds.tobytes(), dtype=bounds.dtype).reshape(2, 3)
        object.__setattr__(self, 'support_bounds', bounds)
        source = self.source_support_bounds
        if source is None:
            source_bounds = bounds
        else:
            source_bounds = bounds_array(source)
            source_bounds = np.frombuffer(source_bounds.tobytes(), dtype=source_bounds.dtype).reshape(2, 3)
        object.__setattr__(self, 'source_support_bounds', source_bounds)
        object.__setattr__(self, 'scene', PlacementScene(self.scene.boxes, source=self.scene.source,
            workspace=self.scene.workspace, known_free=self.scene.known_free))
        if self.admission is not None:
            if (not isinstance(self.admission, ObservedMaterialAdmission)
                    or self.admission.scene_geometry_digest != self.scene.geometry_digest
                    or self.admission.robot_material.capture != self.frame.capture):
                raise ValueError('invalid_observed_admission')
        source_contact = self.source_support_contact
        if source_contact is None and np.array_equal(source_bounds, bounds):
            source_contact = self.support_contact
        object.__setattr__(self, 'source_support_contact', source_contact)
        for contact, footprint in ((source_contact, source_bounds), (self.support_contact, bounds)):
            if contact is not None and (
                    not isinstance(contact, ObservedSupportContact) or self.admission is None
                    or contact.object_ref != self.frame.payload.object_ref
                    or contact.scene_epoch != self.admission.background.scene_epoch
                    or not np.array_equal(contact.patch.bounds, footprint)):
                raise ValueError('invalid_observed_support_contact')
        if self.target_contact is not None:
            contact, payload = self.target_contact, self.frame.payload
            if not isinstance(contact, ObservedTargetContact):
                raise ValueError('invalid_observed_target_contact')
            world_from_reference = payload.pose.world_from_object
            radius = np.linalg.norm(payload.vertices_reference, axis=1).max()
            margin = pose_uncertainty_margin(radius, self.frame.position_error_m, self.frame.rotation_error_rad)
            if ((contact.object_ref, contact.reference_id, contact.geometry_revision)
                    != (payload.object_ref, payload.reference_id, payload.revision)
                    or contact.capture != self.frame.capture or contact.target.kind != 'convex'
                    or not np.array_equal(contact.target.vertices, payload.vertices_reference)
                    or not np.array_equal(contact.target.world_from_local, world_from_reference)
                    or contact.target.offset_m < margin):
                raise ValueError('invalid_observed_target_contact')


@dataclass
class ObservedStoredPlan:
    task: ObservedNativeTask
    initial: ObservedNativeSnapshot
    joints: np.ndarray
    executor: PhysicalPickPlaceExecutor


class _PayloadEnvelope:
    input_mode = 'observed'

    def __init__(self, observe, initial, *, attachment_ee_positions=None, attachment_tracking_error_m=0., retention_translation_m=.02):
        self.observe, self.initial = observe, initial
        frame = initial.frame
        self.vertices = frame.payload.vertices_reference
        ee = (frame.robot_after.world_from_ee[:3, 3][None] if attachment_ee_positions is None
              else np.asarray(attachment_ee_positions, float).reshape(-1, 3))
        radius = np.linalg.norm(self.vertices, axis=1).max()
        lever = np.linalg.norm(frame.payload.pose.world_from_object[:3, 3]-ee, axis=1).max()+radius
        # The calibration radius must cover the relative displacement already
        # permitted by runtime retention. Otherwise a hidden few-mm lever gate
        # rejects a payload before the declared 2 cm retention limit is tested.
        self.lever_limit = lever+frame.position_error_m+attachment_tracking_error_m+retention_translation_m
        self.robot_position = max(frame.robot_before.position_error_m, frame.robot_after.position_error_m)
        self.robot_rotation = max(frame.robot_before.rotation_error_rad, frame.robot_after.rotation_error_rad)
        # Per-capture EE error. Transport needs two captures; retention adds
        # before/current once each. Angular displacement is included in metres.
        self.attachment_error = pose_uncertainty_margin(self.lever_limit+self.robot_position,
            self.robot_position+frame.alignment_error_m, self.robot_rotation+frame.alignment_error_rad)
        self.attachment_rotation = self.robot_rotation+frame.alignment_error_rad

    def observe_payload(self, object_ref):
        current, initial = self.observe().frame, self.initial.frame
        a, b = current.payload, initial.payload
        if ((a.object_ref, a.reference_id, a.revision, a.construction) != (b.object_ref, b.reference_id, b.revision, b.construction)
                or a.object_ref != object_ref or not np.array_equal(a.vertices_reference, b.vertices_reference)
                or shape_identity(a.pose) != shape_identity(b.pose)
                or current.position_error_m > initial.position_error_m
                or current.rotation_error_rad > initial.rotation_error_rad
                or max(current.robot_before.position_error_m, current.robot_after.position_error_m) > self.robot_position
                or max(current.robot_before.rotation_error_rad, current.robot_after.rotation_error_rad) > self.robot_rotation
                or current.alignment_error_m > initial.alignment_error_m
                or current.alignment_error_rad > initial.alignment_error_rad):
            raise ObservationUnavailable('payload_geometry_changed')
        return PayloadGeometryEvidence(a.object_ref, a.reference_id, a.revision, self.vertices, a.pose, a.construction,
            position_error_m=current.position_error_m, rotation_error_rad=current.rotation_error_rad,
            attachment_error_m=self.attachment_error, attachment_rotation_error_rad=self.attachment_rotation,
            attachment_lever_limit_m=self.lever_limit,
            captured_world_from_ee=current.robot_after.world_from_ee)


class _CommandBoundary:
    """Mark possible dispatch before entering transport, not confirmed actuation.

    An exception can follow a successful send with a lost acknowledgement. Only
    failures before this boundary can assert no command; return-time marking
    would turn some possible physical progress into a false no-actuation receipt.
    """
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

    def guarded_place_probe(self, *args, **kwargs):
        self.mark()
        return self.robot.guarded_place_probe(*args, **kwargs)


class ObservedNativeAdapter:
    input_mode = 'observed'

    def __init__(self, robot, *, source, refine, tasks, clock=time.monotonic, max_ik_calls=128, robot_name=''):
        if (not isinstance(robot_name, str) or source.input_mode != 'observed' or not callable(refine) or type(max_ik_calls) is not int
                or not 6 <= max_ik_calls <= 2048 or not all(callable(getattr(source, name, None))
                for name in ('snapshot', 'collect', 'current_revision', 'invalidate'))):
            raise ValueError('invalid_observed_native_provider')
        self.robot, self.source, self.refine, self.clock = robot, source, refine, clock
        self.robot_name = robot_name.strip().casefold()
        self.namespace = uuid.uuid4().hex
        self.task_session = None
        self.tasks = {}
        task_key_namespace = uuid.uuid4().hex
        identities = set()
        for task in tasks:
            if (not isinstance(task, ObservedNativeTask) or not all((task.object_ref, task.support_ref,
                    task.object_name, task.receptacle_name)) or (task.object_ref, task.support_ref) in identities):
                raise ValueError('invalid_observed_native_task')
            identities.add((task.object_ref, task.support_ref))
            self.tasks[f'observed_task:{task_key_namespace}:{len(self.tasks)}'] = task
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

    @staticmethod
    def _payload_scope(snapshot):
        payload = snapshot.frame.payload
        return {'payload_geometry_source': payload.construction,
                'payload_pose_scope': ('declared_shape_equivalence' if isinstance(payload.pose, ShapePoseEvidence)
                                       else 'unique_rigid_pose')}

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
        supports = ((snapshot.source_support_contact, task.source_support_ref or task.support_ref),
                    (snapshot.support_contact, task.support_ref))
        for contact, ref in supports:
            if contact is not None:
                if contact.support_ref != ref:
                    raise ObservationUnavailable('support_geometry_unknown')
                contact.require(snapshot.admission.background, now=self.clock(),
                                static_assumption=snapshot.admission.static_assumption)
        return snapshot

    def discover(self, object_filter='', robot=''):
        with self.lock:
            try:
                robot = robot.strip().casefold()
                if robot and robot != self.robot_name:
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
        previous_admission = ex.collision.admission
        if ((previous_admission is None) != (snapshot.admission is None)
                or (previous_admission is not None
                    and previous_admission.profile_identity != snapshot.admission.profile_identity)):
            raise ObservationUnavailable('robot_material_identity_changed')
        if snapshot.admission is not None:
            snapshot.admission.require(now=self.clock(), model_signature=model_signature(ex.model),
                                       capture_id=snapshot.frame.capture.observation_id)
        old_contact, new_contact = ex.collision.target_contact, snapshot.target_contact
        if ((old_contact is None) != (new_contact is None)
                or (old_contact is not None and old_contact.profile_identity != new_contact.profile_identity)):
            raise ObservationUnavailable('target_contact_identity_changed')
        if new_contact is not None:
            new_contact.require(now=self.clock(), model_signature=model_signature(ex.model),
                                capture_id=snapshot.frame.capture.observation_id)
        for old, current in ((ex.collision.source_support_contact, snapshot.source_support_contact),
                             (ex.collision.support_contact, snapshot.support_contact)):
            if not self._same_support_contact(old, current):
                raise ObservationUnavailable('scene_changed_replan')
            if current is not None:
                current.require(snapshot.admission.background, now=self.clock(),
                                static_assumption=snapshot.admission.static_assumption)
        if not allow_scene_change and snapshot.local_scene is None and (snapshot.scene_revision != ex.collision.scene_revision
                or snapshot.scene.geometry_digest != ex.collision.scene.geometry_digest):
            raise ObservationUnavailable('scene_revision_changed')
        if hasattr(ex, 'observed_support_bounds') and (
                not np.array_equal(ex.observed_support_bounds, snapshot.support_bounds)
                or not np.array_equal(ex.observed_source_support_bounds, snapshot.source_support_bounds)):
            raise ObservationUnavailable('scene_changed_replan')
        ex.collision.scene = snapshot.scene
        ex.collision.scene_revision = snapshot.scene_revision
        # This is the measured capture's original receive time, never 'now'.
        ex.collision.received_monotonic = snapshot.frame.capture.received_monotonic
        ex.collision.admission = snapshot.admission
        ex.collision.local_scene = snapshot.local_scene
        if snapshot.local_scene is not None:
            ex.collision.measured_joints = snapshot.frame.robot_before.joints
        ex.collision.set_target_contact(new_contact)
        ex.collision.set_support_contact(snapshot.support_contact)
        ex.collision.set_source_support_contact(snapshot.source_support_contact)
        return snapshot

    def _certify(self, ex, task, region=None):
        with ex.collision.certify_transfer():
            return self._certify_candidates(ex, task, region)

    def _certify_candidates(self, ex, task, region):
        if not ex.transfer_candidates:
            return self._certify_pair(ex, task, region)
        first_error = None
        failed_grasps = set()
        for candidate_index, candidate in enumerate(ex.transfer_candidates):
            grasp_key = (tuple((np.asarray(p).tobytes(), np.asarray(r).tobytes()) for p, r in candidate.grasp),
                         tuple(candidate.close_joints))
            if grasp_key in failed_grasps:
                continue
            ex.last_certification_phase = None
            ex.grasp_targets = list(candidate.grasp)
            ex.place_targets = list(candidate.place)
            ex.gripper_close_configuration = dict(candidate.close_joints)
            error = self._certify_pair(ex, task, region)
            if error is None:
                return None
            logger.info('Observed candidate %s failed in %s: %s', candidate_index, ex.last_certification_phase, error)
            if ex.last_certification_phase in {'pregrasp', 'grasp', 'lift'}:
                # A different landing pose cannot repair the same rejected
                # pickup. Scope this cache to this certification invocation.
                failed_grasps.add(grasp_key)
            if first_error is None:
                first_error = error
        ex.grasp_paths.clear()
        ex.place_paths.clear()
        return first_error

    def _certify_pair(self, ex, task, region):
        before, payload = ex.data.qpos.copy(), ex.collision.capture_payload()
        clearance = ex._contact_clearance_scope
        try:
            _, error = ex.plan_gripper_open()
            if error:
                return error
            for grasp, phases, targets in ((True, ('pregrasp', 'grasp', 'lift'), ex.grasp_targets),
                                            (False, ('preplace', 'place', 'retreat'), ex.place_targets)):
                paths = []
                for phase, (point, rotation) in zip(phases, targets, strict=True):
                    ex.last_certification_phase = phase
                    if phase == 'lift':
                        ex.collision.set_payload(ex.model, ex.data, task.object_ref, ex.ee_body)
                    if phase == 'retreat':
                        ex.collision.set_payload(ex.model, ex.data, None)
                        with ex.collision.motion_phase('release'):
                            _, error = ex.plan_gripper_open()
                        if error:
                            return error
                    phase_started = time.perf_counter()
                    logger.info('Certifying observed %s target %s', phase, np.asarray(point).tolist())
                    with ex.collision.motion_phase(phase):
                        path, error = ex.plan_pose(point, rotation)
                    if error is None and phase == 'grasp' and ex.collision.target_contact is not None:
                        _, error = ex.plan_gripper_close()
                    logger.info('Observed %s certification took %.3fs: %s', phase,
                                time.perf_counter()-phase_started, error or 'ok')
                    if error:
                        logger.warning('Observed transfer candidate stopped in %s: %s', phase, error)
                        return error
                    if phase == 'place' and region is not None:
                        bounds = ex.collision.possible_payload_bounds(ex.model, ex.data)
                        if np.any(bounds[0, :2] < region[0]) or np.any(bounds[1, :2] > region[1]):
                            return 'placement_region_unavailable'
                    paths.append(path)
                if grasp:
                    ex.grasp_paths = paths
                else:
                    ex.place_paths = paths
            return None
        finally:
            ex._contact_clearance_scope = clearance
            ex.data.qpos[:] = before
            ex.collision.restore_payload(payload)
            mujoco.mj_kinematics(ex.model, ex.data)

    def plan_request(self, request, *, task_ref='', placement_region=None):
        """Resolve supported options before entering the normal certified planner."""
        from .manipulation_request import ManipulationRequest

        if not isinstance(request, ManipulationRequest):
            return response('plan_pick_place', code='invalid_manipulation_hint')
        if request.hint_mode == 'strict' and request.hints.get('approach'):
            return response('plan_pick_place', code='strict_approach_unavailable',
                data={'execution_started': False})
        unsupported = set(request.hints)-{'approach'}
        if request.constraints.get('protected_objects'):
            unsupported.add('protected_objects')
        if unsupported or not getattr(getattr(self, 'refine', None), 'supports_manipulation_request', False):
            return response('plan_pick_place', code='manipulation_options_unsupported',
                data={'execution_started': False, 'unsupported_options': sorted(unsupported)})
        return self.plan(task_ref, request.object_name, request.receptacle_name,
                         placement_region, request=request)

    def plan(self, task_ref='', object_name='', receptacle_name='', placement_region=None, *, request=None):
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
                pickup_bounds = initial.source_support_bounds
                if initial.source_support_contact is not None:
                    pickup_bounds = initial.source_support_contact.pickup_observation_bounds(initial.frame.support_evidence_bounds(),
                        approximate=initial.frame.payload.construction == "approximate_observed_box")
                    if pickup_bounds is None:
                        return self._failure('plan_pick_place', 'task_object_not_settled_on_support')
                frames = tuple(self.source.collect(task.object_ref, task.support_ref, after=initial.frame.capture.latest))
                settling_tolerance = .01 if np.array_equal(pickup_bounds, initial.source_support_bounds) else .015
                supported = evaluate_place(initial.frame, frames, support_bounds=pickup_bounds,
                    after=initial.frame.capture.latest, now=self.clock(), settling_tolerance_m=settling_tolerance)
                if not supported.satisfied:
                    return self._failure('plan_pick_place', 'task_object_not_settled_on_support')
                current = self._observe(task)
                # No path has been certified yet. Use the newer map if it can
                # only reduce clearance and the observed task remains bound.
                self._require_same_initial(initial, current, allow_stricter_map=True)
                initial = current
                if region is not None and (np.any(region[0] < initial.support_bounds[0, :2])
                                           or np.any(region[1] > initial.support_bounds[1, :2])):
                    return self._failure('plan_pick_place', 'placement_region_unavailable')
                if request is not None:
                    ex = self.refine(task, initial, placement_region=region, request=request)
                else:
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
                ex.collision.provider = _PayloadEnvelope(lambda: self._observe(task), initial,
                    attachment_ee_positions=[targets[1][0] for targets in
                        [ex.grasp_targets, *ex.grasp_target_options,
                         *(option.grasp for option in ex.transfer_candidates)]],
                    attachment_tracking_error_m=ex.position_tolerance_m,
                    retention_translation_m=ex.payload_retention_translation_m)
                ex.synchronize(ex.data)
                initial_joints = ex.data.qpos.copy()
                ex.ik_call_budget = self.max_ik_calls
                ex.observed_support_bounds = initial.support_bounds.copy()
                ex.observed_source_support_bounds = initial.source_support_bounds.copy()
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
                    if ex.collision.local_scene is None and self.source.current_revision() != ex.collision.scene_revision:
                        raise ObservationUnavailable('scene_revision_changed')
                    self._refresh_checker(ex, task, allow_scene_change=False)

                ex.synchronize, ex.plan_pose = synchronize, plan_pose
                certificate_region = initial.support_bounds[:, :2] if region is None and ex.transfer_candidates else region
                error = self._certify(ex, task, certificate_region)
                if error:
                    admission = ex.collision.last_admission
                    logger.warning('Observed transfer certification stopped for %s: %s (collision=%s geom=%s admission=%s)',
                                   task.object_ref, error, ex.collision.last_reason, ex.collision.last_blocked_geom,
                                   admission.reason if admission is not None else None)
                    return self._failure('plan_pick_place', error)
                # Recheck after planning without renewing old evidence.
                current = self._observe(task)
                self._require_same_initial(initial, current, allow_stricter_map=current.local_scene is not None)
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
                        if ((a.object_ref, a.reference_id, a.revision, a.construction)
                                != (b.object_ref, b.reference_id, b.revision, b.construction)
                                or not np.array_equal(a.vertices_reference, b.vertices_reference)
                                or frames[-1].capture.calibration_id != current.capture.calibration_id
                                or np.linalg.norm(a.pose.world_from_object[:3, 3] - b.pose.world_from_object[:3, 3]) > .005
                                or shape_identity(a.pose) != shape_identity(b.pose)
                                or evidence_rotation_distance(a.pose, a.pose.world_from_object[:3, :3], b.pose.world_from_object[:3, :3]) > .02):
                            raise ObservationUnavailable('payload_geometry_changed')
                    return frames

                def observe_after(obj, *, after):
                    frames = collect_effect(obj, after=after)
                    if not frames:
                        raise ObservationUnavailable('visual_evidence_incomplete')
                    return frames[-1]

                verifier = ObservedVisualVerifier(object_ref=task.object_ref, support_bounds=target_bounds,
                    observe=observe_effect, collect=collect_effect, observe_after=observe_after, clock=self.clock,
                    retention_rotation_rad=ex.payload_retention_rotation_rad,
                    retention_translation_m=ex.payload_retention_translation_m,
                    settling_tolerance_m=getattr(ex, 'placement_settling_tolerance_m', .01),
                    # Use the same measured occlusion-bracketed footprint that
                    # admitted the resting object, not just the largest visible
                    # rectangle beside it.
                    source_support_bounds=pickup_bounds)
                ex.visual_verifier = verifier
                ex.payload_tracking_changed = getattr(self.source, 'set_carry_prediction', None)
                ex.transport = lambda: PhysicalMotionResult(True, 'stationary', 'transport')
                ref = f'observed_plan:{uuid.uuid4().hex}'
                if len(self.plans) >= 32:
                    self.plans.pop(next(iter(self.plans)))
                self.plans[ref] = ObservedStoredPlan(task, initial, initial_joints, ex)
                return response('plan_pick_place', data=self._data(plan_ref=ref, planned_ops=['grasp', 'place'],
                    completed_ops=[], execution_started=False, planning_ik_calls=ex.ik_solver_calls,
                    guidance={'hint_mode': request.hint_mode, 'applied': request.hints, 'adjustments': [], 'constraints': request.constraints} if request else None,
                    placement_strategy='verify_at_hover' if ex.release_refiner is not None else 'visual',
                    **self._payload_scope(initial)))
            except (ObservationUnavailable, ValueError, RuntimeError, TimeoutError) as exc:
                logger.exception('Observed transfer planning did not complete')
                result = self._failure('plan_pick_place', str(exc))
                if isinstance(exc, ObservationUnavailable) and str(exc) in {
                    'geometry_tracking_no_match', 'geometry_tracking_budget_exhausted',
                    'moving_tracking_time_budget_exhausted', 'moving_tracking_lost_reinitialize',
                    'moving_multiview_association_unknown', 'moving_geometry_ambiguous', 'object_pose_ambiguous',
                }:
                    # Planning issued no commands. Discard the failed identity
                    # track and its handles; the next request must ground a new
                    # scene from fresh measurements. Never do this on execution.
                    self.plans.clear()
                    try:
                        self.source.invalidate()
                    except (ValueError, RuntimeError, TimeoutError):
                        logger.exception('Failed to invalidate the stopped planning source')
                        result['data']['perception_reset'] = False
                    else:
                        result['data']['perception_reset'] = True
                        result['recovery'] = 'replan'
                return result

    @staticmethod
    def _same_support_contact(a, b):
        if a is None or b is None:
            return a is b
        return a.profile_identity == b.profile_identity

    @staticmethod
    def _require_same_initial(initial, current, *, allow_stricter_map=False):
        a, b = initial.frame.payload, current.frame.payload
        map_matches = (initial.scene_revision == current.scene_revision
                       and initial.scene.geometry_digest == current.scene.geometry_digest)
        if not map_matches and allow_stricter_map and initial.admission is not None and current.admission is not None:
            map_matches = current.admission.background.is_conservative_update_of(initial.admission.background)
        if ((initial.admission is None) != (current.admission is None)
                or (initial.admission is not None and (
                    initial.admission.profile_identity != current.admission.profile_identity))
                or (initial.target_contact is None) != (current.target_contact is None)
                or (initial.target_contact is not None and
                    initial.target_contact.profile_identity != current.target_contact.profile_identity)
                or not ObservedNativeAdapter._same_support_contact(initial.support_contact, current.support_contact)
                or not ObservedNativeAdapter._same_support_contact(initial.source_support_contact, current.source_support_contact)
                or not map_matches
                or initial.frame.capture.calibration_id != current.frame.capture.calibration_id
                or not np.array_equal(initial.support_bounds, current.support_bounds)
                or not np.array_equal(initial.source_support_bounds, current.source_support_bounds)
                or (a.object_ref, a.reference_id, a.revision) != (b.object_ref, b.reference_id, b.revision)
                or shape_identity(a.pose) != shape_identity(b.pose)
                or not np.array_equal(a.vertices_reference, b.vertices_reference)
                or np.linalg.norm(a.pose.world_from_object[:3, 3] - b.pose.world_from_object[:3, 3]) > .005
                or evidence_rotation_distance(a.pose, a.pose.world_from_object[:3, :3], b.pose.world_from_object[:3, :3]) > .02):
            logger.warning('Initial evidence changed: position_delta=%.5f rotation_delta=%.5f map_matches=%s source_bounds_match=%s destination_bounds_match=%s',
                np.linalg.norm(a.pose.world_from_object[:3, 3]-b.pose.world_from_object[:3, 3]),
                evidence_rotation_distance(a.pose, a.pose.world_from_object[:3, :3], b.pose.world_from_object[:3, :3]),
                map_matches, np.array_equal(initial.source_support_bounds, current.source_support_bounds),
                np.array_equal(initial.support_bounds, current.support_bounds))
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
            original_gripper_command = ex.command_gripper_joints

            def mark():
                nonlocal started
                started = True

            def command(q):
                # Same possible-dispatch boundary as the gripper transport.
                mark()
                return original_command(q)

            def gripper_command(q):
                mark()
                return original_gripper_command(q)

            try:
                self._require_same_initial(plan.initial, self._observe(task),
                                           allow_stricter_map=plan.initial.local_scene is not None)
                if model_signature(ex.model) != self._model_signature:
                    raise ObservationUnavailable('scene_changed_replan')
                ex.synchronize(ex.data)
                if np.max(np.abs(ex.data.qpos - plan.joints)) > .01:
                    raise ObservationUnavailable('scene_changed_replan')
                ex.robot, ex.command_joints = _CommandBoundary(self.robot, mark), command
                if original_gripper_command is not None:
                    ex.command_gripper_joints = gripper_command
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
                    completed_ops=completed, failed_stage=None, measurements=[e.as_dict() for e in self.effects],
                    **self._payload_scope(plan.initial)))
            except TimeoutError:
                reason = 'command_outcome_unknown' if started else 'observation_unavailable'
                return self._failure('execute_pick_place_plan', reason, started=started,
                                     completed=completed, phase=phase)
            except Exception as exc:
                logger.exception('Observed transfer execution failed during %s', phase)
                # Unknown exceptions after a command cannot assert no actuation.
                self.effects = ex.visual_verifier.results.copy()
                return self._failure('execute_pick_place_plan', str(exc), started=started, completed=completed, phase=phase)
            finally:
                ex.robot, ex.command_joints = original_robot, original_command
                ex.command_gripper_joints = original_gripper_command
                if started and len(completed) < 2:
                    hold = getattr(original_robot, 'hold_position', None)
                    if callable(hold):
                        try:
                            hold()
                        except (ObservationUnavailable, RuntimeError, TimeoutError, ValueError):
                            logger.exception('Could not confirm native arm hold after incomplete transfer')
