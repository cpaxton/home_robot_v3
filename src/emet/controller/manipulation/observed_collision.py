# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Robot-only native planning against supplied, bounded observed evidence.

This consumes geometry; it does not infer a complete object from a visible mask.
A provider must supply a conservative envelope (including pose ambiguity) in an
explicit reference frame. Attaching that envelope is a planning hypothesis, never
an acknowledgement that the physical gripper holds the object.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.shape_pose import ShapePoseEvidence, evidence_rotation_distance, shape_identity
from emet.motion.local_geometry_distance import cell_separations
from emet.motion.observed_admission import ObservedMaterialAdmission, RobotGeometryBound
from emet.motion.observed_contact import CONTACT_PHASES, MOTION_PHASES, ObservedTargetContact
from emet.motion.observed_solids import WorldSolid
from emet.motion.observed_support_contact import ObservedSupportContact
from emet.motion.placement_geometry import (
    OBSERVED_SCENE_SOURCES,
    HeldObject,
    PlacementCollisionChecker,
    PlacementScene,
    box_corners,
)


@dataclass(frozen=True)
class PayloadGeometryEvidence:
    """One coherent geometry/pose snapshot in a provider-defined reference frame.

    ``vertices_reference`` represents the planning payload volume.
    ``approximate_observed_box`` explicitly assumes unseen extent from a padded
    measured box; it is not a certified enclosure. ``known_model_prior`` is an explicit prior tier; ``multiview_bound``
    requires a separately qualified multi-view constructor. These labels record
    provenance and do not prove that the producer built a valid bound.

    Vertices describe canonical material in the reference frame. Object pose
    error stays separate. Attachment errors are per-capture EE-relative bounds;
    their metre value already includes angular displacement at the declared
    lever limit. Transport reserves two captures; retention adds before/current.
    """
    object_ref: str
    reference_id: str
    revision: str
    vertices_reference: np.ndarray
    pose: ObjectPoseEvidence | ShapePoseEvidence
    construction: str
    position_error_m: float = 0.0
    rotation_error_rad: float = 0.0
    attachment_error_m: float = 0.0
    attachment_rotation_error_rad: float = 0.0
    attachment_lever_limit_m: float | None = None
    # Optional robot pose from the SAME qualified capture as the object pose.
    # Runtime retention must not mix camera geometry with newer servo feedback.
    captured_world_from_ee: np.ndarray | None = None

    def __post_init__(self):
        # HeldObject validates finite positive extent; its EE naming is irrelevant
        # here because these vertices are still in the declared reference frame.
        vertices = HeldObject(self.vertices_reference).vertices_ee
        if np.linalg.matrix_rank(vertices - vertices.mean(0), tol=1e-10) < 3:
            raise ValueError('invalid_payload_geometry_evidence')
        if (not self.object_ref or not self.reference_id or not self.revision
                or not isinstance(self.pose, (ObjectPoseEvidence, ShapePoseEvidence))
                or self.construction not in {'known_model_prior', 'multiview_bound', 'approximate_observed_box'}
                or not np.isfinite([self.position_error_m, self.rotation_error_rad]).all()
                or self.position_error_m < 0 or not 0 <= self.rotation_error_rad <= np.pi
                or not np.isfinite([self.attachment_error_m, self.attachment_rotation_error_rad]).all()
                or self.attachment_error_m < 0 or self.attachment_rotation_error_rad < 0):
            raise ValueError('invalid_payload_geometry_evidence')
        if self.attachment_lever_limit_m is not None and (
                not np.isfinite(self.attachment_lever_limit_m) or self.attachment_lever_limit_m <= 0):
            raise ValueError('invalid_payload_geometry_evidence')
        if self.captured_world_from_ee is not None:
            captured = ObjectPoseEvidence(self.captured_world_from_ee, self.pose.received_monotonic,
                                          self.pose.source, self.pose.observation_id).world_from_object
            object.__setattr__(self, 'captured_world_from_ee', captured)
        # Shape records already carry their calibration bounds. Unique rigid
        # poses receive the enclosing visual frame's bounds at the adapter seam.
        object.__setattr__(self, 'position_error_m', max(self.position_error_m,
            self.pose.position_error_m if isinstance(self.pose, ShapePoseEvidence) else 0.))
        object.__setattr__(self, 'rotation_error_rad', max(self.rotation_error_rad,
            self.pose.rotation_error_rad if isinstance(self.pose, ShapePoseEvidence) else 0.))
        vertices = np.frombuffer(vertices.tobytes(), dtype=vertices.dtype).reshape(-1, 3)
        if isinstance(self.pose, ShapePoseEvidence):
            if self.construction != 'known_model_prior':
                raise ValueError('shape_pose_requires_complete_model_prior')
            try:
                planes = ConvexHull(vertices).equations
            except QhullError as error:
                raise ValueError('invalid_payload_shape_envelope') from error
            corners = self.pose.model.vertices_reference
            if np.any(corners @ planes[:, :3].T + planes[:, 3] > 1e-9):
                raise ValueError('payload_does_not_enclose_declared_shape')
        object.__setattr__(self, 'vertices_reference', vertices)

    def require(self, object_ref, *, now=None):
        if object_ref != self.object_ref:
            raise ObservationUnavailable('payload_identity_mismatch')
        return (self.pose.require_shape(now=now) if isinstance(self.pose, ShapePoseEvidence)
                else self.pose.require('observed', now=now))


class ObservedPayloadProvider(Protocol):
    input_mode: str

    def observe_payload(self, object_ref: str) -> PayloadGeometryEvidence | None:
        """Return one coherent measured pose and conservative geometry, or unknown."""
        ...


_SAME_SUPPORT = object()
_PICKUP_PHASES = frozenset({'pregrasp', 'grasp', 'close', 'lift'})


class ObservedNativeCollisionChecker(PlacementCollisionChecker):
    """Native attachment seam with no simulator object-body dependency.

    Model geoms must belong to the robot. Scene freshness and revision are checked
    for every sampled configuration, including the executor's dense revalidation.
    The known-free callback must cover entire query volumes, including calibrated
    robot space. It must not equate absent occupied points with observed free space.

    A raw self-occluding depth map cannot satisfy this contract for the robot's
    starting volume: self surfaces are occupied and their interiors are unknown.
    Removing robot-labelled endpoints leaves unknown cells, not certified free
    space. Optional captured-material admission proves only calibrated solids
    and observed-free newly swept space. Contact policy and live source assembly
    are separately required; synthetic free fixtures only exercise the seam.
    """
    geometry_source = 'observed_voxels'
    input_mode = 'observed'

    def __init__(self, model, *, robot_body, ee_body, scene: PlacementScene,
                 payload_provider: ObservedPayloadProvider, scene_revision: str,
                 current_revision, received_monotonic: float, contact_bodies=(),
                 margin_m=.005, max_age_s=2., clock=time.monotonic, admission=None, target_contact=None,
                 support_contact=None, source_support_contact=_SAME_SUPPORT):
        if (scene.source not in OBSERVED_SCENE_SOURCES or scene.workspace is None
                or not callable(scene.known_free)):
            raise ValueError('observed_known_free_workspace_required')
        if payload_provider.input_mode != 'observed':
            raise ValueError('observed_payload_provider_required')
        if (not scene_revision or not callable(current_revision)
                or not np.isfinite(received_monotonic) or not np.isfinite(max_age_s) or max_age_s <= 0):
            raise ValueError('invalid_observed_scene_evidence')
        super().__init__(model, robot_body=robot_body, ee_body=ee_body, scene=scene,
                         payload=None, contact_bodies=contact_bodies, margin_m=margin_m)
        if any(int(model.geom_bodyid[g]) not in self.robot_ids for g in range(model.ngeom)):
            raise ValueError('observed_robot_only_model_required')
        self.geometry_source = scene.source
        self.provider = payload_provider
        self.scene_revision, self.current_revision = scene_revision, current_revision
        self.received_monotonic, self.max_age_s, self.clock = received_monotonic, max_age_s, clock
        self.payload_body = self.payload_parent = self.payload_transform = None
        self._attachment_evidence = None
        self.last_contacts = []
        self._released_target = None
        if admission is not None and not isinstance(admission, ObservedMaterialAdmission):
            raise ValueError('invalid_observed_admission')
        self.admission = admission
        self.last_admission = None
        self.last_blocked_geom = None
        self._stationary_geometry = {}
        self.local_scene = None
        self.simulation_clearance_slack_m = 0.0
        self.simulation_contact_terminated_close = False
        self.simulation_local_payload_contact = False
        self.simulation_support_contact_tolerance_m = 0.
        self._planning_gripper_contact_corridor = False
        self._starting_volumes = ()
        self.active_departure = None
        self.contact_clearance = None
        self.allow_contact_departure = False
        self.departure_options = {}
        self.recovery_guides = ()
        self.last_departure_diagnostics = None
        self._certify_live_clock = None
        self.measured_joints = None
        self._bound_query_cache = None
        self._signature = model_signature(model)
        self.target_contact = None
        self.target_approach_geom_ids = frozenset()
        self.phase = 'preflight'
        self.set_target_contact(target_contact)
        self._support_departure_clearance = None
        self._released_object_ref = None
        self._released_evidence = None
        self._support_payload_error_m = 0.
        self._payload_pose_error_m = 0.
        self.support_contact = None
        self._source_support_contact = _SAME_SUPPORT
        self._certifying_transfer = False
        self.probe_band = None
        self.probe_contact_verified = False
        self.set_support_contact(support_contact)
        self.set_source_support_contact(source_support_contact)

    def _require_support_contact(self, contact):
        if contact is not None:
            if not isinstance(contact, ObservedSupportContact) or self.admission is None:
                raise ValueError('invalid_observed_support_contact')
            contact.require(self.admission.background, now=self.clock(), static_assumption=self.admission.static_assumption)

    def set_support_contact(self, contact):
        self._require_support_contact(contact)
        self.support_contact = contact

    def set_source_support_contact(self, contact):
        if contact is not _SAME_SUPPORT:
            self._require_support_contact(contact)
        self._source_support_contact = contact

    @property
    def source_support_contact(self):
        if self._source_support_contact is _SAME_SUPPORT:
            return self.support_contact
        return self._source_support_contact

    def _support_for_phase(self, phase):
        if phase in _PICKUP_PHASES:
            return self.source_support_contact
        return self.support_contact

    def set_target_contact(self, contact):
        if contact is not None:
            if not isinstance(contact, ObservedTargetContact):
                raise ValueError('invalid_observed_target_contact')
            contact.require(now=self.clock(), model_signature=self._signature)
            if not contact.fingertip_geom_ids <= set(self.geom_ids):
                raise ValueError('invalid_observed_fingertip_geoms')
        self.target_contact = contact

    @contextmanager
    def motion_phase(self, phase):
        if phase not in MOTION_PHASES:
            raise ValueError('invalid_contact_phase')
        previous, self.phase = self.phase, phase
        try:
            yield
        finally:
            self.phase = previous

    @contextmanager
    def planning_snapshot(self):
        """Plan against one measured snapshot; execution still requires fresh input.

        Retain the real clock for payload evidence. This does not change the
        stored capture time, certify contact, or permit issuing a command.
        """
        live_clock = self.clock
        live_revision = self.current_revision
        previous_live_clock = self._certify_live_clock
        if self.local_scene is not None:
            self._certify_live_clock = previous_live_clock or live_clock
            self.clock = lambda: self.received_monotonic
            self.current_revision = lambda: self.scene_revision
        try:
            yield
        finally:
            self.clock = live_clock
            self.current_revision = live_revision
            self._certify_live_clock = previous_live_clock

    @contextmanager
    def certify_transfer(self):
        """Keep checking the still-measured pickup during a hypothetical release."""
        previous, self._certifying_transfer = self._certifying_transfer, True
        try:
            with self.planning_snapshot():
                yield
        finally:
            self._certifying_transfer = previous

    def _robot_bound(self, model, data, gid):
        cache = self._bound_query_cache
        key = (gid, data.geom_xpos[gid].tobytes(), data.geom_xmat[gid].tobytes())
        if cache is not None and key in cache:
            return cache[key]
        nominal = WorldSolid.from_geom(model, data, gid,
            convex_material_calibration_id='robot-model-collision-envelope')
        if self.admission is None:
            result = nominal.with_offset(np.sqrt(3)*self.margin), 0.
            if cache is not None:
                cache[key] = result
            return result
        bound = self.admission.query_robot_geometry(model, data, gid)
        material = self.admission.robot_material
        if (not isinstance(bound, RobotGeometryBound) or bound.geom_id != gid
                or bound.model_signature != self._signature or bound.calibration_id != material.calibration_id
                or bound.configuration_profile_id != material.configuration_profile_id):
            raise ObservationUnavailable('robot_collision_envelope_unknown')
        reserve = np.sqrt(3)*self.margin + bound.displacement_bound_m
        if bound.solid.offset_m < reserve or not bound.solid.contains_solid(nominal.with_offset(reserve)):
            raise ObservationUnavailable('robot_collision_envelope_unknown')
        result = bound.solid, bound.displacement_bound_m
        if cache is not None:
            cache[key] = result
        return result

    def _target_robot_collides(self, model, data, gid, target=None):
        contact = self.target_contact
        if (contact.approximate_volume and self.simulation_contact_terminated_close
                and self.phase in {'grasp', 'close'}
                and gid in (contact.fingertip_geom_ids | self.target_approach_geom_ids)):
            # A padded partial-view box is a task estimate, not solid object
            # material. Explicit contact-terminated simulation may approach it
            # with the named hand. Surrounding measured surfaces are still
            # checked by _robot_scene_collides; arm links keep target checks.
            return False
        if (gid in (contact.fingertip_geom_ids | self.target_approach_geom_ids) and self.phase in CONTACT_PHASES
                and getattr(self, 'simulation_contact_terminated_close', False)):
            if gid in contact.fingertip_geom_ids and self._planning_gripper_contact_corridor and self.phase == 'close':
                return False  # Motor endpoint is not a reachable material state.
            # Simulation-only contact hypothesis. Keep full target uncertainty
            # for environment planning, but permit named pads to find its nominal
            # surface. Housing still cannot enter nominal material; unrelated
            # geometry retains the complete target uncertainty envelope.
            target = (contact.target if target is None else target).with_offset(0.)
        solid, _ = self._robot_bound(model, data, gid)
        if gid in (contact.fingertip_geom_ids | self.target_approach_geom_ids) and self.phase in CONTACT_PHASES:
            # Gripper approach clearance is not physical penetration. Keep all
            # model/pose error; only named pads may enter the target's shell.
            # Housing geometry must still remain outside the target envelope.
            solid = solid.with_offset(solid.offset_m-np.sqrt(3)*self.margin)
        collides = contact.collides(solid, geom_id=gid, phase=self.phase, target=target)
        if collides and model is not None:
            self.last_blocked_geom = model.geom(gid).name
        return collides

    @contextmanager
    def motion_scope(self, model, data, joint_names, q_start):
        """Bind fixed links to their measured pose for one joint-space query.

        Fixed links retain their existing environmental contact. Moving links
        still check the environment and all robot links through self collision.
        """
        previous = self._stationary_geometry
        previous_volumes = self._starting_volumes
        joint_ids = {model.joint(name).id for name in joint_names}
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, data)
        for name, value in zip(joint_names, q_start, strict=True):
            scratch.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_kinematics(model, scratch)
        stationary = {}
        for gid in self.geom_ids:
            body = int(model.geom_bodyid[gid])
            affected = False
            while body:
                first = int(model.body_jntadr[body])
                count = int(model.body_jntnum[body])
                affected |= bool(joint_ids.intersection(range(first, first + count)))
                body = int(model.body_parentid[body])
            if not affected:
                stationary[gid] = (scratch.geom_xpos[gid].copy(), scratch.geom_xmat[gid].copy())
        if self.measured_joints is not None:
            if np.asarray(self.measured_joints).shape != (model.nq,):
                raise ObservationUnavailable('invalid_joint_feedback')
            scratch.qpos[:] = self.measured_joints
            mujoco.mj_kinematics(model, scratch)
        starting_volumes = tuple(
            self._robot_bound(model, scratch, gid)[0] for gid in self.geom_ids
        ) if self.local_scene is not None else ()
        self._stationary_geometry, self._starting_volumes = stationary, starting_volumes
        try:
            yield
        finally:
            self._stationary_geometry = previous
            self._starting_volumes = previous_volumes

    def make_departure(self, model, data, joint_names, q_start, *, clearance_only=False):
        """Build a measured local-contact baseline for an optional escape search."""
        if (not self.allow_contact_departure or self.local_scene is None or self.payload is not None
                or self.contact_clearance is not None):
            return None
        from emet.motion.contact_departure import ContactDeparture, DepartureMeasurement
        from emet.motion.surface_contact_patches import SurfaceContactPatches

        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, data)
        addresses = [int(model.joint(name).qposadr[0]) for name in joint_names]
        local_scene = self.local_scene
        starting_volumes = tuple(replace(volume, offset_m=volume.offset_m + .001)
                                 for volume in self._starting_volumes)
        moving = [gid for gid in self.geom_ids if gid not in self._stationary_geometry]
        last_query = [None]
        patches = {}
        self.last_departure_diagnostics = None

        def query(q):
            previous_cache = self._bound_query_cache
            self._bound_query_cache = {}
            try:
                return geometry_query(q)
            finally:
                self._bound_query_cache = previous_cache

        def geometry_query(q):
            # Runtime revalidation must use the latest measured map. Cell array
            # indices are not identities: fusion can insert/reorder cells.
            current_scene = self.local_scene
            if current_scene is None or current_scene.resolution != local_scene.resolution or current_scene.error != local_scene.error:
                raise ObservationUnavailable('contact_departure_map_profile_changed')
            scratch.qpos[addresses] = q
            mujoco.mj_kinematics(model, scratch)
            distances, physical_distances, samples, known = {}, {}, [], True
            for gid in moving:
                solid, _ = self._robot_bound(model, scratch, gid)
                if hasattr(current_scene, 'nearby_surfaces'):
                    points, estimated = current_scene.nearby_surfaces(solid)
                else:
                    # Compatibility for evidence adapters supplying their own
                    # local query implementation; no occupied-cell volume is
                    # interpreted as nominal contact.
                    ids = list(current_scene.nearby_distances(solid))
                    points = getattr(current_scene, 'surface_points', current_scene.centers)[ids]
                    estimated = cell_separations(solid.with_offset(0.), points, resolution=1e-6)
                if len(points):
                    if gid not in patches:
                        patches[gid] = SurfaceContactPatches(
                            points, getattr(current_scene, 'surface_points', current_scene.centers),
                            surface_tree=getattr(current_scene, 'surface_tree', None),
                            normal_band_m=min(.01, max(.001, 2*current_scene.error)),
                        )
                    membership = patches[gid].assign(points)
                    for point, distance, patch in zip(points, estimated, membership, strict=True):
                        key = (('environment_patch', gid, int(patch)) if patch >= 0 else
                               ('environment_surface', gid, tuple(np.round(point, 6))))
                        # Min over a patch lets the contacting sample change as
                        # the link slides. Baseline patch identity stays fixed.
                        physical_distances[key] = min(physical_distances.get(key, np.inf), float(distance))
                        padded = float(distance)-solid.offset_m-current_scene.error
                        distances[key] = min(distances.get(key, np.inf), padded)
                if gid not in patches:
                    patches[gid] = SurfaceContactPatches([], [])
                known &= current_scene.query(solid, starting_volumes=starting_volumes, check_occupied=False).allowed
                samples.append(box_corners(solid.bounds))
                if self.target_contact is not None:
                    known &= not self._target_robot_collides(model, scratch, gid)
            mujoco.mj_collision(model, scratch)
            for index in range(scratch.ncon):
                contact = scratch.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                if first in moving or second in moving:
                    distances[('self', min(first, second), max(first, second))] = float(contact.dist)
                    physical_distances[('self', min(first, second), max(first, second))] = float(contact.dist)
            value = DepartureMeasurement(distances, np.concatenate(samples), bool(known), physical_distances)
            last_query[0] = value
            return value

        try:
            policy = ContactDeparture(q_start, query, joint_names=joint_names,
                                      require_initial_contact=not clearance_only, **self.departure_options)
            policy.guide_configurations = self.recovery_guides

            @contextmanager
            def clearance_scope():
                previous = self.contact_clearance
                self.contact_clearance = (patches, local_scene.resolution, local_scene.error,
                                          policy.new_contact_tolerance)
                try:
                    yield
                finally:
                    self.contact_clearance = previous

            policy.clearance_scope = clearance_scope

            def describe(key):
                if key[0] == 'environment_patch':
                    return {'geometry': model.geom(key[1]).name,
                            'point_m': patches[key[1]].anchors[key[2]].tolist(), 'contact_patch': key[2]}
                if key[0] == 'environment_surface':
                    return {'geometry': model.geom(key[1]).name, 'point_m': list(key[2])}
                return {'geometry_pair': [model.geom(gid).name for gid in key[1:]]}

            policy.describe_pair = describe
            physical = policy.baseline.physical_distances or policy.baseline.distances
            self.last_departure_diagnostics = {'contact_representation': 'measured_surface_patches', 'initial_contacts': [
                {**describe(key), 'nominal_separation_m': physical[key],
                 'padded_separation_m': policy.baseline.distances[key]}
                for key in sorted(physical, key=physical.get)[:4]
            ]}
            return policy
        except ValueError as error:
            value = last_query[0]
            if value is not None:
                physical = value.physical_distances or value.distances
                key = min(physical, key=physical.get) if physical else None
                self.last_departure_diagnostics = {
                    'reason': str(error), 'minimum_nominal_surface_separation_m': physical.get(key) if key else None,
                    'new_sweep_known': value.newly_swept_space_known,
                    'voxel_size_m': local_scene.resolution,
                }
                if key and key[0] == 'environment_patch':
                    self.last_departure_diagnostics.update(geometry=model.geom(key[1]).name,
                        point_m=patches[key[1]].anchors[key[2]].tolist())
                elif key and key[0] == 'environment_surface':
                    self.last_departure_diagnostics.update(geometry=model.geom(key[1]).name, point_m=list(key[2]))
            return None

    def _robot_scene_collides(self, model, data, gid, bounds):
        if getattr(self, 'simulation_ignore_environment', False):
            return False
        # After release, check the fresh visual target as well as the retained
        # release hypothesis. A rolling object must not disappear behind the
        # frozen release box while the arm retreats.
        if self.target_contact is not None and self.payload is None:
            if self._target_robot_collides(model, data, gid):
                self.last_blocked_geom = model.geom(gid).name
                return True
        fixed = self._stationary_geometry.get(gid)
        if fixed is not None and np.allclose(data.geom_xpos[gid], fixed[0], rtol=0, atol=1e-9) and np.allclose(
            data.geom_xmat[gid], fixed[1], rtol=0, atol=1e-9
        ):
            return False
        if self.admission is None:
            return super()._robot_scene_collides(model, data, gid, bounds)
        solid, model_error = self._robot_bound(model, data, gid)
        if self.local_scene is not None:
            # Explicit simulation diagnostic: reduce planning reserve only.
            # Keep model-displacement and sensor uncertainty, and all obstacles.
            solid = solid.with_offset(max(model_error, solid.offset_m-self.simulation_clearance_slack_m))
            if getattr(self, 'contact_clearance', None) is not None:
                patches, resolution, error, tolerance = self.contact_clearance
                if self.local_scene.resolution != resolution or self.local_scene.error != error:
                    raise ObservationUnavailable('contact_departure_map_profile_changed')
                patch = patches.get(gid)
            else:
                patch = None
            if patch is not None:
                self.last_admission = self.local_scene.query_contact_clearance(
                    solid, patch, contact_tolerance_m=tolerance, starting_volumes=self._starting_volumes,
                    minimum_padding_m=model_error)
            else:
                self.last_admission = self.local_scene.query(solid, starting_volumes=self._starting_volumes)
            if not self.last_admission.allowed:
                self.last_blocked_geom = model.geom(gid).name
            return not self.last_admission.allowed
        self.last_admission = self.admission.query(solid, now=self.clock(), allow_captured_robot=True,
                                                 free_witness=self._support_for_phase(self.phase))
        if not self.last_admission.allowed:
            self.last_blocked_geom = model.geom(gid).name
        return not self.last_admission.allowed

    def _payload_robot_collides(self, model, data, gid, bounds, payload_bounds):
        if getattr(self, 'simulation_ignore_environment', False):
            return False
        if self.target_contact is None:
            expanded = payload_bounds + np.array([[-1.], [1.]])*self._payload_pose_error_m
            return super()._payload_robot_collides(model, data, gid, bounds, expanded)
        if self.payload is None:
            target = self._released_target
            if target is None:
                pose = np.eye(4)
                pose[:3, 3] = payload_bounds.mean(0)
                target = WorldSolid('box', tuple((payload_bounds[1]-payload_bounds[0])/2), pose)
        else:
            ee = data.body(self.ee_body)
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = ee.xmat.reshape(3, 3), ee.xpos
            target = WorldSolid('convex', (), pose, vertices=self.payload.vertices_ee,
                                offset_m=self._payload_pose_error_m,
                                material_calibration_id='observed-payload-envelope')
        return self._target_robot_collides(model, data, gid, target)

    def _payload_scene_collides(self, model, data, bounds):
        if getattr(self, 'simulation_ignore_environment', False):
            return False
        if self.local_scene is not None and self.simulation_local_payload_contact:
            return self._local_payload_scene_collides(model, data)
        if self.admission is None:
            return super()._payload_scene_collides(model, data,
                bounds + np.array([[-1.], [1.]])*self._payload_pose_error_m)
        pose = np.eye(4)
        pose[:3, 3] = bounds.mean(0)
        solid = WorldSolid('box', tuple((bounds[1]-bounds[0])/2), pose)
        contact = self._support_for_phase(self.phase if self.payload is not None else 'release')
        if contact is None:
            self.last_admission = self.admission.query(replace(solid, offset_m=self._payload_pose_error_m),
                now=self.clock(), allow_captured_robot=True)
        else:
            if self.payload is not None:
                ee = data.body(self.ee_body)
                vertices = self.payload.world_vertices(ee.xpos, ee.xmat)
                ref, phase = self.payload_body, self.phase
            else:
                if self._released_target is None or self._released_object_ref is None:
                    raise ObservationUnavailable('payload_geometry_unknown')
                vertices = self._released_target.vertices
                ref, phase = self._released_object_ref, 'release'
            error = self._payload_pose_error_m + contact.payload_motion_error_m
            self.last_admission = self._query_supported_payload(vertices, bounds, error, ref, phase, contact)
            if self.last_admission.allowed and self.payload is None:
                # A frozen release hypothesis cannot hide a subsequently rolling
                # or displaced object. Check current measured geometry as well.
                evidence, observed_pose = self._evidence(ref)
                current = evidence.vertices_reference @ observed_pose[:3, :3].T + observed_pose[:3, 3]
                current_bounds = np.stack((current.min(0)-self.margin, current.max(0)+self.margin))
                current_contact = self.source_support_contact if self._certifying_transfer else contact
                if current_contact is None:
                    raise ObservationUnavailable('support_geometry_unknown')
                self.last_admission = self._query_supported_payload(current, current_bounds,
                    self._pose_error(evidence) + current_contact.payload_motion_error_m, ref, 'release',
                    current_contact)
        return not self.last_admission.allowed

    def _local_payload_scene_collides(self, model, data):
        """Simulation diagnostic using measured local surfaces for the payload.

        Only a named, measured support return may be contacted. The nominal
        payload cannot enter it beyond the contact cap; uncertainty remains in
        the envelope used against every other obstacle.
        """
        if self.payload is not None:
            ee = data.body(self.ee_body)
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = ee.xmat.reshape(3, 3), ee.xpos
            physical = WorldSolid('convex', (), pose, vertices=self.payload.vertices_ee,
                                  material_calibration_id='observed-payload-envelope')
            candidates = [(physical, self._support_for_phase(self.phase), self.phase)]
        else:
            if self._released_target is None or self._released_object_ref is None:
                raise ObservationUnavailable('payload_geometry_unknown')
            candidates = [(self._released_target.with_offset(0.), self.support_contact, 'release')]
            evidence, observed_pose = self._evidence(self._released_object_ref)
            current = WorldSolid('convex', (), observed_pose, vertices=evidence.vertices_reference,
                                material_calibration_id='observed-payload-envelope')
            contact = self.source_support_contact if self._certifying_transfer else self.support_contact
            candidates.append((current, contact, 'lift' if self._certifying_transfer else 'release'))
        for physical, contact, phase in candidates:
            permit = None
            if contact is not None and phase in {'lift', 'place', 'release'}:
                contact.require(self.admission.background, now=self.clock(), static_assumption=self.admission.static_assumption)
                clearance = contact.physical_clearance(physical)
                cap = contact.max_penetration_m
                if self.simulation_support_contact_tolerance_m:
                    # The explicit simulator contact cap limits nominal
                    # material penetration, not plane uncertainty or AABB
                    # padding. Keep uncertainty against every other obstacle.
                    normal, offset = contact.unit_plane
                    if physical.kind == 'convex':
                        vertices = physical.vertices @ physical.world_from_local[:3, :3].T + physical.world_from_local[:3, 3]
                    else:
                        vertices = box_corners(physical.bounds)
                    clearance = float(np.min(vertices @ normal-offset))
                    cap = self.simulation_support_contact_tolerance_m
                if clearance < -cap - 1e-9:
                    self.last_reason = 'payload_support_penetration'
                    return True

                def permit(point, radius, support=contact):
                    normal, offset = support.unit_plane
                    margin = radius + support.patch.measurement_margin_m
                    if abs(point @ normal - offset) > margin:
                        return False
                    local = (point - support.world_from_camera[:3, 3]) @ support.world_from_camera[:3, :3]
                    if local[2] <= 0:
                        return False
                    uv = support.intrinsic @ local
                    x, y = np.rint(uv[:2] / uv[2]).astype(int)
                    height, width = support.depth.shape
                    if (0 <= x < width and 0 <= y < height and support.support_mask[y, x]
                            and abs(local[2] - support.depth[y, x]) <= margin):
                        return True
                    # Another measured camera may see counter pixels occluded
                    # by the payload in the selected support view. Associate
                    # only a bounded continuation of that original plane;
                    # do not require identical pixels across camera views.
                    return support.measured_surface_distance(point) <= .03 + radius
            padded = physical.with_offset(self._payload_pose_error_m + self.margin)
            self.last_admission = self.local_scene.query(padded, allow_occupied=permit)
            if not self.last_admission.allowed:
                return True
        return False

    def _query_supported_payload(self, vertices, bounds, error, ref, phase, contact):
        lo, hi = vertices.min(0), vertices.max(0)
        pose, padded_pose = np.eye(4), np.eye(4)
        pose[:3, 3], padded_pose[:3, 3] = (lo+hi)/2, bounds.mean(0)
        physical = WorldSolid('box', tuple((hi-lo)/2+error), pose)
        padded = WorldSolid('box', tuple((bounds[1]-bounds[0])/2+error), padded_pose)
        if self.probe_band is not None and self.probe_band.support.profile_identity == contact.profile_identity and (
            phase == "probe" or phase == "release" and self.probe_contact_verified
        ):
            return self.admission.query_probe_payload(
                physical, padded, band=self.probe_band, object_ref=ref, now=self.clock(),
            )
        return self.admission.query_payload(physical, padded, contact=contact,
            object_ref=ref, phase=phase, now=self.clock(), departure_clearance_m=self._support_departure_clearance)

    @staticmethod
    def _pose_error(evidence):
        from emet.controller.manipulation.observed_effects import pose_uncertainty_margin

        # The attachment frame and the queried EE each contribute once.
        return (pose_uncertainty_margin(np.linalg.norm(evidence.vertices_reference, axis=1).max(),
                                       evidence.position_error_m, evidence.rotation_error_rad)
                + 2*evidence.attachment_error_m)

    def _evidence(self, object_ref):
        evidence = self.provider.observe_payload(object_ref)
        if evidence is None:
            raise ObservationUnavailable('payload_geometry_unknown')
        evidence_clock = self._certify_live_clock or self.clock
        pose = evidence.require(object_ref, now=evidence_clock())
        previous = self._attachment_evidence or self._released_evidence
        if previous is not None and (previous.object_ref != evidence.object_ref
                or previous.reference_id != evidence.reference_id
                or previous.revision != evidence.revision
                or previous.construction != evidence.construction
                or evidence.position_error_m > previous.position_error_m
                or evidence.rotation_error_rad > previous.rotation_error_rad
                or evidence.attachment_error_m > previous.attachment_error_m
                or evidence.attachment_rotation_error_rad > previous.attachment_rotation_error_rad
                or evidence.attachment_lever_limit_m != previous.attachment_lever_limit_m
                or shape_identity(previous.pose) != shape_identity(evidence.pose)
                or not np.array_equal(previous.vertices_reference, evidence.vertices_reference)):
            raise ObservationUnavailable('payload_geometry_changed')
        return evidence, pose

    def payload_retention_residual(self, model, data, object_ref):
        """Compare attached geometry, quotienting only certified primitive symmetry."""
        if self.payload_transform is None or self._attachment_evidence is None:
            raise ObservationUnavailable('payload_geometry_unknown')
        evidence, pose = self._evidence(object_ref)
        mujoco.mj_copyData(self._probe, model, data)
        mujoco.mj_kinematics(model, self._probe)
        ee = self._probe.body(self.ee_body)
        captured = evidence.captured_world_from_ee
        ee_rotation = ee.xmat.reshape(3, 3) if captured is None else captured[:3, :3]
        ee_position = ee.xpos if captured is None else captured[:3, 3]
        inverse = ee_rotation.T
        position = inverse @ (pose[:3, 3]-ee_position)
        rotation = inverse @ pose[:3, :3]
        current_vertices = evidence.vertices_reference @ rotation.T + position
        if (evidence.attachment_lever_limit_m is not None
                and np.linalg.norm(current_vertices, axis=1).max() > evidence.attachment_lever_limit_m):
            raise ObservationUnavailable('payload_attachment_outside_calibration')
        distance = float(np.linalg.norm(position-self.payload_transform[:3, 3]))
        angle = evidence_rotation_distance(evidence.pose, self.payload_transform[:3, :3], rotation)
        before = self._attachment_evidence
        distance += before.position_error_m + evidence.position_error_m + before.attachment_error_m + evidence.attachment_error_m
        angle += (before.rotation_error_rad + evidence.rotation_error_rad
                  + before.attachment_rotation_error_rad + evidence.attachment_rotation_error_rad)
        return distance, angle

    def observe_object(self, object_ref):
        evidence, _ = self._evidence(object_ref)
        return evidence.pose

    def observed_geometry_pose(self, object_ref):
        """Geometry representative for explicitly shape-aware native consumers."""
        _, pose = self._evidence(object_ref)
        return pose

    def possible_payload_bounds(self, model, data):
        """Physical payload enclosure including uncertainty, without clearance."""
        if model is not self._model or self.payload is None:
            raise ObservationUnavailable('payload_geometry_unknown')
        ee = data.body(self.ee_body)
        vertices = self.payload.world_vertices(ee.xpos, ee.xmat)
        return np.stack((vertices.min(0)-self._payload_pose_error_m,
                         vertices.max(0)+self._payload_pose_error_m))

    def set_payload(self, model, data, body, parent=None, *, capture_aligned=False):
        """``body`` is an opaque observed handle here, never a MuJoCo body name."""
        if model is not self._model:
            raise ValueError('Collision checker model mismatch')
        if body is None:
            # Retain the released object's possible volume for retreat planning.
            # The next scene revision must replace this with observed geometry.
            if self.payload is not None:
                self._released_object_ref = self.payload_body
                if self.support_contact is not None:
                    self._released_evidence = self._attachment_evidence
                mujoco.mj_copyData(self._probe, model, data)
                mujoco.mj_kinematics(model, self._probe)
                ee = self._probe.body(self.ee_body)
                vertices = self.payload.world_vertices(ee.xpos, ee.xmat)
                self.released_bounds = np.stack((vertices.min(0) - self.margin, vertices.max(0) + self.margin))
                self._released_target = WorldSolid('convex', (), np.eye(4), vertices=vertices,
                    offset_m=self._payload_pose_error_m,
                    material_calibration_id='observed-released-envelope')
            self.payload = None
            self.payload_body = self.payload_parent = self.payload_transform = None
            self._attachment_evidence = None
            self._support_departure_clearance = None
            return
        if parent != self.ee_body:
            raise ValueError('payload_parent_mismatch')
        same_attachment = self.payload_body == body and self._attachment_evidence is not None
        previous_floor = self._support_departure_clearance if same_attachment else None
        previous_error = self._support_payload_error_m if same_attachment else 0.
        previous_pose_error = self._payload_pose_error_m if same_attachment else 0.
        evidence, world_from_reference = self._evidence(body)
        if any(contact is not None and body != contact.object_ref
               for contact in (self.source_support_contact, self.support_contact)):
            raise ObservationUnavailable('support_geometry_unknown')
        if self.target_contact is not None and (body != self.target_contact.object_ref
                or evidence.reference_id != self.target_contact.reference_id
                or evidence.revision != self.target_contact.geometry_revision):
            raise ObservationUnavailable('target_contact_identity_changed')
        mujoco.mj_copyData(self._probe, model, data)
        mujoco.mj_kinematics(model, self._probe)
        ee = self._probe.body(parent)
        ee_from_reference = np.eye(4)
        captured = evidence.captured_world_from_ee if capture_aligned else None
        rotation = (ee.xmat.reshape(3, 3) if captured is None else captured[:3, :3]).T
        position = ee.xpos if captured is None else captured[:3, 3]
        ee_from_reference[:3, :3] = rotation @ world_from_reference[:3, :3]
        ee_from_reference[:3, 3] = rotation @ (world_from_reference[:3, 3] - position)
        vertices = evidence.vertices_reference @ ee_from_reference[:3, :3].T + ee_from_reference[:3, 3]
        if (evidence.attachment_lever_limit_m is not None
                and np.linalg.norm(vertices, axis=1).max() > evidence.attachment_lever_limit_m):
            raise ObservationUnavailable('payload_attachment_outside_calibration')
        payload = HeldObject(vertices, ee_from_reference)
        self.payload = payload
        self.payload_body, self.payload_parent = body, parent
        self.payload_transform = payload.body_pose_ee
        self._attachment_evidence = evidence
        self._payload_pose_error_m = max(previous_pose_error, self._pose_error(evidence))
        source_contact = self.source_support_contact
        if source_contact is not None:
            self._support_payload_error_m = max(previous_error,
                self._pose_error(evidence) + source_contact.payload_motion_error_m)
            # Measured payload evidence, not the hypothetical IK/RRT pose,
            # anchors departure. Query ordering cannot ratchet this floor down.
            observed = WorldSolid('convex', (), world_from_reference, offset_m=self._support_payload_error_m,
                                 vertices=evidence.vertices_reference,
                                 material_calibration_id='observed-payload-envelope')
            self._support_departure_clearance = (source_contact.physical_clearance(observed)
                                                 if previous_floor is None else previous_floor)
        self.released_bounds = None
        self._released_target = None
        self._released_object_ref = None
        self._released_evidence = None

    def capture_payload(self):
        return (self.payload_body, self.payload_parent, self.payload_transform,
                self.payload, self._attachment_evidence,
                None if self.released_bounds is None else self.released_bounds.copy(), self._released_target, self._released_evidence,
                self._payload_pose_error_m, self._support_departure_clearance, self._released_object_ref, self._support_payload_error_m)

    def restore_payload(self, state):
        (self.payload_body, self.payload_parent, self.payload_transform,
         self.payload, self._attachment_evidence, self.released_bounds, self._released_target, self._released_evidence,
         self._payload_pose_error_m, self._support_departure_clearance, self._released_object_ref, self._support_payload_error_m) = state

    def configuration_collides(self, model, data):
        previous = self._bound_query_cache
        self._bound_query_cache = {}
        try:
            return self._configuration_collides(model, data)
        finally:
            self._bound_query_cache = previous

    def _configuration_collides(self, model, data):
        self.last_contacts = []
        self.last_admission = None
        self.last_blocked_geom = None
        age = self.clock() - self.received_monotonic
        if not np.isfinite(age) or not 0 <= age <= self.max_age_s:
            self.last_reason = 'scene_observation_stale'
            return True
        if self.current_revision() != self.scene_revision:
            self.last_reason = 'scene_revision_changed'
            return True
        if self.target_contact is not None:
            self.target_contact.require(now=self.clock(), model_signature=self._signature)
            if self.target_contact.capture.received_monotonic != self.received_monotonic:
                raise ObservationUnavailable('target_contact_identity_changed')
        if self.admission is not None:
            self.admission.require(now=self.clock(), model_signature=self._signature)
            if (self.admission.scene_geometry_digest != self.scene.geometry_digest
                    or self.admission.robot_material.capture.received_monotonic != self.received_monotonic):
                raise ObservationUnavailable('robot_material_identity_changed')
        if self.active_departure is not None:
            addresses = [int(model.joint(name).qposadr[0]) for name in self.active_departure.joint_names]
            collides = not self.active_departure.configuration_allowed(data.qpos[addresses])
            self.last_reason = 'contact_departure_violation' if collides else None
        else:
            collides = super().configuration_collides(model, data)
        if self.current_revision() != self.scene_revision:
            self.last_reason = 'scene_revision_changed'
            return True
        return collides
