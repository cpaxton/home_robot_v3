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
from emet.motion.observed_admission import ObservedMaterialAdmission, RobotGeometryBound
from emet.motion.observed_contact import CONTACT_PHASES, MOTION_PHASES, ObservedTargetContact
from emet.motion.observed_solids import WorldSolid
from emet.motion.observed_support_contact import ObservedSupportContact
from emet.motion.placement_geometry import HeldObject, PlacementCollisionChecker, PlacementScene


@dataclass(frozen=True)
class PayloadGeometryEvidence:
    """One coherent geometry/pose snapshot in a provider-defined reference frame.

    ``vertices_reference`` encloses the possible object volume, not just visible
    samples. ``known_model_prior`` is an explicit prior tier; ``multiview_bound``
    requires a separately qualified multi-view constructor. These labels record
    provenance and do not prove that the producer built a valid bound.
    """
    object_ref: str
    reference_id: str
    revision: str
    vertices_reference: np.ndarray
    pose: ObjectPoseEvidence | ShapePoseEvidence
    construction: str
    position_error_m: float = 0.0
    rotation_error_rad: float = 0.0

    def __post_init__(self):
        # HeldObject validates finite positive extent; its EE naming is irrelevant
        # here because these vertices are still in the declared reference frame.
        vertices = HeldObject(self.vertices_reference).vertices_ee
        if np.linalg.matrix_rank(vertices - vertices.mean(0), tol=1e-10) < 3:
            raise ValueError('invalid_payload_geometry_evidence')
        if (not self.object_ref or not self.reference_id or not self.revision
                or not isinstance(self.pose, (ObjectPoseEvidence, ShapePoseEvidence))
                or self.construction not in {'known_model_prior', 'multiview_bound'}
                or not np.isfinite([self.position_error_m, self.rotation_error_rad]).all()
                or self.position_error_m < 0 or not 0 <= self.rotation_error_rad <= np.pi):
            raise ValueError('invalid_payload_geometry_evidence')
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
                 support_contact=None):
        if (scene.source != 'observed_voxels' or scene.workspace is None
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
        self._signature = model_signature(model)
        self.target_contact = None
        self.phase = 'preflight'
        self.set_target_contact(target_contact)
        self._support_departure_clearance = None
        self._released_object_ref = None
        self._released_evidence = None
        self._support_payload_error_m = 0.
        self.support_contact = None
        self.set_support_contact(support_contact)

    def set_support_contact(self, contact):
        if contact is not None:
            if not isinstance(contact, ObservedSupportContact) or self.admission is None:
                raise ValueError('invalid_observed_support_contact')
            contact.require(self.admission.background, now=self.clock(), static_assumption=self.admission.static_assumption)
        self.support_contact = contact

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

    def _robot_bound(self, model, data, gid):
        nominal = WorldSolid.from_geom(model, data, gid,
            convex_material_calibration_id='robot-model-collision-envelope')
        if self.admission is None:
            return replace(nominal, offset_m=np.sqrt(3)*self.margin), 0.
        bound = self.admission.query_robot_geometry(model, data, gid)
        material = self.admission.robot_material
        if (not isinstance(bound, RobotGeometryBound) or bound.geom_id != gid
                or bound.model_signature != self._signature or bound.calibration_id != material.calibration_id
                or bound.configuration_profile_id != material.configuration_profile_id):
            raise ObservationUnavailable('robot_collision_envelope_unknown')
        reserve = np.sqrt(3)*self.margin + bound.displacement_bound_m
        if bound.solid.offset_m < reserve or not bound.solid.contains_solid(replace(nominal, offset_m=reserve)):
            raise ObservationUnavailable('robot_collision_envelope_unknown')
        return bound.solid, bound.displacement_bound_m

    def _target_robot_collides(self, model, data, gid, target=None):
        contact = self.target_contact
        solid, _ = self._robot_bound(model, data, gid)
        if gid in contact.fingertip_geom_ids and self.phase in CONTACT_PHASES:
            # Clearance padding is not physical penetration. Keep all declared
            # configuration error, but remove only the ordinary planning margin.
            solid = replace(solid, offset_m=solid.offset_m-np.sqrt(3)*self.margin)
        return contact.collides(solid, geom_id=gid, phase=self.phase, target=target)

    def _robot_scene_collides(self, model, data, gid, bounds):
        # After release, check the fresh visual target as well as the retained
        # release hypothesis. A rolling object must not disappear behind the
        # frozen release box while the arm retreats.
        if self.target_contact is not None and self.payload is None:
            if self._target_robot_collides(model, data, gid):
                return True
        if self.admission is None:
            return super()._robot_scene_collides(model, data, gid, bounds)
        solid, _ = self._robot_bound(model, data, gid)
        self.last_admission = self.admission.query(solid, now=self.clock(), allow_captured_robot=True)
        return not self.last_admission.allowed

    def _payload_robot_collides(self, model, data, gid, bounds, payload_bounds):
        if self.target_contact is None:
            return super()._payload_robot_collides(model, data, gid, bounds, payload_bounds)
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
                                material_calibration_id='observed-payload-envelope')
        return self._target_robot_collides(model, data, gid, target)

    def _payload_scene_collides(self, model, data, bounds):
        if self.admission is None:
            return super()._payload_scene_collides(model, data, bounds)
        pose = np.eye(4)
        pose[:3, 3] = bounds.mean(0)
        solid = WorldSolid('box', tuple((bounds[1]-bounds[0])/2), pose)
        if self.support_contact is None:
            self.last_admission = self.admission.query(solid, now=self.clock(), allow_captured_robot=True)
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
            self.last_admission = self._query_supported_payload(vertices, bounds, self._support_payload_error_m, ref, phase)
            if self.last_admission.allowed and self.payload is None:
                # A frozen release hypothesis cannot hide a subsequently rolling
                # or displaced object. Check current measured geometry as well.
                evidence, observed_pose = self._evidence(ref)
                current = evidence.vertices_reference @ observed_pose[:3, :3].T + observed_pose[:3, 3]
                current_bounds = np.stack((current.min(0)-self.margin, current.max(0)+self.margin))
                self.last_admission = self._query_supported_payload(current, current_bounds,
                    self._support_pose_error(evidence), ref, 'release')
        return not self.last_admission.allowed

    def _query_supported_payload(self, vertices, bounds, error, ref, phase):
        lo, hi = vertices.min(0), vertices.max(0)
        pose, padded_pose = np.eye(4), np.eye(4)
        pose[:3, 3], padded_pose[:3, 3] = (lo+hi)/2, bounds.mean(0)
        physical = WorldSolid('box', tuple((hi-lo)/2+error), pose)
        padded = WorldSolid('box', tuple((bounds[1]-bounds[0])/2+error), padded_pose)
        return self.admission.query_payload(physical, padded, contact=self.support_contact,
            object_ref=ref, phase=phase, now=self.clock(), departure_clearance_m=self._support_departure_clearance)

    def _support_pose_error(self, evidence):
        from emet.controller.manipulation.observed_effects import pose_uncertainty_margin

        return (pose_uncertainty_margin(np.linalg.norm(evidence.vertices_reference, axis=1).max(),
                                       evidence.position_error_m, evidence.rotation_error_rad)
                + self.support_contact.payload_motion_error_m)

    def _evidence(self, object_ref):
        evidence = self.provider.observe_payload(object_ref)
        if evidence is None:
            raise ObservationUnavailable('payload_geometry_unknown')
        pose = evidence.require(object_ref, now=self.clock())
        previous = self._attachment_evidence or self._released_evidence
        if previous is not None and (previous.object_ref != evidence.object_ref
                or previous.reference_id != evidence.reference_id
                or previous.revision != evidence.revision
                or previous.construction != evidence.construction
                or evidence.position_error_m > previous.position_error_m
                or evidence.rotation_error_rad > previous.rotation_error_rad
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
        inverse = ee.xmat.reshape(3, 3).T
        position = inverse @ (pose[:3, 3]-ee.xpos)
        rotation = inverse @ pose[:3, :3]
        distance = float(np.linalg.norm(position-self.payload_transform[:3, 3]))
        angle = evidence_rotation_distance(evidence.pose, self.payload_transform[:3, :3], rotation)
        before = self._attachment_evidence
        distance += before.position_error_m + evidence.position_error_m
        angle += before.rotation_error_rad + evidence.rotation_error_rad
        return distance, angle

    def observe_object(self, object_ref):
        evidence, _ = self._evidence(object_ref)
        return evidence.pose

    def observed_geometry_pose(self, object_ref):
        """Geometry representative for explicitly shape-aware native consumers."""
        _, pose = self._evidence(object_ref)
        return pose

    def set_payload(self, model, data, body, parent=None):
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
        evidence, world_from_reference = self._evidence(body)
        if self.support_contact is not None and body != self.support_contact.object_ref:
            raise ObservationUnavailable('support_geometry_unknown')
        if self.target_contact is not None and (body != self.target_contact.object_ref
                or evidence.reference_id != self.target_contact.reference_id
                or evidence.revision != self.target_contact.geometry_revision):
            raise ObservationUnavailable('target_contact_identity_changed')
        mujoco.mj_copyData(self._probe, model, data)
        mujoco.mj_kinematics(model, self._probe)
        ee = self._probe.body(parent)
        ee_from_reference = np.eye(4)
        rotation = ee.xmat.reshape(3, 3).T
        ee_from_reference[:3, :3] = rotation @ world_from_reference[:3, :3]
        ee_from_reference[:3, 3] = rotation @ (world_from_reference[:3, 3] - ee.xpos)
        vertices = evidence.vertices_reference @ ee_from_reference[:3, :3].T + ee_from_reference[:3, 3]
        payload = HeldObject(vertices, ee_from_reference)
        self.payload = payload
        self.payload_body, self.payload_parent = body, parent
        self.payload_transform = payload.body_pose_ee
        self._attachment_evidence = evidence
        if self.support_contact is not None:
            self._support_payload_error_m = max(previous_error, self._support_pose_error(evidence))
            # Measured payload evidence, not the hypothetical IK/RRT pose,
            # anchors departure. Query ordering cannot ratchet this floor down.
            observed = WorldSolid('convex', (), world_from_reference, offset_m=self._support_payload_error_m,
                                 vertices=evidence.vertices_reference,
                                 material_calibration_id='observed-payload-envelope')
            self._support_departure_clearance = (self.support_contact.physical_clearance(observed)
                                                 if previous_floor is None else previous_floor)
        self.released_bounds = None
        self._released_target = None
        self._released_object_ref = None
        self._released_evidence = None

    def capture_payload(self):
        return (self.payload_body, self.payload_parent, self.payload_transform,
                self.payload, self._attachment_evidence,
                None if self.released_bounds is None else self.released_bounds.copy(), self._released_target, self._released_evidence,
                self._support_departure_clearance, self._released_object_ref, self._support_payload_error_m)

    def restore_payload(self, state):
        (self.payload_body, self.payload_parent, self.payload_transform,
         self.payload, self._attachment_evidence, self.released_bounds, self._released_target, self._released_evidence,
         self._support_departure_clearance, self._released_object_ref, self._support_payload_error_m) = state

    def configuration_collides(self, model, data):
        self.last_contacts = []
        self.last_admission = None
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
        collides = super().configuration_collides(model, data)
        if self.current_revision() != self.scene_revision:
            self.last_reason = 'scene_revision_changed'
            return True
        return collides
