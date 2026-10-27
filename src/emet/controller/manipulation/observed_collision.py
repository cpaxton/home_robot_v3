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
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
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
    pose: ObjectPoseEvidence
    construction: str

    def __post_init__(self):
        # HeldObject validates finite positive extent; its EE naming is irrelevant
        # here because these vertices are still in the declared reference frame.
        vertices = HeldObject(self.vertices_reference).vertices_ee
        if np.linalg.matrix_rank(vertices - vertices.mean(0), tol=1e-10) < 3:
            raise ValueError('invalid_payload_geometry_evidence')
        if (not self.object_ref or not self.reference_id or not self.revision
                or not isinstance(self.pose, ObjectPoseEvidence)
                or self.construction not in {'known_model_prior', 'multiview_bound'}):
            raise ValueError('invalid_payload_geometry_evidence')
        vertices = np.frombuffer(vertices.tobytes(), dtype=vertices.dtype).reshape(-1, 3)
        object.__setattr__(self, 'vertices_reference', vertices)

    def require(self, object_ref, *, now=None):
        if object_ref != self.object_ref:
            raise ObservationUnavailable('payload_identity_mismatch')
        return self.pose.require('observed', now=now)


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
    """
    geometry_source = 'observed_voxels'
    input_mode = 'observed'

    def __init__(self, model, *, robot_body, ee_body, scene: PlacementScene,
                 payload_provider: ObservedPayloadProvider, scene_revision: str,
                 current_revision, received_monotonic: float, contact_bodies=(),
                 margin_m=.005, max_age_s=2., clock=time.monotonic):
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

    def _evidence(self, object_ref):
        evidence = self.provider.observe_payload(object_ref)
        if evidence is None:
            raise ObservationUnavailable('payload_geometry_unknown')
        pose = evidence.require(object_ref, now=self.clock())
        previous = self._attachment_evidence
        if previous is not None and (previous.object_ref != evidence.object_ref
                or previous.reference_id != evidence.reference_id
                or previous.revision != evidence.revision
                or previous.construction != evidence.construction
                or not np.array_equal(previous.vertices_reference, evidence.vertices_reference)):
            raise ObservationUnavailable('payload_geometry_changed')
        return evidence, pose

    def observe_object(self, object_ref):
        evidence, _ = self._evidence(object_ref)
        return evidence.pose

    def set_payload(self, model, data, body, parent=None):
        """``body`` is an opaque observed handle here, never a MuJoCo body name."""
        if model is not self._model:
            raise ValueError('Collision checker model mismatch')
        if body is None:
            # Retain the released object's possible volume for retreat planning.
            # The next scene revision must replace this with observed geometry.
            if self.payload is not None:
                mujoco.mj_copyData(self._probe, model, data)
                mujoco.mj_kinematics(model, self._probe)
                ee = self._probe.body(self.ee_body)
                vertices = self.payload.world_vertices(ee.xpos, ee.xmat)
                self.released_bounds = np.stack((vertices.min(0) - self.margin, vertices.max(0) + self.margin))
            self.payload = None
            self.payload_body = self.payload_parent = self.payload_transform = None
            self._attachment_evidence = None
            return
        if parent != self.ee_body:
            raise ValueError('payload_parent_mismatch')
        evidence, world_from_reference = self._evidence(body)
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
        self.released_bounds = None

    def capture_payload(self):
        return (self.payload_body, self.payload_parent, self.payload_transform,
                self.payload, self._attachment_evidence,
                None if self.released_bounds is None else self.released_bounds.copy())

    def restore_payload(self, state):
        (self.payload_body, self.payload_parent, self.payload_transform,
         self.payload, self._attachment_evidence, self.released_bounds) = state

    def configuration_collides(self, model, data):
        self.last_contacts = []
        age = self.clock() - self.received_monotonic
        if not np.isfinite(age) or not 0 <= age <= self.max_age_s:
            self.last_reason = 'scene_observation_stale'
            return True
        if self.current_revision() != self.scene_revision:
            self.last_reason = 'scene_revision_changed'
            return True
        collides = super().configuration_collides(model, data)
        if self.current_revision() != self.scene_revision:
            self.last_reason = 'scene_revision_changed'
            return True
        return collides
