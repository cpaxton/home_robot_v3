# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Visual effects from temporally bounded RGBD and measured robot kinematics.

No simulator contact, body velocity, predicted attachment pose or controller ACK
is an observed effect. Producers must bound capture times in the client monotonic
clock using a qualified clock mapping/pipeline; receive time alone is inadequate.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
from emet.controller.manipulation.shape_pose import ShapePoseEvidence, evidence_rotation_distance, shape_identity
from emet.motion.mujoco_collision import body_subtree
from emet.motion.placement_geometry import bounds_array


def rotation_distance(a, b):
    return float(np.arccos(np.clip((np.sum(a * b) - 1) / 2, -1, 1)))


def rigid_copy(value):
    # Reuse the pose contract's rigid-transform validation and immutable backing.
    return ObjectPoseEvidence(value, 0., 'rgbd_tracking', 'validation').world_from_object


@dataclass(frozen=True)
class CaptureInterval:
    observation_id: str
    earliest: float
    latest: float
    received_monotonic: float
    calibration_id: str

    def __post_init__(self):
        if (not self.observation_id or not self.calibration_id
                or not np.isfinite([self.earliest, self.latest, self.received_monotonic]).all()
                or not self.earliest <= self.latest <= self.received_monotonic):
            raise ValueError('invalid_capture_interval')

    def require(self, *, now, after=None, max_age_s=2.):
        if not 0 <= now - self.earliest <= max_age_s or self.received_monotonic > now:
            raise ObservationUnavailable('object_observation_stale')
        if after is not None and (not np.isfinite(after) or self.earliest <= after):
            raise ObservationUnavailable('capture_motion_alignment_unknown')


@dataclass(frozen=True)
class RobotPoseSample:
    world_from_ee: np.ndarray
    joints: np.ndarray
    received_monotonic: float
    gripper_bounds: np.ndarray | None = None
    source: str = 'measured_joint_feedback'

    def __post_init__(self):
        if self.source not in {'measured_joint_feedback', 'simulator_robot_kinematics_debug'}:
            raise ValueError('invalid_robot_feedback_source')
        pose = rigid_copy(self.world_from_ee)
        joints = np.asarray(self.joints, float)
        if joints.ndim != 1 or not joints.size or not np.isfinite(joints).all() or not np.isfinite(self.received_monotonic):
            raise ValueError('invalid_robot_pose_sample')
        object.__setattr__(self, 'world_from_ee', pose)
        object.__setattr__(self, 'joints', np.frombuffer(joints.tobytes(), dtype=joints.dtype))
        if self.gripper_bounds is not None:
            boxes = np.asarray(self.gripper_bounds, float)
            if (boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or not len(boxes)
                    or not np.isfinite(boxes).all() or np.any(boxes[:, 0] >= boxes[:, 1])):
                raise ValueError('invalid_gripper_geometry')
            object.__setattr__(self, 'gripper_bounds', np.frombuffer(boxes.tobytes(), dtype=boxes.dtype).reshape(boxes.shape))

    @classmethod
    def from_model(cls, model, data, *, ee_body, gripper_bodies, received_monotonic,
                   source='measured_joint_feedback'):
        """Derive complete gripper volumes from calibrated robot FK, not contacts."""
        probe = mujoco.MjData(model)
        mujoco.mj_copyData(probe, model, data)
        mujoco.mj_kinematics(model, probe)
        ids = set().union(*(body_subtree(model, name) for name in gripper_bodies))
        boxes = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) not in ids or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            rotation = probe.geom_xmat[gid].reshape(3, 3)
            center = rotation @ model.geom_aabb[gid, :3] + probe.geom_xpos[gid]
            half = np.abs(rotation) @ model.geom_aabb[gid, 3:]
            boxes.append([center - half, center + half])
        if not boxes:
            raise ObservationUnavailable('gripper_geometry_unknown')
        ee = probe.body(ee_body)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = ee.xmat.reshape(3, 3), ee.xpos
        return cls(pose, probe.qpos, received_monotonic, np.array(boxes), source)


@dataclass(frozen=True)
class VisualEffectFrame:
    payload: PayloadGeometryEvidence
    capture: CaptureInterval
    robot_before: RobotPoseSample
    robot_after: RobotPoseSample
    position_error_m: float
    rotation_error_rad: float
    # Independently bounded camera/encoder timing and between-sample motion.
    # Matching endpoints alone do not establish stillness during exposure.
    alignment_error_m: float | None = None
    alignment_error_rad: float | None = None

    def __post_init__(self):
        if (not np.isfinite([self.position_error_m, self.rotation_error_rad]).all()
                or self.position_error_m < 0 or self.rotation_error_rad < 0):
            raise ValueError('invalid_tracking_error_bounds')

    def require(self, *, now, after=None):
        self.capture.require(now=now, after=after)
        if (self.alignment_error_m is None or self.alignment_error_rad is None
                or not np.isfinite([self.alignment_error_m, self.alignment_error_rad]).all()
                or not 0 <= self.alignment_error_m <= .002 or not 0 <= self.alignment_error_rad <= .01):
            raise ObservationUnavailable('capture_motion_alignment_unknown')
        pose = self.payload.require(self.payload.object_ref, now=now)
        if (self.payload.pose.observation_id != self.capture.observation_id
                or self.payload.pose.received_monotonic != self.capture.received_monotonic):
            raise ObservationUnavailable('capture_identity_mismatch')
        a, b = self.robot_before, self.robot_after
        if a.source != 'measured_joint_feedback' or b.source != 'measured_joint_feedback':
            raise ObservationUnavailable('robot_feedback_unqualified')
        if (not a.received_monotonic <= self.capture.earliest <= self.capture.latest <= b.received_monotonic <= now
                or a.joints.shape != b.joints.shape or np.max(np.abs(a.joints - b.joints)) > .002
                or np.linalg.norm(a.world_from_ee[:3, 3] - b.world_from_ee[:3, 3]) > .002
                or rotation_distance(a.world_from_ee[:3, :3], b.world_from_ee[:3, :3]) > .01):
            raise ObservationUnavailable('capture_motion_alignment_unknown')
        if self.position_error_m > .01 or self.rotation_error_rad > .05:
            raise ObservationUnavailable('object_pose_ambiguous')
        if isinstance(self.payload.pose, ShapePoseEvidence) and (
                self.position_error_m < self.payload.pose.position_error_m
                or self.rotation_error_rad < self.payload.pose.rotation_error_rad):
            raise ObservationUnavailable('object_pose_ambiguous')
        return pose, b.world_from_ee

    def bounds(self):
        pose = self.payload.pose.world_from_object
        if isinstance(self.payload.pose, ShapePoseEvidence):
            box = self.payload.pose.model.world_bounds(pose[:3, :3], pose[:3, 3])
            radius = np.linalg.norm(self.payload.pose.model.half_extents)
            margin = self.position_error_m + 2*radius*np.sin(min(np.pi, self.rotation_error_rad)/2)
            return np.stack((box[0]-margin, box[1]+margin))
        vertices = self.payload.vertices_reference @ pose[:3, :3].T + pose[:3, 3]
        radius = np.linalg.norm(self.payload.vertices_reference, axis=1).max()
        margin = self.position_error_m + 2 * radius * np.sin(min(np.pi, self.rotation_error_rad) / 2)
        return np.stack((vertices.min(0) - margin, vertices.max(0) + margin))


@dataclass(frozen=True)
class VisualEffectResult:
    state: str
    reason: str
    observation_ids: tuple[str, ...] = ()

    @property
    def satisfied(self):
        return self.state == 'satisfied'

    def as_dict(self):
        # Provider exceptions/capture identifiers may contain private paths or
        # device identities. Public receipts expose only safe codes and hashes.
        allowed = {
            'visual_grasp_reference_recorded', 'visual_lift_verified', 'visual_lift_not_verified',
            'visual_placement_verified', 'visual_placement_not_verified', 'visual_evidence_incomplete',
            'object_observation_stale', 'object_state_unknown', 'object_pose_ambiguous',
            'capture_motion_alignment_unknown', 'capture_identity_mismatch', 'capture_replayed',
            'robot_feedback_unqualified',
            'capture_clock_changed', 'payload_identity_mismatch', 'payload_geometry_changed',
            'support_geometry_unknown', 'gripper_geometry_unknown', 'scene_revision_changed',
            'scene_changed_replan', 'observed_scene_unavailable',
        }
        return {'state': self.state, 'reason': self.reason if self.reason in allowed else 'observation_unavailable',
                'observation_ids': ['observation:' + hashlib.sha256(value.encode()).hexdigest()[:16]
                                    for value in self.observation_ids], 'source': 'rgbd_tracking'}


def _frames(reference, frames, *, now, after, min_span_s):
    if reference is None or len(frames) < 2:
        raise ObservationUnavailable('visual_evidence_incomplete')
    # The grasp baseline is historical evidence, not a claim of current pose.
    # Validate it at its actual bracketed receive time without renewing its age.
    if not 0 <= now - reference.capture.earliest <= 30.:
        raise ObservationUnavailable('object_observation_stale')
    reference.require(now=reference.robot_after.received_monotonic)
    last = reference.capture.latest
    seen = {reference.capture.observation_id}
    for frame in frames:
        if frame is None:
            raise ObservationUnavailable('object_state_unknown')
        frame.require(now=now, after=after)
        if frame.capture.calibration_id != reference.capture.calibration_id:
            raise ObservationUnavailable('capture_clock_changed')
        a, b = reference.payload, frame.payload
        if (a.object_ref, a.reference_id, a.revision, a.construction) != (b.object_ref, b.reference_id, b.revision, b.construction):
            raise ObservationUnavailable('payload_geometry_changed')
        if shape_identity(a.pose) != shape_identity(b.pose):
            raise ObservationUnavailable('payload_geometry_changed')
        if not np.array_equal(a.vertices_reference, b.vertices_reference):
            raise ObservationUnavailable('payload_geometry_changed')
        if frame.capture.observation_id in seen or frame.capture.earliest <= last:
            raise ObservationUnavailable('capture_replayed')
        seen.add(frame.capture.observation_id)
        last = frame.capture.latest
    if frames[-1].capture.earliest - frames[0].capture.latest < min_span_s:
        raise ObservationUnavailable('visual_evidence_incomplete')


def evaluate_lift(reference, frames, *, support_top_z, after, now=None):
    """Require separation and measured object/EE co-motion across fresh captures."""
    now = time.monotonic() if now is None else now
    try:
        _frames(reference, frames, now=now, after=after, min_span_s=.2)
        initial, initial_ee = reference.require(now=reference.robot_after.received_monotonic)
        expected = np.linalg.inv(initial_ee) @ initial
        if not np.isfinite(support_top_z):
            raise ObservationUnavailable('support_geometry_unknown')
        ids = tuple(f.capture.observation_id for f in frames)
        for frame in frames:
            obj, ee = frame.require(now=now, after=after)
            relative = np.linalg.inv(ee) @ obj
            error = (frame.position_error_m + reference.position_error_m
                     + frame.alignment_error_m + reference.alignment_error_m)
            angular_error = (frame.rotation_error_rad + reference.rotation_error_rad
                             + frame.alignment_error_rad + reference.alignment_error_rad)
            if (np.linalg.norm(relative[:3, 3] - expected[:3, 3]) + error > .02
                    or evidence_rotation_distance(frame.payload.pose, relative[:3, :3], expected[:3, :3]) + angular_error > .1
                    or frame.bounds()[0, 2] <= support_top_z + .02
                    or ee[2, 3] - initial_ee[2, 3] < .04
                    or obj[2, 3] - initial[2, 3] - error < .04):
                definite_failure = (
                    np.linalg.norm(relative[:3, 3] - expected[:3, 3]) - error > .02
                    or evidence_rotation_distance(frame.payload.pose, relative[:3, :3], expected[:3, :3]) - angular_error > .1
                    or frame.bounds()[1, 2] <= support_top_z + .02
                    or ee[2, 3] - initial_ee[2, 3] < .04
                    or obj[2, 3] - initial[2, 3] + error < .04)
                return VisualEffectResult('violated' if definite_failure else 'unknown', 'visual_lift_not_verified', ids)
        return VisualEffectResult('satisfied', 'visual_lift_verified', ids)
    except ObservationUnavailable as exc:
        return VisualEffectResult('unknown', str(exc))


def evaluate_place(reference, frames, *, support_bounds, after, now=None):
    """Verify footprint/support consistency and settling after measured retreat."""
    now = time.monotonic() if now is None else now
    try:
        _frames(reference, frames, now=now, after=after, min_span_s=.3)
        support = bounds_array(support_bounds)
        ids = tuple(f.capture.observation_id for f in frames)
        anchor = frames[0]
        first = anchor.payload.pose.world_from_object
        for frame in frames:
            obj, ee = frame.require(now=now, after=after)
            box = frame.bounds()
            # The entire possible footprint must fit. Support consistency is
            # visual geometry, not an invented force/contact measurement.
            gripper = frame.robot_after.gripper_bounds
            if gripper is None:
                raise ObservationUnavailable('gripper_geometry_unknown')
            separation = np.maximum(np.maximum(box[0] - gripper[:, 1], gripper[:, 0] - box[1]), 0.)
            error = frame.position_error_m + anchor.position_error_m
            angular_error = frame.rotation_error_rad + anchor.rotation_error_rad
            if (np.any(box[0, :2] < support[0, :2]) or np.any(box[1, :2] > support[1, :2])
                    or box[0, 2] < support[1, 2] - .02 or box[1, 2] <= support[1, 2]
                    or box[0, 2] > support[1, 2] + .02
                    or np.linalg.norm(separation, axis=1).min() - frame.alignment_error_m < .01
                    or np.linalg.norm(obj[:3, 3] - first[:3, 3]) + error > .01
                    or evidence_rotation_distance(frame.payload.pose, obj[:3, :3], first[:3, :3]) + angular_error > .1):
                return VisualEffectResult('violated', 'visual_placement_not_verified', ids)
        return VisualEffectResult('satisfied', 'visual_placement_verified', ids)
    except ObservationUnavailable as exc:
        return VisualEffectResult('unknown', str(exc))


class ObservedVisualVerifier:
    """Execution consumer of live frames and bounded stop-and-observe batches."""
    def __init__(self, *, object_ref, support_bounds, observe, collect, clock=time.monotonic):
        self.object_ref, self.support_bounds = object_ref, bounds_array(support_bounds)
        self.observe, self.collect, self.clock = observe, collect, clock
        self.reference = self.lift_reference = None
        self.results = []

    def _record(self, result):
        self.results.append(result)
        return result

    def begin_grasp(self, *, after):
        self.reference = self.lift_reference = None
        try:
            frame = self.observe(self.object_ref)
            if frame is None:
                raise ObservationUnavailable('object_state_unknown')
            frame.require(now=self.clock(), after=after)
            if frame.payload.object_ref != self.object_ref:
                raise ObservationUnavailable('payload_identity_mismatch')
            self.reference = frame
            return self._record(VisualEffectResult('satisfied', 'visual_grasp_reference_recorded',
                                                   (frame.capture.observation_id,)))
        except ObservationUnavailable as exc:
            return self._record(VisualEffectResult('unknown', str(exc)))

    def verify_lift(self, *, after):
        frames = tuple(self.collect(self.object_ref, after=after))
        result = evaluate_lift(self.reference, frames, support_top_z=self.support_bounds[1, 2],
                               after=after, now=self.clock())
        if result.satisfied:
            self.lift_reference = frames[-1]
        return self._record(result)

    def verify_place(self, *, after):
        frames = tuple(self.collect(self.object_ref, after=after))
        result = evaluate_place(self.lift_reference, frames, support_bounds=self.support_bounds,
                                after=after, now=self.clock())
        return self._record(result)
