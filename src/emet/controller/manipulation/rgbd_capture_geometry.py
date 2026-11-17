# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Keep camera geometry and motor-only robot evidence with its original RGBD.

The explicit simulation profile supplies assumptions, not errors inferred from
fit residuals. This assembles visual-effect inputs; it does not certify a full
collision scene, hardware calibration, or an observed execution provider.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from emet.controller.manipulation.encoder_capture import MotorCapture
from emet.controller.manipulation.gripper_geometry import GripperGeometryEnvelope, bound_stretch_gripper
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import CaptureInterval, VisualEffectFrame, rigid_copy
from emet.controller.manipulation.stretch_encoder_kinematics import (
    EncoderKinematicEstimate,
    estimate_stretch_ee_from_encoders,
)
from emet.core.rgbd_registration import decode_registration, source_points_rgb


@dataclass(frozen=True)
class SimGeometryProfile:
    calibration_id: str
    object_position_error_m: float
    object_rotation_error_rad: float
    robot_position_error_m: float
    robot_rotation_error_rad: float
    scope: str = "simulation_geometry_assumptions"
    schema_version: int = 1

    def __post_init__(self):
        errors = [
            self.object_position_error_m,
            self.object_rotation_error_rad,
            self.robot_position_error_m,
            self.robot_rotation_error_rad,
        ]
        if (
            not isinstance(self.calibration_id, str)
            or not self.calibration_id
            or self.scope != "simulation_geometry_assumptions"
            or type(self.schema_version) is not int
            or self.schema_version != 1
            or not np.isfinite(errors).all()
            or not 0 < errors[0] <= 0.01
            or not 0 < errors[1] <= 0.05
            or errors[2] < 0
            or not 0 <= errors[3] <= np.pi
        ):
            raise ValueError("invalid_sim_geometry_profile")


@dataclass(frozen=True)
class RgbdCaptureGeometry:
    capture: CaptureInterval
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    robot: EncoderKinematicEstimate
    gripper: GripperGeometryEnvelope
    profile: SimGeometryProfile
    source_points_world: np.ndarray | None = None
    source_calibration_id: str = ""
    source_depth: np.ndarray | None = None
    source_intrinsic: np.ndarray | None = None
    world_from_depth: np.ndarray | None = None

    def __post_init__(self):
        k = np.asarray(self.intrinsic, float)
        if (
            not isinstance(self.capture, CaptureInterval)
            or k.shape != (3, 3)
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose([k[0, 1], k[1, 0]], 0)
            or not isinstance(self.robot, EncoderKinematicEstimate)
            or not isinstance(self.robot.motor_capture, MotorCapture)
            or self.robot.motor_capture.capture != self.capture
            or not isinstance(self.gripper, GripperGeometryEnvelope)
            or not isinstance(self.profile, SimGeometryProfile)
            or self.gripper.position_error_m != self.profile.robot_position_error_m
            or self.gripper.rotation_error_rad != self.profile.robot_rotation_error_rad
            or self.gripper.observation_id != self.capture.observation_id
            or self.gripper.received_monotonic != self.capture.received_monotonic
            or not np.array_equal(self.gripper.world_from_ee, self.robot.world_from_ee)
            or not np.array_equal(self.gripper.motor_positions, self.robot.motor_capture.positions)
        ):
            raise ValueError("invalid_rgbd_capture_geometry")
        object.__setattr__(self, "intrinsic", np.frombuffer(k.tobytes(), dtype=float).reshape(3, 3))
        object.__setattr__(self, "world_from_camera", rigid_copy(self.world_from_camera))
        if self.source_points_world is not None:
            points = np.asarray(self.source_points_world, float)
            if points.ndim != 3 or points.shape[-1] != 3 or not self.source_calibration_id:
                raise ValueError("invalid_rgbd_capture_geometry")
            object.__setattr__(
                self, "source_points_world", np.frombuffer(points.tobytes(), dtype=float).reshape(points.shape)
            )
        elif self.source_calibration_id:
            raise ValueError("invalid_rgbd_capture_geometry")
        raw_fields = (self.source_depth, self.source_intrinsic, self.world_from_depth)
        if any(value is not None for value in raw_fields):
            if any(value is None for value in raw_fields) or self.source_points_world is None:
                raise ValueError("invalid_rgbd_capture_geometry")
            depth, intrinsic, pose = (np.asarray(value, float) for value in raw_fields)
            if (
                depth.ndim != 2
                or intrinsic.shape != (3, 3)
                or not np.isfinite(intrinsic).all()
                or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
            ):
                raise ValueError("invalid_rgbd_capture_geometry")
            for name, value in (("source_depth", depth), ("source_intrinsic", intrinsic)):
                object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape))
            object.__setattr__(self, "world_from_depth", rigid_copy(pose))

    @classmethod
    def from_stretch(cls, observation, model, *, profile, now, original_rays=False):
        if not isinstance(profile, SimGeometryProfile) or observation.is_simulation is not True:
            raise ObservationUnavailable("simulation_capture_required")
        robot = estimate_stretch_ee_from_encoders(observation, model, now=now)
        gripper = bound_stretch_gripper(
            model,
            robot,
            position_error_m=profile.robot_position_error_m,
            rotation_error_rad=profile.robot_rotation_error_rad,
        )
        world, identity = None, ""
        raw, raw_k, depth_pose = None, None, None
        if original_rays:
            packet = observation.rgbd_registration
            timing = observation.image_timing["head_cam_left"]
            if (
                not isinstance(packet, dict)
                or packet.get("clock_id") != timing["clock_id"]
                or packet.get("rgb_interval_ns") != timing["capture_interval_ns"]
                or packet.get("depth_interval_ns") != timing["depth_capture_interval_ns"]
            ):
                raise ObservationUnavailable("capture_identity_mismatch")
            decoded = decode_registration(packet)
            if not np.array_equal(np.rot90(decoded["rgb"], -1), observation.rgb):
                raise ObservationUnavailable("rgbd_geometry_capture_mismatch")
            points = np.rot90(source_points_rgb(decoded), -1)
            # The camera pose rotates clockwise with the head images. Convert
            # native optical XYZ to those same upright optical coordinates.
            points = points @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            world = points @ observation.camera_pose[:3, :3].T + observation.camera_pose[:3, 3]
            upright_from_native = np.eye(4)
            upright_from_native[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            depth_pose = observation.camera_pose @ upright_from_native @ decoded["rgb_from_depth"]
            raw, raw_k = decoded["raw_depth"], decoded["depth_k"]
            k, shape = decoded["rgb_k"], decoded["rgb"].shape
            rotated_k = np.array([[k[1, 1], 0, shape[0] - 1 - k[1, 2]], [0, k[0, 0], k[0, 2]], [0, 0, 1]])
            if not np.allclose(rotated_k, observation.camera_K, rtol=0, atol=1e-9):
                raise ObservationUnavailable("rgbd_geometry_capture_mismatch")
            # Compare against the actual wire conversion, before using the
            # unquantized registered sensor depth. No depth is synthesized.
            expected = (np.rot90(decoded["registered_depth"], -1) * 1000).astype(np.uint16).astype(np.float32) / 1000
            if not np.array_equal(expected, observation.depth, equal_nan=True):
                raise ObservationUnavailable("rgbd_geometry_capture_mismatch")
            digest = hashlib.sha256()
            for value in (decoded["depth_k"], decoded["rgb_k"], decoded["rgb_from_depth"]):
                normalized = np.round(value, 12)
                normalized[normalized == 0.] = 0.
                digest.update(normalized.tobytes())
            identity = digest.hexdigest()
        return cls(
            robot.motor_capture.capture,
            observation.camera_K,
            observation.camera_pose,
            robot,
            gripper,
            profile,
            world,
            identity,
            raw,
            raw_k,
            depth_pose,
        )

    @property
    def profile_identity(self):
        # Camera world pose/motor values may change; calibration/model may not.
        return (
            self.profile,
            hashlib.sha256(self.intrinsic.tobytes()).hexdigest(),
            self.gripper.model_signature,
            self.robot.motor_capture.kinematic_signature,
            tuple(self.gripper.geom_ids),
            self.robot.motor_capture.names,
            self.robot.motor_capture.units,
            self.gripper.assumptions,
            self.source_calibration_id,
            tuple(row for row in self.gripper.joint_intervals if row[0] != "joint_gripper_slide"),
        )

    def require(self, view, *, now):
        self.capture.require(now=now)
        if view.capture != self.capture:
            raise ObservationUnavailable("capture_identity_mismatch")
        depth = view.depth.astype(float)
        y, x = np.indices(depth.shape)
        k = self.intrinsic
        camera = np.stack(((x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth), axis=-1)
        expected = camera @ self.world_from_camera[:3, :3].T + self.world_from_camera[:3, 3]
        if self.source_points_world is not None:
            expected = self.source_points_world
        valid = np.isfinite(depth) & (depth > 0)
        if expected.shape != view.world.shape or not np.allclose(view.world[valid], expected[valid], rtol=0, atol=1e-9):
            raise ObservationUnavailable("rgbd_geometry_capture_mismatch")

    def equivalent_capture(self, other):
        """Transport receipt can change on a duplicate; source measurements cannot."""
        if not isinstance(other, RgbdCaptureGeometry) or self.profile_identity != other.profile_identity:
            return False
        if (self.source_depth is None) != (other.source_depth is None):
            return False
        if self.source_depth is not None and not all(
            np.array_equal(a, b, equal_nan=True)
            for a, b in (
                (self.source_depth, other.source_depth),
                (self.source_intrinsic, other.source_intrinsic),
                (self.world_from_depth, other.world_from_depth),
                (self.source_points_world, other.source_points_world),
            )
        ):
            return False
        return all(
            np.array_equal(a, b)
            for a, b in (
                (self.world_from_camera, other.world_from_camera),
                (self.robot.world_from_ee, other.robot.world_from_ee),
                (self.robot.world_from_base, other.robot.world_from_base),
                (self.robot.motor_capture.positions, other.robot.motor_capture.positions),
                (self.gripper.bounds_world, other.gripper.bounds_world),
            )
        )

    def visual_frame(self, tracked, *, robot_model, now):
        if (
            tracked.capture != self.capture
            or tracked.fit.observation_ids != (self.capture.observation_id,)
            or tracked.fit.received_monotonic != self.capture.received_monotonic
        ):
            raise ObservationUnavailable("capture_identity_mismatch")
        profile = self.profile
        pose = tracked.fit.require_shape(
            calibration_id=profile.calibration_id,
            position_error_m=profile.object_position_error_m,
            rotation_error_rad=profile.object_rotation_error_rad,
            now=now,
        )
        payload = PayloadGeometryEvidence(
            tracked.fit.object_ref,
            tracked.reference_id,
            tracked.fit.model.model_id,
            tracked.fit.model.vertices_reference,
            pose,
            "known_model_prior",
        )
        before, after = self.gripper.robot_brackets(self.robot, robot_model, now=now)
        # The opt-in renderer freezes RGBD and motors at one private scene copy.
        # This zero inter-sample motion term is specific to that simulator source.
        frame = VisualEffectFrame(
            payload,
            self.capture,
            before,
            after,
            profile.object_position_error_m,
            profile.object_rotation_error_rad,
            0.0,
            0.0,
        )
        frame.require(now=now)
        return frame
