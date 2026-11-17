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

    @classmethod
    def from_stretch(cls, observation, model, *, profile, now):
        if not isinstance(profile, SimGeometryProfile) or observation.is_simulation is not True:
            raise ObservationUnavailable("simulation_capture_required")
        robot = estimate_stretch_ee_from_encoders(observation, model, now=now)
        gripper = bound_stretch_gripper(
            model,
            robot,
            position_error_m=profile.robot_position_error_m,
            rotation_error_rad=profile.robot_rotation_error_rad,
        )
        return cls(robot.motor_capture.capture, observation.camera_K, observation.camera_pose, robot, gripper, profile)

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
        valid = np.isfinite(depth) & (depth > 0)
        if not np.allclose(view.world[valid], expected[valid], rtol=0, atol=1e-9):
            raise ObservationUnavailable("rgbd_geometry_capture_mismatch")

    def equivalent_capture(self, other):
        """Transport receipt can change on a duplicate; source measurements cannot."""
        if not isinstance(other, RgbdCaptureGeometry) or self.profile_identity != other.profile_identity:
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
