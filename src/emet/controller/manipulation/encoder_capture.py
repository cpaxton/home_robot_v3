# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Timestamped motor readouts; passive-joint geometry is deliberately absent."""

import time
from dataclasses import dataclass

import numpy as np

from emet.controller.manipulation.capture_clock import local_monotonic_clock_id, local_sim_capture_interval
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval

STRETCH_CAPTURE_MOTORS = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper", "head_pan", "head_tilt")
STRETCH_WHEEL_JOINTS = ("joint_left_wheel", "joint_right_wheel")


@dataclass(frozen=True)
class MotorCapture:
    capture: CaptureInterval
    positions: np.ndarray
    kinematic_signature: str
    names: tuple[str, ...] = STRETCH_CAPTURE_MOTORS
    units: str = "actuator_length"

    def __post_init__(self):
        values = np.asarray(self.positions, float)
        if values.shape != (len(self.names),) or not np.isfinite(values).all():
            raise ObservationUnavailable("invalid_joint_feedback")
        object.__setattr__(self, "positions", np.frombuffer(values.tobytes(), dtype=float))

    def as_dict(self):
        return dict(zip(self.names, self.positions.tolist(), strict=True))


def _read_local_sim_encoder_packet(observation, key, source, units, packet_fields, now):
    now = time.monotonic() if now is None else now
    if observation.is_simulation is not True:
        raise ObservationUnavailable("simulation_capture_required")
    timing = (observation.image_timing or {}).get("head_cam_left")
    if not isinstance(timing, dict) or timing.get("source") != "renderer_synchronized_snapshot":
        raise ObservationUnavailable("capture_motion_alignment_unknown")
    stamp = timing.get("timestamp_ns")
    if type(stamp) is not int or stamp <= 0:
        raise ObservationUnavailable("capture_interval_unavailable")
    capture = local_sim_capture_interval(
        timing, received_monotonic=observation.received_monotonic, observation_id=f"capture:{stamp}"
    )
    capture.require(now=now)
    packet = timing.get(key)
    keys = {"schema_version", "source", "clock_id", "interval_ns", "positions", "units"} | packet_fields
    if (
        not isinstance(packet, dict)
        or set(packet) != keys
        or packet["schema_version"] != 1
        or packet["source"] != source
        or packet["units"] != units
        or packet["clock_id"] != local_monotonic_clock_id()
        or packet["interval_ns"] != timing.get("capture_interval_ns")
        or packet["interval_ns"] != timing.get("depth_capture_interval_ns")
        or stamp != timing["capture_interval_ns"][0]
    ):
        raise ObservationUnavailable("capture_motion_alignment_unknown")
    return capture, packet


def read_local_sim_motor_capture(observation, *, now=None):
    """Read synchronized motor channels in actuator lengths for Stretch FK."""
    capture, packet = _read_local_sim_encoder_packet(
        observation, "motor_sample", "simulator_stretch_motor_encoders", "actuator_length",
        {"kinematic_signature"}, now,
    )
    if (
        not isinstance(packet["kinematic_signature"], str)
        or not packet["kinematic_signature"]
        or not isinstance(packet["positions"], dict)
        or set(packet["positions"]) != set(STRETCH_CAPTURE_MOTORS)
    ):
        raise ObservationUnavailable("capture_motion_alignment_unknown")
    try:
        values = np.array([packet["positions"][name] for name in STRETCH_CAPTURE_MOTORS], float)
    except (TypeError, ValueError) as error:
        raise ObservationUnavailable("invalid_joint_feedback") from error
    return MotorCapture(capture, values, packet["kinematic_signature"])


@dataclass(frozen=True)
class WheelCapture:
    capture: CaptureInterval
    left_rad: float
    right_rad: float

    def __post_init__(self):
        if not np.isfinite([self.left_rad, self.right_rad]).all():
            raise ObservationUnavailable("invalid_wheel_feedback")


def read_local_sim_wheel_capture(observation, *, now=None):
    """Read unwrapped left and right wheel angles from the RGB-D capture."""
    capture, packet = _read_local_sim_encoder_packet(
        observation, "wheel_sample", "simulator_stretch_wheel_encoders", "radian", set(), now,
    )
    positions = packet["positions"]
    if not isinstance(positions, dict) or set(positions) != set(STRETCH_WHEEL_JOINTS):
        raise ObservationUnavailable("invalid_wheel_feedback")
    if any(type(value) not in (int, float) for value in positions.values()):
        raise ObservationUnavailable("invalid_wheel_feedback")
    return WheelCapture(capture, *(positions[name] for name in STRETCH_WHEEL_JOINTS))
