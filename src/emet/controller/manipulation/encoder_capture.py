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


def read_local_sim_motor_capture(observation, *, now=None):
    """Same-sample ideal simulator encoders; hardware needs its own clock mapper.

    Arm is one measured total extension. Finger angles, rubber deformation and
    individual telescope lengths are not claimed as measured. No EE/gripper
    geometry or zero physical pose uncertainty is inferred from these readouts.
    """
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
    packet = timing.get("motor_sample")
    keys = {"schema_version", "source", "clock_id", "interval_ns", "positions", "units", "kinematic_signature"}
    if (
        not isinstance(packet, dict)
        or set(packet) != keys
        or packet["schema_version"] != 1
        or packet["source"] != "simulator_stretch_motor_encoders"
        or packet["units"] != "actuator_length"
        or not isinstance(packet["kinematic_signature"], str)
        or not packet["kinematic_signature"]
        or packet["clock_id"] != local_monotonic_clock_id()
        or packet["interval_ns"] != timing.get("capture_interval_ns")
        or packet["interval_ns"] != timing.get("depth_capture_interval_ns")
        or stamp != timing["capture_interval_ns"][0]
        or not isinstance(packet["positions"], dict)
        or set(packet["positions"]) != set(STRETCH_CAPTURE_MOTORS)
    ):
        raise ObservationUnavailable("capture_motion_alignment_unknown")
    try:
        values = np.array([packet["positions"][name] for name in STRETCH_CAPTURE_MOTORS], float)
    except (TypeError, ValueError) as error:
        raise ObservationUnavailable("invalid_joint_feedback") from error
    return MotorCapture(capture, values, packet["kinematic_signature"])
