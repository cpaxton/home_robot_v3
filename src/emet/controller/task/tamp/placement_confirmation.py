# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Confirm a held payload's final placement before opening the gripper."""

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
from emet.motion.mujoco_arm_ik import solve_pose_ik
from emet.motion.observed_support_contact import ObservedProbeBand
from emet.motion.placement import validated_dense_path
from emet.motion.placement_geometry import box_corners


def _probe_path(ex, start, rotation, travel):
    before = ex.data.qpos.copy()
    route = []
    try:
        with ex.collision.motion_phase("probe"):
            for distance in np.linspace(0, travel, int(np.ceil(travel / .000245)) + 1)[1:]:
                result = solve_pose_ik(
                    ex.model, ex.data, ee_body=ex.ee_body, joint_names=ex.joint_names,
                    target_pos=start - [0., 0., distance], target_rotation=rotation,
                    max_iters=60, tol_m=.000015, tol_rad=.002, coupled_groups=ex.coupled_groups,
                )
                if not result.success:
                    return ()
                edge = [ex.data.qpos[ex.qadr].copy(), result.qpos[ex.qadr].copy()]
                if validated_dense_path(ex.model, ex.data, ex.joint_names, edge, ex.collision, joint_step=.005) is None:
                    return ()
                ex.data.qpos[:] = result.qpos
                route.append(result.qpos[ex.qadr].copy())
        return route
    finally:
        ex.data.qpos[:] = before
        mujoco.mj_kinematics(ex.model, ex.data)


def _retreat_held(ex, point, rotation):
    # Recheck from the actual stopped joint feedback rather than assuming the
    # last issued probe waypoint was reached.
    try:
        paths, error = ex._replan_at_measured_pose(
            ("probe",), ((point, rotation),), object_body=ex.payload_body, grasp=False,
        )
        if error is None:
            return ex._execute_path("probe", paths[0]).success
    except (ObservationUnavailable, RuntimeError, TimeoutError):
        pass
    return False


def _additional_view(ex, hover, rotation, offset, check_surface, bounds):
    target = hover + [offset, 0, .035]
    paths, error = ex._replan_at_measured_pose(
        ("preplace",), ((target, rotation),), object_body=ex.payload_body, grasp=False,
    )
    if error is not None:
        return False
    outcome = ex._execute_path("preplace", paths[0])
    if not outcome.success:
        raise ObservationUnavailable(outcome.message)
    verified = check_surface(bounds)
    if not _retreat_held(ex, hover, rotation):
        raise ObservationUnavailable("probe_retreat_unavailable")
    return verified


def confirm_placement(ex, *, calibration_id, half_band_m, capture_payload, check_surface):
    """Verify the vertical landing at the hover, then try one guarded probe."""
    checker = ex.collision
    ex.synchronize(ex.data)
    contact = checker.support_contact
    band = ObservedProbeBand(contact, half_band_m)
    body = ex.data.body(ex.ee_body)
    hover, rotation = body.xpos.copy(), body.xmat.reshape(3, 3).copy()
    bounds = checker.possible_payload_bounds(ex.model, ex.data)
    if bounds is None:
        return PhysicalMotionResult(False, "payload_not_retained", "place_confirmation")
    normal, offset = contact.unit_plane
    gap = float(np.min(box_corners(bounds) @ normal - offset))
    if not 0 <= gap <= .0097:
        return PhysicalMotionResult(False, "probe_contact_outside_band", "place_confirmation")
    confirmed = check_surface(bounds) if check_surface is not None else False
    if check_surface is not None and not confirmed:
        for side in (.035, -.035):
            if _additional_view(ex, hover, rotation, side, check_surface, bounds):
                confirmed = True
                break
    if confirmed and gap <= .001:
        ex.event(phase="placement_confirmation", method="visual")
        return PhysicalMotionResult(True, "ok", "place_confirmation")

    goal = hover - [0., 0., gap]
    if confirmed:
        paths, error = ex._replan_at_measured_pose(
            ("place",), ((goal, rotation),), object_body=ex.payload_body, grasp=False,
        )
        if error is None:
            outcome = ex._execute_path("place", paths[0])
            ex.event(phase="placement_confirmation", method="visual", controller_success=outcome.success)
            return outcome

    ex.synchronize(ex.data)
    if np.linalg.norm(ex.data.body(ex.ee_body).xpos - hover) > .0004:
        return PhysicalMotionResult(False, "probe_tracking_deviation", "place_confirmation")
    travel = min(gap + half_band_m, .0097)
    checker.probe_band = band
    route = _probe_path(ex, hover, rotation, travel)
    if not route:
        checker.probe_band = None
        return PhysicalMotionResult(False, "probe_path_unavailable", "place_confirmation")

    def retention():
        ex.synchronize(ex.data)
        frame = capture_payload()
        if not ex.payload_retained():
            raise ObservationUnavailable("probe_payload_tracking_lost")
        return frame.capture.latest

    try:
        receipt = ex.robot.guarded_place_probe(
            route, contact_band_m=(max(0., gap - half_band_m), travel),
            calibration_id=calibration_id, retention_capture=retention,
        )
        if receipt["status"] == "contact" and receipt["reason"] == "support_contact":
            retention()
            checker.probe_contact_verified = True
            ex.event(phase="placement_confirmation", method="guarded_contact", receipt=receipt)
            return PhysicalMotionResult(True, "ok", "place_confirmation")
        if receipt["reason"] not in ("probe_unexpected_force", "probe_tracking_deviation"):
            _retreat_held(ex, hover, rotation)
        return PhysicalMotionResult(False, receipt["reason"], "place_confirmation")
    finally:
        if not checker.probe_contact_verified:
            checker.probe_band = None
