# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""World-frame transfer candidates built from a calibrated grasp and live geometry."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import pose_uncertainty_margin, rigid_copy

from .observed_arm import TransferCandidate


@dataclass(frozen=True)
class CalibratedGrasp:
    """Object-local tool frame and finite gripper target from separate calibration."""

    object_from_ee: np.ndarray
    approach_axis_ee: tuple[float, float, float]
    close_joints: tuple[tuple[str, float], ...]
    approach_m: float = 0.08
    lift_m: float = 0.15
    retreat_m: float | None = None
    world_aligned: bool = False
    transit_clearance_m: float | None = None
    keep_current_heading: bool = False

    def __post_init__(self):
        object.__setattr__(self, "object_from_ee", rigid_copy(self.object_from_ee))
        axis = np.asarray(self.approach_axis_ee, dtype=float)
        retreat = self.lift_m if self.retreat_m is None else self.retreat_m
        if (
            (self.transit_clearance_m is not None and (
                not np.isfinite(self.transit_clearance_m) or not .01 <= self.transit_clearance_m <= .15))
            or type(self.keep_current_heading) is not bool
            or self.keep_current_heading and not self.world_aligned
            or type(self.world_aligned) is not bool
            or axis.shape != (3,)
            or not np.isclose(np.linalg.norm(axis), 1.0)
            or not np.isfinite([self.approach_m, self.lift_m, retreat]).all()
            or min(self.approach_m, self.lift_m, retreat) <= 0
        ):
            raise ValueError("invalid_grasp_calibration")
        object.__setattr__(self, "retreat_m", retreat)


def candidates_from_supports(
    snapshot,
    grasps,
    *,
    placement_region=None,
    yaw_options=(0.0, np.pi / 2),
    placement_gap_m=0.0,
    position_fractions=None,
    limit=8,
):
    """Propose poses only. The shared executor certifies contact and the swept route."""
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("invalid_candidate_budget")
    if not np.isfinite(placement_gap_m) or not 0 <= placement_gap_m <= 0.01:
        raise ValueError("invalid_placement_gap")
    if position_fractions is not None:
        position_fractions = np.asarray(position_fractions, float)
        if (position_fractions.ndim != 2 or position_fractions.shape[1] != 2
                or not 1 <= len(position_fractions) <= 9 or not np.isfinite(position_fractions).all()
                or np.any(np.abs(position_fractions) > .5)):
            raise ValueError('invalid_placement_position_fractions')
    if snapshot.support_contact is None:
        raise ObservationUnavailable("support_geometry_unknown")
    bounds = snapshot.support_bounds[:, :2]
    if placement_region is not None:
        requested = np.asarray(placement_region, dtype=float)
        if requested.shape != (2, 2) or not np.isfinite(requested).all() or np.any(requested[0] >= requested[1]):
            raise ValueError("invalid_placement_region")
        bounds = np.array([np.maximum(bounds[0], requested[0]), np.minimum(bounds[1], requested[1])])
    if np.any(bounds[0] >= bounds[1]):
        return ()
    vertices = snapshot.frame.payload.vertices_reference
    source_pose = snapshot.frame.payload.pose.world_from_object
    margin = pose_uncertainty_margin(
        np.linalg.norm(vertices, axis=1).max(), snapshot.frame.position_error_m, snapshot.frame.rotation_error_rad
    )
    plane = snapshot.support_contact.patch.plane
    profiles = tuple(grasps)
    if not all(isinstance(grasp, CalibratedGrasp) for grasp in profiles):
        raise ValueError("invalid_grasp_calibration")
    results = []
    for yaw_index, yaw in enumerate(yaw_options):
        if not np.isfinite(yaw):
            raise ValueError("invalid_placement_orientation")
        landing_rotation = Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix() @ source_pose[:3, :3]
        rotated = vertices @ landing_rotation.T
        low = bounds[0] - rotated[:, :2].min(axis=0) + margin
        high = bounds[1] - rotated[:, :2].max(axis=0) - margin
        if np.any(low >= high):
            continue
        # Rank the centers of the free footprint first, then sample the longer
        # feasible axis. These are pose proposals, not a free-space assertion.
        axis = int(np.argmax(high - low))
        fractions = position_fractions
        if fractions is None:
            fractions = np.zeros((3, 2))
            fractions[:, axis] = (0., -.25, .25)
        for fraction_index, fraction in enumerate(fractions):
            xy = (low + high) / 2 + fraction * (high - low)
            floor = (rotated[:, :2] + xy) @ plane[:2] + plane[2]
            landing_z = np.max(floor - rotated[:, 2]) + placement_gap_m
            landing_pose = np.eye(4)
            landing_pose[:3, :3] = landing_rotation
            landing_pose[:3, 3] = (*xy, landing_z)
            for grasp_index, grasp in enumerate(profiles):
                if grasp.world_aligned:
                    # A calibrated top-down action need not tilt with a noisy
                    # object-axis estimate. Geometry/evidence stays measured.
                    action_frame = np.eye(4)
                    action_frame[:3, 3] = source_pose[:3, 3]
                    pick = action_frame @ grasp.object_from_ee
                    if grasp.keep_current_heading:
                        shape = getattr(snapshot.frame.payload.pose, 'model', None)
                        if getattr(shape, 'symmetry', None) not in {'axial_unoriented', 'axial_oriented', 'all_rotations'}:
                            raise ObservationUnavailable('grasp_heading_symmetry_unknown')
                        current = snapshot.frame.robot_before.world_from_ee[:3, :3]
                        delta = np.arctan2(current[1, 0], current[0, 0])-np.arctan2(pick[1, 0], pick[0, 0])
                        pick[:3, :3] = Rotation.from_rotvec([0., 0., delta]).as_matrix() @ pick[:3, :3]
                    place = landing_pose @ np.linalg.inv(source_pose) @ pick
                else:
                    pick = source_pose @ grasp.object_from_ee
                    place = landing_pose @ grasp.object_from_ee
                p, r = pick[:3, 3], pick[:3, :3]
                q, s = place[:3, 3], place[:3, :3]
                rise = np.array([0.0, 0.0, grasp.lift_m])
                retreat = np.array([0.0, 0.0, grasp.retreat_m])
                approach = r @ np.asarray(grasp.approach_axis_ee) * grasp.approach_m
                preplace = q + rise
                # Recessed destinations must not lower the carried object
                # below the source counter before crossing the rim.
                preplace[2] = max(preplace[2], (p + rise)[2])
                if grasp.transit_clearance_m is not None:
                    source_support = getattr(snapshot, 'source_support_contact', None)
                    if source_support is None:
                        raise ObservationUnavailable('support_geometry_unknown')
                    source_vertices = vertices @ source_pose[:3, :3].T + source_pose[:3, 3]
                    source_plane = source_support.patch.plane
                    source_height = np.max(source_vertices[:, :2] @ source_plane[:2] + source_plane[2])
                    payload_bottom_from_tool = rotated[:, 2].min()+landing_z-q[2]
                    preplace[2] = max(q[2]+grasp.transit_clearance_m,
                        source_height-payload_bottom_from_tool+grasp.transit_clearance_m)
                results.append((fraction_index, yaw_index, grasp_index,
                    TransferCandidate(
                        ((p - approach, r), (p, r), (p + rise, r)),
                        ((preplace, s), (q, s), (q + retreat, s)),
                        grasp.close_joints,
                    )
                ))
    # Try distinct wrist orientations at the central landing before spending
    # the candidate budget on small translations of one blocked orientation.
    return tuple(entry[3] for entry in sorted(results, key=lambda entry: entry[:3])[:limit])
