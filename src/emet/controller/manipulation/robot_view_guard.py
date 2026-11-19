# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Reject camera associations with possible robot returns; never certify ownership.

Outer geometry is sufficient to flag interference, not to remove pixels from a
target, label endpoints self, or declare space free. Unchecked robot links remain
unchecked; absence of a flag is not an execution or correspondence certificate.
"""

import hashlib
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from emet.controller.manipulation.gripper_geometry import bound_stretch_gripper
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval, pose_uncertainty_margin, rigid_copy
from emet.motion.observed_solids import WorldSolid
from emet.motion.placement_geometry import box_corners
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.depth_registration import register_depth


def rigid_gripper_outer_solids(model, estimate, *, now, position_error_m, rotation_error_rad):
    """Model-only rigid housing/camera geometry anchored to the motor-derived EE.

    Descendants across ANY tool joint are excluded from this guard's scope.
    Their absence never makes them collision-free or qualifies passive geometry.
    Mesh convex hulls are OUTER envelopes here, never guaranteed inner material.
    """
    envelope = bound_stretch_gripper(
        model, estimate, position_error_m=position_error_m, rotation_error_rad=rotation_error_rad
    )
    envelope.robot_brackets(estimate, model, now=now)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_SG3_gripper_body")
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, STRETCH_ARM_CHAIN.ee_body)
    if root < 0 or ee < 0:
        raise ObservationUnavailable("gripper_geometry_unknown")
    rigid = {root}
    for bid in range(root + 1, model.nbody):
        if int(model.body_parentid[bid]) in rigid and not model.body_jntnum[bid]:
            rigid.add(bid)
    probe = mujoco.MjData(model)
    mujoco.mj_kinematics(model, probe)
    nominal_ee = np.eye(4)
    nominal_ee[:3, :3], nominal_ee[:3, 3] = probe.xmat[ee].reshape(3, 3), probe.xpos[ee]
    world_from_nominal = estimate.world_from_ee @ np.linalg.inv(nominal_ee)
    ids, solids = [], []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in rigid:
            continue
        nominal = WorldSolid.from_geom(model, probe, gid, convex_material_calibration_id="robot-collision-outer-only")
        solid = replace(nominal, world_from_local=world_from_nominal @ nominal.world_from_local)
        radius = np.linalg.norm(box_corners(solid.bounds) - estimate.world_from_ee[:3, 3], axis=1).max()
        solids.append(replace(solid, offset_m=pose_uncertainty_margin(radius, position_error_m, rotation_error_rad)))
        ids.append(gid)
    return tuple(ids), tuple(solids), envelope.model_signature


@dataclass(frozen=True)
class RobotViewGuard:
    capture: CaptureInterval
    registered_depth: np.ndarray
    source_points_world: np.ndarray
    registered_points_world: np.ndarray
    geom_ids: tuple[int, ...]
    outer_solids: tuple[WorldSolid, ...]
    model_signature: str
    calibration_id: str
    registration_profile: str
    point_error_m: float
    robot_position_error_m: float = 0.0
    robot_rotation_error_rad: float = 0.0

    def __post_init__(self):
        depth, points = np.asarray(self.registered_depth), np.asarray(self.source_points_world, float)
        registered = np.asarray(self.registered_points_world, float)
        if (
            not isinstance(self.capture, CaptureInterval)
            or depth.ndim != 2
            or points.shape != (*depth.shape, 3)
            or registered.shape != points.shape
            or not self.geom_ids
            or len(self.geom_ids) != len(self.outer_solids)
            or len(set(self.geom_ids)) != len(self.geom_ids)
            or any(type(g) is not int or g < 0 for g in self.geom_ids)
            or any(not isinstance(s, WorldSolid) or s.offset_m < 0 for s in self.outer_solids)
            or not all(
                isinstance(s, str) and s for s in (self.model_signature, self.calibration_id, self.registration_profile)
            )
            or not np.isfinite([self.point_error_m, self.robot_position_error_m, self.robot_rotation_error_rad]).all()
            or min(self.point_error_m, self.robot_position_error_m, self.robot_rotation_error_rad) < 0
            or self.robot_rotation_error_rad > np.pi
        ):
            raise ValueError("invalid_robot_view_guard")
        valid = np.isfinite(depth) & (depth > 0)
        if (
            not np.isfinite(points[valid]).all()
            or np.any(np.isfinite(points[~valid]))
            or not np.isfinite(registered[valid]).all()
        ):
            raise ValueError("invalid_robot_view_guard")
        for name, value in (
            ("registered_depth", depth),
            ("source_points_world", points),
            ("registered_points_world", registered),
        ):
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape))
        object.__setattr__(self, "geom_ids", tuple(self.geom_ids))
        object.__setattr__(self, "outer_solids", tuple(self.outer_solids))

    @classmethod
    def from_registered_depth(
        cls,
        *,
        capture,
        raw_depth,
        depth_k,
        rgb_k,
        rgb_from_depth,
        world_from_rgb,
        output_shape,
        geom_ids,
        outer_solids,
        model_signature,
        calibration_id,
        point_error_m,
        robot_position_error_m=0.0,
        robot_rotation_error_rad=0.0,
    ):
        rgb_k = np.asarray(rgb_k, float)
        if (
            rgb_k.shape != (3, 3)
            or not np.allclose(rgb_k[2], [0.0, 0.0, 1.0])
            or not np.allclose([rgb_k[0, 1], rgb_k[1, 0]], 0.0)
        ):
            raise ValueError("invalid_observed_camera_geometry")
        aligned, provenance = register_depth(
            raw_depth, depth_k, rgb_k, rgb_from_depth, output_shape, return_provenance=True
        )
        pose = rigid_copy(world_from_rgb) @ rigid_copy(rgb_from_depth)
        raw_depth = np.asarray(raw_depth, float)
        source = provenance["source_pixel_index"]
        valid = source >= 0
        sy, sx = np.divmod(source[valid], raw_depth.shape[1])
        rays = np.stack((sx, sy, np.ones(len(sx))), axis=-1) @ np.linalg.inv(depth_k).T
        points = (rays * raw_depth[sy, sx, None]) @ pose[:3, :3].T + pose[:3, 3]
        world = np.full((*output_shape, 3), np.nan)
        world[valid] = points
        y, x = np.indices(output_shape, dtype=float)
        k = np.asarray(rgb_k, float)
        rgb_points = np.stack(
            (
                (x - k[0, 2]) * (aligned.astype(float) / k[0, 0]),
                (y - k[1, 2]) * (aligned.astype(float) / k[1, 1]),
                aligned,
            ),
            axis=-1,
        )
        rgb_pose = rigid_copy(world_from_rgb)
        registered = rgb_points @ rgb_pose[:3, :3].T + rgb_pose[:3, 3]
        digest = hashlib.sha256(repr((raw_depth.shape, tuple(output_shape))).encode())
        for value in (depth_k, rgb_k, rgb_from_depth):
            # Rigid-pair FK subtraction can differ at machine epsilon as the
            # robot moves. Normalize only the calibration fingerprint, never
            # the actual projection or measured endpoint coordinates.
            normalized = np.round(np.asarray(value, float), 12)
            normalized[normalized == 0.0] = 0.0  # Canonicalize signed FK roundoff zero.
            digest.update(normalized.tobytes())
        return cls(
            capture,
            aligned,
            world,
            registered,
            tuple(geom_ids),
            tuple(outer_solids),
            model_signature,
            calibration_id,
            digest.hexdigest(),
            point_error_m,
            robot_position_error_m,
            robot_rotation_error_rad,
        )

    @property
    def profile_identity(self):
        return (
            self.geom_ids,
            self.model_signature,
            self.calibration_id,
            self.registration_profile,
            self.point_error_m,
            self.robot_position_error_m,
            self.robot_rotation_error_rad,
        )

    def require(self, capture, depth, world, *, now):
        self.capture.require(now=now)
        if self.capture != capture or not np.array_equal(depth, self.registered_depth, equal_nan=True):
            raise ObservationUnavailable("robot_view_capture_mismatch")
        valid = np.isfinite(self.registered_depth) & (self.registered_depth > 0)
        if np.asarray(world).shape != self.registered_points_world.shape or not np.allclose(
            np.asarray(world)[valid], self.registered_points_world[valid], rtol=0, atol=1e-10, equal_nan=True
        ):
            raise ObservationUnavailable("robot_view_capture_mismatch")

    def require_no_interference(self, mask):
        points = self.source_points_world[np.asarray(mask, bool)]
        if not np.isfinite(points).all():
            raise ObservationUnavailable("rgbd_correspondence_unavailable")
        if any(np.any(solid.lower_distance(points) <= self.point_error_m) for solid in self.outer_solids):
            raise ObservationUnavailable("rgbd_robot_interference")
