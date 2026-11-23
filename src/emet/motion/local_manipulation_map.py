# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Sparse local RGB-D geometry for arm sweeps, independent of navigation voxels."""

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.motion.depth_surface_validity import supported_depth
from emet.motion.local_geometry_distance import cell_separations
from emet.motion.observed_admission import AdmissionDecision
from emet.motion.placement_geometry import box_corners


class LocalManipulationMap:
    def __init__(self, observations, masks, *, resolution_m=.005, depth_error_m=.003, previous=None,
                 simulation_allow_unknown=False):
        if not 0 < resolution_m <= .01 or not 0 <= depth_error_m <= .02:
            raise ValueError('invalid_local_manipulation_resolution')
        self.resolution = float(resolution_m)
        self.error = float(depth_error_m)
        self.simulation_allow_unknown = bool(simulation_allow_unknown)
        self.last_collision = None
        points, self.views = [], []
        if previous is not None:
            if previous.resolution != self.resolution or previous.error != self.error:
                raise ValueError('local_map_profile_changed')
            points.append(previous.surface_points)
            self.views.extend(previous.views[-6:])
        for name, observation in observations.items():
            _, depth, world = capture_arrays(observation)
            valid = supported_depth(depth)
            depth = np.where(valid, depth, 0.)
            excluded = np.asarray(masks[name], bool)
            if excluded.shape != depth.shape:
                raise ValueError('invalid_local_manipulation_mask')
            points.append(world[valid & ~excluded])
            self.views.append((depth, np.asarray(observation.camera_K), np.asarray(observation.camera_pose)))
        measured = np.concatenate(points)
        keys = np.floor(measured / self.resolution).astype(np.int64)
        if len(keys):
            origin = keys.min(axis=0)
            shape = keys.max(axis=0) - origin + 1
            flat = np.ravel_multi_index((keys - origin).T, tuple(shape))
            unique, representatives = np.unique(flat, return_index=True)
            measured = measured[representatives]
            keys = np.column_stack(np.unravel_index(unique, tuple(shape))) + origin
        # Keep measured surface locations separately: voxel centres/volumes
        # are useful for conservative broad-phase queries, not contact identity.
        self.surface_points = measured
        self.surface_tree = cKDTree(measured)
        self.centers = (keys + .5) * self.resolution
        self.tree = cKDTree(self.centers)
        self.radius = np.sqrt(3) * self.resolution / 2 + self.error

    def free_box(self, bounds):
        corners = box_corners(bounds)
        for depth, intrinsic, pose in self.views:
            local = (corners - pose[:3, 3]) @ pose[:3, :3]
            if np.any(local[:, 2] <= 0):
                continue
            projected = local @ intrinsic.T
            uv = projected[:, :2] / projected[:, 2:]
            low = np.floor(uv.min(0) + .5).astype(int)
            high = np.floor(uv.max(0) + .5).astype(int)
            if np.any(low < 0) or np.any(high >= depth.shape[::-1]):
                continue
            pixels = depth[low[1]:high[1]+1, low[0]:high[0]+1]
            if np.all(np.isfinite(pixels) & (pixels > local[:, 2].max() + self.error)):
                return True
        return False

    def query_contact_clearance(self, solid, patches, *, contact_tolerance_m=0., starting_volumes=(), minimum_padding_m=0.):
        """Consume planning reserve only at original, nominally clear patches.

        New surfaces retain model/depth uncertainty and cannot acquire contact
        allowance. Unknown swept space remains subject to measured free rays.
        """
        bounds = solid.bounds
        center, half = bounds.mean(0), (bounds[1]-bounds[0])/2
        ids = np.asarray(self.tree.query_ball_point(center, np.linalg.norm(half)+self.radius), dtype=int)
        allowed = set()
        if len(ids):
            points = self.surface_points[ids]
            associated = patches.assign(points) >= 0
            distances = solid.with_offset(0.).lower_distance(points)
            # A new measured surface that clears both model and depth error
            # may consume travel reserve without being called a new contact.
            # Only an original patch may enter the nominal contact allowance.
            clear = distances > minimum_padding_m+self.error
            contact = associated & (distances >= -contact_tolerance_m)
            allowed = set(ids[clear | contact].tolist())

        return self.query(solid, starting_volumes=starting_volumes, excluded_occupied_ids=allowed,
                          free_space_solid=solid.with_offset(minimum_padding_m))

    def nearby_surfaces(self, solid, *, reach_m=.01):
        """Measured samples and nominal distances, without occupied-cell volume."""
        bounds = solid.bounds
        center, half = bounds.mean(0), (bounds[1]-bounds[0])/2
        ids = self.surface_tree.query_ball_point(center, np.linalg.norm(half)+self.error+reach_m)
        points = self.surface_points[ids]
        if not len(points):
            return np.empty((0, 3)), np.empty(0)
        keep = np.all((points >= bounds[0]-self.error-reach_m) &
                      (points <= bounds[1]+self.error+reach_m), axis=1)
        points = points[keep]
        nominal = solid.with_offset(0.).lower_distance(points)
        keep = nominal-solid.offset_m-self.error < reach_m
        return points[keep], nominal[keep]

    def nearby_distances(self, solid, *, reach_m=.01):
        bounds = solid.bounds
        center, half = bounds.mean(0), (bounds[1] - bounds[0]) / 2
        ids = self.tree.query_ball_point(center, np.linalg.norm(half) + self.radius + reach_m)
        if ids:
            points = self.centers[ids]
            keep = np.all((points >= bounds[0] - self.radius - reach_m)
                          & (points <= bounds[1] + self.radius + reach_m), axis=1)
            ids = np.asarray(ids)[keep].tolist()
        if not ids:
            return {}
        distances = cell_separations(solid, self.centers[ids], resolution=self.resolution) - solid.offset_m - self.error
        return {int(index): float(distance) for index, distance in zip(ids, distances, strict=True)
                if distance < reach_m}

    def query(self, solid, *, starting_volumes=(), max_nodes=20000, check_occupied=True, allow_occupied=None, free_space_solid=None, excluded_occupied_ids=()):
        self.last_collision = None
        bounds = solid.bounds
        center, half = bounds.mean(0), (bounds[1] - bounds[0]) / 2
        nearby = self.tree.query_ball_point(center, np.linalg.norm(half) + self.radius)
        if nearby:
            points = self.centers[nearby]
            keep = np.all((points >= bounds[0] - self.radius) & (points <= bounds[1] + self.radius), axis=1)
            nearby = np.asarray(nearby)[keep].tolist()
        if excluded_occupied_ids:
            nearby = [index for index in nearby if index not in excluded_occupied_ids]
        if check_occupied and nearby:
            lower = solid.lower_distance(self.centers[nearby])
            ids = np.asarray(nearby)[np.argsort(lower)]
            ids = ids[np.sort(lower) <= self.radius]
            # Most colliding samples can be rejected by a few close cells.
            # Clear samples still inspect every candidate with the same kernel.
            for batch in (ids[:32], ids[32:]):
                if not len(batch):
                    continue
                distances = cell_separations(solid, self.centers[batch], resolution=self.resolution)
                blocked = distances <= solid.offset_m + self.error
                if allow_occupied is not None:
                    for index in np.flatnonzero(blocked):
                        if allow_occupied(self.centers[batch[index]], self.radius):
                            blocked[index] = False
                if np.any(blocked):
                    index = int(np.flatnonzero(blocked)[np.argmin(distances[blocked])])
                    self.last_collision = {
                        "point_m": self.centers[batch[index]].tolist(),
                        "estimated_separation_m": float(distances[index]),
                        "padding_m": float(solid.offset_m + self.error),
                        "solid_bounds_m": solid.bounds.tolist(),
                    }
                    return AdmissionDecision(False, 'occupied_local_surface')
        if self.simulation_allow_unknown:
            return AdmissionDecision(True, 'simulation_unknown_space_optimism')
        if free_space_solid is not None:
            # Contact consumes the travel buffer, not the robot's calibrated
            # geometry uncertainty or the sensor's free-ray error band.
            solid = free_space_solid
            bounds = solid.bounds
        same_shape = [volume for volume in starting_volumes if volume.kind == solid.kind
                     and volume.dimensions == solid.dimensions
                     and (volume.kind != 'convex' or np.array_equal(volume.vertices, solid.vertices))]
        for volume in same_shape:
            first, second = volume.world_from_local, solid.world_from_local
            radius = np.linalg.norm(box_corners(volume.bounds) - first[:3, 3], axis=1).max()
            displacement = np.linalg.norm(first[:3, 3] - second[:3, 3])
            displacement += np.linalg.norm(first[:3, :3] - second[:3, :3], ord=2) * radius
            if solid.offset_m + displacement <= volume.offset_m + .001:
                return AdmissionDecision(True, 'starting_volume_within_motion_slack')
        starting_volumes = tuple(replace(volume, offset_m=0.) if volume.kind == 'convex' and volume.offset_m > 0
                                 else volume for volume in starting_volumes)
        pending, nodes = [bounds], 0
        while pending:
            bounds = pending.pop()
            nodes += 1
            if nodes > max_nodes:
                return AdmissionDecision(False, 'local_query_budget_exhausted', nodes)
            if solid.disjoint_box(bounds):
                continue
            if any(volume.contains_box(bounds) for volume in starting_volumes) or self.free_box(bounds):
                continue
            axis = int(np.argmax(bounds[1] - bounds[0]))
            if bounds[1, axis] - bounds[0, axis] <= .001:
                return AdmissionDecision(False, 'unknown_local_sweep', nodes)
            first, second = bounds.copy(), bounds.copy()
            first[1, axis] = second[0, axis] = bounds[:, axis].mean()
            pending.extend((first, second))
        return AdmissionDecision(True, 'local_observed_clearance', nodes)
