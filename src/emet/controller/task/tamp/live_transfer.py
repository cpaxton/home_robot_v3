# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Live RGB-D scene assembly shared by stopped-base transfer ports."""

import threading
import time
import uuid
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.continuous_rgbd import ContinuousRgbdSource, TrackedRgbdCapture
from emet.controller.manipulation.geometry_models import AxialMeshSurface, PrimitiveModel
from emet.controller.manipulation.multiview_rgbd_tracking import MultiViewRgbdTracker, SynchronizedRgbdView
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import VisualEffectFrame
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_support import refresh_support_patch, support_mask_from_seed
from emet.motion.observed_workspace import ObservedOccupancyPartition, ObservedWorkspace

from .observed_capture import capture_arrays
from .observed_source import (
    ObservedTransferEvidence,
    ObservedTransferSource,
    fit_transfer_support,
    partition_transfer_depth,
)


def _tracking_interior(foreground):
    """Prefer interior depth, retaining thin measured fragments for fit validation."""
    eroded = binary_erosion(foreground)
    return eroded if eroded.sum() >= 25 else foreground


@dataclass(frozen=True)
class TransferSelection:
    """Frame-bound foreground and two distinct planar support hypotheses."""

    object_camera: str
    object_mask: np.ndarray
    source_camera: str
    source_seed: dict
    destination_camera: str
    destination_seed: dict
    capture_id: str
    source_mask: np.ndarray | None = None
    destination_mask: np.ndarray | None = None


@dataclass(frozen=True)
class TransferSample:
    capture: object
    observations: dict
    robot_before: object
    robot_after: object
    material: object
    query_robot_geometry: object
    robot_guards: dict
    qualify_surface: object | None = None


def synchronized_views(sample):
    return {
        name: SynchronizedRgbdView(*capture_arrays(obs), sample.capture, sample.robot_guards.get(name))
        for name, obs in sample.observations.items()
    }


def fuse_transfer_partitions(partitions):
    """Fuse simultaneous depth views once, retaining all occupied owners."""
    workspace = ObservedWorkspace.fuse_static(p.workspace for p in partitions)
    owners = {}
    for partition in partitions:
        for name, indices in partition.owners:
            owners.setdefault(name, set()).update(indices)
    points = np.concatenate([p.points_world for p in partitions])
    labels = np.concatenate([p.point_labels for p in partitions])
    return ObservedOccupancyPartition(
        workspace, tuple((name, tuple(sorted(ids))) for name, ids in sorted(owners.items())), points, labels,
    )


def _tracking_cameras(profile, available):
    """Separate semantic initialization view from ongoing tracking views."""
    requested = profile.get("tracking_views")
    if requested is not None:
        if (not isinstance(requested, (tuple, list)) or not requested
                or any(not isinstance(name, str) or name not in available for name in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError("invalid_tracking_views")
        return tuple(requested)
    if profile.get("tracking_reference_only"):
        return (profile["grounding"]["object"]["camera"],)
    return tuple(available)


class LiveTransferScene:
    """Bind live qualified measurements and a bounded static scene to a task.

    A cross-camera sample has one original exposure interval. Fixed support
    witnesses keep their own timestamps. The pose stream runs independently
    of repeated planner reads.
    """

    def __init__(
        self, *, arm, task, read_sample, select, payload_model, fingertip_geom_ids, profile, clock=time.monotonic,
    ):
        self.arm, self.task, self.read_sample, self.select = arm, task, read_sample, select
        self.payload_model, self.profile, self.clock = payload_model, profile, clock
        self.target_surface_association_m = float(profile.get("target_surface_association_m", .008))
        if not np.isfinite(self.target_surface_association_m) or not 0 <= self.target_surface_association_m <= .03:
            raise ValueError("invalid_target_surface_association")
        bounds = np.asarray(profile["workspace"], dtype=float)
        resolution = float(profile["resolution_m"])
        if (
            bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0]) or not np.isfinite(resolution) or resolution <= 0
        ):
            raise ValueError("invalid_observed_workspace")
        extent = np.ceil((bounds[1] - bounds[0]) / resolution - 1e-10)
        self.workspace = np.stack((bounds[0], bounds[0] + extent * resolution))
        enclosure = profile.get("payload_axial_enclosure")
        self.enclosure = PrimitiveModel("cylinder", tuple(enclosure)) if enclosure is not None else None
        self.signature = model_signature(arm.model)
        self.epoch = "transfer:" + uuid.uuid4().hex
        self.stream = None
        self.samples = {}
        self.last_evidence = None
        self.initial_views = None
        self.diagnostics = {}
        self.blockers = []
        self.local_scene = None
        self._map_state = None
        self._map_thread = None
        self._map_stop = threading.Event()
        self._map_failure = None
        self.source = ObservedTransferSource(
            read=self.read, collect=self.collect, current_revision=self.current_revision, invalidate=self.invalidate,
            model_signature=self.signature, fingertip_geom_ids=fingertip_geom_ids, clock=clock,
            check_placement_surface=self.check_placement_surface,
        )

    def _read_views(self):
        sample = self.read_sample(self.epoch)
        views = synchronized_views(sample)
        if self.initial_views is not None and sample.capture.observation_id == self.initial_capture_id:
            if not ContinuousRgbdSource._same_packet(self.initial_views, views):
                raise ObservationUnavailable("capture_replayed")
            return None
        if self.initial_views is not None and self.profile.get("tracking_geometry_window_m") is not None:
            fit = self.stream.tracker.latest
            if fit is None:
                raise ObservationUnavailable("moving_tracking_lost_reinitialize")
            centers = fit.fit.poses[:, :3, 3]
            allowance = float(self.profile["tracking_geometry_window_m"])
            if not 0 < allowance <= .03:
                raise ValueError("invalid_tracking_geometry_window")
            radius = float(np.linalg.norm(self.payload_model.half_extents)) + allowance
            masks = {}
            tracking_cameras = _tracking_cameras(self.profile, sample.observations)
            for name, view in views.items():
                if name not in tracking_cameras:
                    continue
                distance = np.linalg.norm(view.world[..., None, :] - centers, axis=-1).min(axis=-1)
                foreground = (view.depth >= .1) & (distance <= radius)
                # A geometry window is only a search region. Gripper pixels
                # inside it must not bypass the track's original appearance.
                foreground &= self.stream.tracker.appearance_mask(view.rgb)
                if self.enclosure is not None:
                    pose = fit.fit.poses[0]
                    local = (view.world - pose[:3, 3]) @ pose[:3, :3]
                    foreground &= self.enclosure.distance(local.reshape(-1, 3)).reshape(view.depth.shape) <= .008
                # Visible support surfaces are independent measurements and
                # cannot contribute to the tracked object's foreground.
                for support in self.supports:
                    patch = support.patch
                    height = view.world[..., :2] @ patch.plane[:2] + patch.plane[2]
                    on_plane = np.abs(view.world[..., 2] - height) <= .008
                    foreground &= ~on_plane
                foreground = _tracking_interior(foreground)
                if foreground.sum() >= 25:
                    masks[name] = replace(view, foreground_mask=foreground)
            if not masks:
                self.diagnostics["last_empty_tracking_view"] = {
                    "capture_id": sample.capture.observation_id,
                    "counts": {name: int(np.count_nonzero(
                        (view.depth >= .1) & (np.linalg.norm(view.world[..., None, :] - centers, axis=-1).min(axis=-1) <= radius)
                    )) for name, view in views.items() if name in tracking_cameras},
                }
            views = masks
        elif self.initial_views is not None and "box" in self.profile["grounding"]["object"]:
            # A measured stationary camera may reapply its diagnostic pixel ROI
            # to the new depth. Arm motion leaves this scope and uses tracking.
            camera = self.profile["grounding"]["object"]["camera"]
            current = sample.observations[camera].camera_pose
            if np.linalg.norm(current - self.initial_camera_pose) <= .002:
                view = views[camera]
                plane = self.supports[0].patch.plane
                height = view.world[..., :2] @ plane[:2] + plane[2]
                h, w = view.depth.shape
                box = np.asarray(self.profile["grounding"]["object"]["box"])
                x0, y0 = np.floor(box[:2] * [w, h] / 1000).astype(int)
                x1, y1 = np.ceil(box[2:] * [w, h] / 1000).astype(int)
                mask = np.zeros((h, w), bool)
                mask[y0:y1, x0:x1] = (view.depth[y0:y1, x0:x1] >= .1) & (
                    view.world[y0:y1, x0:x1, 2] - height[y0:y1, x0:x1] > .008
                )
                components, _ = label(mask)
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                viable = np.flatnonzero(sizes >= 25)
                if len(viable) != 1:
                    raise ObservationUnavailable("moving_mask_ambiguous" if len(viable) else "moving_mask_lost")
                foreground = binary_erosion(components == viable[0])
                # The original grounded camera verifies the stationary
                # object. Both camera observations remain available to the map.
                views = {camera: replace(view, foreground_mask=foreground)}
        if sample.capture.observation_id in self.samples:
            return views
        self.samples[sample.capture.observation_id] = sample
        while len(self.samples) > 16:
            self.samples.pop(next(iter(self.samples)))
        return views

    def _bootstrap(self):
        started = self.clock()
        views = self._read_views()
        capture = next(iter(views.values())).capture
        sample = self.samples[capture.observation_id]
        if self.profile.get("simulation_allow_unknown") and not any(
            getattr(observation, "is_simulation", False) for observation in sample.observations.values()
        ):
            raise ValueError("unknown_space_override_requires_simulator")
        read_at = self.clock()
        selected = self.select(sample, self.task)
        if selected.capture_id != capture.observation_id:
            raise ObservationUnavailable("capture_identity_mismatch")
        profile = self.profile
        self.diagnostics = {
            "capture_id": capture.observation_id, "foreground_points": int(selected.object_mask.sum()),
            "support_bounds_m": {},
            "bootstrap_s": {"capture_read": read_at - started, "camera_age_at_read": read_at - capture.earliest},
            "local_collision_resolution_m": profile.get("local_resolution_m"),
            "robot_surface_tolerance_m": profile.get("robot_geometry", {}).get("surface_tolerance_m", 0.),
            "collision_proxy_self_association": profile.get("robot_geometry", {}).get("collision_proxy_self_association", False),
            "simulation_collision_bypass": profile.get("simulation_ignore_environment", False),
            "contact_departure_enabled": profile.get("contact_departure", False),
            "tracking_single_view_fallback": profile.get("tracking_single_view_fallback", False),
            "tracking_max_update_s": profile.get("tracking_max_update_s", 1.0),
            "payload_prior_scale": profile.get("payload_prior_scale", 1.0),
            "native_query_warmup_s": getattr(self, "native_query_warmup_s", 0.),
        }
        supports = []
        labels = {name: {} for name in views}
        for camera, seed, measured_mask, ref in (
            (selected.source_camera, selected.source_seed, selected.source_mask, self.task.source_support_ref),
            (selected.destination_camera, selected.destination_seed, selected.destination_mask, self.task.support_ref),
        ):
            observation = sample.observations[camera]
            try:
                mask = support_mask_from_seed(observation, seed, measurement_margin_m=profile["support_margin_m"])
                if measured_mask is not None:
                    mask &= measured_mask
            except ObservationUnavailable:
                if measured_mask is None:
                    raise
                mask = measured_mask
            backing = profile.get("support_backing", {}).get(ref, {})
            supports.append(fit_transfer_support(
                observation, mask, capture=capture, object_ref=self.task.object_ref, support_ref=ref,
                scene_epoch=self.epoch, calibration_id=profile["calibration_id"],
                measurement_margin_m=profile["support_margin_m"],
                payload_motion_error_m=profile["payload_motion_error_m"],
                backing_depth_m=backing.get("depth_m", 0.0), backing_prior_id=backing.get("id", ""),
            ))
            self.diagnostics["support_bounds_m"][ref] = supports[-1].patch.bounds.tolist()
            bounds = supports[-1].patch.bounds
            world = views[camera].world
            labels[camera][ref] = mask & np.all((world >= bounds[0]) & (world <= bounds[1]), axis=-1)
        labels[selected.object_camera][self.task.object_ref] = selected.object_mask
        initial_poses = ()
        if self.enclosure is not None:
            points = views[selected.object_camera].world[selected.object_mask]
            if len(points) < 25:
                raise ObservationUnavailable("transfer_grounding_unavailable")
            xy = (np.quantile(points[:, :2], .05, axis=0) + np.quantile(points[:, :2], .95, axis=0)) / 2
            plane = supports[0].patch.plane
            z = xy @ plane[:2] + plane[2] - self.payload_model.vertices_reference[:, 2].min()
            seed_poses = np.repeat(np.eye(4)[None], 8, axis=0)
            seed_poses[:, :3, :3] = Rotation.from_euler("z", np.arange(8) * 45.0, degrees=True).as_matrix()
            seed_poses[:, :3, 3] = [*xy, z]
            initial_poses = seed_poses
        tracker = MultiViewRgbdTracker(
            self.payload_model, reference_view=selected.object_camera, reference=views[selected.object_camera],
            mask=selected.object_mask, scene_epoch=self.epoch, object_ref=self.task.object_ref, clock=self.clock,
            initial_poses=initial_poses,
            max_starts=profile.get("tracking_max_starts", 6),
            max_fit_points=profile.get("tracking_max_points", 180),
            max_residual_m=profile.get("simulation_fit_residual_m", .003),
            max_update_s=profile.get("tracking_max_update_s", 1.0), max_gap_s=2.0,
            single_view_fallback=profile.get("tracking_single_view_fallback", False),
            best_fit_hypothesis=profile.get("simulation_best_fit_pose", False),
            best_fit_view=profile.get("simulation_best_fit_view", False),
            max_pixels=profile.get("tracking_max_pixels", 100000),
            component_policy=(
                "complete_prior_union" if self.enclosure is not None and isinstance(self.payload_model, AxialMeshSurface)
                else "single_component"
            ),
        )
        fitted = self.clock()
        self.diagnostics["bootstrap_s"]["ground_and_fit"] = fitted - read_at
        self.supports = supports
        self.diagnostics["fit"] = tracker.latest.fit.as_dict()
        pose = self._pose(tracker.latest.fit)
        self.diagnostics["pose_scope"] = "axial_enclosure" if self.enclosure is not None else "unique_rigid_pose"
        self.diagnostics["world_from_payload"] = pose.world_from_object.tolist()
        self.assumption = StaticBackgroundAssumption(
            self.epoch, capture.earliest, capture.earliest + profile["static_s"], profile["calibration_id"],
        )
        # Tracker initialization covers the reference view. Subsequent packets
        # use all cameras, with the same capture identity across views. Start
        # the pose stream before the initial voxel carve to bound capture lag.
        self.stream = ContinuousRgbdSource(
            tracker, initial_views={selected.object_camera: views[selected.object_camera]},
            read_views=self._read_views, history_size=12, clock=self.clock,
            max_transient_fit_rejections=self.profile.get("transient_fit_rejections", 0),
        )
        self.initial_capture_id = capture.observation_id
        self.initial_views = TrackedRgbdCapture(tracker.latest, tuple(sorted(views.items())))
        self.initial_camera_pose = np.array(sample.observations[selected.object_camera].camera_pose, copy=True)
        try:
            self.stream.start()
            self.background = self._fuse(sample, labels, None)
            self.diagnostics["bootstrap_s"]["first_map"] = self.clock() - fitted
        except (ObservationUnavailable, ValueError, RuntimeError, TimeoutError):
            self.invalidate()
            raise

    def _fuse(self, sample, masks, previous):
        profile = self.profile
        if sample.qualify_surface is not None and not len(sample.material.measured_surface_points):
            sample = replace(sample, material=sample.qualify_surface())
        if previous is not None and profile.get("simulation_ignore_environment"):
            # Explicit simulator diagnostic retains the original map geometry
            # and its acquisition ages while advancing capture provenance.
            return replace(previous, latest_capture=sample.capture,
                           capture_ids=previous.capture_ids | {sample.capture.observation_id},
                           capture_history=previous.capture_history + (sample.capture,))
        if sample.material is None or sample.query_robot_geometry is None:
            raise ObservationUnavailable("robot_material_geometry_unavailable")
        masks = {name: dict(entries) for name, entries in masks.items()}
        for name, obs in sample.observations.items():
            _, depth, world = capture_arrays(obs)
            valid = (depth >= 0.1) & np.isfinite(world).all(axis=-1)
            points = world[valid]
            own = np.zeros(len(points), dtype=bool)
            for solid in sample.material.inner_solids:
                bounds = solid.bounds
                within = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
                own[within] |= solid.lower_distance(points[within]) <= -profile["depth_margin_m"]
            own |= sample.material.matches_surface(points, point_error_m=profile["depth_margin_m"])
            mask = np.zeros(depth.shape, dtype=bool)
            mask[valid] = own
            masks[name]["robot"] = mask
        partitions = tuple(
            partition_transfer_depth(
                observation, masks=masks[name], workspace=self.workspace,
                resolution_m=profile["resolution_m"], depth_margin_m=profile["depth_margin_m"],
                current_grid=previous.workspace if previous is not None else None,
                require_surface_support=profile.get("depth_neighbor_support", False),
            )
            for name, observation in sample.observations.items()
        )
        fused = fuse_transfer_partitions(partitions)
        common = {
            "excluded_refs": (self.task.object_ref, "robot"), "scene_epoch": self.epoch,
            "capture": sample.capture, "now": sample.capture.received_monotonic, "robot_material": sample.material,
            "point_error_m": profile["depth_margin_m"],
        }
        if not self.assumption.started_monotonic <= sample.capture.earliest <= self.clock() <= self.assumption.expires_monotonic:
            raise ObservationUnavailable("background_epoch_or_capture_invalid")
        if previous is None:
            return ObservedBackground.from_capture(fused, **common)
        updated = previous.update(fused, expand_free=False, retain_new_obstacles=True, **common)
        if updated.geometry_revision != previous.geometry_revision:
            changes = self.diagnostics.setdefault("recent_map_changes", [])
            old_cells, new_cells = previous.workspace.cells, updated.workspace.cells
            old_owners = dict(previous.occupied_owners)
            changes.append({
                "capture_id": sample.capture.observation_id,
                "new_occupied_cells": int(np.count_nonzero((old_cells != 2) & (new_cells == 2))),
                "lost_free_cells": int(np.count_nonzero((old_cells == 1) & (new_cells == 0))),
                "changed_owners": sorted(
                    ref for ref, cells in updated.occupied_owners if cells != old_owners.get(ref)
                ),
            })
            del changes[:-8]
        conflicts = np.flatnonzero((previous.workspace.cells == 1) & (updated.workspace.cells == 2))
        if len(conflicts):
            # Publish the stricter geometry before rejecting this operation so
            # every prior plan fails revision checks, even on a repeated capture.
            self.background = updated
            owners = {ref: int(np.intersect1d(conflicts, ids).size) for ref, ids in updated.occupied_owners}
            self.diagnostics["background_conflict"] = {
                "free_to_occupied_cells": len(conflicts),
                "owners": {ref: count for ref, count in owners.items() if count},
                "returns": self._conflicting_returns(sample, masks, previous.workspace, conflicts),
            }
            if not self.profile.get("simulation_ignore_environment"):
                raise ObservationUnavailable("background_scene_changed")
        return updated

    def _conflicting_returns(self, sample, masks, grid, conflicts):
        """Trace a bounded set of collision-map contradictions to measured pixels."""
        rows = []
        first = self.samples.get(self.initial_capture_id)
        arrays = {name: capture_arrays(observation) for name, observation in sample.observations.items()}
        for index in conflicts[:12]:
            coordinate = np.array(np.unravel_index(index, grid.cells.shape))
            low = grid.origin + coordinate * grid.resolution
            high = low + grid.resolution
            row = {"bounds_m": [low.tolist(), high.tolist()], "cameras": {}}
            rows.append(row)
            for name, observation in sample.observations.items():
                _, depth, world = arrays[name]
                inside = (depth >= .1) & (depth <= 5.) & np.all((world >= low) & (world < high), axis=-1)
                y, x = np.nonzero(inside)
                if not len(y):
                    continue
                values = depth[y, x]
                connected = np.zeros(len(values), dtype=bool)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    valid = (ny >= 0) & (ny < depth.shape[0]) & (nx >= 0) & (nx < depth.shape[1])
                    neighbor = depth[ny[valid], nx[valid]]
                    connected[valid] |= (neighbor >= .1) & (np.abs(neighbor - values[valid]) <= .01)
                camera_row = {
                    "pixels": len(y), "pixel_bounds_xy": [[int(x.min()), int(y.min())], [int(x.max()), int(y.max())]],
                    "depth_range_m": [float(values.min()), float(values.max())],
                    "world_range_m": [world[inside].min(0).tolist(), world[inside].max(0).tolist()],
                    "pixels_with_depth_consistent_neighbor": int(connected.sum()),
                    "masks": {ref: int((inside & mask).sum()) for ref, mask in masks[name].items()},
                }
                if first is not None:
                    initial = first.observations[name]
                    pose = initial.camera_pose
                    points = (world[inside] - pose[:3, 3]) @ pose[:3, :3]
                    front = points[:, 2] > 0
                    projected = points[front] @ initial.camera_K.T
                    pixel = np.floor(projected[:, :2] / projected[:, 2:] + .5).astype(int)
                    valid = np.all((pixel >= 0) & (pixel < initial.depth.shape[::-1]), axis=1)
                    x0, y0 = pixel[valid].T
                    offsets = initial.depth[y0, x0] - points[front][valid, 2]
                    if len(offsets):
                        camera_row["initial_depth_minus_current_return_m"] = [float(offsets.min()), float(offsets.max())]
                    camera_row["camera_translation_m"] = float(np.linalg.norm(
                        observation.camera_pose[:3, 3] - pose[:3, 3]
                    ))
                row["cameras"][name] = camera_row
        return rows

    def _tracked_masks(self, sample, tracked):
        masks = {name: {} for name in sample.observations}
        for name, obs in sample.observations.items():
            _, depth, world = capture_arrays(obs)
            for contact in self.supports:
                patch = contact.patch
                mask = (depth >= .1) & np.all((world >= patch.bounds[0]) & (world <= patch.bounds[1]), axis=-1)
                height = world[..., :2] @ patch.plane[:2] + patch.plane[2]
                masks[name][contact.support_ref] = mask & (np.abs(world[..., 2] - height) <= patch.measurement_margin_m)
        for name, mask in tracked.fit.masks:
            masks[name][self.task.object_ref] = mask
        if self.enclosure is not None:
            pose = tracked.fit.fit.poses[0]
            for name, observation in sample.observations.items():
                _, depth, world = capture_arrays(observation)
                local = (world-pose[:3, 3]) @ pose[:3, :3]
                consistent = (depth >= .1) & (self.enclosure.distance(local.reshape(-1, 3)).reshape(depth.shape)
                                              <= self.target_surface_association_m)
                masks[name][self.task.object_ref] = masks[name].get(self.task.object_ref, False) | consistent
        return masks

    def _pose(self, fit):
        profile = self.profile
        if profile.get("simulation_best_fit_pose") and len(fit.poses) > 1:
            # Explicit diagnostic hypothesis selection, never a claim that the
            # other measured fits have become impossible. Keep the full fit in
            # the trace, and keep age, observability and search-boundary checks.
            self.diagnostics["best_fit_pose_selection"] = fit.as_dict()
            index = int(np.argmin(fit.rms_m))
            fit = replace(fit, poses=fit.poses[index:index+1], rms_m=fit.rms_m[index:index+1],
                          nullspaces=(fit.nullspaces[index],))
        if self.enclosure is not None:
            return fit.require_axial_enclosure(
                self.enclosure, calibration_id=profile["calibration_id"], now=self.clock(),
                position_error_m=profile["position_error_m"], rotation_error_rad=profile["rotation_error_rad"],
            )
        return fit.require_unique(
            now=self.clock(), position_error_m=profile["position_error_m"], rotation_error_rad=profile["rotation_error_rad"],
        )

    def _frame(self, tracked):
        sample = self.samples.get(tracked.fit.capture.observation_id)
        if sample is None:
            raise ObservationUnavailable("capture_identity_mismatch")
        profile = self.profile
        fit = tracked.fit
        pose = self._pose(fit.fit)
        geometry = self.enclosure if self.enclosure is not None else self.payload_model
        payload = PayloadGeometryEvidence(
            self.task.object_ref, fit.reference_id, self.payload_model.model_id,
            geometry.vertices_reference, pose, "known_model_prior",
        )
        frame = VisualEffectFrame(
            payload, sample.capture, sample.robot_before, sample.robot_after,
            profile["position_error_m"], profile["rotation_error_rad"],
            profile["alignment_error_m"], profile["alignment_error_rad"],
        )
        frame.require(now=self.clock())
        return sample, frame

    def read(self, object_ref, support_ref):
        if (object_ref, support_ref) != (self.task.object_ref, self.task.support_ref):
            raise ObservationUnavailable("payload_identity_mismatch")
        try:
            return self._read_evidence()
        except (ObservationUnavailable, ValueError, RuntimeError, TimeoutError) as exc:
            self.diagnostics["blocked_reason"] = str(exc)
            self.blockers = [*self.blockers[-7:], str(exc)]
            raise

    def _build_local_scene(self, sample, tracked, previous):
        from emet.motion.local_manipulation_map import LocalManipulationMap

        if sample.qualify_surface is not None and not len(sample.material.measured_surface_points):
            sample = replace(sample, material=sample.qualify_surface())
        excluded = {}
        tracked_masks = dict(tracked.fit.masks)
        for name, observation in sample.observations.items():
            _, depth, world = capture_arrays(observation)
            own = sample.material.matches_surface(world.reshape(-1, 3), point_error_m=self.profile["depth_margin_m"])
            excluded[name] = own.reshape(depth.shape) | tracked_masks.get(name, np.zeros(depth.shape, bool))
            if self.enclosure is not None:
                pose = tracked.fit.fit.poses[0]
                local = (world - pose[:3, 3]) @ pose[:3, :3]
                excluded[name] |= self.enclosure.distance(local.reshape(-1, 3)).reshape(depth.shape) <= self.target_surface_association_m
        return LocalManipulationMap(
            sample.observations, excluded, resolution_m=self.profile["local_resolution_m"],
            depth_error_m=self.profile["depth_margin_m"], previous=previous,
            simulation_allow_unknown=self.profile.get("simulation_allow_unknown", False),
        )

    def _update_local_maps(self):
        while not self._map_stop.wait(.1):
            try:
                tracked = self.stream.snapshot()
                sample, _ = self._frame(tracked)
                if sample.qualify_surface is not None:
                    sample = replace(sample, material=sample.qualify_surface())
                background, local = self._map_state
                if sample.capture.latest <= background.latest_capture.latest:
                    continue
                try:
                    background = self._fuse(sample, self._tracked_masks(sample, tracked), background)
                except ObservationUnavailable as exc:
                    if str(exc) != "background_scene_changed":
                        raise
                    background = self.background
                    self._map_failure = str(exc)
                local = self._build_local_scene(sample, tracked, local)
                self.diagnostics["background_processing_lag_s"] = self.clock() - sample.capture.earliest
                self._map_state = (background, local)
            except (ObservationUnavailable, ValueError, RuntimeError, TimeoutError) as exc:
                self._map_failure = str(exc)
                return

    def _read_evidence(self):
        if self.stream is None:
            self._bootstrap()
        tracked = self.stream.snapshot()
        sample, frame = self._frame(tracked)
        if self.profile.get("local_resolution_m") is not None:
            if self._map_state is None:
                local = self._build_local_scene(sample, tracked, None)
                self._map_state = (self.background, local)
                self._map_stop.clear()
                self._map_thread = threading.Thread(target=self._update_local_maps, name="observed-local-map", daemon=True)
                self._map_thread.start()
            if self._map_failure is not None:
                failure = self._map_failure
                if failure == "background_scene_changed":
                    self._map_failure = None
                raise ObservationUnavailable(failure)
            background, local = self._map_state
            # The map retains acquisition times under the bounded static-scene
            # assumption. Current robot/object state comes from the pose stream.
            sample, frame = self._frame(self.stream.snapshot())
        else:
            if self.last_evidence is not None and self.last_evidence.frame.capture == sample.capture:
                return self.last_evidence
            if self.background.latest_capture != sample.capture:
                self.background = self._fuse(sample, self._tracked_masks(sample, tracked), self.background)
            background, local = self.background, None
        self.last_evidence = ObservedTransferEvidence(
            frame, background, sample.material, *self.supports, sample.query_robot_geometry, self.assumption, local,
        )
        return self.last_evidence

    def collect(self, object_ref, support_ref, *, after):
        try:
            self.read(object_ref, support_ref)
            return tuple(self._frame(packet)[1] for packet in self.stream.collect(after=after, min_span_s=.3))
        except ObservationUnavailable as exc:
            self.blockers = [*self.blockers[-7:], f"collect: {exc}"]
            raise

    def check_placement_surface(self, bounds):
        try:
            _, packet = self.stream.collect(after=self.clock(), min_span_s=.05, timeout_s=1.2)
        except ObservationUnavailable as exc:
            if str(exc) == "visual_evidence_incomplete":
                return False
            raise
        sample, _ = self._frame(packet)
        reference = self.supports[1].patch
        for name in ("wrist", "scene"):
            observation = sample.observations.get(name)
            if observation is None:
                continue
            try:
                patch = refresh_support_patch(observation, reference, observation_id=sample.capture.observation_id)
            except ObservationUnavailable:
                continue
            margin = patch.measurement_margin_m
            if np.all(bounds[0, :2] >= patch.bounds[0, :2] + margin) and np.all(
                bounds[1, :2] <= patch.bounds[1, :2] - margin
            ):
                return True
        return False

    def current_revision(self):
        background = self._map_state[0] if self._map_state is not None else getattr(self, "background", None)
        return background.geometry_revision if self.last_evidence is not None and background is not None else ""

    def invalidate(self):
        self._map_stop.set()
        if self._map_thread is not None:
            self._map_thread.join(timeout=5.)
            if self._map_thread.is_alive():
                raise RuntimeError("local_map_shutdown_timeout")
        self._map_thread, self._map_state, self._map_failure = None, None, None
        if self.stream is not None:
            self.stream.close()
            rejection = getattr(self.stream, "last_fit_rejection", None)
            if rejection is not None:
                self.last_tracking_rejection = rejection
            self.diagnostics["best_fit_hypothesis_selections"] = self.stream.tracker.hypothesis_selections
            self.diagnostics["best_fit_view_selections"] = self.stream.tracker.view_selections
            self.diagnostics["last_view_candidates"] = {
                name: fit.as_dict() for name, fit in self.stream.tracker.last_view_candidates}
            candidate = getattr(self.stream.tracker, "last_candidate_fit", None)
            if candidate is not None:
                self.diagnostics["last_candidate_fit"] = candidate.as_dict()
            self.diagnostics["tracking_captures"] = self.stream.processed_captures
            self.diagnostics["duplicate_packets"] = self.stream.duplicate_packets
            self.diagnostics["rejected_fit_captures"] = self.stream.rejected_fit_captures
            self.diagnostics["tracking_view_rejections"] = self.stream.tracker.last_view_rejections
            self.diagnostics["recent_capture_timings_s"] = [
                {
                    "since_seed": packet.fit.capture.earliest - self.assumption.started_monotonic,
                    "delivery": packet.fit.capture.received_monotonic - packet.fit.capture.earliest,
                    "fit": packet.fit.processing_s,
                }
                for packet in self.stream.history()
            ]
        self.stream, self.last_evidence = None, None
        self.samples.clear()
        self.epoch = "transfer:" + uuid.uuid4().hex
        self.initial_views = None
        self.local_scene = None
