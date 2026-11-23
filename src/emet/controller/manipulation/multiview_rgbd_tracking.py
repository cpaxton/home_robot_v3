# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Same-sample RGBD view handoff without commanded poses or per-view pose fiction.

A single partial view may supply measured surfaces while another preserves full
geometry. No candidate is selected by trimming/fitting competing components. A
camera with unknown association contributes nothing; all-view/global loss latches.
These remain diagnostic associations, not calibrated execution certificates.
"""

import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from emet.controller.manipulation.geometry_models import AxialMeshSurface, PrimitiveModel
from emet.controller.manipulation.geometry_tracking import GeometryFit, fit_geometry
from emet.controller.manipulation.moving_rgbd_tracking import MovingRgbdTracker, _arrays, _frozen
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.manipulation.rgbd_capture_geometry import RgbdCaptureGeometry
from emet.controller.manipulation.robot_view_guard import RobotViewGuard
from emet.controller.manipulation.shape_pose import shape_rotation_distance


@dataclass(frozen=True)
class SynchronizedRgbdView:
    rgb: np.ndarray
    depth: np.ndarray
    world: np.ndarray
    capture: CaptureInterval
    robot_guard: RobotViewGuard | None = None
    geometry: RgbdCaptureGeometry | None = None
    foreground_mask: np.ndarray | None = None

    def __post_init__(self):
        arrays = _arrays(self.rgb, self.depth, self.world)
        for name, array in zip(("rgb", "depth", "world"), arrays, strict=True):
            object.__setattr__(self, name, _frozen(array))
        if self.foreground_mask is not None:
            mask = np.asarray(self.foreground_mask)
            if (
                mask.dtype != bool or mask.shape != self.depth.shape or mask.sum() < 25
                or not np.isfinite(self.world[mask]).all() or np.any(self.depth[mask] <= 0)
            ):
                raise ValueError("invalid_moving_reference_mask")
            object.__setattr__(self, "foreground_mask", _frozen(mask))
        if self.robot_guard is not None and not isinstance(self.robot_guard, RobotViewGuard):
            raise ValueError("invalid_robot_view_guard")
        if self.geometry is not None:
            if not isinstance(self.geometry, RgbdCaptureGeometry):
                raise ValueError("invalid_rgbd_capture_geometry")
            self.geometry.require(self, now=self.capture.received_monotonic)


@dataclass(frozen=True)
class MultiViewMaskFit:
    fit: GeometryFit
    capture: CaptureInterval
    masks: tuple[tuple[str, np.ndarray], ...]
    view_rejections: tuple[tuple[str, str], ...]
    reference_id: str
    processing_s: float
    clipped_views: tuple[str, ...] = ()
    component_policy: str = "single_component"
    appearance_model: str = "chromaticity"
    execution_qualified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "masks", tuple((name, _frozen(mask)) for name, mask in self.masks))

    def as_dict(self):
        return {
            **self.fit.as_dict(),
            "views_used": [name for name, _ in self.masks],
            "mask_pixels_by_view": {name: int(mask.sum()) for name, mask in self.masks},
            "view_rejections": dict(self.view_rejections),
            "reference_id": self.reference_id,
            "processing_s": self.processing_s,
            "clipped_views": list(self.clipped_views),
            "component_policy": self.component_policy,
            "appearance_model": self.appearance_model,
            "execution_qualified": False,
        }


class MultiViewRgbdTracker(MovingRgbdTracker):
    """One semantic/appearance reference, continuously observed across cameras.

    All views must describe ONE source-synchronized sample. The producer owns
    capture/encoder alignment and calibration; matching receive times cannot
    establish this. New camera views associate against the still-live measured
    geometry and fixed appearance prior, never restart a lost global identity.

    ``complete_prior_union`` fits every reachable component, including singleton
    fragments, against one declared primitive. No component subset is selected.
    ``allow_clipped_views`` admits only measured points at the image edge; it
    neither fills hidden geometry nor waives pose observability. Both profiles
    require a complete primitive prior and remain diagnostic.
    """

    def __init__(
        self,
        model,
        *,
        reference_view,
        reference,
        mask,
        scene_epoch,
        allow_clipped_views=False,
        component_policy="single_component",
        single_view_fallback=False,
        best_fit_hypothesis=False,
        best_fit_view=False,
        **kwargs,
    ):
        if not isinstance(reference_view, str) or not reference_view:
            raise ValueError("invalid_rgbd_view_identity")
        if type(allow_clipped_views) is not bool or (allow_clipped_views and not isinstance(model, PrimitiveModel)):
            raise ValueError("clipped_views_require_declared_primitive")
        if component_policy not in {"single_component", "complete_prior_union"} or (
            component_policy == "complete_prior_union" and not isinstance(model, (PrimitiveModel, AxialMeshSurface))
        ):
            raise ValueError("invalid_component_policy")
        capture = reference.capture
        if type(single_view_fallback) is not bool:
            raise ValueError("invalid_single_view_fallback")
        if type(best_fit_hypothesis) is not bool:
            raise ValueError("invalid_best_fit_hypothesis")
        if type(best_fit_view) is not bool or best_fit_view and not single_view_fallback:
            raise ValueError("invalid_best_fit_view")
        self.best_fit_view = best_fit_view
        self.view_selections = 0
        self.last_view_candidates = ()
        self.best_fit_hypothesis = best_fit_hypothesis
        self.hypothesis_selections = 0
        self.single_view_fallback = single_view_fallback
        self.allow_clipped_views = allow_clipped_views
        self.component_policy = component_policy
        self.last_view_rejections = ()
        self._guard_profiles = {
            reference_view: (reference.robot_guard.profile_identity if reference.robot_guard is not None else None)
        }
        self._geometry_profiles = {reference_view: reference.geometry.profile_identity if reference.geometry else None}
        capture.require(now=kwargs.get("clock", time.monotonic)())
        if reference.geometry is not None:
            reference.geometry.require(reference, now=kwargs.get("clock", time.monotonic)())
        if reference.robot_guard is not None:
            reference.robot_guard.require(
                capture, reference.depth, reference.world, now=kwargs.get("clock", time.monotonic)()
            )
            reference.robot_guard.require_no_interference(mask)
        super().__init__(
            model,
            observation_id=capture.observation_id,
            received_monotonic=capture.received_monotonic,
            scene_epoch=scene_epoch,
            rgb=reference.rgb,
            depth=reference.depth,
            world=reference.world,
            mask=mask,
            **kwargs,
        )
        seed = self.latest
        if self.best_fit_hypothesis and len(seed.fit.poses) > 1:
            seed = replace(seed, fit=self._preferred_hypothesis(seed.fit))
            self.hypothesis_selections += 1
        capture.require(now=self.clock())
        self.latest = MultiViewMaskFit(
            seed.fit,
            capture,
            ((reference_view, seed.mask),),
            (),
            seed.reference_id,
            seed.processing_s,
            component_policy=self.component_policy,
            appearance_model=self.appearance_model,
        )

    @staticmethod
    def _preferred_hypothesis(fit):
        index = int(np.argmin(fit.rms_m))
        return replace(fit, poses=fit.poses[index:index+1], rms_m=fit.rms_m[index:index+1],
                       nullspaces=(fit.nullspaces[index],))

    def _require_fit(self, fit):
        if self.best_fit_hypothesis and len(fit.poses) > 1:
            self.last_candidate_fit = fit
            fit = self._preferred_hypothesis(fit)
        MovingRgbdTracker._require_fit(fit)

    def update(self, views, *, scene_epoch):
        previous, self.latest = self.latest, None
        if previous is None:
            raise ObservationUnavailable("moving_tracking_lost_reinitialize")
        started = time.perf_counter()
        self.last_view_rejections = ()
        if not 1 <= len(views) <= 4 or any(not isinstance(name, str) or not name for name in views):
            raise ObservationUnavailable("invalid_rgbd_views")
        capture = next(iter(views.values())).capture
        if any(view.capture != capture for view in views.values()):
            raise ObservationUnavailable("multiview_capture_mismatch")
        capture.require(now=self.clock(), after=previous.capture.latest)
        # Either measurement may have occurred anywhere in its capture interval.
        # Reachability needs the largest elapsed time, not the minimum gap.
        dt = capture.latest - previous.capture.earliest
        if (
            not views
            or not 0 < dt <= self.gap
            or scene_epoch != self.scene_epoch
            or capture.calibration_id != previous.capture.calibration_id
            or capture.observation_id in self._seen
            or len(self._seen) >= 4096
        ):
            raise ObservationUnavailable("moving_tracking_epoch_or_capture_invalid")
        self._seen.add(capture.observation_id)
        centers = previous.fit.poses[:, :3, 3]
        displacement = self.speed * dt + self.margin
        masks, points, rejected, clipped = [], [], [], []
        for name, view in views.items():
            rgb, depth, world = view.rgb, view.depth, view.world
            geometry_profile = view.geometry.profile_identity if view.geometry else None
            if name in self._geometry_profiles and geometry_profile != self._geometry_profiles[name]:
                raise ObservationUnavailable("rgbd_geometry_profile_changed")
            self._geometry_profiles[name] = geometry_profile
            if view.geometry is not None:
                view.geometry.require(view, now=self.clock())
            profile = view.robot_guard.profile_identity if view.robot_guard is not None else None
            if name in self._guard_profiles and profile != self._guard_profiles[name]:
                raise ObservationUnavailable("robot_view_profile_changed")
            self._guard_profiles[name] = profile
            if view.robot_guard is not None:
                view.robot_guard.require(capture, depth, world, now=self.clock())
            try:
                if view.foreground_mask is not None:
                    mask = view.foreground_mask
                    radius = float(np.linalg.norm(self.model.half_extents))
                    if np.any(cKDTree(centers).query(world[mask])[0] > radius + displacement):
                        raise ObservationUnavailable("moving_mask_motion_bound_exceeded")
                else:
                    mask = self._candidate_mask(
                        rgb, depth, world, centers=centers, displacement=displacement,
                        allow_image_boundary=self.allow_clipped_views,
                        merge_reachable=self.component_policy == "complete_prior_union",
                    )
                if view.robot_guard is not None:
                    view.robot_guard.require_no_interference(mask)
            except ObservationUnavailable as error:
                rejected.append((name, str(error)))
                continue
            masks.append((name, mask))
            points.append(world[mask])
            if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                clipped.append(name)
        self.last_view_rejections = tuple(rejected)
        if not masks:
            raise ObservationUnavailable("moving_multiview_association_unknown")
        def fit_points(selected):
            candidate = fit_geometry(
                selected, self.model, object_ref=self.object_ref,
                observation_ids=(capture.observation_id,), received_monotonic=capture.received_monotonic,
                initial_poses=previous.fit.poses[:12], max_evaluations=self.max_evaluations,
                max_starts=self.max_starts, max_points=self.max_fit_points, max_residual_m=self.max_residual_m)
            self.last_candidate_fit = candidate
            if self.best_fit_hypothesis and len(candidate.poses) > 1:
                candidate = self._preferred_hypothesis(candidate)
                self.hypothesis_selections += 1
            self._require_fit(candidate)
            return candidate

        try:
            fit = fit_points(np.concatenate(points))
        except ObservationUnavailable as error:
            if not self.single_view_fallback or len(points) < 2 or str(error) not in {
                'geometry_tracking_no_match', 'geometry_tracking_unobservable',
                'geometry_tracking_budget_exhausted', 'moving_geometry_ambiguous',
            }:
                raise
            qualified = []
            for (name, mask), selected in zip(masks, points, strict=True):
                try:
                    candidate = fit_points(selected)
                    qualified.append((name, mask, candidate))
                except ObservationUnavailable as rejected_fit:
                    rejected.append((name, str(rejected_fit)))
            if not qualified:
                raise error
            pose_centers = np.concatenate([entry[2].poses[:, :3, 3] for entry in qualified])
            self.last_view_candidates = tuple((name, candidate) for name, _, candidate in qualified)
            centers_conflict = np.linalg.norm(np.ptp(pose_centers, axis=0)) > .005
            best = min(qualified, key=lambda entry: float(np.min(entry[2].rms_m)))
            representative = best[2].poses[0, :3, :3]
            rotations_conflict = any(shape_rotation_distance(self.model, representative, pose[:3, :3]) > .03
                   for _, _, candidate in qualified for pose in candidate.poses)
            if centers_conflict or rotations_conflict:
                if not self.best_fit_view:
                    raise ObservationUnavailable('moving_geometry_ambiguous') from error
                self.view_selections += 1
                rejected.extend((name, 'diagnostic_conflicting_view_not_selected')
                                for name, _, _ in qualified if name != best[0])
            name, mask, fit = best
            rejected.extend((other, 'fusion_rejected_single_view_fallback')
                            for other, _ in masks if other != name and other not in {row[0] for row in rejected})
            masks = [(name, mask)]
            clipped = [camera for camera in clipped if camera == name]
            self.last_candidate_fit = fit
            self.last_view_rejections = tuple(rejected)
        if np.any(cKDTree(centers).query(fit.poses[:, :3, 3])[0] > displacement):
            raise ObservationUnavailable("moving_mask_motion_bound_exceeded")
        capture.require(now=self.clock())
        elapsed = time.perf_counter() - started
        if elapsed > self.max_update_s:
            raise ObservationUnavailable("moving_tracking_time_budget_exhausted")
        self.latest = MultiViewMaskFit(
            fit,
            capture,
            tuple(masks),
            tuple(rejected),
            self.reference_id,
            elapsed,
            tuple(clipped),
            self.component_policy,
            self.appearance_model,
        )
        return self.latest
