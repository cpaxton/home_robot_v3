# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Same-sample RGBD view handoff without commanded poses or per-view pose fiction.

A single partial view may supply measured surfaces while another preserves full
geometry. No candidate is selected by trimming/fitting competing components. A
camera with unknown association contributes nothing; all-view/global loss latches.
These remain diagnostic associations, not calibrated execution certificates.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.geometry_tracking import GeometryFit, fit_geometry
from emet.controller.manipulation.moving_rgbd_tracking import MovingRgbdTracker, _arrays, _frozen
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval


@dataclass(frozen=True)
class SynchronizedRgbdView:
    rgb: np.ndarray
    depth: np.ndarray
    world: np.ndarray
    capture: CaptureInterval

    def __post_init__(self):
        arrays = _arrays(self.rgb, self.depth, self.world)
        for name, array in zip(("rgb", "depth", "world"), arrays, strict=True):
            object.__setattr__(self, name, _frozen(array))


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
        **kwargs,
    ):
        if not isinstance(reference_view, str) or not reference_view:
            raise ValueError("invalid_rgbd_view_identity")
        if type(allow_clipped_views) is not bool or (allow_clipped_views and not isinstance(model, PrimitiveModel)):
            raise ValueError("clipped_views_require_declared_primitive")
        if component_policy not in {"single_component", "complete_prior_union"} or (
            component_policy == "complete_prior_union" and not isinstance(model, PrimitiveModel)
        ):
            raise ValueError("invalid_component_policy")
        capture = reference.capture
        self.allow_clipped_views = allow_clipped_views
        self.component_policy = component_policy
        self.last_view_rejections = ()
        capture.require(now=kwargs.get("clock", time.monotonic)())
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
        dt = capture.earliest - previous.capture.latest
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
            try:
                mask = self._candidate_mask(
                    rgb,
                    depth,
                    world,
                    centers=centers,
                    displacement=displacement,
                    allow_image_boundary=self.allow_clipped_views,
                    merge_reachable=self.component_policy == "complete_prior_union",
                )
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
        fit = fit_geometry(
            np.concatenate(points),
            self.model,
            object_ref=self.object_ref,
            observation_ids=(capture.observation_id,),
            received_monotonic=capture.received_monotonic,
            initial_poses=previous.fit.poses[:12],
            max_evaluations=self.max_evaluations,
        )
        self._require_fit(fit)
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
