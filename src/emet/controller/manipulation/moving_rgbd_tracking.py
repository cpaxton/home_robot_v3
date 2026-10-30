# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Moving-instance mask candidates from measured RGBD and bounded continuity.

No commanded object/EE transform, simulator body mask or scene pose is accepted.
Lost tracking is latched; a new semantic initialization/track is required. These
are candidate associations, not calibrated identity or execution certificates.
"""

import time
import uuid
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from emet.controller.manipulation.geometry_tracking import GeometryFit, fit_geometry
from emet.controller.manipulation.native_inputs import ObservationUnavailable


def _frozen(array):
    array = np.asarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _arrays(rgb, depth, world):
    rgb, depth, world = np.asarray(rgb), np.asarray(depth), np.asarray(world)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8 or world.shape != (*depth.shape, 3):
        raise ValueError("invalid_moving_rgbd_inputs")
    return rgb, depth, world


def _chroma(rgb):
    values = np.asarray(rgb, float)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1.0)


def _neutral_specular(rgb):
    """Remove a neutral additive highlight, then normalize diffuse intensity.

    This declared lighting model is suitable only for saturated coloured targets.
    It is not invariant to arbitrary illuminants, material changes or clipping.
    """
    values = np.asarray(rgb, float)
    return _chroma(values - values.min(axis=-1, keepdims=True))


def _saturated(rgb):
    values = np.asarray(rgb, float)
    return np.ptp(values, axis=-1) >= 0.2 * np.maximum(values.max(axis=-1), 1.0)


def _components(appearance, world, *, edge_m, max_pixels):
    """Whole measured components. Two-pixel links bridge holes without filling them."""
    count = int(appearance.sum())
    if count > max_pixels:
        raise ObservationUnavailable("moving_mask_pixel_budget_exhausted")
    if count == 0:
        return ()
    index = np.full(appearance.shape, -1, dtype=np.int32)
    index[appearance] = np.arange(count)
    rows, columns = [], []
    h, w = appearance.shape
    for dy in range(3):
        for dx in range(-2, 3):
            if dy == 0 and dx <= 0:
                continue
            ya, yb = slice(0, h - dy), slice(dy, h)
            xa, xb = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            linked = appearance[ya, xa] & appearance[yb, xb]
            linked &= np.linalg.norm(world[ya, xa] - world[yb, xb], axis=-1) <= edge_m
            rows.append(index[ya, xa][linked])
            columns.append(index[yb, xb][linked])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(count, count)).tocsr()
    _, labels = connected_components(graph, directed=False)
    pixels = np.flatnonzero(appearance)
    order = np.argsort(labels, kind="stable")
    return np.split(pixels[order], np.flatnonzero(np.diff(labels[order])) + 1)


@dataclass(frozen=True)
class MovingMaskFit:
    fit: GeometryFit
    mask: np.ndarray
    reference_id: str
    max_displacement_m: float
    processing_s: float
    appearance_model: str = "chromaticity"
    execution_qualified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "mask", _frozen(self.mask))

    def as_dict(self):
        return {
            **self.fit.as_dict(),
            "mask_pixels": int(self.mask.sum()),
            "reference_id": self.reference_id,
            "max_displacement_m": self.max_displacement_m,
            "processing_s": self.processing_s,
            "appearance_model": self.appearance_model,
            "association_scope": "bounded_continuity_candidate_not_global_reidentification",
            "execution_qualified": False,
        }


class MovingRgbdTracker:
    """A fixed semantic appearance prior plus fresh geometry; no automatic re-ID.

    The caller declares a motion bound and supplies coherent current world points.
    All appearance components intersecting the reachable volume compete, including
    fragments too small to fit. The default single-view profile stops on multiple
    candidates before fitting. The explicit multi-view complete-prior profile
    may validate their entire union. Geometry never trims a mixed component into
    an apparent object. Neutral-specular appearance is separately opt-in and
    requires saturated colour; default chromaticity behavior is unchanged.
    """

    def __init__(
        self,
        model,
        *,
        object_ref,
        rgb,
        depth,
        world,
        mask,
        observation_id,
        received_monotonic,
        scene_epoch,
        max_speed_m_s=0.25,
        max_gap_s=1.5,
        association_margin_m=0.005,
        chroma_distance=0.03,
        max_edge_m=0.012,
        max_pixels=100000,
        max_update_s=0.75,
        max_evaluations=6000,
        appearance_model="chromaticity",
        clock=time.monotonic,
    ):
        if (
            not object_ref
            or not scene_epoch
            or not observation_id
            or not np.isfinite(
                [max_speed_m_s, max_gap_s, association_margin_m, chroma_distance, max_edge_m, max_update_s]
            ).all()
            or not 0 < max_speed_m_s <= 1
            or not 0 < max_gap_s <= 2
            or not 0 < association_margin_m <= 0.02
            or not 0 < chroma_distance <= 0.2
            or not 0 < max_edge_m <= 0.03
            or max_update_s <= 0
            or type(max_pixels) is not int
            or max_pixels < 25
            or appearance_model not in {"chromaticity", "neutral_specular"}
        ):
            raise ValueError("invalid_moving_tracking_profile")
        self.model, self.object_ref, self.scene_epoch, self.clock = model, object_ref, scene_epoch, clock
        self.speed, self.gap, self.margin = max_speed_m_s, max_gap_s, association_margin_m
        self.chroma_distance, self.edge_m, self.max_pixels = chroma_distance, max_edge_m, max_pixels
        self.max_update_s, self.max_evaluations = max_update_s, max_evaluations
        self.appearance_model = appearance_model
        self._colour_features = _neutral_specular if appearance_model == "neutral_specular" else _chroma
        self.reference_id = "moving_reference:" + uuid.uuid4().hex
        self.latest = None
        self._seen = {observation_id}
        self._received = received_monotonic
        self._require_time(received_monotonic)
        rgb, depth, world = _arrays(rgb, depth, world)
        mask = np.asarray(mask)
        if (
            mask.dtype != bool
            or mask.shape != depth.shape
            or mask.sum() < 25
            or not np.isfinite(world[mask]).all()
            or not np.isfinite(depth[mask]).all()
            or np.any(depth[mask] <= 0)
        ):
            raise ValueError("invalid_moving_reference_mask")
        # Learn appearance from measured interior pixels. RGB antialiasing at a
        # silhouette can mix foreground colour with background depth; admitting
        # those mixed colours creates false competing background components.
        # This changes only the colour reference, never fills/trims current 3D
        # components or discards a competing candidate on geometric fit quality.
        interior = binary_erosion(mask)
        if interior.sum() < 25:
            raise ObservationUnavailable("moving_reference_appearance_unavailable")
        colors = rgb[interior]
        if np.median(colors.sum(axis=-1)) < 60 or (
            appearance_model == "neutral_specular" and not _saturated(colors).all()
        ):
            raise ObservationUnavailable("moving_reference_appearance_unavailable")
        chroma = self._colour_features(colors)
        self._appearance = cKDTree(chroma[np.linspace(0, len(chroma) - 1, min(256, len(chroma))).astype(int)])
        self._min_brightness = max(30.0, float(np.median(colors.sum(axis=-1)) * 0.25))
        started = time.perf_counter()
        fit = fit_geometry(
            world[mask],
            model,
            object_ref=object_ref,
            observation_ids=(observation_id,),
            received_monotonic=received_monotonic,
            max_evaluations=max_evaluations,
        )
        self._require_fit(fit)
        self._require_time(received_monotonic)
        self.latest = MovingMaskFit(
            fit, mask, self.reference_id, 0.0, time.perf_counter() - started, self.appearance_model
        )

    def _require_time(self, received):
        now = self.clock()
        if not np.isfinite([received, now]).all() or not 0 <= now - received <= 2.0:
            raise ObservationUnavailable("object_observation_stale")

    @staticmethod
    def _require_fit(fit):
        if (
            not fit.translation_observable
            or fit.search_boundary_hit
            or np.max(np.ptp(fit.poses[:, :3, 3], axis=0)) > 0.005
        ):
            raise ObservationUnavailable("moving_geometry_ambiguous")

    def _candidate_mask(
        self, rgb, depth, world, *, centers, displacement, allow_image_boundary=False, merge_reachable=False
    ):
        """Associate a whole component without asserting full pose observability."""
        rgb, depth, world = _arrays(rgb, depth, world)
        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(world).all(-1)
        appearance = valid & (rgb.sum(axis=-1) >= self._min_brightness)
        if self.appearance_model == "neutral_specular":
            appearance &= _saturated(rgb)
        ys, xs = np.where(appearance)
        distances = self._appearance.query(self._colour_features(rgb[ys, xs]))[0]
        appearance[ys, xs] &= distances <= self.chroma_distance
        parts = _components(appearance, world, edge_m=self.edge_m, max_pixels=self.max_pixels)
        radius = float(np.linalg.norm(self.model.half_extents))
        reachable = []
        for pixels in parts:
            points = world.reshape(-1, 3)[pixels]
            if np.any(cKDTree(centers).query(points)[0] <= radius + displacement):
                reachable.append(pixels)
        if not reachable or (len(reachable) != 1 and not merge_reachable):
            raise ObservationUnavailable("moving_mask_ambiguous" if reachable else "moving_mask_lost")
        mask = np.zeros(depth.shape, bool)
        # An explicit complete-prior profile may fit ALL reachable fragments as
        # one solid. Never select a subset by fit quality or discard tiny parts:
        # unrelated returns must invalidate the union during geometry validation.
        mask.flat[np.concatenate(reachable)] = True
        if not allow_image_boundary and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
            raise ObservationUnavailable("moving_mask_image_boundary")
        if mask.sum() < 25:
            raise ObservationUnavailable("moving_mask_insufficient")
        return mask

    def update(self, rgb, depth, world, *, observation_id, received_monotonic, scene_epoch):
        previous, self.latest = self.latest, None
        if previous is None:
            raise ObservationUnavailable("moving_tracking_lost_reinitialize")
        started = time.perf_counter()
        self._require_time(received_monotonic)
        dt = received_monotonic - self._received
        if (
            not observation_id
            or observation_id in self._seen
            or not 0 < dt <= self.gap
            or scene_epoch != self.scene_epoch
            or len(self._seen) >= 4096
        ):
            raise ObservationUnavailable("moving_tracking_epoch_or_capture_invalid")
        self._seen.add(observation_id)
        self._received = received_monotonic
        rgb, depth, world = _arrays(rgb, depth, world)
        centers = previous.fit.poses[:, :3, 3]
        displacement = self.speed * dt + self.margin
        mask = self._candidate_mask(rgb, depth, world, centers=centers, displacement=displacement)
        fit = fit_geometry(
            world[mask],
            self.model,
            object_ref=self.object_ref,
            observation_ids=(observation_id,),
            received_monotonic=received_monotonic,
            initial_poses=previous.fit.poses[:12],
            max_evaluations=self.max_evaluations,
        )
        self._require_fit(fit)
        if np.any(cKDTree(centers).query(fit.poses[:, :3, 3])[0] > displacement):
            raise ObservationUnavailable("moving_mask_motion_bound_exceeded")
        self._require_time(received_monotonic)
        elapsed = time.perf_counter() - started
        if elapsed > self.max_update_s:
            raise ObservationUnavailable("moving_tracking_time_budget_exhausted")
        self.latest = MovingMaskFit(fit, mask, self.reference_id, displacement, elapsed, self.appearance_model)
        return self.latest
