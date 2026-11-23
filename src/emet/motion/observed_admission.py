# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Age-preserving environment evidence and captured robot-material ownership.

No map cell becomes free because it is labelled self. Queries cover actual solid
geometry, so unknown AABB corners need not be mistaken for robot material.
Newly swept material still needs measured free space. Unresolved subdivisions
and exhausted budgets reject. Capture/calibration bounds remain supplied priors.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_solids import WorldSolid

if TYPE_CHECKING:
    from emet.controller.manipulation.observed_effects import CaptureInterval
    from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption


@dataclass(frozen=True)
class CapturedRobotMaterial:
    """Guaranteed material at a measured capture, not a future candidate pose.

    Each erosion is an independently justified Hausdorff-motion/error bound for
    that solid. Outer joint-interval envelopes cannot be supplied as ownership.
    A mesh needs an explicit convex-material declaration or solid decomposition.
    """
    capture: CaptureInterval
    model_signature: str
    calibration_id: str
    scene_epoch: str
    solids: tuple[WorldSolid, ...]
    uncertainty_m: tuple[float, ...]
    configuration_profile_id: str
    source: str = 'measured_joint_feedback'
    surface_tolerance_m: float = 0.0
    measured_surface_points: np.ndarray | None = None

    def __post_init__(self):
        # Deferred to avoid the effects -> collision -> admission import cycle.
        from emet.controller.manipulation.observed_effects import CaptureInterval

        solids, error = tuple(self.solids), tuple(map(float, self.uncertainty_m))
        if (not isinstance(self.capture, CaptureInterval)
                or not all(isinstance(v, str) and v for v in
                           (self.model_signature, self.calibration_id, self.scene_epoch, self.configuration_profile_id))
                or self.source != 'measured_joint_feedback' or not solids or len(error) != len(solids)
                or not np.isfinite(self.surface_tolerance_m) or not 0 <= self.surface_tolerance_m <= .03
                or not np.isfinite(error).all() or min(error) < 0
                or any(not isinstance(s, WorldSolid) or s.offset_m != 0 for s in solids)):
            raise ValueError('invalid_robot_material_evidence')
        object.__setattr__(self, 'solids', solids)
        object.__setattr__(self, 'uncertainty_m', error)
        if self.measured_surface_points is not None:
            points = np.asarray(self.measured_surface_points, float)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError('invalid_robot_surface_evidence')
            frozen = np.frombuffer(points.tobytes(), dtype=float).reshape(points.shape)
            object.__setattr__(self, 'measured_surface_points', frozen)

    @cached_property
    def _surface_tree(self):
        if self.measured_surface_points is None or not len(self.measured_surface_points):
            return None
        return cKDTree(self.measured_surface_points)

    def require(self, *, now, model_signature, scene_epoch):
        self.capture.require(now=now)
        if model_signature != self.model_signature or scene_epoch != self.scene_epoch:
            raise ObservationUnavailable('robot_material_identity_changed')

    @property
    def inner_solids(self):
        return tuple(replace(solid, offset_m=-error) for solid, error in
                     zip(self.solids, self.uncertainty_m, strict=True))

    def matches_surface(self, points, *, point_error_m):
        """Associate measured primitive surfaces within their combined error.

        This association never expands the material used for swept clearance.
        Convex plane lower bounds cannot establish surface proximity.
        """
        if not np.isfinite(point_error_m) or point_error_m < 0:
            raise ValueError('invalid_self_material_qualification')
        matched = np.zeros(len(points), dtype=bool)
        for solid, error in zip(self.solids, self.uncertainty_m, strict=True):
            if solid.kind != 'convex':
                matched |= np.abs(solid.lower_distance(points)) <= max(error + point_error_m, self.surface_tolerance_m)
        if self._surface_tree is not None and len(points):
            finite = np.isfinite(points).all(axis=1)
            matched[finite] |= self._surface_tree.query(points[finite], distance_upper_bound=1e-9)[0] <= 1e-9
        return matched

    @property
    def profile_identity(self):
        return (self.model_signature, self.calibration_id, self.configuration_profile_id,
                self.scene_epoch, self.capture.calibration_id, self.surface_tolerance_m,
                self.measured_surface_points is not None)


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str
    subdivisions: int = 0


@dataclass(frozen=True)
class RobotGeometryBound:
    """Outer candidate volume and the external calibration bounding its error."""
    solid: WorldSolid
    geom_id: int
    model_signature: str
    calibration_id: str
    configuration_profile_id: str
    displacement_bound_m: float

    def __post_init__(self):
        if (not isinstance(self.solid, WorldSolid) or type(self.geom_id) is not int or self.geom_id < 0
                or not all(isinstance(v, str) and v for v in
                           (self.model_signature, self.calibration_id, self.configuration_profile_id))
                or not np.isfinite(self.displacement_bound_m) or self.displacement_bound_m < 0):
            raise ValueError('invalid_robot_collision_bound')


@dataclass(frozen=True, init=False)
class ObservedEnvironmentAdmission:
    """Strict map/ray/support queries with no robot ownership or sensor claim."""
    background: ObservedBackground
    static_assumption: StaticBackgroundAssumption | None
    max_nodes: int
    min_extent: float
    _owned: tuple[WorldSolid, ...]

    def __init__(self, background, *, static_assumption=None, max_nodes=20_000, min_extent_m=.0005):
        from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption

        if (not isinstance(background, ObservedBackground) or type(max_nodes) is not int or max_nodes < 1
                or not np.isfinite(min_extent_m) or min_extent_m <= 0
                or static_assumption is not None and not isinstance(static_assumption, StaticBackgroundAssumption)):
            raise ValueError('invalid_observed_admission')
        object.__setattr__(self, 'background', background)
        object.__setattr__(self, 'static_assumption', static_assumption)
        object.__setattr__(self, 'max_nodes', max_nodes)
        object.__setattr__(self, 'min_extent', min_extent_m)
        object.__setattr__(self, '_owned', ())

    def _require_material(self, *, now):
        raise ObservationUnavailable('robot_material_geometry_unavailable')

    def static_free_bounds(self, bounds, *, now):
        """Prove a closed box free for the remaining lifetime of this static map."""
        assumption = self.static_assumption
        if (
            assumption is None
            or assumption.scene_epoch != self.background.scene_epoch
            or not assumption.started_monotonic <= now <= assumption.expires_monotonic
        ):
            return False
        grid = self.background.workspace
        region = grid.query_region(bounds)
        if region is None or not np.all(grid.cells[region] == 1):
            return False
        stamps = self.background.observed_monotonic[region]
        return bool(np.all((assumption.started_monotonic <= stamps) & (stamps <= now)))

    def query(self, solid: WorldSolid, *, now, allow_captured_robot=False, free_witness=None):
        if free_witness is not None:
            from emet.motion.observed_support_contact import ObservedSupportContact

            if not isinstance(free_witness, ObservedSupportContact):
                raise ValueError('invalid_observed_support_contact')
            free_witness.require(self.background, now=now, static_assumption=self.static_assumption)
        return self._query(solid, now=now, allow_captured_robot=allow_captured_robot, free_witness=free_witness)

    def query_payload(self, physical, padded, *, contact, object_ref, phase, now, departure_clearance_m=None,
                      allow_captured_robot=False):
        from emet.motion.observed_support_contact import ObservedSupportContact

        if not isinstance(contact, ObservedSupportContact) or not padded.contains_solid(physical):
            raise ValueError('invalid_observed_support_contact')
        contact.require(self.background, now=now, static_assumption=self.static_assumption)
        active = contact if contact.eligible(physical, object_ref=object_ref, phase=phase,
                                             departure_clearance_m=departure_clearance_m) else None
        return self._query(padded, now=now, allow_captured_robot=allow_captured_robot,
                           support_contact=active, free_witness=contact)

    def query_probe_payload(self, physical, padded, *, band, object_ref, now):
        from emet.motion.observed_support_contact import ObservedProbeBand

        if not isinstance(band, ObservedProbeBand) or not padded.contains_solid(physical):
            raise ValueError("invalid_probe_contact_band")
        contact = band.support
        if contact.object_ref != object_ref:
            raise ValueError("payload_identity_mismatch")
        contact.require(self.background, now=now, static_assumption=self.static_assumption)
        return self._query(padded, now=now, allow_captured_robot=False, support_contact=contact,
                           free_witness=contact, probe_band=band)

    def _query(self, solid, *, now, allow_captured_robot, support_contact=None, free_witness=None, probe_band=None):
        if allow_captured_robot:
            self._require_material(now=now)
        grid = self.background.workspace
        bounds = solid.bounds
        grid_bounds = np.stack((grid.origin, grid.origin + np.array(grid.cells.shape)*grid.resolution))
        clipped = np.stack((np.maximum(bounds[0], grid_bounds[0]), np.minimum(bounds[1], grid_bounds[1])))
        region = grid.query_region(clipped) if np.all(clipped[0] < clipped[1]) else None
        nodes = 0
        pending = []
        support_pending = False
        witness = support_contact if support_contact is not None else free_witness
        support_cells = (self.background.exclusively_owned_cells(witness.support_ref)
                         if witness is not None else frozenset())
        if region is not None:
            cells = grid.cells[region]
            stamps = self.background.observed_monotonic[region]
            assumption = self.static_assumption
            fresh = ((0 <= now-stamps) & (now-stamps <= 2.) if assumption is None else
                (assumption.scene_epoch == self.background.scene_epoch
                 and assumption.started_monotonic <= now <= assumption.expires_monotonic)
                & (assumption.started_monotonic <= stamps) & (stamps <= now))
            # Fast path: a solid whose closed AABB lies entirely inside the map
            # and over only fresh, free cells is provably clear, so the ordinary
            # subdivision/ownership query is unnecessary. It only ever accepts;
            # any unknown/stale/occupied cell, support contact, captured-robot
            # ownership or out-of-map volume falls through unchanged.
            if (support_contact is None and not allow_captured_robot
                    and np.all(bounds[0] >= grid_bounds[0]) and np.all(bounds[1] <= grid_bounds[1])
                    and np.all(cells == 1) and np.all(fresh)):
                return AdmissionDecision(True, 'observed_free_fast_path', nodes)
            # Preserve obstacle/mixed evidence before any ownership shortcut.
            for local in np.argwhere(cells == 2):
                nodes += 1
                if nodes > self.max_nodes:
                    return AdmissionDecision(False, 'admission_budget_exhausted', nodes)
                index = tuple(v+s.start for v, s in zip(local, region, strict=True))
                cell = grid.origin + np.stack((np.array(index), np.array(index)+1))*grid.resolution
                if solid.disjoint_box(cell):
                    continue
                if np.ravel_multi_index(index, grid.cells.shape) in support_cells and fresh[tuple(local)]:
                    support_pending = True
                    continue
                return AdmissionDecision(False, 'occupied_environment', nodes)
        # Environment evidence wins even when an occupied cell also intersects
        # captured material. Only then can a whole-solid containment proof pass.
        owned = self._owned if allow_captured_robot else ()
        if not support_pending and any(material.contains_solid(solid) for material in owned):
            return AdmissionDecision(True, 'captured_robot_material')
        if np.any(bounds[0] < grid_bounds[0]) or np.any(bounds[1] > grid_bounds[1]):
            return AdmissionDecision(False, 'outside_observed_workspace')
        if region is not None:
            for local in np.argwhere(~((cells == 1) & fresh)):
                index = np.array([v+s.start for v, s in zip(local, region, strict=True)])
                cell = grid.origin + np.stack((index, index+1))*grid.resolution
                is_support = (cells[tuple(local)] == 2
                              and np.ravel_multi_index(tuple(index), grid.cells.shape) in support_cells
                              and fresh[tuple(local)])
                subbox = np.stack((np.maximum(cell[0], bounds[0]), np.minimum(cell[1], bounds[1])))
                pending.append((subbox, is_support, cells[tuple(local)] == 0))
        while pending:
            cell, is_support, is_unknown = pending.pop()
            nodes += 1
            if nodes > self.max_nodes:
                return AdmissionDecision(False, 'admission_budget_exhausted', nodes)
            if solid.disjoint_box(cell):
                continue
            if probe_band is not None and (is_support or is_unknown) and probe_band.covers_box(cell):
                continue
            if is_support:
                if witness.covers_box(cell, allow_support_material=support_contact is not None):
                    continue
            elif any(material.contains_box(cell) for material in owned):
                continue
            elif support_contact is not None and support_contact.covers_box(cell):
                # The finite backing-layer prior is continuous across voxel
                # boundaries. Only an eligible shallow payload contact can
                # use it; the map cell remains unknown and obstacles win above.
                continue
            elif free_witness is not None and free_witness.covers_box(cell, allow_support_material=False):
                # A coarse cell may be unknown while this queried subvolume
                # is strictly in front of every original measured return.
                # This cannot admit material behind a ray or mixed occupancy.
                continue
            axis = int(np.argmax(cell[1]-cell[0]))
            if cell[1, axis]-cell[0, axis] <= self.min_extent:
                return AdmissionDecision(False, 'unknown_swept_volume', nodes)
            a, b = cell.copy(), cell.copy()
            a[1, axis] = b[0, axis] = float(np.mean(cell[:, axis]))
            pending.extend(((a, is_support, is_unknown), (b, is_support, is_unknown)))
        return AdmissionDecision(True, 'observed_free_or_captured_material', nodes)


@dataclass(frozen=True, init=False)
class ObservedMaterialAdmission(ObservedEnvironmentAdmission):
    """Preserve environment occupancy before consulting captured self material.

    ``background`` must already separate self endpoints by provenance. Mixed
    self/environment cells remain occupied. Ownership never edits that map and
    its acquisition time is not renewed by a fresh foreground observation.
    """
    background: ObservedBackground
    robot_material: CapturedRobotMaterial
    static_assumption: StaticBackgroundAssumption | None
    max_nodes: int
    min_extent: float
    _owned: tuple[WorldSolid, ...]
    query_robot_geometry: object
    scene_geometry_digest: str

    def __init__(self, background: ObservedBackground, robot_material: CapturedRobotMaterial,
                 *, static_assumption: StaticBackgroundAssumption | None = None,
                 max_nodes=20_000, min_extent_m=.0005, query_robot_geometry=None):
        from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption

        if (type(max_nodes) is not int or max_nodes < 1 or not np.isfinite(min_extent_m)
                or min_extent_m <= 0 or not isinstance(background, ObservedBackground)
                or not isinstance(robot_material, CapturedRobotMaterial)
                or background.scene_epoch != robot_material.scene_epoch
                or background.latest_capture.calibration_id != robot_material.capture.calibration_id
                or (background.self_material_profile is not None
                    and background.self_material_profile != robot_material.profile_identity)
                or (static_assumption is not None and not isinstance(static_assumption, StaticBackgroundAssumption))
                or not callable(query_robot_geometry)):
            raise ValueError('invalid_observed_admission')
        object.__setattr__(self, 'background', background)
        object.__setattr__(self, 'robot_material', robot_material)
        object.__setattr__(self, 'static_assumption', static_assumption)
        object.__setattr__(self, 'max_nodes', max_nodes)
        object.__setattr__(self, 'min_extent', min_extent_m)
        object.__setattr__(self, '_owned', robot_material.inner_solids)
        # This model-only callback supplies the OUTER candidate geometry,
        # including its calibrated unmeasured-joint/pose bounds. It does not
        # construct or replace the captured ownership geometry.
        object.__setattr__(self, 'query_robot_geometry', query_robot_geometry)
        object.__setattr__(self, 'scene_geometry_digest', background.workspace.placement_scene().geometry_digest)

    @property
    def profile_identity(self):
        return self.robot_material.profile_identity

    def require(self, *, now, model_signature, capture_id=None):
        self.robot_material.require(now=now, model_signature=model_signature,
                                    scene_epoch=self.background.scene_epoch)
        if capture_id is not None and capture_id != self.robot_material.capture.observation_id:
            raise ObservationUnavailable('capture_identity_mismatch')

    def _require_material(self, *, now):
        self.robot_material.require(now=now, model_signature=self.robot_material.model_signature,
                                    scene_epoch=self.background.scene_epoch)

    def query_payload(self, physical, padded, *, contact, object_ref, phase, now, departure_clearance_m=None,
                      allow_captured_robot=True):
        return super().query_payload(physical, padded, contact=contact, object_ref=object_ref, phase=phase,
            now=now, departure_clearance_m=departure_clearance_m, allow_captured_robot=allow_captured_robot)
