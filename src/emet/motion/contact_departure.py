# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded, directed departure from the contacts at a measured start pose."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepartureMeasurement:
    distances: dict
    surface_points: np.ndarray
    newly_swept_space_known: bool = True
    physical_distances: dict | None = None


class ContactDeparture:
    """One baseline shared by every branch, interpolation and execution check.

    The query supplies signed physical distances for nearby pairs and samples
    covering affected link surfaces. Positive distances denote separation.
    """

    def __init__(self, start, query, *, clearance_m=.005, slack_m=.001,
                 maximum_penetration_m=.005, maximum_travel_m=.03, joint_names=(),
                 new_contact_tolerance_m=0., search_timeout_s=2., patch_slack_m=.005, require_initial_contact=True):
        self.start = np.asarray(start, float).copy()
        self.joint_names = tuple(joint_names)
        self.query = query
        self.clearance = float(clearance_m)
        self.slack = float(slack_m)
        self.maximum_penetration = float(maximum_penetration_m)
        self.maximum_travel = float(maximum_travel_m)
        self.new_contact_tolerance = float(new_contact_tolerance_m)
        self.search_timeout_s = float(search_timeout_s)
        self.patch_slack = float(patch_slack_m)
        self.rejections = {}
        self.last_rejection = None
        if (not np.isfinite(self.start).all() or not callable(query)
                or not np.isfinite([clearance_m, slack_m, maximum_penetration_m, maximum_travel_m]).all()
                or min(clearance_m, slack_m, maximum_penetration_m, maximum_travel_m) <= 0
                or not np.isfinite([new_contact_tolerance_m, search_timeout_s, patch_slack_m]).all()
                or not 0 <= patch_slack_m <= .005
                or not 0 <= new_contact_tolerance_m <= min(.001, maximum_penetration_m)
                or not 0 < search_timeout_s <= 30):
            raise ValueError('invalid_contact_departure')
        self.baseline = self._measurement(self.start)
        self.pairs = {key: value for key, value in self.baseline.distances.items() if value < clearance_m}
        physical = self.baseline.physical_distances or self.baseline.distances
        if (require_initial_contact and not self.pairs) or min(physical.values(), default=np.inf) < -maximum_penetration_m:
            raise ValueError('unsupported_departure_start')

    def _measurement(self, q):
        value = self.query(np.asarray(q, float))
        if (not isinstance(value, DepartureMeasurement)
                or not all(np.isfinite(distance) for distance in value.distances.values())
                or value.surface_points.ndim != 2 or value.surface_points.shape[1] != 3
                or not len(value.surface_points) or not np.isfinite(value.surface_points).all()):
            raise ValueError('invalid_departure_measurement')
        return value

    def _reject(self, reason, **details):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        self.last_rejection = {"reason": reason, **details}
        pair = details.get('pair')
        if pair is not None and hasattr(self, 'describe_pair'):
            self.last_rejection.update(self.describe_pair(pair))
        return False

    def _allowed(self, value):
        if not value.newly_swept_space_known or value.surface_points.shape != self.baseline.surface_points.shape:
            return self._reject('unknown_sweep_or_changed_geometry')
        travel = np.linalg.norm(value.surface_points - self.baseline.surface_points, axis=1)
        if travel.max() > self.maximum_travel:
            return self._reject('surface_travel_limit', travel_m=float(travel.max()))
        physical = value.physical_distances or value.distances
        if any(distance < -self.maximum_penetration for distance in physical.values()):
            return self._reject('penetration_limit', distance_m=float(min(physical.values())))
        for key, distance in value.distances.items():
            baseline_physical = (self.baseline.physical_distances or {}).get(key)
            if (isinstance(key, tuple) and key[0] == 'environment_patch'
                    and baseline_physical is not None):
                # Contact is a measured surface patch, not a voxel. Tangential
                # motion may change its closest sample. Bound nominal worsening
                # against the original baseline; do not lock inflated padding.
                floor = min(0., baseline_physical)-max(self.slack, self.patch_slack)
                if physical[key] < floor:
                    return self._reject('patch_deepened', pair=key,
                                        distance_m=float(physical[key]), floor_m=float(floor))
                continue
            if value.physical_distances is not None and (
                key not in self.pairs or baseline_physical is not None and baseline_physical > 0
            ):
                # A nearby but physically clear surface is planning reserve,
                # not an existing penetration whose padding must stay fixed.
                # It may be approached while separating elsewhere, but never
                # entered. Tangency alone is not nominal penetration.
                if value.physical_distances.get(key, distance) < -self.new_contact_tolerance:
                    return self._reject('new_physical_contact', pair=key,
                                        distance_m=float(value.physical_distances.get(key, distance)))
                continue
            floor = (self.pairs[key] - self.slack
                     if key in self.pairs else self.clearance)
            if distance < floor:
                return self._reject('baseline_worsened', pair=key, distance_m=float(distance), floor_m=float(floor))
        return True

    def _measurement_cache(self):
        # The geometry query is deterministic within one synchronous edge or
        # interpolation call. Never retain this cache across runtime updates.
        cache = {}

        def measure(q):
            key = np.asarray(q, float).tobytes()
            if key not in cache:
                cache[key] = self._measurement(q)
            return cache[key]

        return measure

    def validate_edge(self, start, end):
        """Validate ordered samples with bounded link-surface displacement."""
        start, end = np.asarray(start, float), np.asarray(end, float)
        if start.shape != self.start.shape or end.shape != self.start.shape:
            return False
        if not np.isfinite([start, end]).all():
            return False
        pending = [(start, end, 0)]
        measure = self._measurement_cache()
        while pending:
            a, b, depth = pending.pop()
            midpoint = (a + b) / 2
            first, middle, last = (measure(q) for q in (a, midpoint, b))
            if not all(self._allowed(value) for value in (first, middle, last)):
                return False
            clear = all(value >= self.clearance for value in first.distances.values())
            if clear and any(value < self.clearance for sample in (middle, last) for value in sample.distances.values()):
                return False
            displacement = max(np.linalg.norm(middle.surface_points - first.surface_points, axis=1).max(),
                               np.linalg.norm(last.surface_points - middle.surface_points, axis=1).max())
            if displacement > .001 or np.max(np.abs(b - a)) > .002:
                if depth >= 16:
                    return False
                pending.extend(((midpoint, b, depth + 1), (a, midpoint, depth + 1)))
        return True

    def configuration_allowed(self, q):
        return self._allowed(self._measurement(q))

    def execution_waypoints(self, path):
        measure = self._measurement_cache()
        result = [np.asarray(path[0], float)]
        for target in path[1:]:
            pending = [(result[-1], np.asarray(target, float), 0)]
            while pending:
                first, last, depth = pending.pop()
                a, b = measure(first), measure(last)
                travel = np.linalg.norm(a.surface_points - b.surface_points, axis=1).max()
                if travel > .002 or np.max(np.abs(last - first)) > .005:
                    if depth >= 16:
                        raise ValueError('departure_execution_sampling_exhausted')
                    middle = (first + last) / 2
                    pending.extend(((middle, last, depth + 1), (first, middle, depth + 1)))
                else:
                    result.append(last)
        return result

    def sample_separating(self, rng):
        """Bias a local joint proposal toward increasing the worst clearance."""
        gradient = np.zeros_like(self.start)
        for index in range(len(gradient)):
            delta = np.zeros_like(gradient)
            delta[index] = .001
            forward = self._measurement(self.start + delta)
            backward = self._measurement(self.start - delta)
            gradient[index] = (min(forward.distances.get(key, self.clearance) for key in self.pairs)
                               - min(backward.distances.get(key, self.clearance) for key in self.pairs))
        magnitude = np.linalg.norm(gradient)
        if magnitude <= 1e-12:
            gradient = rng.normal(size=len(gradient))
            magnitude = np.linalg.norm(gradient)
        return self.start + gradient / magnitude * .03
