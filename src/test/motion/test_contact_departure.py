# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np

from emet.motion.contact_departure import ContactDeparture, DepartureMeasurement


def measurement(q):
    return DepartureMeasurement({'original': float(q[0]) - .002}, np.array([[q[0], q[1], 0.]]))


def test_initial_contact_accepts_separation_and_tangential_motion():
    departure = ContactDeparture([0., 0.], measurement)
    assert departure.validate_edge([0., 0.], [0., .01])
    assert departure.validate_edge([0., .01], [.01, .01])
    assert not departure.validate_edge([.01, .01], [0., .01])


def test_slack_is_relative_to_original_baseline_for_every_branch():
    departure = ContactDeparture([0., 0.], measurement)
    assert departure.validate_edge([0., 0.], [-.0008, 0.])
    assert not departure.validate_edge([-.0008, 0.], [-.0016, 0.])
    assert not departure.validate_edge([0., 0.], [0., .04])


def test_new_contact_and_unknown_sweep_reject():
    def query(q):
        result = measurement(q)
        return DepartureMeasurement({**result.distances, 'other': .01 - abs(q[1])}, result.surface_points,
                                    newly_swept_space_known=q[0] <= .02)

    departure = ContactDeparture([0., 0.], query)
    assert not departure.validate_edge([0., 0.], [0., .01])
    assert not departure.validate_edge([0., 0.], [.025, 0.])


def test_padding_overlap_is_distinct_from_new_physical_contact_during_escape():
    def query(q):
        original = float(q[0]) - .02
        distances = {'original': original}
        physical = {'original': .002 + float(q[0])}
        if q[1] > .001:
            distances['nearby'] = -.001
            physical['nearby'] = .01 if q[1] < .01 else -.0001
        return DepartureMeasurement(distances, np.array([[q[0], q[1], 0.]]), True, physical)

    departure = ContactDeparture([0., 0.], query)
    assert departure.validate_edge([0., 0.], [0., .005])
    assert not departure.validate_edge([0., 0.], [0., .015])


def test_clear_neighbors_can_consume_padding_without_new_penetration():
    def query(q):
        return DepartureMeasurement(
            {'contact': float(q[0])-.002, 'neighbor': .002-float(q[1])},
            np.array([[q[0], q[1], 0.]]), True,
            {'contact': float(q[0])-.002, 'neighbor': .015-float(q[1])},
        )

    departure = ContactDeparture([0., 0.], query)
    assert departure.validate_edge([0., 0.], [.005, .01])
    assert not departure.validate_edge([0., 0.], [.005, .02])
    assert not departure.validate_edge([0., 0.], [-.002, .005])


def test_measurement_reuse_is_local_to_one_edge_not_future_geometry():
    calls = {}
    shift = [0.]

    def query(q):
        key = tuple(q)
        calls[key] = calls.get(key, 0) + 1
        return DepartureMeasurement({'surface': float(q[0])-.002+shift[0]}, np.array([[q[0], 0., 0.]]))

    departure = ContactDeparture([0.], query)
    calls.clear()
    assert departure.validate_edge([0.], [.005])
    assert max(calls.values()) == 1
    shift[0] = -.01
    assert not departure.validate_edge([0.], [.005])


def test_new_contact_numerical_tolerance_is_explicit_and_bounded():
    def query(q):
        distances, physical = {'original': -.002}, {'original': -.002}
        if q[0] > .001:
            distances['neighbor'] = -.01
            physical['neighbor'] = -.0005 if q[0] < .01 else -.002
        return DepartureMeasurement(distances, np.array([[q[0], 0., 0.]]), True, physical)

    assert not ContactDeparture([0.], query).validate_edge([0.], [.005])
    departure = ContactDeparture([0.], query, new_contact_tolerance_m=.001)
    assert departure.validate_edge([0.], [.005])
    assert not departure.validate_edge([0.], [.015])


def test_surface_patch_allows_changing_contact_sample_without_locking_padding():
    patch = ('environment_patch', 7, 0)

    def query(q):
        physical = {patch: -.001-float(q[0])}
        if q[1] > .015:
            physical[('environment_surface', 7, (.05, 0., 0.))] = -.001
        return DepartureMeasurement({key: value-.015 for key, value in physical.items()},
                                    np.array([[q[0], q[1], 0.]]), True, physical)

    departure = ContactDeparture([0., 0.], query, slack_m=.003)
    assert departure.validate_edge([0., 0.], [.002, .01])
    assert departure.validate_edge([0., 0.], [.0035, .01])
    assert not departure.validate_edge([0., 0.], [.0045, .01])
    assert not departure.validate_edge([0., 0.], [.002, .02])


def test_execution_sampling_uses_bounded_motor_steps_with_full_edge_revalidation():
    departure = ContactDeparture([0., 0.], measurement)
    path = departure.execution_waypoints([[0., 0.], [.01, 0.]])
    assert len(path) < 20  # Avoid dozens of sub-mm motor waits for a 1 cm escape.
    np.testing.assert_allclose(path[0], [0., 0.])
    np.testing.assert_allclose(path[-1], [.01, 0.])
    assert max(np.linalg.norm(b-a) for a, b in zip(path[:-1], path[1:], strict=True)) <= .002
    assert all(departure.validate_edge(a, b) for a, b in zip(path[:-1], path[1:], strict=True))
