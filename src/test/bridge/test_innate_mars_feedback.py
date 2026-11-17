"""Feedback timestamps/frames must survive republishing and malformed reads."""

import math

import numpy as np
import pytest
from innate_mars_bridge.feedback import FeedbackStore


def populate(store, stamp=10**9):
    store.arm([f"joint{i}" for i in range(6, 0, -1)], list(range(6)), [0.1] * 6, [2] * 6, stamp)
    store.odom(
        [1, 2, 0],
        [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        [0.1, 0, 0.02, 0.03, 0.04, 0.05],
        stamp,
        "odom",
        "base_link",
    )


def test_rotates_measured_child_twist_into_world_joint_layout_without_losing_six_dofs():
    store = FeedbackStore()
    populate(store)
    data = store.snapshot(10**9)
    assert data["complete_and_fresh"] and not data["units_verified"] and not data["motor_read_validity_verified"]
    np.testing.assert_allclose(data["samples"]["odom"]["base_velocity_world"], [0, 0.1, 0.05], atol=1e-10)
    np.testing.assert_allclose(data["samples"]["odom"]["twist_child"], [0.1, 0, 0.02, 0.03, 0.04, 0.05])
    assert data["samples"]["arm"]["position"] == [5, 4, 3, 2, 1, 0]


def test_missing_stale_future_and_invalid_sources_do_not_become_fresh_on_snapshot():
    store = FeedbackStore()
    assert not store.snapshot(10**9)["complete_and_fresh"]
    populate(store)
    assert not store.snapshot(2 * 10**9)["complete_and_fresh"]
    assert not store.snapshot(10**9 - 1)["complete_and_fresh"]
    store.arm([f"joint{i}" for i in range(1, 7)], [0] * 6, [], [], 2 * 10**9)
    arm = store.snapshot(2 * 10**9)["samples"]["arm"]
    assert not arm["fresh"] and not arm["structurally_valid"] and "position" not in arm
    assert arm["sequence"] == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_invalidates_instead_of_reusing_previous_valid_measurement(bad):
    store = FeedbackStore()
    populate(store)
    store.odom([bad, 0, 0], [0, 0, 0, 1], [0] * 6, 10**9, "odom", "base_link")
    assert not store.snapshot(10**9)["complete_and_fresh"]
    assert "base_pose" not in store.snapshot(10**9)["samples"]["odom"]


def test_returned_snapshots_cannot_modify_store():
    store = FeedbackStore()
    populate(store)
    store.snapshot(10**9)["samples"]["arm"]["position"][0] = 99
    assert store.snapshot(10**9)["samples"]["arm"]["position"][0] == 5
