# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import runpy
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

# Load the actual module without remote/__init__.py's ROS hardware imports.
MarsNavigationClient = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "innate_mars_bridge/innate_mars_bridge/remote/modules/nav.py")
)["MarsNavigationClient"]


def result(value):
    future = Future()
    future.set_result(value)
    return future


@pytest.mark.parametrize("status,expected", [(4, "succeeded"), (5, "cancelled"), (6, "aborted"), (0, "unknown")])
def test_nav2_status_is_preserved(status, expected):
    nav = MarsNavigationClient(Mock())
    nav._ros.get_base_pose_xyt.return_value = np.zeros(3)
    generation = nav._begin_goal(np.zeros(3), "nav2_spin")
    nav._on_result(result(SimpleNamespace(status=status)), generation)
    assert nav.terminal_status() == expected
    assert nav.cancel_navigation(timeout_s=0) is (expected != "unknown")
    assert nav._wait_for_goal(timeout_s=0.01, target_xyt=np.zeros(3)) is (expected == "succeeded")


def test_rejection_is_not_arrival_and_late_callback_cannot_complete_new_goal():
    nav = MarsNavigationClient(Mock())
    first = nav._begin_goal(np.zeros(3), "nav2_spin")
    nav._on_goal_response(result(SimpleNamespace(accepted=False)), first)
    assert nav.terminal_status() == "rejected"
    assert not nav._wait_for_goal(timeout_s=0.01, target_xyt=np.zeros(3))
    second = nav._begin_goal(np.ones(3), "nav2_spin")
    nav._on_result(result(SimpleNamespace(status=4)), first)
    assert nav.terminal_status() is None and not nav.at_goal()
    nav._on_result(result(SimpleNamespace(status=6)), second)
    assert nav.terminal_status() == "aborted"


def test_cancel_before_acceptance_cancels_late_handle_but_waits_for_result():
    nav = MarsNavigationClient(Mock())
    generation = nav._begin_goal(np.zeros(3), "nav2_spin")
    assert not nav.cancel_navigation(timeout_s=0)
    outcome = Future()
    handle = SimpleNamespace(accepted=True, cancel_goal_async=Mock(), get_result_async=lambda: outcome)
    nav._on_goal_response(result(handle), generation)
    handle.cancel_goal_async.assert_called_once()
    assert nav.terminal_status() is None
    outcome.set_result(SimpleNamespace(status=5))
    assert nav.cancel_navigation(timeout_s=0)
