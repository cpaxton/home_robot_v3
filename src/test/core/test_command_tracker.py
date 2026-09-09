# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import pytest

from emet.core.command_tracker import CommandTracker


def envelope(tracker, sequence=0, session="client"):
    return {**tracker.metadata(), "client_session_id": session, "sequence": sequence}


def test_retries_dispatch_once_and_return_latest_receipt():
    tracker = CommandTracker()
    request = envelope(tracker)
    payload = {"xyt": [0, 0, 0.785]}
    assert tracker.accept(request, payload)[0]
    tracker.transition("client", 0, "running")
    for _ in range(10):
        dispatch, receipt = tracker.accept(request, payload)
        assert not dispatch and receipt["status"] == "running"
    tracker.transition("client", 0, "succeeded")
    assert tracker.accept(request, payload)[1]["status"] == "succeeded"
    with pytest.raises(ValueError, match="immutable"):
        tracker.transition("client", 0, "failed")


def test_conflict_restart_and_eviction_never_reexecute():
    tracker = CommandTracker(capacity=2)
    request = envelope(tracker)
    tracker.accept(request, {"head_to": [0, 0]})
    with pytest.raises(ValueError, match="different payload"):
        tracker.accept(request, {"head_to": [0, 1]})
    with pytest.raises(ValueError, match="boot"):
        CommandTracker().accept(request, {})
    for seq in (1, 2):
        tracker.accept(envelope(tracker, seq), {})
    assert len(tracker.snapshot()) == 2
    with pytest.raises(ValueError, match="evicted"):
        tracker.accept(request, {"head_to": [0, 0]})


def test_busy_cancellation_and_deadline():
    now = [0.0]
    tracker = CommandTracker(clock=lambda: now[0])
    tracker.accept(envelope(tracker), {"xyt": [0, 0, 1], "nav_timeout_s": 2})
    dispatch, receipt = tracker.accept(envelope(tracker, 1), {"xyt": [0, 0, 2]})
    assert not dispatch and receipt["status"] == "rejected"
    assert tracker.expired_navigation() is None
    now[0] = 2
    assert tracker.expired_navigation()["sequence"] == 0
    tracker.transition("client", 0, "cancelled")
    assert tracker.expired_navigation() is None
    assert tracker.accept(envelope(tracker, 2), {"xyt": [0, 0, 3]})[0]


def test_active_receipt_survives_eviction_and_snapshots_are_copies():
    tracker = CommandTracker(capacity=2)
    tracker.accept(envelope(tracker), {"xyt": [0, 0, 1]})
    for seq in range(1, 5):
        tracker.accept(envelope(tracker, seq), {"say": "test"})
    snapshot = tracker.snapshot()
    assert snapshot[0]["sequence"] == 0
    snapshot[0]["status"] = "succeeded"
    assert tracker.snapshot()[0]["status"] == "accepted"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline_does_not_claim_session(timeout):
    tracker = CommandTracker()
    with pytest.raises(ValueError):
        tracker.accept(envelope(tracker), {"xyt": [0, 0, 1], "nav_timeout_s": timeout})
    assert tracker.accept(envelope(tracker, session="other"), {})[0]
