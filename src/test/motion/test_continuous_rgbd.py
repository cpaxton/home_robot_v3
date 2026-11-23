# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Producer lifetime and capture accounting, independently of geometry fitting."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.manipulation.continuous_rgbd import ContinuousRgbdSource
from emet.controller.manipulation.multiview_rgbd_tracking import SynchronizedRgbdView
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval


def views(stamp, name="first"):
    cap = CaptureInterval(name, stamp, stamp, stamp, "test-clock")
    return {"head": SynchronizedRgbdView(np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)), np.ones((8, 8, 3)), cap)}


class TrackingBoundary:
    """Emulate the tracker's measured-frame boundary; no synthetic geometry claim."""

    gap, scene_epoch = 1.5, "scene"

    def __init__(self, initial, clock):
        self.clock = clock
        self.latest = self.result(initial)

    @staticmethod
    def result(packet):
        return SimpleNamespace(capture=packet["head"].capture, masks=(("head", np.ones((8, 8), bool)),))

    def update(self, packet, *, scene_epoch):
        cap = packet["head"].capture
        cap.require(now=self.clock(), after=self.latest.capture.latest, max_age_s=self.gap)
        self.latest = self.result(packet)
        return self.latest


def source(read, clock):
    initial = views(clock())
    return ContinuousRgbdSource(
        TrackingBoundary(initial, clock), initial_views=initial, read_views=read, clock=clock, poll_s=0.005
    )


def test_producer_advances_original_frames_while_consumer_is_idle():
    produced = threading.Event()
    count = [0]

    def read():
        count[0] += 1
        if count[0] >= 4:
            produced.set()
        return views(time.monotonic(), str(count[0]))

    stream = source(read, time.monotonic).start()
    barrier = time.monotonic()
    try:
        assert produced.wait(2.0)
        # No consumer call drove any of those acquisitions.
        first, last = stream.collect(after=barrier, min_span_s=0.01, timeout_s=1.0)
        assert last.fit.capture.earliest - first.fit.capture.latest >= 0.01
        assert first.fit.capture.earliest > barrier
        assert stream.processed_captures >= 4
    finally:
        stream.close()
    assert not stream._thread.is_alive()


@pytest.mark.parametrize("empty", [True, False])
def test_missing_or_cached_packets_cannot_keep_source_fresh(empty):
    clock = [10.0]
    packet = views(10.0)
    stream = source(lambda: None if empty else packet, lambda: clock[0])
    assert stream.poll_once() is False
    assert stream.snapshot().fit.capture.earliest == 10.0
    clock[0] = 11.6
    with pytest.raises(ObservationUnavailable, match="object_observation_stale"):
        stream.poll_once()
    clock[0] = 11.7
    stream.read_views = lambda: views(clock[0], "fresh")
    with pytest.raises(ObservationUnavailable, match="object_observation_stale"):
        stream.poll_once()


def test_same_id_with_changed_pixels_is_replayed_and_latches():
    stream = source(lambda: views(10.0), lambda: 10.0)
    changed = views(10.0)
    changed["head"] = replace(changed["head"], rgb=np.ones((8, 8, 3), np.uint8))
    stream.read_views = lambda: changed
    with pytest.raises(ObservationUnavailable, match="capture_replayed"):
        stream.poll_once()
    with pytest.raises(ObservationUnavailable, match="capture_replayed"):
        stream.snapshot()
    assert stream.tracker.latest is None


def test_retransmission_changes_receipt_only_and_preserves_original_freshness():
    clock = [10.0]
    packet = views(10.0)
    stream = source(lambda: packet, lambda: clock[0])
    clock[0] = 11.0
    packet["head"] = replace(packet["head"], capture=replace(packet["head"].capture, received_monotonic=11.0))
    assert not stream.poll_once()
    assert stream.duplicate_packets == 1
    assert stream.snapshot().fit.capture.received_monotonic == 10.0
    clock[0] = 11.6
    with pytest.raises(ObservationUnavailable, match="object_observation_stale"):
        stream.poll_once()


def test_same_id_with_changed_source_interval_is_replay_even_with_same_pixels():
    packet = views(10.0)
    stream = source(lambda: packet, lambda: 10.1)
    packet["head"] = replace(
        packet["head"], capture=replace(packet["head"].capture, earliest=10.05, latest=10.05, received_monotonic=10.05)
    )
    with pytest.raises(ObservationUnavailable, match="capture_replayed"):
        stream.poll_once()


def test_collection_skips_stale_history_and_returns_two_current_original_frames():
    clock = [10.0]
    stream = source(lambda: views(clock[0], str(clock[0])), lambda: clock[0])
    for clock[0] in [10.8, 11.6, 12.4]:
        assert stream.poll_once()
    first, last = stream.collect(after=9.0, min_span_s=0.2, timeout_s=0.1)
    assert first.fit.capture.earliest == 11.6
    assert last.fit.capture.earliest == 12.4


def test_collection_timeout_is_no_visual_evidence_and_not_new_capture():
    stream = source(lambda: None, time.monotonic)
    with pytest.raises(ObservationUnavailable, match="visual_evidence_incomplete"):
        stream.collect(after=time.monotonic(), min_span_s=0.01, timeout_s=0.02)
    assert stream.processed_captures == 1
    stream.close()


def test_reader_failure_and_invalidation_stop_without_reinitialization():
    def read():
        raise ObservationUnavailable("moving_mask_lost")

    stream = source(read, lambda: 10.0)
    with pytest.raises(ObservationUnavailable, match="moving_mask_lost"):
        stream.poll_once()
    with pytest.raises(ObservationUnavailable, match="moving_mask_lost"):
        stream.start()
    second = source(lambda: None, lambda: 10.0)
    second.invalidate()
    with pytest.raises(ObservationUnavailable, match="scene_changed_replan"):
        second.snapshot()


def test_close_prevents_reading_or_restarting_even_without_worker():
    calls = []
    stream = source(lambda: calls.append("read"), lambda: 10.0)
    stream.close()
    for operation in (stream.snapshot, stream.poll_once, stream.start):
        with pytest.raises(ObservationUnavailable, match="rgbd_source_closed"):
            operation()
    assert calls == []


def test_blocked_reader_shutdown_is_reported_and_can_be_joined_later():
    entered, release = threading.Event(), threading.Event()

    def read():
        entered.set()
        release.wait(2.0)
        return None

    stream = source(read, time.monotonic).start()
    try:
        assert entered.wait(1.0)
        with pytest.raises(RuntimeError, match="shutdown_timeout"):
            stream.close(timeout_s=0.01)
    finally:
        release.set()
        stream.close()
    assert not stream._thread.is_alive()


def test_consumer_waits_for_inflight_fresh_capture_without_latching_old_age():
    clock = [10.]
    acquiring, release = threading.Event(), threading.Event()

    def read():
        acquiring.set()
        assert release.wait(2.)
        return views(11.4, 'fresh')

    stream = source(read, lambda: clock[0])
    producer = threading.Thread(target=stream.poll_once)
    producer.start()
    assert acquiring.wait(1.)
    clock[0] = 11.6
    result = []
    consumer = threading.Thread(target=lambda: result.append(stream.snapshot()))
    consumer.start()
    release.set()
    producer.join(2.)
    consumer.join(2.)
    assert not producer.is_alive() and not consumer.is_alive()
    assert result[0].fit.capture.observation_id == 'fresh'
    assert result[0].fit.capture.earliest == 11.4
    assert stream._failure is None
    stream.close()


def test_transient_fit_rejection_never_refreshes_or_publishes_failed_evidence():
    clock = [10.]
    initial = views(clock[0])
    tracker = TrackingBoundary(initial, lambda: clock[0])
    update = tracker.update
    failures = [1]

    def fit(packet, *, scene_epoch):
        if failures[0]:
            failures[0] -= 1
            tracker.latest = None  # Real tracker invalidates the candidate before fitting.
            raise ObservationUnavailable('geometry_tracking_no_match')
        return update(packet, scene_epoch=scene_epoch)

    tracker.update = fit
    stream = ContinuousRgbdSource(tracker, initial_views=initial,
        read_views=lambda: views(clock[0], str(clock[0])), clock=lambda: clock[0], max_transient_fit_rejections=2)
    clock[0] = 10.1
    assert not stream.poll_once()
    assert stream.snapshot().fit.capture.received_monotonic == 10.
    assert stream.processed_captures == 1 and stream.rejected_fit_captures == 1
    _, rejected, reason = stream.last_fit_rejection
    assert reason == 'geometry_tracking_no_match'
    assert rejected[0][1].capture.earliest == 10.1
    clock[0] = 10.2
    assert stream.poll_once()
    assert stream.snapshot().fit.capture.received_monotonic == 10.2
    assert stream.consecutive_fit_rejections == 0
    failures[0] = 3
    for stamp in (10.3, 10.4):
        clock[0] = stamp
        assert not stream.poll_once()
    clock[0] = 10.5
    with pytest.raises(ObservationUnavailable, match='geometry_tracking_no_match'):
        stream.poll_once()
    with pytest.raises(ObservationUnavailable):
        stream.snapshot()


def test_transient_fit_policy_cannot_extend_evidence_age():
    clock = [10.]
    initial = views(clock[0])
    tracker = TrackingBoundary(initial, lambda: clock[0])

    def reject(*args, **kwargs):
        raise ObservationUnavailable('geometry_tracking_no_match')

    tracker.update = reject
    stream = ContinuousRgbdSource(tracker, initial_views=initial,
        read_views=lambda: views(clock[0], str(clock[0])), clock=lambda: clock[0], max_transient_fit_rejections=2)
    clock[0] = 10.1
    assert not stream.poll_once()
    clock[0] = 11.6
    with pytest.raises(ObservationUnavailable):
        stream.snapshot()


def test_transient_fit_policy_does_not_swallow_identity_failure():
    clock = [10.]
    initial = views(clock[0])
    tracker = TrackingBoundary(initial, lambda: clock[0])

    def reject(*args, **kwargs):
        raise ObservationUnavailable('moving_mask_motion_bound_exceeded')

    tracker.update = reject
    stream = ContinuousRgbdSource(tracker, initial_views=initial,
        read_views=lambda: views(clock[0], str(clock[0])), clock=lambda: clock[0], max_transient_fit_rejections=2)
    clock[0] = 10.1
    with pytest.raises(ObservationUnavailable, match='moving_mask_motion_bound_exceeded'):
        stream.poll_once()
    assert stream.rejected_fit_captures == 0


@pytest.mark.parametrize('changed', [False, True])
def test_retransmitted_rejected_capture_is_not_refitted_or_counted_twice(changed):
    clock = [10.]
    packet = views(clock[0])
    tracker = TrackingBoundary(packet, lambda: clock[0])
    update, attempted = tracker.update, []

    def fit(views, *, scene_epoch):
        identifier = views['head'].capture.observation_id
        assert identifier not in attempted  # Real trackers reject repeated IDs.
        attempted.append(identifier)
        if identifier == 'rejected':
            tracker.latest = None
            raise ObservationUnavailable('geometry_tracking_no_match')
        return update(views, scene_epoch=scene_epoch)

    tracker.update = fit
    stream = ContinuousRgbdSource(tracker, initial_views=packet, read_views=lambda: packet,
                                  clock=lambda: clock[0], max_transient_fit_rejections=2)
    clock[0] = 10.1
    packet = views(10.1, 'rejected')
    assert not stream.poll_once()
    if changed:
        packet['head'] = replace(packet['head'], rgb=np.ones((8, 8, 3), np.uint8))
        with pytest.raises(ObservationUnavailable, match='capture_replayed'):
            stream.poll_once()
    else:
        assert not stream.poll_once()
        assert attempted == ['rejected']
        assert stream.rejected_fit_captures == 1 and stream.duplicate_packets == 1
        assert stream.snapshot().fit.capture.earliest == 10.
        clock[0] = 10.2
        packet = views(10.2, 'accepted')
        assert stream.poll_once()
        assert stream.snapshot().fit.capture.earliest == 10.2
    stream.close()
