# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Continuously process original camera samples independently of tool consumers.

This publishes tracked geometry candidates, not collision or execution evidence.
Readers must return bounded-I/O immutable synchronized views. No capture is
retimed, no failed identity restarts, and cached packets cannot renew freshness.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from emet.controller.manipulation.multiview_rgbd_tracking import MultiViewMaskFit, SynchronizedRgbdView
from emet.controller.manipulation.native_inputs import ObservationUnavailable


@dataclass(frozen=True)
class TrackedRgbdCapture:
    fit: MultiViewMaskFit
    views: tuple[tuple[str, SynchronizedRgbdView], ...]


class ContinuousRgbdSource:
    input_mode = "observed"

    def __init__(self, tracker, *, initial_views, read_views, poll_s=0.05, history_size=128, clock=time.monotonic, max_transient_fit_rejections=0):
        if (
            type(max_transient_fit_rejections) is not int or not 0 <= max_transient_fit_rejections <= 2
            or tracker.latest is None
            or not callable(read_views)
            or type(history_size) is not int or history_size < 2
            or not np.isfinite(poll_s)
            or poll_s <= 0
            or not isinstance(initial_views, dict)
            or not initial_views
            or set(initial_views) != {name for name, _ in tracker.latest.masks}
            or any(
                not isinstance(v, SynchronizedRgbdView) or v.capture != tracker.latest.capture
                for v in initial_views.values()
            )
        ):
            raise ValueError("invalid_continuous_rgbd_source")
        self.tracker, self.read_views, self.clock, self.poll_s = tracker, read_views, clock, poll_s
        self._condition, self._poll_lock = threading.Condition(threading.RLock()), threading.Lock()
        self._stop, self._thread = threading.Event(), None
        self._latest = TrackedRgbdCapture(tracker.latest, tuple(sorted(initial_views.items())))
        self._history = deque((self._latest,), maxlen=history_size)
        self._failure, self.processed_captures, self.duplicate_packets = None, 1, 0
        self._closed = False
        self.max_transient_fit_rejections = max_transient_fit_rejections
        self.consecutive_fit_rejections = self.rejected_fit_captures = 0

    def _fail(self, reason):
        with self._condition:
            self._failure = self._failure or reason
            self._latest = None
            self.tracker.latest = None
            self._stop.set()
            self._condition.notify_all()

    @staticmethod
    def _same_packet(previous, current):
        if set(dict(previous.views)) != set(current):
            return False
        for name, old in previous.views:
            new = current[name]
            if (old.geometry is None) != (new.geometry is None):
                return False
            if old.geometry is not None and not old.geometry.equivalent_capture(new.geometry):
                return False
            if any(
                getattr(old.capture, key) != getattr(new.capture, key)
                for key in ("observation_id", "earliest", "latest", "calibration_id")
            ):
                return False
            if any(
                not np.array_equal(a, b, equal_nan=True)
                for a, b in ((old.rgb, new.rgb), (old.depth, new.depth), (old.world, new.world))
            ):
                return False
            if not np.array_equal(old.foreground_mask, new.foreground_mask):
                return False
            a, b = old.robot_guard, new.robot_guard
            if (a is None) != (b is None):
                return False
            if a is not None:
                if a.profile_identity != b.profile_identity:
                    return False
                if any(
                    not np.array_equal(x, y, equal_nan=True)
                    for x, y in (
                        (a.source_points_world, b.source_points_world),
                        (a.registered_points_world, b.registered_points_world),
                    )
                ):
                    return False
                for x, y in zip(a.outer_solids, b.outer_solids, strict=True):
                    if (
                        x.kind != y.kind
                        or x.dimensions != y.dimensions
                        or x.offset_m != y.offset_m
                        or not np.array_equal(x.world_from_local, y.world_from_local)
                        or not np.array_equal(x.vertices, y.vertices)
                    ):
                        return False
        return True

    def poll_once(self):
        """One producer iteration; consumers never perform acquisition here."""
        with self._poll_lock:
            with self._condition:
                self.snapshot(producer=True)
                previous = self._latest
            try:
                views = self.read_views()
                if views is None:
                    self.snapshot(producer=True)
                    return False
                if (
                    not isinstance(views, dict)
                    or not 1 <= len(views) <= 4
                    or any(not isinstance(v, SynchronizedRgbdView) for v in views.values())
                ):
                    raise ObservationUnavailable("invalid_rgbd_views")
                for view in views.values():
                    view.capture.require(now=self.clock(), max_age_s=min(2.0, self.tracker.gap))
                    if view.geometry is not None:
                        view.geometry.require(view, now=self.clock())
                    if view.robot_guard is not None:
                        view.robot_guard.require(view.capture, view.depth, view.world, now=self.clock())
                cap = next(iter(views.values())).capture
                if cap.observation_id == previous.fit.capture.observation_id:
                    if not self._same_packet(previous, views):
                        raise ObservationUnavailable("capture_replayed")
                    with self._condition:
                        self.duplicate_packets += 1
                        self.snapshot(producer=True)
                    return False
                try:
                    fit = self.tracker.update(views, scene_epoch=self.tracker.scene_epoch)
                except ObservationUnavailable as error:
                    # Diagnostic replay only: rejected measurements never
                    # replace the accepted track or renew its timestamp.
                    self.last_fit_rejection = (getattr(previous.fit, "fit", None), tuple(sorted(views.items())), str(error))
                    if (str(error) not in {
                        "geometry_tracking_no_match", "geometry_tracking_unobservable",
                        "geometry_tracking_budget_exhausted", "moving_geometry_ambiguous",
                    } or self.consecutive_fit_rejections >= self.max_transient_fit_rejections):
                        raise
                    # Reject the new fit, not the last actual measurement. Its
                    # timestamp and age limit remain unchanged; no evidence is
                    # published for the rejected capture.
                    previous.fit.capture.require(now=self.clock(), max_age_s=min(2.0, self.tracker.gap))
                    with self._condition:
                        if self._failure or self._closed:
                            raise ObservationUnavailable(self._failure or "rgbd_source_closed") from error
                        # Trackers clear their candidate before fitting. Restore
                        # only the already accepted, still-fresh measurement so
                        # the next real capture can attempt association again.
                        self.tracker.latest = previous.fit
                        self.consecutive_fit_rejections += 1
                        self.rejected_fit_captures += 1
                    return False
                self.consecutive_fit_rejections = 0
                result = TrackedRgbdCapture(fit, tuple(sorted(views.items())))
                with self._condition:
                    if self._failure or self._closed:
                        raise ObservationUnavailable(self._failure or "rgbd_source_closed")
                    self._latest = result
                    self._history.append(result)
                    self.processed_captures += 1
                    self._condition.notify_all()
                return True
            except Exception as error:
                reason = str(error) if isinstance(error, ObservationUnavailable) else "observed_scene_unavailable"
                self._fail(reason)
                raise

    def snapshot(self, *, producer=False):
        deadline = time.monotonic() + 1.0
        with self._condition:
            while True:
                if self._closed:
                    raise ObservationUnavailable("rgbd_source_closed")
                if self._failure:
                    raise ObservationUnavailable(self._failure)
                try:
                    self._latest.fit.capture.require(now=self.clock(), max_age_s=min(2.0, self.tracker.gap))
                    return self._latest
                except ObservationUnavailable as error:
                    if not producer and self._poll_lock.locked():
                        # A producer may be fitting a newer, still-current
                        # capture. Wait for it without publishing old evidence
                        # or killing the in-flight identity track.
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            self._condition.wait(min(.05, remaining))
                            continue
                        raise
                    self._fail(str(error))
                    raise

    def history(self):
        """Immutable diagnostic archive; historical entries are not fresh evidence."""
        with self._condition:
            return tuple(self._history)

    def collect(self, *, after, min_span_s=0.2, timeout_s=2.0):
        if not np.isfinite([after, min_span_s, timeout_s]).all() or min(min_span_s, timeout_s) <= 0:
            raise ValueError("invalid_capture_collection")
        deadline = self.clock() + timeout_s
        with self._condition:
            while True:
                self.snapshot()
                now, age = self.clock(), min(2.0, self.tracker.gap)
                eligible = [
                    r for r in self._history if r.fit.capture.earliest > after and now - r.fit.capture.earliest <= age
                ]
                if len(eligible) >= 2:
                    first, last = eligible[0], eligible[-1]
                    if last.fit.capture.earliest - first.fit.capture.latest >= min_span_s:
                        for r in (first, last):
                            r.fit.capture.require(now=self.clock(), after=after, max_age_s=min(2.0, self.tracker.gap))
                        return first, last
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise ObservationUnavailable("visual_evidence_incomplete")
                self._condition.wait(min(self.poll_s, remaining))

    def start(self):
        with self._condition:
            self.snapshot()
            if self._thread is not None:
                raise RuntimeError("rgbd_source_already_started")
            self._thread = threading.Thread(target=self._run, name="observed-rgbd-source", daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                return  # poll_once has latched the failure for every consumer.
            self._stop.wait(self.poll_s)

    def invalidate(self):
        self._fail("scene_changed_replan")

    def close(self, *, timeout_s=2.0):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("invalid_shutdown_timeout")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise RuntimeError("rgbd_source_shutdown_timeout")
