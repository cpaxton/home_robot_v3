# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Local evidence epochs; camera timestamps are identities, not synced clocks."""
import math
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameTicket:
    revision: int
    received_monotonic: float
    identity_available: bool
    motion_epoch: int = 0


class ObservedFrameGate:
    """Prevent replayed/in-flight pre-motion frames from renewing task evidence.

    Capture timestamps are ordered only within their advertised source/domain.
    Their epoch is never compared with the client's monotonic clock. A new
    capture identity does not certify capture-after-motion or execution safety.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self.revision = 0
        self.motion_epoch = 0
        self._stream = None
        self._stamp = None
        self._received = None
        self._barrier = None
        self._require_identity = False

    def invalidate(self):
        with self.lock:
            self.revision += 1
            self.motion_epoch += 1
            self._barrier = self._stamp
            self._require_identity = True

    def begin(self, received_monotonic, image_timing=None):
        with self.lock:
            self.revision += 1
            if (isinstance(received_monotonic, bool) or not isinstance(received_monotonic, (int, float))
                    or not math.isfinite(received_monotonic)):
                return None
            # Legacy discovery remains read-only. After motion, missing capture
            # identity cannot establish that a cached image has been replaced.
            if not image_timing:
                if self._require_identity:
                    return None
                return FrameTicket(self.revision, received_monotonic, False, self.motion_epoch)
            if not isinstance(image_timing, dict):
                return None
            camera = 'head_cam_left'
            timing = image_timing.get(camera)
            if not isinstance(timing, dict):
                return None
            stamp = timing.get('timestamp_ns')
            source, domain = timing.get('source'), timing.get('clock_domain')
            if (timing.get('available') is not True or type(stamp) is not int or stamp <= 0
                    or not isinstance(source, str) or not source or not isinstance(domain, str) or not domain):
                return None
            stream = (camera, source, domain)
            if self._stream is not None and stream != self._stream:
                return None  # Reconnect/new clock requires a new provider.
            if self._stamp is not None and stamp < self._stamp:
                return None
            if self._barrier is not None and stamp <= self._barrier:
                return None
            if stamp == self._stamp:
                received_monotonic = min(received_monotonic, self._received)
            else:
                self._received = received_monotonic
            self._stream, self._stamp = stream, stamp
            return FrameTicket(self.revision, received_monotonic, True, self.motion_epoch)

    def current(self, ticket):
        with self.lock:
            return isinstance(ticket, FrameTicket) and ticket.revision == self.revision
