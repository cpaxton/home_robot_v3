# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bounded, thread-safe command receipts; no robot or transport dependencies.

At-most-once dispatch is guaranteed only within one server boot. A new boot
must never silently resume a command whose physical outcome is unknown.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4

PROTOCOL_VERSION = 2
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "rejected"})


class CommandTracker:
    def __init__(self, capacity: int = 256, *, clock=time.monotonic):
        if capacity < 2:
            raise ValueError("receipt capacity must be at least two")
        self.boot_id = uuid4().hex
        self.capacity = capacity
        self._clock = clock
        self._lock = threading.RLock()
        self._receipts: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._session: str | None = None
        self._retired_sessions: set[str] = set()
        self._high_water = -1
        self._active: tuple[str, int] | None = None

    def metadata(self):
        return {"version": PROTOCOL_VERSION, "server_boot_id": self.boot_id}

    def accept(self, envelope: dict, payload: dict) -> tuple[bool, dict]:
        """Return (dispatch, receipt). Invalid envelopes never consume identity."""
        if (
            not isinstance(envelope, dict)
            or envelope.get("version") != PROTOCOL_VERSION
            or envelope.get("server_boot_id") != self.boot_id
        ):
            raise ValueError("incompatible protocol or server boot")
        session, sequence = envelope.get("client_session_id"), envelope.get("sequence")
        if not isinstance(session, str) or not session or len(session) > 128:
            raise ValueError("invalid client session")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("invalid command sequence")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        navigation = "xyt" in payload
        timeout = float(payload.get("nav_timeout_s", 30.0))
        if navigation and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("navigation timeout must be positive and finite")
        key = (session, sequence)
        with self._lock:
            if session in self._retired_sessions:
                previous = self._receipts.get(key)
                if previous is not None and previous["payload_digest"] == digest:
                    return False, deepcopy(previous)
                raise ValueError("retired client session; execution forbidden")
            if self._session is None and len(self._retired_sessions) >= self.capacity:
                raise ValueError("session capacity exhausted; restart server before acquiring control")
            if self._session is not None and session != self._session:
                raise ValueError("another client owns this server session")
            previous = self._receipts.get(key)
            if previous is not None:
                if previous["payload_digest"] != digest:
                    raise ValueError("command identity reused with different payload")
                return False, deepcopy(previous)
            if sequence <= self._high_water:
                raise ValueError("stale or evicted command; execution forbidden")
            self._session = session
            self._high_water = sequence
            busy = self._active is not None and not set(payload) <= {"cancel_navigation", "say"}
            receipt = {
                **self.metadata(),
                "client_session_id": session,
                "sequence": sequence,
                "payload_digest": digest,
                "status": "rejected" if busy else "accepted",
                "reason": "navigation busy" if busy else None,
                "revision": 0,
                "deadline": self._clock() + timeout if navigation else None,
            }
            self._receipts[key] = receipt
            if navigation and not busy:
                self._active = key
            while len(self._receipts) > self.capacity:
                victim = next(k for k in self._receipts if k != self._active)
                del self._receipts[victim]
            return not busy, deepcopy(receipt)

    def release_control(self, session: str):
        """Explicit handoff only after confirmed stop; retired identities never execute again."""
        with self._lock:
            if session != self._session or self._active is not None:
                raise ValueError("cannot release control with active navigation or mismatched owner")
            self._retired_sessions.add(session)
            self._session = None
            self._high_water = -1

    def transition(
        self, session: str, sequence: int, status: str, *, reason=None, result=None, release_navigation=True
    ) -> dict:
        with self._lock:
            key = (session, sequence)
            receipt = self._receipts[key]
            if receipt["status"] in TERMINAL:
                if receipt["status"] != status:
                    raise ValueError("terminal command outcome is immutable")
                return deepcopy(receipt)
            if status not in TERMINAL | {"running"}:
                raise ValueError("invalid command transition")
            receipt.update(status=status, reason=reason, revision=receipt["revision"] + 1)
            if result is not None:
                receipt["result"] = deepcopy(result)
            if status in TERMINAL and self._active == key and release_navigation:
                self._active = None
            return deepcopy(receipt)

    def expired_navigation(self) -> dict | None:
        """Expiry requires adapter cancellation; it is not itself proof of stopping."""
        with self._lock:
            active = self._receipts.get(self._active)
            if active is not None and self._clock() >= active["deadline"]:
                return deepcopy(active)
            return None

    def snapshot(self) -> list[dict]:
        with self._lock:
            return deepcopy(list(self._receipts.values()))
