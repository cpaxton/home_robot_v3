# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Server command lifecycle. All robot motion is delegated to explicit adapter hooks."""

from __future__ import annotations

import math
import threading

from emet.core.command_tracker import CommandTracker


class CommandRuntime:
    def initialize_commands(self):
        self.command_tracker = CommandTracker()
        self._command_lock = threading.RLock()
        self._navigation_command = None
        self._navigation_fault = False
        self._command_error = None

    def command_message(self, message):
        if message is None:
            return None
        with self._command_lock:
            return {
                **message,
                "command_protocol": self.command_tracker.metadata(),
                "command_receipts": self.command_tracker.snapshot(),
                "command_error": self._command_error,
            }

    def dispatch_command(self, action):
        with self._command_lock:
            if not isinstance(action, dict):
                return False
            identity = action.get("command", {})
            if not isinstance(identity, dict):
                return False
            payload = {k: v for k, v in action.items() if k not in {"command", "step"}}
            try:
                if "step" in action and (type(action["step"]) is not int or action["step"] < 0):
                    raise ValueError("invalid telemetry step")
                if "xyt" in payload:
                    for flag in ("nav_relative", "nav_world", "nav_teleport", "nav_blocking"):
                        if flag in payload and type(payload[flag]) is not bool:
                            raise ValueError(f"{flag} must be boolean")
                    if payload.get("nav_relative") and payload.get("nav_world"):
                        raise ValueError("relative and world navigation frames are mutually exclusive")
                    if set(payload) - {
                        "xyt",
                        "nav_relative",
                        "nav_world",
                        "nav_teleport",
                        "nav_blocking",
                        "nav_timeout_s",
                    }:
                        raise ValueError("navigation must be a standalone command")
                    goal = payload["xyt"]
                    if not isinstance(goal, list) or len(goal) != 3 or not all(math.isfinite(float(v)) for v in goal):
                        raise ValueError("navigation goal must contain three finite coordinates")
                dispatch, receipt = self.command_tracker.accept(identity, payload)
            except (ValueError, TypeError, OverflowError) as exc:
                self._command_error = {**identity, "reason": str(exc)}
                return False
            if not dispatch:
                return False
            session, sequence = receipt["client_session_id"], receipt["sequence"]
            try:
                if payload == {"release_control": True}:
                    self.command_tracker.release_control(session)
                    self.command_tracker.transition(session, sequence, "succeeded")
                elif "cancel_navigation" in payload:
                    target = payload["cancel_navigation"]
                    if (
                        set(payload) != {"cancel_navigation"}
                        or not isinstance(target, dict)
                        or set(target) != {"client_session_id", "sequence"}
                    ):
                        raise ValueError("invalid cancellation target")
                    current = self._navigation_command
                    if current is None:
                        terminal = next(
                            (
                                r
                                for r in self.command_tracker.snapshot()
                                if all(r.get(k) == target.get(k) for k in ("client_session_id", "sequence"))
                            ),
                            None,
                        )
                        if terminal is None or not (
                            terminal["status"] in {"succeeded", "cancelled"}
                            or terminal.get("result", {}).get("stop_confirmed")
                        ):
                            raise ValueError("cancellation target has no confirmed stop")
                    elif target != {"client_session_id": current[0], "sequence": current[1]}:
                        raise ValueError("cancellation does not identify the active navigation command")
                    else:
                        self._finish_cancel("cancelled", "client cancellation")
                    if self._navigation_fault:
                        raise RuntimeError("stop could not be confirmed")
                    self.command_tracker.transition(session, sequence, "succeeded")
                elif "xyt" in payload:
                    # Install identity before starting the adapter; dispatch is serialized.
                    self._navigation_command = (session, sequence, None)
                    context = self.start_navigation_command({"nav_timeout_s": 30.0, **payload})
                    self._navigation_command = (session, sequence, context)
                    self.command_tracker.transition(session, sequence, "running", result=context)
                else:
                    self.handle_action({**payload, "step": action.get("step", sequence)})
                    self.command_tracker.transition(session, sequence, "succeeded")
            except Exception as exc:
                if "xyt" in payload:
                    self._finish_cancel("failed", f"navigation start failed: {exc}")
                else:
                    self.command_tracker.transition(session, sequence, "failed", reason=str(exc))
            self._last_step = max(self._last_step, action.get("step", sequence))
            return True

    def poll_navigation_command(self):
        with self._command_lock:
            if self._navigation_command is None or self._navigation_fault:
                return
            if self.command_tracker.expired_navigation() is not None:
                self._finish_cancel("failed", "motion deadline exceeded")
                return
            session, sequence, context = self._navigation_command
            try:
                outcome = self.navigation_command_result(context)
            except Exception as exc:
                self._finish_cancel("failed", f"navigation status failed: {exc}")
                return
            if outcome is not None:
                status, result = outcome
                if status == "succeeded":
                    self.command_tracker.transition(session, sequence, status, result=result)
                    self._navigation_command = None
                else:
                    self._finish_cancel("failed", str(result))

    def _finish_cancel(self, status, reason):
        session, sequence, _ = self._navigation_command
        try:
            stopped = self.cancel_navigation_command() is True
        except Exception:
            stopped = False
        self._navigation_fault = not stopped
        self.command_tracker.transition(
            session,
            sequence,
            status if stopped else "failed",
            reason=reason if stopped else f"{reason}; stop unconfirmed",
            result={"stop_confirmed": stopped},
            release_navigation=stopped,
        )
        if stopped:
            self._navigation_command = None

    def start_navigation_command(self, action):
        raise NotImplementedError("navigation contract is not implemented by this adapter")

    def navigation_command_result(self, context):
        raise NotImplementedError

    def cancel_navigation_command(self):
        return False
