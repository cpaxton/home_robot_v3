"""Opt-in reactive motion lifecycle, with the same ownership as navigation.

No simulation, ML or robot dependencies belong in this module.
"""

import math

REACTIVE_KEYS = frozenset({"reactive_goal", "update_reactive_goal", "cancel_reactive_goal"})


class ReactiveRuntime:
    def supports_reactive_control(self):
        return False

    def validate_reactive_goal(self, goal):
        raise NotImplementedError

    def _validate_reactive_payload(self, payload):
        keys = REACTIVE_KEYS.intersection(payload)
        if not keys:
            return
        if len(keys) != 1 or not self.supports_reactive_control():
            raise ValueError("reactive control is not supported or command is ambiguous")
        key = next(iter(keys))
        if set(payload) - ({key, "reactive_timeout_s"} if key == "reactive_goal" else {key}):
            raise ValueError("reactive command must be standalone")
        if key == "reactive_goal":
            timeout = float(payload.get("reactive_timeout_s", 10.0))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("reactive timeout must be positive and finite")
            self.validate_reactive_goal(payload[key])
        else:
            value = payload[key]
            if key == "update_reactive_goal":
                if not isinstance(value, dict) or set(value) != {"target", "goal"}:
                    raise ValueError("update requires target identity and goal")
                self.validate_reactive_goal(value["goal"])
                value = value["target"]
            if (
                not isinstance(value, dict)
                or set(value) != {"client_session_id", "sequence"}
                or not isinstance(value["client_session_id"], str)
                or type(value["sequence"]) is not int
            ):
                raise ValueError("invalid reactive target identity")

    def _dispatch_reactive(self, payload, session, sequence):
        if "reactive_goal" in payload:
            self._reactive_command = (session, sequence, None)
            try:
                context = self.start_reactive_command(payload["reactive_goal"])
                self._reactive_command = (session, sequence, context)
                self.command_tracker.transition(session, sequence, "running", result=context)
            except Exception as exc:
                self._finish_reactive_cancel("failed", f"reactive start failed: {exc}")
            return
        current = self._reactive_command
        update = "update_reactive_goal" in payload
        value = payload["update_reactive_goal"] if update else payload["cancel_reactive_goal"]
        target = value["target"] if update else value
        if current is None:
            terminal = next(
                (r for r in self.command_tracker.snapshot() if all(r.get(k) == v for k, v in target.items())), None
            )
            if update or terminal is None or not terminal.get("result", {}).get("stop_confirmed"):
                raise ValueError("target is not an active reactive command with a confirmed stop")
        elif target != {"client_session_id": current[0], "sequence": current[1]}:
            raise ValueError("target does not identify active reactive command")
        elif self._reactive_fault:
            raise RuntimeError("reactive stop unconfirmed; restart required")
        elif update:
            if self.command_tracker.expired_navigation() is not None:
                self._finish_reactive_cancel("failed", "motion deadline exceeded")
                raise RuntimeError("reactive deadline exceeded")
            try:
                context = self.update_reactive_command(value["goal"])
            except Exception:
                self._finish_reactive_cancel("failed", "reactive update failed")
                raise
            self._reactive_command = (current[0], current[1], context)
            self.command_tracker.transition(current[0], current[1], "running", result=context)
        else:
            self._finish_reactive_cancel("cancelled", "client cancellation")
        if self._reactive_fault:
            raise RuntimeError("reactive stop unconfirmed")
        self.command_tracker.transition(session, sequence, "succeeded")

    def poll_reactive_command(self):
        # Caller owns _command_lock; controller stepping never acquires that lock.
        if self._reactive_command is None or self._reactive_fault:
            return
        if self.command_tracker.expired_navigation() is not None:
            self._finish_reactive_cancel("failed", "motion deadline exceeded")
            return
        try:
            outcome = self.reactive_command_result(self._reactive_command[2])
        except Exception as exc:
            self._finish_reactive_cancel("failed", f"reactive status failed: {exc}")
            return
        if outcome is not None:
            status, result = outcome
            self._finish_reactive_cancel(
                status, "measured completion" if status == "succeeded" else "controller failed", result=result
            )

    def _finish_reactive_cancel(self, status, reason, *, result=None):
        session, sequence, context = self._reactive_command
        try:
            stopped = self.cancel_reactive_command() is True
        except Exception:
            stopped = False
        self._reactive_fault = not stopped
        self.command_tracker.transition(
            session,
            sequence,
            status if stopped else "failed",
            reason=reason if stopped else f"{reason}; stop unconfirmed",
            result={**(context or {}), **(result or {}), "stop_confirmed": stopped},
            release_navigation=stopped,
        )
        if stopped:
            self._reactive_command = None

    def start_reactive_command(self, goal):
        raise NotImplementedError

    def update_reactive_command(self, goal):
        raise NotImplementedError

    def reactive_command_result(self, context):
        raise NotImplementedError

    def cancel_reactive_command(self):
        return False
