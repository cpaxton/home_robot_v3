# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared strict command transport for robot clients; telemetry steps are not receipts."""

import math
import time
from uuid import uuid4

from emet.core.command_tracker import PROTOCOL_VERSION, TERMINAL


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def _messages(client):
    return [m for name in ("_state", "_obs", "_servo") if isinstance(m := getattr(client, name, None), dict)]


def _check_peer(client, boot=None):
    protocols = [m["command_protocol"] for m in _messages(client) if "command_protocol" in m]
    if not protocols or any(p.get("version") != PROTOCOL_VERSION for p in protocols):
        raise RuntimeError("Bridge lacks command protocol v2; deploy matching core and bridge before motion")
    boots = {p.get("server_boot_id") for p in protocols}
    if None in boots or len(boots) != 1 or (boot is not None and boot not in boots):
        raise RuntimeError("Server boot changed; command outcome unknown, automatic replay forbidden")
    return next(iter(boots))


def command_receipt(client, action):
    identity = action["command"]
    _check_peer(client, identity["server_boot_id"])
    found = []
    for message in _messages(client):
        error = message.get("command_error")
        if isinstance(error, dict) and all(error.get(k) == v for k, v in identity.items()):
            raise RuntimeError(error["reason"])
        found.extend(r for r in message.get("command_receipts", []) if all(r.get(k) == v for k, v in identity.items()))
    latest = max(found, key=lambda r: r["revision"], default=None)
    # Cached image streams may arrive after fast state: never regress the receipt.
    previous = getattr(client, "_command_receipt", None)
    if previous and all(previous.get(k) == v for k, v in identity.items()):
        if latest is None or previous["revision"] > latest["revision"]:
            latest = previous
    if latest is not None:
        client._command_receipt = latest
    return latest


def send_command(client, payload, *, timeout=5.0, reliable=True):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("acknowledgement timeout must be positive and finite")
    with client._act_lock:
        boot = _check_peer(client, getattr(client, "_command_boot", None))
        client._command_boot = boot
        if not hasattr(client, "_command_session"):
            client._command_session = uuid4().hex
        sequence = getattr(client, "_command_sequence", 0)
        client._command_sequence = sequence + 1
        step = max(0, int(client._iter), int(getattr(client, "_last_step", -1)) + 1)
        client._iter = step + 1
        action = _plain({k: v for k, v in payload.items() if k not in {"command", "step"}})
        action.update(
            step=step,
            command={
                "version": PROTOCOL_VERSION,
                "server_boot_id": boot,
                "client_session_id": client._command_session,
                "sequence": sequence,
            },
        )
        if "xyt" in action:
            client._last_navigation_command = action
        deadline = time.monotonic() + timeout
        client.send_message(action)
        while reliable:
            receipt = command_receipt(client, action)
            if receipt is not None:
                if receipt["status"] in {"failed", "rejected", "cancelled"}:
                    raise RuntimeError(f"Command {sequence} {receipt['status']}: {receipt.get('reason')}")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Command {sequence} acknowledgement timed out; outcome unknown")
            time.sleep(0.05)
            _check_peer(client, boot)
            client.send_message(action)
        return action


def wait_navigation(client, action, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = command_receipt(client, action)
        if receipt and receipt["status"] in TERMINAL:
            return receipt["status"] == "succeeded"
        time.sleep(0.05)
    identity = action["command"]
    send_command(client, {"cancel_navigation": {k: identity[k] for k in ("client_session_id", "sequence")}})
    receipt = command_receipt(client, action)
    if not receipt or receipt["status"] not in TERMINAL:
        raise TimeoutError("Navigation expired; cancellation outcome unknown")
    return receipt["status"] == "succeeded"


def close_command_session(client):
    """Best-effort safe handoff before closing telemetry; uncertainty retains ownership."""
    if not hasattr(client, "_command_session"):
        return
    import logging

    try:
        action = getattr(client, "_last_navigation_command", None)
        if action is not None:
            receipt = command_receipt(client, action)
            if receipt is None or receipt["status"] not in TERMINAL:
                identity = action["command"]
                send_command(client, {"cancel_navigation": {k: identity[k] for k in ("client_session_id", "sequence")}})
        send_command(client, {"release_control": True}, timeout=2.0)
    except (RuntimeError, TimeoutError) as exc:
        logging.getLogger(__name__).warning("Control handoff not confirmed: %s", exc)
