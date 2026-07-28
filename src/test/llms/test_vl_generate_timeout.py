# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Hard wall-clock timeout for Qwen3-VL ``model.generate``."""

from __future__ import annotations

import time

import pytest

from emet.llms.qwen3_vl_client import (
    VlGenerateTimeoutError,
    _generate_with_heartbeat,
    resolve_vl_generate_timeout_s,
)
from emet.llms.repetition_stop import HardTimeStop


def test_resolve_vl_generate_timeout_default(monkeypatch):
    monkeypatch.delenv("EMET_VL_GENERATE_TIMEOUT_S", raising=False)
    assert resolve_vl_generate_timeout_s() == 180.0


def test_resolve_vl_generate_timeout_disabled(monkeypatch):
    monkeypatch.setenv("EMET_VL_GENERATE_TIMEOUT_S", "0")
    assert resolve_vl_generate_timeout_s() is None


def test_resolve_vl_generate_timeout_custom(monkeypatch):
    monkeypatch.setenv("EMET_VL_GENERATE_TIMEOUT_S", "90")
    assert resolve_vl_generate_timeout_s() == 90.0


def test_hard_time_stop_fires():
    stop = HardTimeStop(0.05)
    time.sleep(0.06)
    assert stop(None, None) is True  # type: ignore[arg-type]
    assert stop.fired is True


def test_generate_with_heartbeat_hard_timeout():
    def _slow() -> str:
        time.sleep(2.0)
        return "done"

    with pytest.raises(VlGenerateTimeoutError, match="exceeded 0.3"):
        _generate_with_heartbeat(
            _slow,
            input_len=10,
            max_new=8,
            has_vision=True,
            heartbeat_s=0.1,
            timeout_s=0.3,
        )


def test_generate_with_heartbeat_completes_under_timeout():
    out = _generate_with_heartbeat(
        lambda: "ok",
        input_len=1,
        max_new=1,
        has_vision=False,
        heartbeat_s=0,
        timeout_s=2.0,
    )
    assert out == "ok"
