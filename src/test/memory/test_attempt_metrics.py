# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for repeat-failure metrics and manip ledger helpers."""

from __future__ import annotations

from types import SimpleNamespace

from emet.memory.graph_eqa.attempt_ledger import AttemptRecord
from emet.memory.graph_eqa.attempt_metrics import (
    record_manip_attempt,
    summarize_repeat_failures,
)


def _rec(**kwargs) -> AttemptRecord:
    defaults = {
        "action_kind": "navigate",
        "outcome": "failed",
        "status_code": "no_path",
        "note": "",
        "step": 0,
        "target_node_id": None,
        "obs_id": None,
        "xyz": None,
        "source": "eqa",
        "question_id": None,
        "phrase": "",
    }
    defaults.update(kwargs)
    return AttemptRecord(**defaults)


def test_summarize_repeat_failures_by_node_id():
    rows = [
        _rec(target_node_id=3, step=0),
        _rec(target_node_id=3, step=1),
        _rec(target_node_id=7, step=2),
        _rec(target_node_id=3, outcome="ok", status_code="ok", step=3),
    ]
    stats = summarize_repeat_failures(rows)
    assert stats.n_attempts == 4
    assert stats.n_failures == 3
    assert stats.n_repeat_failures == 1
    assert stats.n_wasted_rounds == 1
    assert stats.by_kind == {"navigate": 1}


def test_summarize_repeat_failures_by_obs_then_xy():
    rows = [
        _rec(obs_id=9, step=0),
        _rec(obs_id=9, step=1),
        _rec(xyz=(1.0, 2.0, 0.5), step=2),
        _rec(xyz=(1.1, 2.05, 0.9), step=3),  # within 0.25 m planar → repeat
        _rec(xyz=(5.0, 5.0, 0.5), step=4),
    ]
    stats = summarize_repeat_failures(rows, xyz_tol_m=0.25)
    assert stats.n_failures == 5
    assert stats.n_repeat_failures == 2  # obs 9 once + xy once


def test_summarize_prefers_stable_target_and_distinguishes_verify_views():
    rows = [
        _rec(
            action_kind="navigate",
            target_kind="place",
            target_id="place_sink",
            obs_id=9,
        ),
        _rec(
            action_kind="navigate",
            target_kind="place",
            target_id="place_sink",
            obs_id=81,
        ),
        _rec(
            action_kind="verify",
            target_kind="place",
            target_id="place_sink",
            view_id="view_1",
        ),
        _rec(
            action_kind="verify",
            target_kind="place",
            target_id="place_sink",
            view_id="view_2",
        ),
    ]

    stats = summarize_repeat_failures(rows)

    assert stats.n_repeat_failures == 1
    assert stats.by_kind == {"navigate": 1}


def test_summarize_filters_kinds():
    rows = [
        _rec(action_kind="navigate", target_node_id=1),
        _rec(action_kind="pick", phrase="cup", outcome="failed"),
        _rec(action_kind="pick", phrase="cup", outcome="failed"),
    ]
    stats = summarize_repeat_failures(rows, kinds={"pick"})
    assert stats.n_attempts == 2
    assert stats.n_repeat_failures == 1
    assert stats.by_kind == {"pick": 1}


def test_record_manip_attempt_writes_when_enabled():
    calls: list[dict] = []

    class FakeGM:
        def record_attempt(self, **kwargs):
            calls.append(kwargs)

    gm = FakeGM()
    record_manip_attempt(
        gm,
        action_kind="pick",
        success=False,
        phrase="mug",
        status_code="controller_failed",
        note="test",
        xyz=(1.0, 2.0, 3.0),
        source="eqa",
    )
    assert len(calls) == 1
    assert calls[0]["action_kind"] == "pick"
    assert calls[0]["outcome"] == "failed"
    assert calls[0]["xyz"] == (1.0, 2.0, 3.0)


def test_record_manip_attempt_noop_without_memory():
    record_manip_attempt(None, action_kind="place", success=True)
    record_manip_attempt(SimpleNamespace(), action_kind="place", success=True)
