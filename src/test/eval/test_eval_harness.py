# Copyright (c) Chris Paxton 2026
"""Tests for emet.eval.harness crash policy and progress helpers."""

from __future__ import annotations

from pathlib import Path

from emet.eval.harness import (
    CrashPolicy,
    CrashStreak,
    count_scored_units,
    detect_host_freeze,
    native_signal_name,
    update_eval_progress,
    write_crash_marker,
)


def test_crash_policy_from_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("NATIVE_CRASH_ABORT", raising=False)
    monkeypatch.delenv("NATIVE_CRASH_POLICY", raising=False)
    p = CrashPolicy.from_env({})
    assert p.policy == "skip"
    assert p.retries == 1
    assert p.streak_abort == 2


def test_crash_policy_abort_alias(monkeypatch) -> None:
    p = CrashPolicy.from_env({"NATIVE_CRASH_ABORT": "1"})
    assert p.policy == "abort"


def test_crash_streak_abort() -> None:
    policy = CrashPolicy(policy="skip", streak_abort=2)
    streak = CrashStreak()
    streak.record_crash("14")
    assert not streak.should_abort(policy)
    streak.record_crash("48")
    assert streak.should_abort(policy)
    streak.record_ok()
    streak.record_crash("49")
    assert not streak.should_abort(policy)


def test_native_signal_name() -> None:
    assert native_signal_name(139) == "SIGSEGV"
    assert native_signal_name(0) is None


def test_write_crash_marker_and_progress(tmp_path: Path) -> None:
    write_crash_marker(tmp_path, "classic", 14, returncode=139, signal_name="SIGSEGV")
    assert (tmp_path / "classic_q14.CRASH").is_file()
    (tmp_path / "classic_q2.jsonl").write_text('{"question_id":2,"correct":true}\n')
    assert count_scored_units(tmp_path, ["classic"], [2, 14]) == 1
    update_eval_progress(
        tmp_path,
        units_done=1,
        units_total=64,
        phase="classic",
        current_id="14",
        units_failed=1,
    )
    data = (tmp_path / "progress.json").read_text()
    assert '"units_failed": 1' in data


def test_detect_host_freeze(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text('{"phase":"classic","current_id":"48","units_done":26,"units_total":64}\n')
    (tmp_path / "classic_q48.jsonl").write_text("")
    (tmp_path / "classic.log").write_bytes(b"hello\x00\x00\x00\x00")
    info = detect_host_freeze(tmp_path)
    assert info is not None
    assert info["question_id"] == "48"
    assert info["log_trailing_nuls"] >= 3
