# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Overnight resume skips DONE phases and passes RESUME=1 (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emet.eval import hmeqa_overnight as overnight


def test_phase_done_and_scored_helpers(tmp_path: Path) -> None:
    out = tmp_path / "bal32"
    out.mkdir()
    assert overnight._phase_done(out) is False
    assert overnight._has_scored_units(out) is False
    (out / "classic_q2.jsonl").write_text("{}\n", encoding="utf-8")
    assert overnight._has_scored_units(out) is True
    (out / "classic_q49.jsonl").write_text("", encoding="utf-8")
    assert overnight._has_scored_units(out) is True  # non-empty sibling still counts
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    assert overnight._phase_done(out) is True


def test_run_overnight_skips_done_holdout_and_resumes_bal32(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "overnight"
    hold = base / "holdout8"
    bal = base / "bal32"
    hold.mkdir(parents=True)
    bal.mkdir(parents=True)
    (hold / "DONE").write_text("ok\n", encoding="utf-8")
    (hold / "h2h_summary.json").write_text(
        json.dumps(
            {
                "classic": {"accuracy": 0.625, "n": 8, "correct": 5},
                "agentic": {"accuracy": 0.625, "n": 8, "correct": 5},
            }
        ),
        encoding="utf-8",
    )
    (base / "gate.json").write_text(
        json.dumps(
            {
                "need_retune": False,
                "reason": "ok",
                "proceed_bal32": True,
                "classic": {"accuracy": 0.625, "n": 8, "correct": 5},
                "agentic": {"accuracy": 0.625, "n": 8, "correct": 5},
            }
        ),
        encoding="utf-8",
    )
    (bal / "classic_q2.jsonl").write_text(
        json.dumps({"question_id": 2, "correct": False}) + "\n",
        encoding="utf-8",
    )

    calls: list[dict[str, Any]] = []

    def fake_h2h(out: Path, **kwargs: Any) -> int:
        calls.append({"out": out, **kwargs})
        (out / "DONE").write_text("ok\n", encoding="utf-8")
        (out / "h2h_summary.json").write_text(
            json.dumps(
                {
                    "classic": {"accuracy": 0.3, "n": 1, "correct": 0},
                    "agentic": {"accuracy": 0.0, "n": 0, "correct": 0},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(overnight, "_run_h2h", fake_h2h)
    monkeypatch.setattr(overnight, "_summarize", lambda _out: None)
    monkeypatch.setattr(overnight, "_status_note", lambda *a, **k: None)

    rc = overnight.run_overnight(base=base, skip_bal32=False)
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["out"] == bal
    assert calls[0]["resume"] is True
    assert calls[0]["arms"] == "classic,agentic"


def test_run_overnight_skips_both_done_phases(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "overnight"
    hold = base / "holdout8"
    bal = base / "bal32"
    hold.mkdir(parents=True)
    bal.mkdir(parents=True)
    (hold / "DONE").write_text("ok\n", encoding="utf-8")
    (bal / "DONE").write_text("ok\n", encoding="utf-8")
    (hold / "h2h_summary.json").write_text("{}", encoding="utf-8")
    (bal / "h2h_summary.json").write_text(
        json.dumps(
            {
                "classic": {"accuracy": 0.4, "n": 32, "correct": 13},
                "agentic": {"accuracy": 0.4, "n": 32, "correct": 13},
            }
        ),
        encoding="utf-8",
    )
    (base / "gate.json").write_text(
        json.dumps({"need_retune": False, "proceed_bal32": True}),
        encoding="utf-8",
    )

    calls: list[Any] = []

    def _record_h2h(*a: Any, **k: Any) -> int:
        calls.append((a, k))
        return 0

    monkeypatch.setattr(overnight, "_run_h2h", _record_h2h)
    monkeypatch.setattr(overnight, "_summarize", lambda _out: None)
    monkeypatch.setattr(overnight, "_status_note", lambda *a, **k: None)

    rc = overnight.run_overnight(base=base)
    assert rc == 0
    assert calls == []
    gate = json.loads((base / "gate.json").read_text(encoding="utf-8"))
    assert gate.get("bal32_classic", {}).get("n") == 32
