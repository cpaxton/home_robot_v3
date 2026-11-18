# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Keep pending, infrastructure failures and incorrect answers distinct."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/summarize_benchmark_refresh.py"
spec = importlib.util.spec_from_file_location("refresh_summary", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_assigned_denominators_include_all_outcomes(tmp_path):
    tasks = [{"id": f"eqa-lazy_graph-q{i}", "family": "eqa"} for i in range(4)]
    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": tasks, "source_sha": "frozen"}))
    rows = [
        {"id": t["id"], "source_sha": "frozen", "complete": ok, "success": success}
        for t, ok, success in zip(tasks, [True, True, False], [True, False, None], strict=False)
    ]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    table, groups = module.summarize(tmp_path)
    g = groups["eqa/lazy_graph"]
    assert (g["assigned"], g["complete"], g["errors"], g["pending"], g["success"]) == (4, 2, 1, 1, 1)
    assert g["accuracy_among_complete"] == 0.5
    assert all(t["sim_time_s"] is None for t in table)
    assert len(table) == 4
    (tmp_path / "results.jsonl").write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        module.summarize(tmp_path)
