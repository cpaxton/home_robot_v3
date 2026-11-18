import json
import runpy
from collections import Counter
from pathlib import Path

import pytest


def script():
    return runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/eval_wbc_motion_head.py"))


def test_confirmation_is_fresh_balanced_and_valid_without_selecting_cases():
    from emet.rl.manipulation.arm_splines import SplineLevel, load_manifest, validate_case

    namespace = script()
    spec = load_manifest(Path(__file__).resolve().parents[3] / "configs/benchmarks/mars_arm_splines_v2.json")
    cases = namespace["confirmation_cases"](spec)
    assert len({c["id"] for c in cases}) == len({c["seed"] for c in cases}) == 32
    assert Counter(c["level"] for c in cases) == {0: 16, 1: 8, 2: 8}
    assert not {c["seed"] for c in cases} & {c["seed"] for c in spec["cases"]}
    for case in cases:
        validate_case(case, [SplineLevel(**x) for x in spec["levels"]])
    for level in range(3):
        assert len(set(Counter(c["posture"] for c in cases if c["level"] == level).values())) == 1
    with pytest.raises(ValueError, match="overlap"):
        namespace["confirmation_cases"]({"cases": cases})


def test_unfinished_or_changed_source_fit_is_rejected_before_simulator(tmp_path, monkeypatch):
    namespace = script()
    namespace["run"].__globals__["source_hashes"] = lambda: {"source": "current"}
    fit = tmp_path / "fit"
    fit.mkdir()
    for report in (
        {"complete": False, "sources": {"source": "current"}},
        {"complete": True, "sources": {"source": "old"}},
    ):
        (fit / "report.json").write_text(json.dumps(report))
        output = tmp_path / "output"
        with pytest.raises(ValueError, match="source-matched"):
            namespace["run"](fit, "unused-initializer", output)
        assert not output.exists()
