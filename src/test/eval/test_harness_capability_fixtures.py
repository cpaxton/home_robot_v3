# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
from collections import Counter
from pathlib import Path

import pytest

from emet.eval.harness_capability_fixtures import load_development_cases, qualify_geometry

ROOT = Path(__file__).resolve().parents[3]
CASES = load_development_cases(ROOT / "configs/eval/harness_capability_dev.json")


def test_development_population_is_six_predeclared_pairs():
    assert set(Counter(c["family"] for c in CASES).values()) == {2}
    assert len({c["family"] for c in CASES}) == 6


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_real_cpu_ray_and_contact_interventions(case):
    row = qualify_geometry(case)
    assert row["status"] == "geometry_control_passed"
    assert row["active_agent_qualified"] is False
    assert row["manipulation_qualified"] is False
