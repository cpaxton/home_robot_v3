# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "relation_audit", Path(__file__).resolve().parents[3] / "scripts/audit_placement_relations.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize(
    "box", [None, [], [True, 0, 10, 10], [0, 0, 1001, 100], [20, 0, 10, 10], [0, 0, float("nan"), 10]]
)
def test_invalid_box(box):
    assert not audit.valid_box(box)


def test_relations_and_missing_anchor():
    left, right = [0, 0, 200, 200], [800, 0, 1000, 200]
    observation = {"countertops": [left, right], "stove": [[400, 0, 600, 400]], "refrigerator": []}
    assert audit.valid_localization(observation)
    assert audit.spatial_selection(observation, "stove", "left") == left
    assert audit.spatial_selection(observation, "stove", "right") == right
    assert audit.spatial_selection(observation, "refrigerator", "right") is None
    observation["countertops"].append([700, 0, 800, 200])
    assert audit.spatial_selection(observation, "stove", "right") is None


def test_margin_ambiguity_and_invalid_schema():
    observation = {"countertops": [[450, 0, 600, 200]], "stove": [[400, 0, 600, 400]], "refrigerator": []}
    assert audit.spatial_selection(observation, "stove", "right") is None
    assert not audit.valid_localization({})
    assert not audit.valid_localization({**observation, "stove": "bad"})


def test_scoring():
    box = [0, 0, 100, 100]
    assert audit.score(box, [box])["correct"]
    assert audit.score(None, [box])["correct"] is False
    assert audit.score(None, [])["correct"]
    assert not audit.score(box, [])["correct"]
