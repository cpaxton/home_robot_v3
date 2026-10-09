# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("surface_audit", scripts / "audit_surface_verification.py")
audit = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(audit)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("value", [None, [True], [0, 0], [2], ["0"]])
def test_invalid_ids_fail_closed(value):
    with pytest.raises(ValueError):
        audit.accepted_ids({"countertop_ids": value}, {0, 1})


def test_individual_cannot_accept_unshown_candidate():
    with pytest.raises(ValueError):
        audit.accepted_ids({"countertop_ids": [1]}, {0})
    assert audit.accepted_ids({"countertop_ids": []}, {0}) == []
    assert audit.accepted_ids({"countertop_ids": [0, 1]}, {0, 1}) == [0, 1]


def test_boxes_are_plain_normalized_numbers():
    regions = [{"id": 0, "bbox_xyxy": [10, 20, 30, 40]}]
    assert audit.selected_boxes(regions, [0], 100, 200) == [[100.0, 100.0, 300.0, 200.0]]
    assert audit.selected_boxes(regions, [], 100, 200) == []
