# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU tests for OVMM GT object selection (no sim)."""

from __future__ import annotations

import numpy as np

from emet.eval.ovmm_probe_targets import (
    pick_find_object,
    pick_view_targets,
    resolve_phrases,
)


def test_pick_find_object_skips_sugar_cube() -> None:
    placements = {
        "obj_main": {"cat": "sugar_cube", "pos": np.array([0.3, -0.4, 0.93])},
        "distr_bottle": {"cat": "bottle", "pos": np.array([0.5, -0.3, 0.95])},
        "cab_main": {"cat": "cab", "pos": np.array([2.25, -0.2, 1.85])},
    }
    picked = pick_find_object(placements)
    assert picked is not None
    assert picked["cat"] == "bottle"
    assert picked["body"] == "distr_bottle"
    none = pick_find_object({"obj_main": {"cat": "sugar_cube", "pos": np.array([0.3, -0.4, 0.9])}})
    assert none is None


def test_pick_view_targets_object_and_nearest_cabinet() -> None:
    placements = {
        "obj_main": {"cat": "can", "pos": np.array([0.5, 0.1, 0.9])},
        "cab_far": {"cat": "cabinet", "pos": np.array([4.0, 0.0, 1.0])},
        "cab_near": {"cat": "kitchen cabinet", "pos": np.array([0.8, 0.1, 1.0])},
        "counter_0": {"cat": "counter", "pos": np.array([0.4, 0.0, 0.9])},
    }
    targets = pick_view_targets(placements, object_body="obj_main")
    assert [t["id"] for t in targets] == ["object", "cabinet"]
    assert targets[0]["cat"] == "can"
    assert targets[1]["body"] == "cab_near"
    assert targets[1].get("yaw_only") is True
    with_counter = pick_view_targets(placements, include_counter=True)
    assert [t["id"] for t in with_counter] == ["object", "cabinet", "counter"]


def test_resolve_phrases_uses_findable_cat_not_sugar_cube() -> None:
    phrases = resolve_phrases(
        None,
        {
            "obj_main": {"cat": "sugar_cube", "pos": np.array([0.3, -0.4, 0.9])},
            "distr_bowl": {"cat": "bowl", "pos": np.array([0.4, -0.3, 0.9])},
        },
    )
    assert phrases[0] == "bowl"
    assert "sugar cube" not in {p.lower() for p in phrases}
    same = resolve_phrases(["can"], {"obj_main": {"cat": "can"}}, object_body="obj_main")
    assert same == ["can"]
    skipped = resolve_phrases(["sugar cube", "cabinet"], None)
    assert skipped == ["cabinet"]
