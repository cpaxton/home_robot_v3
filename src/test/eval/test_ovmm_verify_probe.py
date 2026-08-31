# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU tests for live verify-probe geometry and score gates (no sim)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from emet.eval.ovmm_verify_probe import backup_distance_m, interpret_scores, standoff_xyt


def test_standoff_faces_object_from_spawn_side() -> None:
    obj = np.array([1.0, 0.0, 0.9], dtype=np.float64)
    robot = np.array([0.0, 0.0], dtype=np.float64)
    xyt = standoff_xyt(obj, robot, standoff_m=1.0)
    np.testing.assert_allclose(xyt[:2], [0.0, 0.0], atol=1e-6)
    assert abs(xyt[2]) < 1e-6
    xyt2 = standoff_xyt(obj, np.array([1.0, 2.0]), standoff_m=1.0)
    np.testing.assert_allclose(xyt2[:2], [1.0, 1.0], atol=1e-6)
    assert abs(xyt2[2] + math.pi / 2.0) < 1e-6


def test_backup_distance_only_when_too_close() -> None:
    obj = np.array([0.3, -0.4, 0.9])
    need = 1.6 - float(np.linalg.norm([0.3, -0.4]))
    assert backup_distance_m(obj, np.array([0.0, 0.0]), min_standoff_m=1.6) == pytest.approx(need)
    assert backup_distance_m(obj, np.array([0.0, 3.0]), min_standoff_m=1.6) == 0.0


def test_interpret_scores_yoloe_or_siglip() -> None:
    miss = interpret_scores(yoloe_score=0.05, yoloe_bbox=(1, 2, 3, 4), dense_max=0.08)
    assert miss["yoloe_hit"] is False
    assert miss["siglip_present"] is False
    assert miss["verified"] is False
    yoloe = interpret_scores(yoloe_score=0.22, yoloe_bbox=(1, 2, 3, 4), dense_max=0.05)
    assert yoloe["yoloe_hit"] is True
    assert yoloe["verified"] is True
    sig = interpret_scores(yoloe_score=0.0, yoloe_bbox=None, dense_max=0.14)
    assert sig["siglip_present"] is True
    assert sig["verified"] is True
