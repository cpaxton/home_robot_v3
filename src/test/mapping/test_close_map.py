# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU-only tests for the occupancy-plane close-look map."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.mapping.close_map import (
    CloseDistanceMap,
    CloseLookQuery,
    close_map_catalog_fields,
    close_map_from_voxel_map,
    decide_close_look,
    format_close_map_hint,
)


def _map(*, res: float = 0.1) -> CloseDistanceMap:
    return CloseDistanceMap(grid_size=(32, 32), origin_xy=(16.0, 16.0), resolution_m=res)


def _pose_looking_plus_x(*, cam_xyz=(0.0, 0.0, 0.4)) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(cam_xyz, dtype=np.float64)
    pose[:3, 2] = (1.0, 0.0, 0.0)
    return pose


def test_decide_close_look_no_map_does_not_stay_or_escape() -> None:
    d = decide_close_look(None, approaches_left=4, nav_blocked=False, attempts=0)
    assert d.stay is False and d.escape is False
    assert d.reason == "no_map"


def test_decide_close_look_stays_until_resolved_or_exhausted() -> None:
    q = CloseLookQuery(
        x=0.0,
        y=0.0,
        radius_m=0.35,
        n_hit_cells=3,
        n_resolved_cells=0,
        min_cam_dist_m=1.2,
        aimed_hit=False,
        resolved=False,
    )
    stay = decide_close_look(q, approaches_left=3, nav_blocked=False, attempts=0)
    assert stay.stay is True and stay.escape is False
    assert stay.reason == "unresolved_stay"

    blocked = decide_close_look(q, approaches_left=3, nav_blocked=True, attempts=0)
    assert blocked.escape is True and blocked.reason == "escape_unreachable"

    exhausted = decide_close_look(q, approaches_left=0, nav_blocked=False, attempts=4)
    assert exhausted.escape is True and exhausted.reason == "escape_exhausted"

    resolved_q = CloseLookQuery(
        x=0.0,
        y=0.0,
        radius_m=0.35,
        n_hit_cells=2,
        n_resolved_cells=1,
        min_cam_dist_m=0.4,
        aimed_hit=True,
        resolved=True,
    )
    done = decide_close_look(resolved_q, approaches_left=3, nav_blocked=False, attempts=1)
    assert done.stay is False and done.escape is False
    assert done.reason == "resolved"


def test_close_distance_map_aimed_near_hit_is_resolved() -> None:
    cm = _map()
    pose = _pose_looking_plus_x()
    # 0.30 m along the optical axis, within default r_close_m=0.55.
    pts = np.array([[0.30, 0.0, 0.4]], dtype=np.float64)
    assert cm.update_from_view(pose, pts) >= 1
    q = cm.query_xy(0.30, 0.0)
    assert q.n_hit_cells >= 1
    assert q.aimed_hit is True
    assert q.resolved is True
    assert q.min_cam_dist_m is not None and q.min_cam_dist_m < 0.55


def test_close_distance_map_far_hit_is_not_resolved() -> None:
    cm = _map()
    pose = _pose_looking_plus_x()
    pts = np.array([[1.20, 0.0, 0.4]], dtype=np.float64)
    assert cm.update_from_view(pose, pts) >= 1
    q = cm.query_xy(1.20, 0.0)
    assert q.n_hit_cells >= 1
    assert q.resolved is False


def test_close_map_from_voxel_map_and_hint() -> None:
    assert close_map_from_voxel_map(None) is None
    assert close_map_from_voxel_map(SimpleNamespace()) is None
    cm = _map()
    voxel = SimpleNamespace(close_map=cm)
    assert close_map_from_voxel_map(voxel) is cm
    hint = format_close_map_hint(cm, 0.0, 0.0)
    assert "NOT resolved" in hint
    assert format_close_map_hint(None, 0.0, 0.0) == ""
    fields = close_map_catalog_fields(voxel, 0.0, 0.0)
    assert fields is not None
    assert fields["resolved"] is False
    assert "aimed" in fields
    assert "min_cam_m" in fields
    assert close_map_catalog_fields(None, 0.0, 0.0) is None
