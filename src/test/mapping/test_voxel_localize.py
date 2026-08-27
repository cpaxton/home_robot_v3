# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU tests for DynaMem localize_text → XYZ."""

from __future__ import annotations

import numpy as np

from emet.mapping.voxel_localize import (
    localize_text_xyz,
    localize_text_xyz_from_phrases,
    voxel_map_from_agent,
)


class _Voxel:
    def __init__(self, hits: dict[str, list[float] | None]) -> None:
        self.hits = {k.lower(): v for k, v in hits.items()}
        self._last_localize_stats: dict[str, object] = {}

    def localize_text(self, text, debug=False, return_debug=False):
        q = str(text or "").strip().lower()
        pt = self.hits.get(q)
        self._last_localize_stats = {
            "query": text,
            "max_cosine": 0.22 if pt is not None else 0.05,
            "yoloe_hit": pt is not None,
        }
        return None if pt is None else np.asarray(pt, dtype=np.float64)


def test_localize_text_xyz_returns_point_and_stats():
    vm = _Voxel({"red cylinder": [0.08, -0.55, 0.6]})
    xyz, stats = localize_text_xyz(vm, "red cylinder")
    assert xyz is not None
    assert np.allclose(xyz, [0.08, -0.55, 0.6])
    assert stats["yoloe_hit"] is True


def test_localize_text_xyz_from_phrases_prefers_full_phrase():
    vm = _Voxel(
        {
            "cylinder": [9.0, 9.0, 9.0],
            "red cylinder": [0.08, -0.55, 0.6],
        }
    )
    xyz, used, _stats = localize_text_xyz_from_phrases(vm, ["red cylinder", "cylinder"])
    assert used == "red cylinder"
    assert xyz is not None
    assert np.allclose(xyz, [0.08, -0.55, 0.6])


def test_localize_text_xyz_rejects_non_points():
    xyz, _stats = localize_text_xyz(None, "red cylinder")
    assert xyz is None
    xyz, _stats = localize_text_xyz(_Voxel({}), "red cylinder")
    assert xyz is None


def test_voxel_map_from_agent_uses_attribute_or_getter():
    class _Agent:
        voxel_map = _Voxel({"blue cube": [-0.02, -0.55, 0.6]})

    assert voxel_map_from_agent(_Agent()) is _Agent.voxel_map

    class _Getter:
        def get_voxel_map(self):
            return _Voxel({"x": [1.0, 2.0, 3.0]})

    assert voxel_map_from_agent(_Getter()) is not None
    assert voxel_map_from_agent(None) is None
