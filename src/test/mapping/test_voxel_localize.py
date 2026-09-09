# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU tests for DynaMem localize_text → XYZ."""

from __future__ import annotations

import numpy as np

from emet.mapping.voxel_localize import (
    CURRENT_VIEW_OBS_ID,
    VOXEL_HYP_OBS_BASE,
    is_current_view_sentinel,
    is_proposal_handle,
    localize_confidence,
    localize_text_xyz,
    localize_text_xyz_from_phrases,
    pin_localize_xyz,
    pin_phrases_after_mapping,
    pinned_localize_xyz,
    pinned_xyz_from_phrases,
    voxel_map_from_agent,
    voxel_proposal_id,
)


class _Voxel:
    def __init__(self, hits: dict[str, list[float] | None]) -> None:
        self.hits = {k.lower(): v for k, v in hits.items()}
        self._last_localize_stats: dict[str, object] = {}
        self.n_live = 0

    def localize_text(self, text, debug=False, return_debug=False):
        self.n_live += 1
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


def test_voxel_map_from_agent_uses_getter_when_attr_lacks_localize():
    class _Occupancy:
        pass

    semantic = _Voxel({"blue cube": [-0.02, -0.55, 0.6]})

    class _Agent:
        voxel_map = _Occupancy()

        def get_voxel_map(self):
            return semantic

    assert voxel_map_from_agent(_Agent()) is semantic


def test_localize_text_xyz_pins_first_hit_and_ignores_later_live():
    vm = _Voxel({"red cylinder": [0.08, -0.55, 0.6]})
    xyz, stats = localize_text_xyz(vm, "red cylinder")
    assert xyz is not None
    assert stats["from_pin"] is False
    assert vm.n_live == 1

    vm.hits["red cylinder"] = [9.0, 9.0, 9.0]
    xyz2, stats2 = localize_text_xyz(vm, "red cylinder")
    assert xyz2 is not None
    assert np.allclose(xyz2, [0.08, -0.55, 0.6])
    assert stats2["from_pin"] is True
    assert vm.n_live == 1


def test_localize_text_xyz_pin_survives_live_miss():
    vm = _Voxel({"blue cube": [-0.02, -0.55, 0.6]})
    localize_text_xyz(vm, "blue cube")
    vm.hits.clear()
    xyz, stats = localize_text_xyz(vm, "blue cube")
    assert xyz is not None
    assert np.allclose(xyz, [-0.02, -0.55, 0.6])
    assert stats["from_pin"] is True
    assert vm.n_live == 1


def test_localize_text_xyz_from_phrases_prefers_pin_over_other_live_hits():
    vm = _Voxel(
        {
            "red cylinder": [0.08, -0.55, 0.6],
            "cylinder": [9.0, 9.0, 9.0],
        }
    )
    pin_localize_xyz(vm, "red cylinder", [0.08, -0.55, 0.6], {"yoloe_hit": True})
    vm.hits["red cylinder"] = None
    xyz, used, stats = localize_text_xyz_from_phrases(vm, ["red cylinder", "cylinder"])
    assert used == "red cylinder"
    assert xyz is not None
    assert np.allclose(xyz, [0.08, -0.55, 0.6])
    assert stats["from_pin"] is True
    assert vm.n_live == 0


def test_pinned_localize_xyz_roundtrip():
    vm = _Voxel({})
    assert pinned_localize_xyz(vm, "table")[0] is None
    pin_localize_xyz(vm, "table", [0.0, -0.5, 0.7])
    xyz, stats = pinned_localize_xyz(vm, "table")
    assert xyz is not None
    assert np.allclose(xyz, [0.0, -0.5, 0.7])
    assert stats["from_pin"] is True


def test_pin_phrases_after_mapping_queries_each_phrase_once():
    vm = _Voxel(
        {
            "red cylinder": [0.08, -0.55, 0.6],
            "blue cube": [-0.02, -0.55, 0.6],
        }
    )
    hits = pin_phrases_after_mapping(vm, ["red cylinder", "blue cube", "red cylinder"])
    assert hits == {"red cylinder": True, "blue cube": True}
    assert vm.n_live == 2
    vm.hits.clear()
    xyz, stats = localize_text_xyz(vm, "blue cube")
    assert xyz is not None
    assert np.allclose(xyz, [-0.02, -0.55, 0.6])
    assert stats["from_pin"] is True
    assert vm.n_live == 2


def test_pinned_xyz_from_phrases_does_not_live_query():
    vm = _Voxel({"red cylinder": [0.08, -0.55, 0.6]})
    pin_localize_xyz(vm, "red cylinder", [0.08, -0.55, 0.6], {"yoloe_hit": True})
    vm.hits.clear()

    def _boom(*_a, **_k):
        raise AssertionError("pin lookup must not call localize_text")

    vm.localize_text = _boom
    xyz, used, stats = pinned_xyz_from_phrases(vm, ["red cylinder", "cylinder"])
    assert used == "red cylinder"
    assert xyz is not None
    assert np.allclose(xyz, [0.08, -0.55, 0.6])
    assert stats["from_pin"] is True
    assert vm.n_live == 0


def test_proposal_handle_and_localize_confidence():
    assert is_proposal_handle(VOXEL_HYP_OBS_BASE)
    assert is_proposal_handle(voxel_proposal_id(1))
    assert not is_proposal_handle(1)
    assert not is_proposal_handle("nope")
    assert not is_proposal_handle(CURRENT_VIEW_OBS_ID)
    assert not is_proposal_handle(-2_000_000)
    assert not is_proposal_handle(-1_000_000)
    assert is_current_view_sentinel(CURRENT_VIEW_OBS_ID)
    assert not is_current_view_sentinel(VOXEL_HYP_OBS_BASE)
    assert voxel_proposal_id(0) == VOXEL_HYP_OBS_BASE
    assert voxel_proposal_id(1) == VOXEL_HYP_OBS_BASE - 1
    assert localize_confidence({"yoloe_hit": True, "max_cosine": 0.22}) == 1.0
    assert localize_confidence({"yoloe_hit": False, "max_cosine": 0.31}) == 0.31
    assert localize_confidence(None) is None
