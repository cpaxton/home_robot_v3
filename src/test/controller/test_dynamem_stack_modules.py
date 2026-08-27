# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU checks that DynaMem / Dynagraph / LazyGraph share the split modules."""

from __future__ import annotations

from emet.app.graph_nav_cli import configure_graph_nav, main
from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.controller_dynamem import DynamemController
from emet.controller.controller_lazy_graph import LazyGraphController
from emet.controller.dynamem.controller import DynamemController as Packed
from emet.mapping.voxel.dynamem_eqa import DynamemVoxelEQAMixin
from emet.mapping.voxel.dynamem_localize import DynamemVoxelLocalizeMixin
from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


def test_dynamem_controller_is_concrete_after_bind():
    """``__new__`` tests skip ``__init__``; ABC must be cleared after bind."""
    assert not DynamemController.__abstractmethods__
    agent = DynamemController.__new__(DynamemController)
    agent.voxel_map = object()
    assert agent.get_voxel_map() is agent.voxel_map


def test_dynamem_controller_facade_is_shim():
    assert DynamemController is Packed
    for name in (
        "update",
        "execute_action",
        "process_text",
        "create_obstacle_map",
        "look_around",
        "run_eqa",
        "navigate_to_target_pose",
        "describe_head_camera_scene_text",
        "_head_to_sweep",
        "_normalize_scene_rgb_u8",
    ):
        assert hasattr(DynamemController, name), name


def test_graph_stack_inheritance():
    assert issubclass(DynagraphController, DynamemController)
    assert issubclass(LazyGraphController, DynagraphController)


def test_voxel_map_uses_localize_and_eqa_mixins():
    assert issubclass(SparseVoxelMap, DynamemVoxelLocalizeMixin)
    assert issubclass(SparseVoxelMap, DynamemVoxelEQAMixin)
    assert hasattr(SparseVoxelMap, "localize_text")
    assert hasattr(SparseVoxelMap, "query_answer")
    assert hasattr(SparseVoxelMap, "find_alignment_over_model")


def test_graph_nav_cli_configure_swaps_controller():
    configure_graph_nav(LazyGraphController, product="LazyGraph")
    try:
        from emet.app.graph_nav_cli import _controller_cls, _product

        assert _controller_cls() is LazyGraphController
        assert _product() == "LazyGraph"
        assert main is not None
    finally:
        configure_graph_nav(DynagraphController, product="Dynagraph")
