# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Dynagraph VLM-context panel — mock-only (no rerun-sdk native import)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from emet.visualization.dynagraph_context import (
    build_dynagraph_context_markdown,
    log_vlm_context_to_visualizer,
    send_graph_memory_rerun_blueprint,
)


def test_build_dynagraph_context_markdown_includes_prompt_and_image_n():
    mug = SimpleNamespace(
        node_id=1,
        labels=["mug"],
        obs_id=7,
        xyz=[0.1, 0.2, 0.3],
        is_viewpoint=False,
        is_frontier=False,
        support_count=2,
    )
    gm = SimpleNamespace(
        get_nodes=lambda: [mug],
        last_eqa_obs_ids=[7, 9],
        last_eqa_prompt_node_count=4,
        last_eqa_prompt_regions=2,
        last_eqa_prompt_text="Question: where is the mug?\nSCENE_GRAPH:\n mug [#1 img 7]",
        last_eqa_look_obs_id=7,
        last_eqa_action_obs_id=9,
        last_router_state_text="Investigate: mug investigated=0",
        last_eqa_spatial_rag={"n_regions": 2, "n_nodes": 4, "radius_m": 1.5, "seed_node_ids": [1]},
        last_agentic_decision=None,
        last_eqa_raw="answer: A",
        last_eqa_parsed=("", "A", True, "", ""),
        last_relevant_images=[],
        last_eqa_nav_fallback_count=0,
    )
    md = build_dynagraph_context_markdown(gm)
    assert "Image 1" in md and "obs 7" in md
    assert "look" in md and "action" in md
    assert "SCENE_GRAPH" in md
    assert "Investigate:" in md
    assert "mug" in md
    assert "Prompt nodes:** 4" in md


def test_build_dynagraph_context_markdown_none():
    md = build_dynagraph_context_markdown(None)
    assert "no graph memory" in md


def test_log_vlm_context_skips_duplicate_mosaic(monkeypatch):
    class _Viz:
        enabled = True

        def __init__(self):
            self.texts: list[tuple[str, str]] = []
            self.images: list[tuple[str, np.ndarray]] = []

        def log_text(self, name, text):
            self.texts.append((name, text))

        def log_custom_2d_image(self, name, img):
            self.images.append((name, img))

        def clear_identity(self, name):
            return None

    mosaic = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "emet.visualization.dynagraph_context._eqa_context_mosaic",
        lambda _entries: mosaic,
    )
    gm = SimpleNamespace(
        get_nodes=lambda: [],
        last_eqa_obs_ids=[3],
        last_eqa_prompt_text="Question: x",
        last_eqa_parsed=("", "", False, "", ""),
        last_relevant_images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        last_eqa_look_obs_id=None,
        last_eqa_action_obs_id=None,
        last_eqa_prompt_node_count=0,
        last_eqa_prompt_regions=0,
        last_eqa_spatial_rag=None,
        last_agentic_decision=None,
        last_eqa_raw="",
        last_router_state_text="",
        last_eqa_nav_fallback_count=0,
    )
    viz = _Viz()
    log_vlm_context_to_visualizer(viz, gm)
    log_vlm_context_to_visualizer(viz, gm)
    assert viz.texts[-1][0] == "world/dynagraph/context"
    assert len(viz.images) == 1
    assert viz.images[0][0] == "world/dynagraph/context/mosaic"


def test_log_vlm_context_skips_when_enabled_is_not_true():
    vis = MagicMock()
    gm = MagicMock()
    log_vlm_context_to_visualizer(vis, gm)
    vis.log_text.assert_not_called()
    vis.log_custom_2d_image.assert_not_called()


def test_send_graph_memory_blueprint_skipped_when_disabled():
    vis = SimpleNamespace(enabled=False)
    send_graph_memory_rerun_blueprint(vis)


def _install_fake_rerun_blueprint(monkeypatch) -> dict:
    captured: dict = {}

    class _View:
        def __init__(self, *args, **kwargs):
            self.origin = kwargs.get("origin")
            self.name = kwargs.get("name")
            self.contents = kwargs.get("contents")

        def __repr__(self):
            return f"View(origin={self.origin!r})"

    class _Layout:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __str__(self):
            return " ".join(str(a) for a in self.args) + " " + str(self.kwargs)

    fake_rrb = ModuleType("rerun.blueprint")
    fake_rrb.Spatial3DView = _View
    fake_rrb.TextDocumentView = _View
    fake_rrb.Spatial2DView = _View
    fake_rrb.Vertical = _Layout
    fake_rrb.Horizontal = _Layout
    fake_rrb.TimePanel = lambda **k: "time"
    fake_rrb.Blueprint = _Layout

    fake_rr = ModuleType("rerun")
    fake_rr.send_blueprint = lambda bp, **k: captured.update(bp=bp)
    fake_rr.blueprint = fake_rrb

    monkeypatch.setitem(sys.modules, "rerun", fake_rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", fake_rrb)
    return captured


def test_send_graph_memory_blueprint_includes_context_origin(monkeypatch):
    captured = _install_fake_rerun_blueprint(monkeypatch)
    vis = SimpleNamespace(enabled=True, collapse_panels=True)
    send_graph_memory_rerun_blueprint(vis)
    assert "bp" in captured
    dumped = str(captured["bp"])
    assert "world/dynagraph/context" in dumped
    assert "world/dynagraph/context/mosaic" in dumped
    assert "origin='world'" in dumped or 'origin="world"' in dumped
