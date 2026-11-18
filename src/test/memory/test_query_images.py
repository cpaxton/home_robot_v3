# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU tests for agentic VLM query PNG dumps."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from emet.memory.graph_eqa.query_images import dump_query_rgb, query_images_dir


def test_dump_query_rgb_writes_canonical_and_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_AGENTIC_QUERY_IMAGES", "1")
    monkeypatch.setenv("EMET_AGENTIC_QUERY_IMAGES_DIR", str(tmp_path))
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[2:6, 2:6] = (200, 10, 10)
    ex = SimpleNamespace(_round=3, _trace_meta={"ovmm_phase": "find_object"})
    paths = dump_query_rgb(ex, 7, rgb, kind="vlm_assess")
    assert paths["rgb_png"].endswith("rgb_0007.png")
    assert "vlm_assess_find_object_r03_obs0007.png" in paths["query_png"]
    assert (tmp_path / "rgb_0007.png").is_file()
    assert (tmp_path / "vlm_assess_find_object_r03_obs0007.png").is_file()


def test_query_images_dir_off(monkeypatch, tmp_path):
    monkeypatch.setenv("EMET_AGENTIC_QUERY_IMAGES", "0")
    monkeypatch.setenv("EMET_AGENTIC_QUERY_IMAGES_DIR", str(tmp_path))
    assert query_images_dir(SimpleNamespace()) is None


def test_context_dump_preserves_canonical_current_pixels(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_AGENTIC_QUERY_IMAGES_DIR", str(tmp_path))
    monkeypatch.setenv("EMET_AGENTIC_QUERY_IMAGES", "1")
    ex = SimpleNamespace(_round=1)
    current = np.zeros((8, 8, 3), dtype=np.uint8)
    context = np.full_like(current, 77)
    current_paths = dump_query_rgb(ex, 7, current)
    context_paths = dump_query_rgb(ex, 7, context, kind="vlm_assess_image_2", write_canonical=False)
    assert "rgb_png" not in context_paths
    np.testing.assert_array_equal(np.asarray(Image.open(current_paths["rgb_png"])), current)
    np.testing.assert_array_equal(np.asarray(Image.open(context_paths["query_png"])), context)
