# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for VLM frontier scoring helpers (controller_graph_eqa)."""

from emet.controller.controller_graph_eqa import _parse_image_pick


def test_parse_image_pick():
    assert _parse_image_pick("2", 4) == 1
    assert _parse_image_pick("Image 3 looks most promising.", 6) == 2
    assert _parse_image_pick("I would pick image 1.", 3) == 0
    assert _parse_image_pick("7", 6) is None  # out of range
    assert _parse_image_pick("0", 6) is None  # 1-based
    assert _parse_image_pick("none of these", 4) is None
    assert _parse_image_pick("", 4) is None
