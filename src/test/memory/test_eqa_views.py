# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa.eqa_views import crop_rgb_tight_bbox, eqa_look_is_spent


def test_eqa_look_spent_after_sub_meter_revisit():
    assert eqa_look_is_spent([3.7]) is False
    assert eqa_look_is_spent([3.7, 0.25]) is True
    assert eqa_look_is_spent([1.2, 1.1, 1.0], nav_attempts=3) is True
    assert eqa_look_is_spent([0.25]) is False


def test_crop_rgb_tight_bbox_drops_full_frame_boxes():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, 5:] = (0, 0, 255)
    crop = crop_rgb_tight_bbox(rgb, (5, 0, 10, 10))
    assert crop is not None
    assert crop.shape[1] < 10
    assert crop_rgb_tight_bbox(rgb, (0, 0, 10, 10)) is None
