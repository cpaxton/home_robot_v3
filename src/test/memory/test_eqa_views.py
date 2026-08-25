# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np
import pytest

from emet.memory.graph_eqa.eqa_views import crop_rgb_tight_bbox, eqa_look_is_spent, spread_obs_ids_xy
from emet.memory.graph_eqa.graph_eqa_siglip import flatten_find_all_images


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


def test_spread_obs_ids_xy_picks_farthest_cluster_second():
    xyz = {
        1: np.array([0.0, 0.0]),
        2: np.array([0.2, 0.0]),
        3: np.array([8.0, 8.0]),
    }
    out = spread_obs_ids_xy([1, 2, 3], xyz.get, max_n=2)
    assert out[0] == 1
    assert out[1] == 3


def test_flatten_find_all_images_restores_score_order():
    import torch

    ids = torch.tensor([2, 5, 1])
    aligns = torch.tensor([0.21, 0.40, 0.30])
    points = torch.tensor([[2.0, 0.0, 0.0], [5.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    ranked = flatten_find_all_images(ids, points, aligns)
    assert [voc for _s, voc, _p in ranked] == [5, 1, 2]
    assert ranked[0][0] == pytest.approx(0.40, abs=1e-5)
    assert flatten_find_all_images(torch.tensor([]), torch.zeros(0, 3), torch.tensor([])) == []
