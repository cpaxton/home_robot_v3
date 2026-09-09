# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import Mock

import numpy as np
import pytest

from emet.memory.surface_candidates import candidate_mask, surface_candidate_image, surface_candidates
from emet.memory.vlm_region_grounding import select_supported_region


def scene():
    depth = np.full((60, 80), 2.0)
    depth[15:45, 20:50] = 1.0
    return np.zeros((60, 80, 3), dtype=np.uint8), depth


def test_depth_layers_separate_object_and_support_with_noise_and_holes():
    rgb, depth = scene()
    depth += np.random.default_rng(0).normal(0, 0.005, depth.shape)
    depth[25:28, 30:33] = np.nan
    regions = surface_candidates(depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4)
    assert len(regions) == 2
    masks = [candidate_mask(r, depth.shape) for r in regions]
    assert not np.any(masks[0] & masks[1])
    assert masks[0].sum() == 891
    assert np.all(depth[masks[0]] < 1.1)
    assert not any(mask[26, 31] for mask in masks)
    assert surface_candidate_image(rgb, regions).size == (512, 280)


def test_disconnected_equal_depth_objects_are_not_merged():
    _, depth = scene()
    depth[4:11, 60:70] = 1.0
    regions = surface_candidates(depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4)
    assert len(regions) == 3


def test_touching_coplanar_objects_are_not_falsely_claimed_separable():
    depth = np.ones((40, 40))
    regions = surface_candidates(depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4)
    assert len(regions) == 1  # geometry cannot distinguish these objects
    # An optional class-agnostic segmentation provider can offer distinct masks.
    left = np.zeros_like(depth, dtype=bool)
    left[:, :20] = True
    regions = surface_candidates(depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4, proposal_masks=[left, ~left])
    assert len(regions) == 2


def test_appearance_boundaries_separate_depth_connected_target_and_support():
    rgb, depth = scene()
    depth[:] = 1  # no geometric separation at all
    rgb[:] = [120, 90, 55]
    rgb[15:45, 20:50] = [190, 20, 20]
    regions = surface_candidates(depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4, rgb=rgb)
    assert sorted(r["points"] for r in regions) == [900, 3900]
    target = next(r for r in regions if r["points"] == 900)
    assert np.all(rgb[candidate_mask(target, depth.shape)] == [190, 20, 20])


def test_indistinguishable_touching_surfaces_remain_one_proposal():
    rgb = np.full((40, 40, 3), 120, dtype=np.uint8)
    regions = surface_candidates(np.ones((40, 40)), [0, 0, 1000, 1000], min_depth=0.25, max_depth=4, rgb=rgb)
    assert len(regions) == 1  # semantics/multi-view still needed; no invented split


def test_missing_depth_has_no_geometry_even_with_external_masks():
    depth = np.full((40, 40), np.nan)
    assert (
        surface_candidates(
            depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4, proposal_masks=[np.ones_like(depth, bool)]
        )
        == []
    )


def test_proposal_overflow_fails_instead_of_dropping_objects():
    depth = np.full((60, 60), np.nan)
    for row in range(3):
        for col in range(3):
            depth[row * 20 : row * 20 + 8, col * 20 : col * 20 + 8] = 1
    with pytest.raises(ValueError, match="too many"):
        surface_candidates(depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4)


@pytest.mark.parametrize(
    "reply",
    [
        '{"selected_id":true,"target_unambiguous":true}',
        '{"selected_id":99,"target_unambiguous":true}',
        '{"selected_id":0,"target_unambiguous":false}',
        '{"selected_id":[0,1],"target_unambiguous":true}',
        "not JSON",
    ],
)
def test_ambiguous_or_invalid_selection_never_admits_geometry(reply):
    rgb, depth = scene()
    client = Mock(side_effect=['{"verified":true,"box":[0,0,1000,1000]}', reply])
    _, mask, audit = select_supported_region(
        rgb, depth, "mug", "mug", client=client, min_depth=0.25, max_depth=4, strategy="depth_candidates"
    )
    assert not audit["valid"] and (mask == -1).all()
    assert client.call_count == 2


def test_selected_mask_not_vlm_point_supplies_geometry():
    rgb, depth = scene()
    client = Mock(
        side_effect=[
            '{"verified":true,"box":[0,0,1000,1000],"point":[999,999]}',
            '{"selected_id":1,"target_unambiguous":true}',
        ]
    )
    _, mask, audit = select_supported_region(
        rgb, depth, "mug", "mug", client=client, min_depth=0.25, max_depth=4, strategy="depth_candidates"
    )
    assert audit["valid"]
    assert (mask == 0).sum() == 900
    assert np.all(depth[mask == 0] == 1)
    assert len(client.call_args.args[0]) == 4  # prompt, original, two separate candidate images


def test_absent_target_does_not_trigger_proposal_selection():
    rgb, depth = scene()
    client = Mock(return_value='{"verified":false}')
    assert not select_supported_region(
        rgb, depth, "mug", "mug", client=client, min_depth=0.25, max_depth=4, strategy="depth_candidates"
    )[2]["valid"]
    client.assert_called_once()
