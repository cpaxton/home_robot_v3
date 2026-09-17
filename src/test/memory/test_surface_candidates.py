# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import Mock

import numpy as np
import pytest

from emet.memory.surface_candidates import candidate_mask, surface_candidate_image, surface_candidates


def test_candidate_panel_excludes_rgb_outside_its_support():
    from emet.memory.surface_candidates import surface_candidate_panels

    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[:] = [255, 0, 0]
    rgb[15:25, 15:25] = [0, 0, 255]
    mask = np.zeros((40, 40), dtype=bool)
    mask[15:25, 15:25] = True
    regions = surface_candidates(
        np.ones((40, 40)), [0, 0, 1000, 1000], min_depth=0.25, max_depth=4, proposal_masks=[mask]
    )
    panel = np.asarray(surface_candidate_panels(rgb, regions)[0])
    assert np.any(np.all(panel == [0, 0, 255], axis=-1))
    assert not np.any((panel[..., 0] > 0) & (panel[..., 1] == 0) & (panel[..., 2] == 0))


from emet.memory.vlm_region_grounding import select_supported_region


def scene():
    depth = np.full((60, 80), 2.0)
    depth[15:45, 20:50] = 1.0
    return np.zeros((60, 80, 3), dtype=np.uint8), depth


def test_unrelated_fragmented_proposal_reports_budget_failure_before_semantics():
    rgb, depth = scene()
    target = np.zeros_like(depth, dtype=bool)
    target[2:8, 2:8] = True
    clutter = np.zeros_like(target)
    for column in range(8):
        clutter[50:56, column * 10 : column * 10 + 6] = True
    client = Mock(side_effect=AssertionError("overflow must not become semantic absence"))
    _, mask, audit = select_supported_region(
        rgb,
        depth,
        "object",
        "object",
        client=client,
        min_depth=0.0,
        max_depth=4,
        strategy="depth_candidates",
        proposal_masks=[target, clutter],
    )
    assert not audit["valid"] and audit["failure_kind"] == "candidate_overflow"
    assert np.all(mask == -1)
    client.assert_not_called()


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


def test_external_union_mask_cannot_join_depth_separated_objects():
    depth = np.ones((20, 20))
    depth[:, 10:] = 0.6
    proposal = np.ones_like(depth, dtype=bool)
    regions = surface_candidates(depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4, proposal_masks=[proposal])
    assert len(regions) == 2
    masks = [candidate_mask(r, depth.shape) for r in regions]
    assert all(np.ptp(depth[mask]) == 0 for mask in masks)
    assert np.array_equal(masks[0] | masks[1], proposal)


@pytest.mark.parametrize("accepted", [True, False])
def test_tracking_filters_whole_surfaces_before_semantics_but_still_requires_verification(accepted):
    depth = np.ones((20, 20))
    depth[:, 10:] = 0.6
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    client = Mock(return_value='{"selected_id":0,"target_unambiguous":' + str(accepted).lower() + "}")
    _, mask, audit = select_supported_region(
        rgb,
        depth,
        "pear",
        "pear",
        client=client,
        min_depth=0.25,
        max_depth=4,
        strategy="depth_candidates",
        presentation="support_only",
        proposal_masks=[np.ones_like(depth, dtype=bool)],
        candidate_filter=lambda candidate: not candidate[:, 10:].any(),
    )
    assert len(audit["candidate_filter"]["rejected_candidates"]) == 1
    assert len(audit["surface_candidates"]) == 1
    assert len(audit["surface_selection"]["image_order"]) == 1
    client.assert_called_once()
    assert audit["valid"] is accepted
    assert np.all(mask[:, 10:] == -1)
    if accepted:
        assert np.all(mask[:, :10] == 0)


def test_tracking_with_no_associated_surface_does_not_ask_vlm_to_choose():
    client = Mock()
    _, mask, audit = select_supported_region(
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.ones((20, 20)),
        "pear",
        "pear",
        client=client,
        min_depth=0.25,
        max_depth=4,
        strategy="depth_candidates",
        proposal_masks=[np.ones((20, 20), dtype=bool)],
        candidate_filter=lambda candidate: False,
    )
    assert not audit["valid"]
    assert np.all(mask == -1)
    client.assert_not_called()
    assert len(audit["candidate_filter"]["rejected_candidates"]) == 1


def test_external_mask_keeps_smooth_sloped_geometry_and_ignores_color_texture():
    depth = np.tile(np.linspace(0.5, 1.0, 20), (20, 1))
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[:, 10:] = 255
    regions = surface_candidates(
        depth, [0, 0, 1000, 1000], min_depth=0.25, max_depth=4, rgb=rgb, proposal_masks=[np.ones_like(depth, bool)]
    )
    assert len(regions) == 1
    assert regions[0]["points"] == 400


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
