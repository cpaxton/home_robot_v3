# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from build_grounding_dataset import object_geom_ids
from grounding_ablation import BoxAblation, expand_box
from score_grounding_dataset import score

from emet.memory.vlm_region_grounding import select_candidate_surface


def test_expansion_clips_and_rejects_malformed_boxes():
    assert expand_box([100, 200, 500, 600], 0.25) == [0, 100, 600, 700]
    for box in (None, [0, 0, 0, 5], [True, 0, 50, 50], [0, 0, 1001, 20]):
        with pytest.raises(ValueError):
            expand_box(box, 0.25)


def test_verifier_rejection_stands_and_does_not_include_original_reason():
    client = Mock(side_effect=['{"verified":true,"box":[100,100,500,500],"reason":"SECRET"}', '{"verified":false}'])
    ablation = BoxAblation(client, np.zeros((80, 60, 3), dtype=np.uint8), "cup", "verify")
    assert '"verified": false' in ablation(["locate"])
    assert "SECRET" not in client.call_args.args[0][0]
    assert len(client.call_args.args[0]) == 4


def test_object_mask_includes_descendants_not_neighbor():
    model = Mock(nbody=5, body_parentid=np.array([0, 0, 1, 2, 0]), geom_bodyid=np.array([1, 2, 3, 4]))
    assert object_geom_ids(model, 1).tolist() == [0, 1, 2]


def test_repair_is_bounded_and_invalid_coordinates_abstain():
    client = Mock(side_effect=['{"verified":true,"box":[100,100,500,500]}', '{"verified":true,"box":[0,0,1001,500]}'])
    ablation = BoxAblation(client, np.zeros((80, 60, 3), dtype=np.uint8), "cup", "repair")
    assert json.loads(ablation(["locate"]))["verified"] is False
    assert client.call_count == 2


def test_scoring_distinguishes_purity_recall_and_false_acceptance(tmp_path):
    gt = np.zeros((10, 10), dtype=bool)
    gt[2:8, 2:8] = True
    np.savez_compressed(tmp_path / "truth.npz", mask=gt)
    truth = [
        {
            "input": {"arrays": "frame.npz", "query": "cup"},
            "mask_file": str(tmp_path / "truth.npz"),
            "scene": "fixture",
            "split": "test",
            "target": "cup",
            "view": {},
            "visible_pixels": 36,
        }
    ]
    (tmp_path / "truth.json").write_text(json.dumps(truth))
    result = [{"input": truth[0]["input"], "audit": {"valid": True}, "selection": {"box": [200, 200, 800, 800]}}]
    (tmp_path / "results.json").write_text(json.dumps(result))
    patch = np.zeros_like(gt)
    patch[3:5, 3:5] = True
    np.savez_compressed(tmp_path / "0-support.npz", mask=patch)
    row = score(tmp_path / "truth.json", tmp_path / "results.json")[0]
    assert row["purity"] == row["box_recall"] == 1
    assert row["surface_recall"] == pytest.approx(4 / 36)
    assert row["pure_surface"] and not row["false_accept"]
    np.savez_compressed(tmp_path / "0-support.npz", mask=~gt)
    assert score(tmp_path / "truth.json", tmp_path / "results.json")[0]["false_accept"]


@pytest.mark.parametrize("accept", [True, False])
def test_external_proposals_need_qwen_acceptance_and_measured_depth(accept):
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    proposed = np.ones((1, 20, 20), dtype=bool)
    client = Mock(return_value=json.dumps({"selected_id": 0, "target_unambiguous": accept}))
    _, support, audit = select_candidate_surface(
        rgb,
        np.ones((20, 20)),
        "cup",
        "cup",
        client=client,
        min_depth=0.25,
        max_depth=4,
        proposal_masks=proposed,
    )
    assert audit["valid"] == accept
    assert np.any(support == 0) == accept
    assert client.call_count == 1  # no Qwen-box dependency
    client.reset_mock()
    _, _, audit = select_candidate_surface(
        rgb,
        np.full((20, 20), np.nan),
        "cup",
        "cup",
        client=client,
        min_depth=0.25,
        max_depth=4,
        proposal_masks=proposed,
    )
    assert not audit["valid"]
    client.assert_not_called()


def test_live_segmenter_uses_pixel_box_and_context_without_clipping_refined_mask():
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    masks = np.zeros((1, 40, 80), dtype=bool)
    masks[0, 5:30, 10:65] = True
    segmenter = Mock()
    segmenter.segment.return_value = masks
    client = Mock(
        side_effect=['{"verified":true,"box":[250,250,500,500]}', '{"selected_id":0,"target_unambiguous":true}']
    )
    _, support, audit = select_candidate_surface(
        rgb,
        np.ones((40, 80)),
        "cup",
        "cup",
        client=client,
        min_depth=0.25,
        max_depth=4,
        segmenter=segmenter,
        presentation="context",
        whole_object=True,
    )
    assert np.array_equal(segmenter.segment.call_args.args[1], [[20, 10, 40, 20]])
    assert np.array_equal(support == 0, masks[0])
    assert audit["proposal_source"] == "box_segmenter"
    assert audit["surface_selection"]["presentation"] == "context"
    assert len(client.call_args.args[0]) == 4  # prompt + original + context + isolated
    assert client.call_args.kwargs["max_new_tokens"] == 512
    assert "ENTIRE visible extent" in client.call_args_list[0].args[0][0]


def test_bad_localization_never_calls_segmenter():
    segmenter = Mock()
    client = Mock(return_value='{"verified":true,"box":[500,0,100,1000]}')
    _, _, audit = select_candidate_surface(
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.ones((20, 20)),
        "cup",
        "cup",
        client=client,
        min_depth=0.25,
        max_depth=4,
        segmenter=segmenter,
    )
    assert not audit["valid"]
    segmenter.segment.assert_not_called()
