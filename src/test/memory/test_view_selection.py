# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views


def frame(i, *, timestamp=None, x=0, yaw=0, **kwargs):
    pose = np.eye(4)
    angle = np.radians(yaw)
    # Rotate optical +Z around world Y.
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    pose[0, 3] = x
    return ViewFrame(
        str(i),
        "episode:map-v1",
        float(i if timestamp is None else timestamp),
        np.full((4, 4, 3), i, dtype=np.uint8),
        camera_pose=pose,
        **kwargs,
    )


def ids(result):
    return [v.frame.view_id for v in result.views]


def test_budget_cut_opposite_basis_does_not_break_retained_packet():
    current = frame(3)
    other = frame(1)
    reflected = other.camera_pose.copy()
    reflected[:, 0] *= -1
    other = replace(other, camera_pose=reflected)
    result = select_views([other, current], current_id="3", config=SelectionConfig(max_candidates=1, max_images=1))
    assert ids(result) == ["3"]
    assert result.excluded["1"] == "candidate_budget"


def test_candidate_budget_does_not_discard_preturn_pin():
    result = select_views(
        [frame(1), frame(2)], current_id="2", pre_turn_id="1", config=SelectionConfig(max_candidates=1, max_images=1)
    )
    assert result.excluded.get("1") != "candidate_budget"
    assert ids(result) == ["2"]


def test_current_and_pre_turn_pinned_with_honest_labels_and_budget():
    result = select_views(
        [frame(1), frame(2), frame(3)], current_id="3", pre_turn_id="1", config=SelectionConfig(max_images=2)
    )
    assert ids(result) == ["3", "1"]
    assert [v.role for v in result.views] == ["current", "pre_turn"]
    assert "2.00s" in result.prompt_labels()
    assert "not proof of current visibility" in result.prompt_labels()
    assert result.excluded == {"2": "image_budget"}
    json.dumps(result.to_dict(), allow_nan=False)
    one = select_views([frame(1), frame(3)], current_id="3", pre_turn_id="1", config=SelectionConfig(max_images=1))
    assert ids(one) == ["3"]
    assert not one.metrics["pre_turn_retained"]


@pytest.mark.parametrize("policy", ["recency", "pose", "embedding", "coverage", "hybrid"])
def test_policy_determinism_and_missing_features(policy):
    frames = [replace(frame(i), camera_pose=None) for i in (1, 2, 3)]
    config = SelectionConfig(policy=policy, max_images=2)
    a = select_views(frames, current_id="3", config=config)
    b = select_views(frames[::-1], current_id="3", config=config)
    assert ids(a) == ids(b) == ["3", "2"]
    assert a.views[1].features["pose"] is None
    assert a.metrics["candidate_coverage_fraction"] is None
    assert a.metrics["mean_view_angle_deg"] is None


def test_pose_selection_values_heading_changes_not_identical_recent_frames():
    result = select_views(
        [frame(1, yaw=90), frame(2, yaw=2), frame(3)],
        current_id="3",
        config=SelectionConfig(policy="pose", max_images=2),
    )
    assert ids(result) == ["3", "1"]
    assert result.metrics["mean_view_angle_deg"] == pytest.approx(90)
    translated = select_views(
        [frame(1, x=2), frame(2), frame(3)], current_id="3", config=SelectionConfig(policy="pose", max_images=2)
    )
    assert ids(translated) == ["3", "1"]
    assert translated.metrics["mean_translation_m"] == 2


def test_embedding_novelty_and_query_relevance_are_separate_ablations():
    frames = [
        frame(1, embedding=np.array([0.0, 1.0]), embedding_space="encoder:v1"),
        frame(2, embedding=np.array([1.0, 0.0]), embedding_space="encoder:v1"),
        frame(3, embedding=np.array([1.0, 0.0]), embedding_space="encoder:v1"),
    ]
    novelty = select_views(frames, current_id="3", config=SelectionConfig(policy="embedding", max_images=2))
    assert ids(novelty) == ["3", "1"]
    assert novelty.metrics["mean_embedding_distance"] == pytest.approx(0.5)
    relevance = select_views(
        frames,
        current_id="3",
        query_embedding=np.array([1.0, 0.0]),
        embedding_space="encoder:v1",
        config=SelectionConfig(policy="hybrid", max_images=2, weights=(0, 0, 0, 0, 1, 0)),
    )
    assert ids(relevance) == ["3", "2"]
    with pytest.raises(ValueError, match="space"):
        select_views(frames, current_id="3", query_embedding=np.ones(2), embedding_space="other")
    with pytest.raises(ValueError, match="incompatible"):
        select_views([frames[0], replace(frames[-1], embedding_space="other")], current_id="3")


def test_coverage_is_marginal_union_not_repeated_cell_count():
    frames = [
        frame(1, visible_cells=frozenset({"a", "b"})),
        frame(2, visible_cells=frozenset({"c"})),
        frame(3, visible_cells=frozenset({"a", "b"})),
    ]
    result = select_views(frames, current_id="3", config=SelectionConfig(policy="coverage", max_images=2))
    assert ids(result) == ["3", "2"]
    assert result.views[1].features["coverage"] == pytest.approx(1 / 3)
    assert result.metrics["candidate_coverage_fraction"] == 1


def test_saliency_only_ablation():
    result = select_views(
        [frame(1, saliency=1), frame(2, saliency=0), frame(3)],
        current_id="3",
        config=SelectionConfig(policy="hybrid", max_images=2, weights=(0, 0, 0, 0, 0, 1)),
    )
    assert ids(result) == ["3", "1"]


def test_scope_age_and_candidate_limits_are_explicit():
    frames = [frame(1, timestamp=-100), replace(frame(2), scope_id="other"), frame(3), frame(4), frame(5)]
    result = select_views(frames, current_id="4", config=SelectionConfig(max_images=1, max_candidates=1))
    assert result.excluded == {"1": "too_old", "2": "different_scope", "5": "future_frame", "3": "candidate_budget"}
    with pytest.raises(ValueError, match="invalid pre-turn"):
        select_views(frames, current_id="4", pre_turn_id="2")
    with pytest.raises(ValueError, match="pre-turn view must be retained"):
        select_views(frames, current_id="4", pre_turn_id="unknown")


def test_exact_duplicate_pixels_cannot_consume_pinned_context_slot():
    current = frame(3)
    duplicate = replace(frame(1), rgb=current.rgb)
    result = select_views([duplicate, frame(2), current], current_id="3", pre_turn_id="1")
    assert ids(result) == ["3", "2"]
    assert result.excluded["1"] == "duplicate_pixels"


def test_snapshot_owns_pixels_pose_and_features():
    rgb = np.full((4, 4, 3), 12, dtype=np.uint8)
    pose = np.eye(4)
    embedding = np.array([1.0, 2.0])
    view = ViewFrame("x", "s", 0, rgb, pose, embedding, "e")
    rgb[:] = 0
    pose[:] = 0
    embedding[:] = 0
    assert view.rgb[0, 0, 0] == 12
    assert view.camera_pose[3, 3] == 1
    assert view.embedding[0] == 1
    with pytest.raises(ValueError):
        view.rgb[0, 0, 0] = 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestamp_s": float("nan")},
        {"camera_pose": np.zeros((4, 4))},
        {"embedding": np.zeros(2), "embedding_space": "e"},
        {"saliency": 2},
        {"rgb": np.zeros((0, 0, 3), dtype=np.uint8)},
    ],
)
def test_invalid_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(frame(1), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_images": 0},
        {"max_candidates": 2},
        {"max_age_s": -1},
        {"policy": "magic"},
        {"weights": (0, 0, 0, 0, 0, 0)},
        {"weights": (1,)},
        {"angle_scale_deg": float("nan")},
    ],
)
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SelectionConfig(**kwargs)


def test_current_identity_must_be_unambiguous():
    with pytest.raises(ValueError, match="current view"):
        select_views([], current_id="1")
    with pytest.raises(ValueError, match="duplicate view IDs"):
        select_views([frame(1), frame(1)], current_id="1")


def test_same_time_ties_are_stable_and_roll_adds_no_view_diversity():
    roll = np.eye(4)
    roll[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    frames = [replace(frame(1, timestamp=2), camera_pose=roll), frame(2), frame(3)]
    result = select_views(frames, current_id="3", config=SelectionConfig(max_images=2))
    assert ids(result) == ["3", "1"]
    assert result.views[1].features["pose"] == pytest.approx(0)


def test_numpy_scalar_metadata_serializes_without_nan():
    result = select_views([replace(frame(1), timestamp_s=np.float32(1), saliency=np.float32(0.5))], current_id="1")
    json.dumps(result.to_dict(), allow_nan=False)


def test_frozen_manifest_comparison_runs_all_policies_without_models(tmp_path):
    manifest = {"current_id": "3", "frames": []}
    for f in [frame(1, yaw=90), frame(2), frame(3)]:
        image_path = f"{f.view_id}.png"
        Image.fromarray(f.rgb).save(tmp_path / image_path)
        manifest["frames"].append(
            {
                "view_id": f.view_id,
                "scope_id": f.scope_id,
                "timestamp_s": f.timestamp_s,
                "image_path": image_path,
                "camera_pose": f.camera_pose.tolist(),
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    script = Path(__file__).resolve().parents[3] / "scripts/compare_view_selection.py"
    compare = runpy.run_path(str(script))["compare_manifest"]
    results = compare(path, max_images=2)
    assert len(results) == 5
    assert results["pose"]["views"][1]["view_id"] == "1"
    assert results["recency"]["views"][1]["view_id"] == "2"
    assert results["embedding"]["metrics"]["feature_support"]["embedding"] == 0
    json.dumps(results, allow_nan=False)
