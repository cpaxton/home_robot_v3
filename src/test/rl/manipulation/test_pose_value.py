import numpy as np
import pytest
import torch

from emet.rl.manipulation.pose_value import (
    PoseConditionedValue,
    decide_pose_candidates,
    pose_candidate_features,
)


def transform(position=(0, 0, 0), yaw=0):
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = position
    return result


def test_pose_features_are_current_ee_relative_and_mask_padding():
    history = np.arange(12, dtype=np.float32).reshape(3, 4)
    current = transform((1, 2, 0), np.pi / 2)
    reachable = transform((1, 3, 0), np.pi)
    padded = np.full((4, 4), np.nan)
    features = pose_candidate_features(history, current, [reachable, padded], [True, False])

    assert features.shape == (2, history.size + 9)
    assert np.allclose(features[0, : history.size], history.ravel())
    # World +Y is current-frame -X after a +90 degree EE yaw.
    assert np.allclose(features[0, history.size : history.size + 3], [1, 0, 0], atol=1e-6)
    assert np.allclose(features[1], 0)


def test_pose_features_reject_bad_real_poses_and_bad_mask():
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        pose_candidate_features(np.zeros((2, 3)), np.eye(4), [np.diag([2, 1, 1, 1])], [True])
    with pytest.raises(ValueError, match="boolean"):
        pose_candidate_features(np.zeros((2, 3)), np.eye(4), [np.eye(4)], [1])


def test_value_gate_selects_safe_lower_bound_and_abstains_on_uncertainty():
    result = decide_pose_candidates(
        [[0.9, 0.95, np.nan], [0.85, 0.1, 0.9], [0.88, 0.9, 0.9]],
        np.array([True, True, False]),
        minimum_value=0.5,
        maximum_disagreement=0.1,
    )

    assert result.accepted_index == 0
    assert result.decisions == ("accepted", "unknown", "unknown")
    rejected = decide_pose_candidates(
        [[0.2], [0.3]], np.array([True]), minimum_value=0.5, maximum_disagreement=0.1
    )
    assert rejected.accepted_index is None
    assert rejected.decisions == ("rejected_low_value",)


def test_pose_value_is_permutation_equivariant_and_masks_padding():
    torch.manual_seed(1)
    model = PoseConditionedValue(history_features=4, robot_count=2).eval()
    history = torch.randn(1, 5, 4)
    poses = torch.randn(1, 3, 9)
    mask = torch.tensor([[True, True, False]])
    poses[0, 2] = torch.nan
    robot = torch.tensor([0])

    scores = model(history, poses, mask, robot)
    permuted = model(history, poses[:, [1, 0, 2]], mask[:, [1, 0, 2]], robot)
    assert torch.allclose(scores[:, :2], permuted[:, [1, 0]], atol=1e-6)
    assert scores[0, 2] == 0
