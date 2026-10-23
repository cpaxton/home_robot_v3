import numpy as np
import pytest

from emet.rl.pose_targets import PoseTargets


def test_pose_targets_support_base_constraint_and_variable_masked_hands():
    targets = PoseTargets(
        base=[1.0, -0.5, 0.2],
        base_mask=np.array([False, False, False]),
        hands=np.arange(27, dtype=np.float32).reshape(3, 9),
        hand_mask=np.array([True, False, True]),
    )
    tokens, mask = targets.actor_tokens(max_hands=4)
    assert tokens.shape == (5, 10)
    np.testing.assert_array_equal(mask, [False, True, False, True, False])
    np.testing.assert_array_equal(tokens[2], np.r_[targets.hands[1], 0])
    assert tokens[0, 7:10].tolist() == [0.0, 0.0, 0.0]


def test_pose_targets_reject_empty_or_malformed_contracts():
    with pytest.raises(ValueError, match="at least one target"):
        PoseTargets([0, 0, 0], np.zeros(3, bool), np.zeros((2, 9)), np.zeros(2, bool))
    with pytest.raises(ValueError, match="finite"):
        PoseTargets([0, 0, np.nan], np.ones(3, bool), np.zeros((1, 9)), np.ones(1, bool))
    with pytest.raises(ValueError, match="boolean"):
        PoseTargets([0, 0, 0], np.ones(3), np.zeros((1, 9)), np.ones(1, dtype=bool))
