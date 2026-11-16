import numpy as np
import pytest
import torch

from emet.rl.manipulation.calibration import scale_hand_feedback
from emet.rl.manipulation.feedback_policy import FeedbackAction
from emet.rl.manipulation.profiles import PROFILES


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_hand_gain_changes_only_arm_response_to_hand_pose_error(robot):
    p = PROFILES[robot]
    width = 24 + p.arm_size + 6
    head = FeedbackAction(3, width, p.action_size)
    with torch.no_grad():
        head.gain.weight.fill_(0.01)
        head.gain.bias.fill_(0.02)
        head.grip.weight.fill_(0.03)
    signals = torch.zeros((3, 3 + width))
    signals[:, :3] = 0.2
    signals[0, 3] = 0.1  # base pose
    signals[1, 3 + 3] = 0.1  # hand pose
    signals[2, 3 + 24] = 0.1  # measured velocity
    before = head(signals).detach()
    scale_hand_feedback(head, p, 2)
    after = head(signals).detach()
    torch.testing.assert_close(before[[0, 2]], after[[0, 2]])
    torch.testing.assert_close(before[:, -1], after[:, -1])
    offset = 2 if p.wheel_bodies else 0
    if offset:
        torch.testing.assert_close(before[:, :offset], after[:, :offset])
    torch.testing.assert_close(torch.atanh(after[1, offset:-1]), 2 * torch.atanh(before[1, offset:-1]))
    assert torch.isfinite(after).all() and torch.max(abs(after)) <= 1


def test_invalid_gain_rejected_without_modifying_the_actor():
    head = FeedbackAction(3, 35, 8)
    before = {k: v.clone() for k, v in head.state_dict().items()}
    for factor in [0, -1, 4.01, np.nan, True]:
        with pytest.raises(ValueError):
            scale_hand_feedback(head, PROFILES["innate_mars"], factor)
    for k, v in head.state_dict().items():
        torch.testing.assert_close(v, before[k])
