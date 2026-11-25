import numpy as np

from emet.rl.manipulation.reward_config import WholeBodyRewardConfig, mars_folded_pose_cost


def test_folded_cost_prefers_leaving_observed_fold():
    parked = [90, -69, 90, -17, 0]
    recovering = [90, -67, 87.7, -17, 0]
    supported = [90, -66, 82, -17, 0]
    open_arm = [90, -55, 65, -17, 0]
    costs = mars_folded_pose_cost(np.deg2rad([parked, recovering, supported, open_arm]))
    assert costs[0] > costs[1] > costs[2] > costs[3] == 0

    # Time spent near the fold also matters when two paths visit the same poses.
    dwell = np.repeat(costs[0], 20)
    recovery = np.linspace(costs[0], costs[-1], 20)
    assert dwell.sum() > recovery.sum()


def test_folded_weight_keeps_legacy_reward_records_compatible():
    legacy = WholeBodyRewardConfig().record()
    assert "mars_folded_pose_weight" not in legacy
    assert WholeBodyRewardConfig.from_record(legacy).mars_folded_pose_weight == 0
    configured = WholeBodyRewardConfig(mars_folded_pose_weight=0.25)
    assert WholeBodyRewardConfig.from_record(configured.record()) == configured
