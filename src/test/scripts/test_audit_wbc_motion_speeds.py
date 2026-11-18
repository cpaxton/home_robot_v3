"""Speed coverage follows physical timestamps, not video frames or sample counts."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "speed_audit", Path(__file__).resolve().parents[3] / "scripts/audit_wbc_motion_speeds.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_speed_bins_weight_physical_time_and_keep_passive_time_visible():
    # 1s moving at the limit, followed by 9s stationary; two equal-count samples
    # must not imply half the physical exposure was at speed.
    result = module.joint_speed_statistics([0, 1, 10], [[0], [1], [1]], [1])
    assert result["peak_joint_speed_rad_s"] == [1]
    assert result["mean_absolute_joint_speed_rad_s"] == [0.1]
    np.testing.assert_allclose(result["speed_bin_joint_time_fraction"], [0.9, 0, 0, 0, 0, 0, 0.1])


@pytest.mark.parametrize("times,limits", [([0, 0], [1]), ([0, np.nan], [1]), ([0, 1], [0])])
def test_invalid_time_or_cap_fails_closed(times, limits):
    with pytest.raises(ValueError):
        module.joint_speed_statistics(times, [[0], [1]], limits)
