import numpy as np

from emet.rl.manipulation.mars_folded_region import audit_mars_folded_path


def test_fold_region_allows_inward_exit_and_rejects_return():
    parked, partial, open_arm = np.deg2rad([[90, -69, 90, -17, 0], [90, -66, 82, -17, 0], [90, -55, 65, -17, 0]])
    assert not audit_mars_folded_path([parked, partial])["accepted"]
    assert audit_mars_folded_path([parked, partial, open_arm])["accepted"]
    assert not audit_mars_folded_path([open_arm, parked, open_arm])["accepted"]


def test_fold_region_bounds_between_waypoints_and_tracking_error():
    start, finish = np.deg2rad([[90, -70, 79, 0, 0], [90, -57, 90, 0, 0]])
    result = audit_mars_folded_path([start, finish], uncertainty_rad=0.0)
    assert result["start_upper_bound"] < 0.5
    assert result["end_upper_bound"] < 0.5
    assert result["violating_segments"] == [0]

    path = np.deg2rad([[90, -69, 77, 0, 0], [90, -69, 79, 0, 0]])
    assert audit_mars_folded_path(path, uncertainty_rad=0.0)["accepted"]
    assert not audit_mars_folded_path(path)["accepted"]
