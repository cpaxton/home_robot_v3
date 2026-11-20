import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.workspace_coverage import (
    balanced_goal_weights,
    goal_coverage,
    nearest_orientation_gaps,
    path_coverage,
    spatial_path,
)


def test_equal_per_cell_weights_include_last_edge_and_exclude_outside():
    edges = [[0, 1, 2]]
    points = [[0], [0.25], [0.75], [2], [2.1]]
    report = goal_coverage(points, edges)
    assert report["occupied_bins"] == 2
    assert report["outside_grid"] == 1
    assert report["distance_to_uniform"] == pytest.approx(0.25)
    np.testing.assert_allclose(balanced_goal_weights(points, edges), [1 / 6, 1 / 6, 1 / 6, 0.5, 0])
    subset = goal_coverage(points, edges, eligible=[1])
    assert subset["outside_eligible_workspace"] == 3
    assert subset["distance_to_uniform"] == 0


def test_path_votes_independent_of_dwell_and_endpoint_overlap():
    edges = [[0, 1, 2]]
    report = path_coverage([[[0.5], [1.5], [0.5]], [[0.5]] * 40], edges)
    assert report["occupied_bins"] == 2
    assert [row["mass"] for row in report["bins"]] == pytest.approx([1.5, 0.5])
    assert goal_coverage([[0.5], [0.5]], edges)["occupied_bins"] == 1


def test_base_start_and_body_hand_frames_remove_start_yaw_and_translation():
    base = [[4, 5, np.pi / 2], [4, 5.2, np.pi / 2]]
    # The hand remains 0.3 m straight ahead of the base, at height 0.4 m.
    rot = Rotation.from_euler("z", np.pi / 2).as_matrix()[:, :2].ravel()
    hand = [[4, 5.3, 0.4, *rot], [4, 5.5, 0.4, *rot]]
    path, _ = spatial_path(hand, base, "base_start")
    np.testing.assert_allclose(path, [[0, 0], [0.2, 0]], atol=1e-8)
    path, rotation = spatial_path(hand, base, "body_hand")
    np.testing.assert_allclose(path, [[0.3, 0, 0.4]] * 2, atol=1e-8)
    np.testing.assert_allclose(rotation.magnitude(), 0, atol=1e-8)
    path, _ = spatial_path(hand, base, "start_base_hand")
    np.testing.assert_allclose(path, [[0.3, 0, 0.4], [0.5, 0, 0.4]], atol=1e-8)
    path, _ = spatial_path(hand, base, "initial_hand")
    np.testing.assert_allclose(path, [[0, 0, 0], [0.2, 0, 0]], atol=1e-8)


def test_orientation_gaps_handle_quaternion_sign_and_identical_poses():
    same = Rotation.from_quat([[0, 0, 0, 1], [0, 0, 0, -1]])
    np.testing.assert_array_equal(nearest_orientation_gaps(same), [0, 0])
    separated = Rotation.from_euler("z", [0, np.pi / 3])
    np.testing.assert_allclose(nearest_orientation_gaps(separated), np.pi / 3)
