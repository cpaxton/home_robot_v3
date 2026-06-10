# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for OVMM find-phase GT oracles (no sim required)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from emet.eval.ovmm_find_phase import (
    resolve_find_phase_nav_step_timeout,
    FindPhaseRunConfig,
    _query_variants,
    category_matches,
    compute_find_phase_metrics,
    create_find_phase_agent,
    distance_to_placement_xy,
    horizontal_coords,
    localization_pred_fields,
    pick_find_object_gt_body,
    pred_xyz_to_json_list,
    query_find_phase_localization,
    score_find_object,
    score_find_recep,
)


def _default_table_placements():
    return {
        "table": {"cat": "table", "pos": [0.0, -0.5, 0.5]},
        "object1": {"cat": "blue cube", "pos": [-0.1, -0.55, 0.6]},
        "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6]},
    }


def test_category_matches_substring():
    assert category_matches("red cylinder", "red cylinder")
    assert category_matches("red", "red cylinder")
    assert category_matches("cabinet", "kitchen cabinet door")
    assert not category_matches("sofa", "table")


def test_pick_find_object_prefers_start_recep():
    placements = {
        "apple_a": {"cat": "apple", "pos": [1.0, 0.0, 0.5]},
        "apple_b": {"cat": "apple", "pos": [0.05, -0.5, 0.6]},
        "counter": {"cat": "counter", "pos": [0.0, -0.5, 0.9]},
    }
    body = pick_find_object_gt_body(placements, "apple", "counter")
    assert body == "apple_b"


def test_score_find_object_success_within_radius():
    placements = _default_table_placements()
    pred = np.array([0.08, -0.55, 0.6])
    out = score_find_object(
        pred,
        placements,
        "red cylinder",
        "table",
        radius_m=0.75,
    )
    assert out["find_object_success"] is True
    assert out["localization_err_obj_m"] == 0.0
    assert out["gt_object_body"] == "object2"


def test_score_find_object_failure_far():
    placements = _default_table_placements()
    pred = np.array([2.0, 2.0, 0.0])
    out = score_find_object(
        pred,
        placements,
        "red cylinder",
        "table",
        radius_m=0.75,
    )
    assert out["find_object_success"] is False
    assert out["localization_err_obj_m"] > 0.75


def test_score_find_recep_any_matching_body():
    placements = {
        "cab_a": {"cat": "cabinet", "pos": [0.0, 1.0, 0.5]},
        "cab_b": {"cat": "upper cabinet", "pos": [0.5, 1.0, 1.2]},
    }
    pred = np.array([0.1, 1.05, 0.0])
    out = score_find_recep(pred, placements, "cabinet", radius_m=0.75)
    assert out["find_recep_success"] is True
    assert out["localization_err_recep_m"] < 0.2


def test_pick_find_object_respects_gt_body():
    placements = _default_table_placements()
    body = pick_find_object_gt_body(
        placements,
        "wrong query",
        "table",
        object_gt_body="object2",
    )
    assert body == "object2"


def test_query_variants_includes_gt_cat():
    variants = _query_variants("cab", {"cab_main": {"cat": "cab"}})
    assert "cab" in variants


def test_distance_to_bounds_habitat_xz():
    placement = {
        "cat": "table",
        "pos": [1.0, 0.5, 2.0],
        "bounds": [[0.5, 0.0, 1.5], [1.5, 1.0, 2.5]],
        "frame": "habitat_yup",
    }
    err = distance_to_placement_xy([1.0, 0.5, 2.0], placement, frame="habitat_xz")
    assert err == 0.0
    err2 = distance_to_placement_xy([2.0, 0.5, 2.0], placement, frame="habitat_xz")
    assert abs(err2 - 0.5) < 1e-6
    assert horizontal_coords([1.0, 0.5, 2.0], frame="habitat_xz").tolist() == [1.0, 2.0]


def test_pred_xyz_to_json_list():
    assert pred_xyz_to_json_list(None) is None
    assert pred_xyz_to_json_list([0.1, -0.5]) == [0.1, -0.5, 0.0]
    assert pred_xyz_to_json_list(np.array([0.08, -0.55, 0.6])) == [0.08, -0.55, 0.6]
    fields = localization_pred_fields([0.08, -0.55, 0.6], None)
    assert fields["pred_obj_xyz"] == [0.08, -0.55, 0.6]
    assert fields["pred_recep_xyz"] is None


@dataclass
class _FakeNode:
    labels: list[str]
    xyz: list[float]


class _FakeGraph:
    def __init__(self, nodes: list[_FakeNode]):
        self._nodes = nodes

    def get_nodes(self):
        return self._nodes


class _FakeVoxel:
    def __init__(self, xyz: list[float] | None):
        self._xyz = xyz

    def localize_text(self, text, debug=False, return_debug=False):
        if return_debug:
            return self._xyz, ""
        return self._xyz


class _FakeMemory:
    def __init__(self, graph: _FakeGraph | None = None):
        self._graph = graph

    def localize_text(self, text):
        from emet.memory.adapters import LocalizeResult

        return LocalizeResult(point_xyz=None, success=False, extra_info={})

    def check_memory_for_object(self, text):
        from emet.memory.adapters import CheckMemoryResult

        return CheckMemoryResult(confidence=0.0, location_xyz=None, extra_info={})

    def list_objects(self):
        return []


def test_query_find_phase_localization_voxel_source():
    memory = _FakeMemory()
    voxel = _FakeVoxel([0.1, -0.5, 0.6])
    xyz, ok, q_used, source = query_find_phase_localization(
        memory,
        "red cylinder",
        voxel_map=voxel,
        prefer_voxel=True,
    )
    assert ok is True
    assert source == "voxel"
    assert xyz is not None
    assert q_used == "red cylinder"


def test_query_find_phase_localization_graph_near_recep_source():
    placements = {
        "table": {"cat": "table", "pos": [0.0, -1.0, 0.24]},
        "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6]},
    }
    graph = _FakeGraph([_FakeNode(labels=["red cylinder"], xyz=[0.08, -0.55, 0.6])])
    memory = _FakeMemory(graph=graph)
    voxel = _FakeVoxel(None)
    xyz, ok, _, source = query_find_phase_localization(
        memory,
        "red cylinder",
        placements=placements,
        near_recep="table",
        voxel_map=voxel,
        prefer_voxel=True,
    )
    assert ok is True
    assert source == "graph_near_recep"
    assert xyz is not None


def test_query_find_phase_localization_miss_source_none():
    memory = _FakeMemory()
    voxel = _FakeVoxel(None)
    xyz, ok, _, source = query_find_phase_localization(
        memory,
        "red cylinder",
        voxel_map=voxel,
        prefer_voxel=True,
    )
    assert ok is False
    assert xyz is None
    assert source is None


def test_compute_find_phase_partial_success():
    placements = _default_table_placements()
    metrics = compute_find_phase_metrics(
        obj_pred_xyz=[0.08, -0.55, 0.6],
        recep_pred_xyz=[0.0, -0.5, 0.5],
        placements=placements,
        object_query="red cylinder",
        start_recep="table",
        goal_recep="table",
        radius_m=0.75,
    )
    assert metrics["find_object_success"] is True
    assert metrics["find_recep_success"] is True
    assert metrics["find_partial_success"] == 1.0


def test_find_phase_run_config_fair_defaults():
    cfg = FindPhaseRunConfig()
    assert cfg.use_sensor_perception is False
    assert cfg.prefer_voxel is True


def test_resolve_find_phase_nav_step_timeout():
    assert resolve_find_phase_nav_step_timeout(cpu_only=False, sim_kind="") == 15.0
    assert resolve_find_phase_nav_step_timeout(cpu_only=True, sim_kind="") == 45.0
    assert resolve_find_phase_nav_step_timeout(cpu_only=False, sim_kind="robocasa") == 30.0
    assert resolve_find_phase_nav_step_timeout(cpu_only=False, sim_kind="molmospaces") == 30.0
    assert resolve_find_phase_nav_step_timeout(cpu_only=False, sim_kind="", override=99.0) == 99.0


@patch("emet.controller.controller_dynagraph.DynagraphController")
def test_create_find_phase_agent_dynagraph_disables_sensor_perception_by_default(mock_cls):
    mock_agent = MagicMock()
    mock_cls.return_value = mock_agent
    robot = MagicMock()
    create_find_phase_agent(robot, {}, "dynagraph")
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["use_sensor_perception"] is False
    mock_agent.start.assert_called_once()


@patch("emet.controller.controller_graph_eqa.GraphEQAController")
def test_create_find_phase_agent_graph_eqa_disables_sensor_perception_by_default(mock_cls):
    mock_agent = MagicMock()
    mock_cls.return_value = mock_agent
    robot = MagicMock()
    create_find_phase_agent(robot, {}, "graph_eqa")
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["use_sensor_perception"] is False


@patch("emet.eval.ovmm_find_phase.collect_scaling_diagnostics")
@patch("emet.eval.ovmm_find_phase.compute_find_phase_metrics")
@patch("emet.eval.ovmm_find_phase.query_find_phase_localization")
@patch("emet.eval.ovmm_find_phase.run_mapping_protocol")
@patch("emet.eval.ovmm_find_phase.create_find_phase_agent")
@patch("emet.app.robot_cli.create_robot_client_from_cli")
@patch("subprocess.Popen")
@patch("emet.config.sim_launch_config.load_sim_launch_config_from_path")
def test_run_episode_find_phase_includes_timing_fields(
    mock_load_sim,
    mock_popen,
    mock_robot_client,
    mock_create_agent,
    mock_mapping,
    mock_query,
    mock_compute_metrics,
    mock_scaling,
):
    from emet.config.sim_launch_config import SimLaunchDefaultMujoco
    from emet.eval.ovmm_find_phase import FindPhaseEpisode, run_episode_find_phase

    mock_load_sim.return_value = SimLaunchDefaultMujoco(robot="stretch", port_offset=0)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mock_robot = MagicMock()
    mock_robot.get_emet_session.return_value = {}
    mock_robot_client.return_value = mock_robot

    mock_agent = MagicMock()
    mock_agent.graph_memory = None
    mock_agent.voxel_map = None
    mock_create_agent.return_value = mock_agent
    mock_mapping.return_value = 3

    obj_xyz = np.array([0.08, -0.55, 0.6])
    recep_xyz = np.array([0.0, -0.5, 0.5])
    mock_query.side_effect = [
        (obj_xyz, True, "red cylinder", "voxel"),
        (recep_xyz, True, "table", "voxel"),
    ]
    mock_compute_metrics.return_value = {
        "find_object_success": True,
        "find_recep_success": True,
        "find_partial_success": 1.0,
        "localization_err_obj_m": 0.0,
        "localization_err_recep_m": 0.0,
    }
    mock_scaling.return_value = {"episode_wall_s": 12.5, "n_controller_steps": 3}

    episode = FindPhaseEpisode(
        id="test_ep",
        tier="S0",
        sim="configs/sim/default_table_stretch.yaml",
        object="red cylinder",
        start_recep="table",
        goal_recep="table",
        explore_steps=3,
        success_radius_m=0.75,
    )
    run_cfg = FindPhaseRunConfig(backend="dynamem", seed=7)

    with patch(
        "emet.memory.graph_eqa.sim_ground_truth_graph.read_sim_object_placements",
        return_value={},
    ):
        with patch("socket.create_connection"):
            with patch("time.sleep"):
                with patch("emet.core.parameters.get_parameters", return_value={}):
                    with patch(
                        "emet.simulation.mujoco_serve_argv.prepare_mujoco_server_argv",
                        return_value=[],
                    ):
                        with patch(
                            "emet.eval.ovmm_find_phase.get_memory_backend_for_agent",
                            return_value=MagicMock(),
                        ):
                            result = run_episode_find_phase(episode, run_cfg)

    for key in ("init_wall_s", "mapping_wall_s", "query_wall_s", "episode_wall_s"):
        assert key in result
        assert isinstance(result[key], float)
    assert result["use_sensor_perception"] is False
    assert result["prefer_voxel"] is True
    assert result["seed"] == 7
