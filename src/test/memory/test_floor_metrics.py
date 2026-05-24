# Copyright (c) Hello Robot, Inc. All rights reserved.

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from emet.memory.floor_metrics import (
    FLOOR_METRICS_JSON,
    compare_explored_floor_metrics,
    compute_explored_floor_metrics,
    format_floor_metrics_summary,
    load_floor_metrics,
    write_floor_metrics_json,
)
from emet.memory.headless_export import export_graph_eqa_dir


def test_compute_explored_floor_metrics_from_bool_grid():
    explored = np.zeros((4, 5), dtype=bool)
    explored[1:3, 2:4] = True

    vm = MagicMock()
    vm.grid_resolution = 0.05
    vm.grid_origin = np.array([10.0, 20.0, 0.0])
    vm.get_2d_map.return_value = (np.zeros_like(explored), explored)

    metrics = compute_explored_floor_metrics(vm, robot="innate_mars")
    assert metrics["explored_cell_count"] == 4
    assert metrics["explored_area_m2"] == pytest.approx(4 * 0.05 * 0.05)
    assert metrics["explored_grid_shape"] == [4, 5]
    assert metrics["grid_origin_xy"] == [10.0, 20.0]


def test_compare_explored_floor_metrics_match():
    left = {"explored_cell_count": 100, "explored_area_m2": 0.25, "robot": "innate_mars"}
    right = {"explored_cell_count": 100, "explored_area_m2": 0.25, "robot": "stretch"}
    out = compare_explored_floor_metrics(left, right)
    assert out["match"] is True
    assert out["cell_delta"] == 0


def test_compare_explored_floor_metrics_area_tolerance():
    left = {"explored_cell_count": 100, "explored_area_m2": 1.0, "robot": "a"}
    right = {"explored_cell_count": 98, "explored_area_m2": 0.97, "robot": "b"}
    out = compare_explored_floor_metrics(left, right, atol_cells=5, rtol_area=0.05)
    assert out["match"] is True


def test_write_and_load_floor_metrics_json(tmp_path):
    metrics = {"explored_cell_count": 12, "explored_area_m2": 0.03, "robot": "galaxea_r1"}
    write_floor_metrics_json(tmp_path, metrics)
    loaded = load_floor_metrics(tmp_path)
    assert loaded["explored_cell_count"] == 12
    assert (tmp_path / FLOOR_METRICS_JSON).is_file()


def test_export_graph_eqa_dir_writes_floor_metrics(tmp_path):
    graph = MagicMock()
    graph.get_nodes.return_value = []
    graph.get_edges.return_value = []
    graph.get_observations.return_value = []

    explored = np.zeros((10, 10), dtype=bool)
    explored[3:7, 3:7] = True
    vm = MagicMock()
    vm.grid_resolution = 0.05
    vm.grid_origin = np.array([50.0, 50.0, 0.0])
    vm.get_2d_map.return_value = (np.zeros_like(explored), explored)
    vm.semantic_memory = None
    vm.observations = []

    text = export_graph_eqa_dir(graph, vm, str(tmp_path), robot="innate_mars")
    assert "Explored floor" in text
    loaded = json.loads((tmp_path / FLOOR_METRICS_JSON).read_text(encoding="utf-8"))
    assert loaded["explored_cell_count"] == 16
    assert format_floor_metrics_summary(loaded).startswith("robot='innate_mars'")
