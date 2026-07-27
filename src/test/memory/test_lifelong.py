# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for lifelong checkpoint load and local start-pose refine."""

from __future__ import annotations

import numpy as np

from emet.memory.adapters import GraphEQABackend
from emet.memory.format import VOXEL_PICKLE_FILENAME
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.lifelong import (
    apply_se2_to_graph,
    apply_se2_to_memory,
    load_lifelong_checkpoint,
    refine_start_pose,
    save_lifelong_checkpoint,
    se2_matrix,
    transform_points_xyz,
)

PARAMS = {
    "dynagraph_merge_xy_m": 0.0,
    "dynagraph_staleness_horizon": 8,
}


def _rgb() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _make_memory() -> GraphEQAMemory:
    return GraphEQAMemory(parameters=dict(PARAMS), defer_llm_clients=True)


def test_se2_apply_moves_graph_nodes():
    mem = _make_memory()
    mem.set_graph_timestep(5)
    mem.add_observation(_rgb(), np.array([1.0, 0.0, 0.5]), ["mug"])
    t = se2_matrix(0.3, -0.1, np.deg2rad(15.0))
    n = apply_se2_to_graph(mem, t)
    assert n >= 1
    nodes = [x for x in mem.get_nodes() if not x.is_viewpoint]
    assert len(nodes) == 1
    expected = transform_points_xyz(np.array([[1.0, 0.0, 0.5]]), t)[0]
    np.testing.assert_allclose(nodes[0].xyz, expected, atol=1e-6)


def test_refine_recovers_small_fudge():
    # Structured cloud (two walls) — closer to a real map than isotropic noise.
    xs = np.linspace(-1.0, 1.0, 80)
    ys = np.linspace(-1.0, 1.0, 80)
    wall_x = np.column_stack([np.full_like(ys, -1.0), ys, np.zeros_like(ys)])
    wall_y = np.column_stack([xs, np.full_like(xs, 1.0), np.zeros_like(xs)])
    floor = np.column_stack([np.repeat(xs[::4], 5), np.tile(ys[::16], xs[::4].size), np.zeros(xs[::4].size * 5)])
    saved = np.concatenate([wall_x, wall_y, floor], axis=0)
    dx, dy, dyaw = 0.25, -0.12, np.deg2rad(12.0)
    t_gt = se2_matrix(dx, dy, dyaw)
    live = transform_points_xyz(saved, t_gt)

    result = refine_start_pose(
        saved,
        live,
        min_points=64,
        max_xy_m=0.75,
        max_yaw_rad=0.5,
        grid_xy_step_m=0.1,
        grid_yaw_step_rad=0.1,
    )
    assert result.accepted, result.reason
    assert result.translation_xy_m < 0.45
    assert abs(result.yaw_rad) < 0.4
    aligned = transform_points_xyz(saved, result.transform)
    err = float(np.mean(np.linalg.norm(aligned - live, axis=1)))
    assert err < 0.05, f"mean point error {err:.4f} m"


def test_refine_rejects_large_translation():
    xs = np.linspace(-1.0, 1.0, 60)
    ys = np.linspace(-1.0, 1.0, 60)
    saved = np.concatenate(
        [
            np.column_stack([np.full_like(ys, -1.0), ys, np.zeros_like(ys)]),
            np.column_stack([xs, np.full_like(xs, 1.0), np.zeros_like(xs)]),
        ],
        axis=0,
    )
    live = transform_points_xyz(saved, se2_matrix(2.5, 0.0, 0.0))
    result = refine_start_pose(
        saved,
        live,
        min_points=64,
        max_xy_m=0.75,
        max_yaw_rad=0.5,
        min_fitness=0.4,
        max_rmse_m=0.25,
    )
    assert not result.accepted, result
    np.testing.assert_allclose(result.transform, np.eye(4), atol=1e-8)


def test_load_lifelong_checkpoint_restores_final_step(tmp_path):
    mem = _make_memory()
    mem.set_graph_timestep(40)
    mem.add_observation(_rgb(), np.array([1.0, 2.0, 0.5]), ["mug"])
    path = tmp_path / "ckpt"
    GraphEQABackend(mem).save(str(path), final_step=42)

    class _Ctrl:
        def __init__(self):
            self.graph_memory = _make_memory()
            self.obs_count = 0
            self.voxel_map = None

        def get_voxel_map(self):
            return None

    ctrl = _Ctrl()
    info = load_lifelong_checkpoint(ctrl, path, refine_start=False)
    assert info["graph_loaded"] is True
    assert info["graph_nodes"] >= 1
    assert info["final_step"] == 42
    assert info["verify"]["ok"] is True
    assert ctrl.obs_count == 42
    assert ctrl.graph_memory._graph_timestep == 42
    nodes = [n for n in ctrl.graph_memory.get_nodes() if not n.is_viewpoint]
    assert len(nodes) == 1
    assert nodes[0].labels == ["mug"]


def test_load_restores_frontier_flag_from_label_heuristic(tmp_path):
    """Old exports omit is_frontier; label 'frontier' must still restore as frontier."""
    import json

    mem = _make_memory()
    mem.add_observation(_rgb(), np.array([0.0, 0.0, 0.0]), ["mug"])
    path = tmp_path / "ckpt"
    GraphEQABackend(mem).save(str(path), final_step=3)
    graph_path = path / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    data["nodes"].append(
        {
            "node_id": 99,
            "labels": ["frontier"],
            "xyz": [1.0, 1.0, 0.0],
            "obs_id": 1,
            "last_seen": 3,
            "support_count": 1,
        }
    )
    graph_path.write_text(json.dumps(data), encoding="utf-8")

    class _Ctrl:
        def __init__(self):
            self.graph_memory = _make_memory()
            self.obs_count = 0
            self.voxel_map = None

        def get_voxel_map(self):
            return None

    ctrl = _Ctrl()
    info = load_lifelong_checkpoint(ctrl, path, refine_start=False)
    assert info["verify"]["ok"] is True
    frontiers = [n for n in ctrl.graph_memory.get_nodes() if n.is_frontier]
    assert len(frontiers) >= 1
    assert frontiers[0].labels[0] == "frontier"


def test_verify_lifelong_restore_raises_on_empty_graph(tmp_path):
    import json

    import pytest

    from emet.memory.lifelong import verify_lifelong_restore

    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "manifest.json").write_text(
        '{"has_graph": true, "has_voxel_pickle": false, "version": 1, "backend": "graph_eqa"}',
        encoding="utf-8",
    )
    (path / "graph.json").write_text(
        json.dumps(
            {
                "nodes": [{"node_id": 1, "labels": ["mug"], "xyz": [0, 0, 0], "obs_id": 1}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    class _Ctrl:
        graph_memory = type("G", (), {"get_nodes": lambda self: []})()
        voxel_map = None

        def get_voxel_map(self):
            return None

    with pytest.raises(RuntimeError, match="graph nodes"):
        verify_lifelong_restore(
            _Ctrl(),
            path,
            {
                "graph_loaded": True,
                "voxel_pickle_loaded": False,
                "voxel_points": 0,
                "semantic_points": 0,
            },
            strict=True,
        )


def test_save_lifelong_writes_voxel_flag_when_requested(tmp_path):
    mem = _make_memory()
    mem.add_observation(_rgb(), np.array([0.5, 0.5, 0.2]), ["bowl"])

    class _FakeVoxel:
        def write_to_pickle(self, filename: str) -> None:
            from pathlib import Path

            Path(filename).write_bytes(b"fake-pkl")

    class _Ctrl:
        def __init__(self):
            self.graph_memory = mem
            self.obs_count = 7
            self.voxel_map = _FakeVoxel()

        def get_voxel_map(self):
            return self.voxel_map

    out = tmp_path / "out"
    save_lifelong_checkpoint(_Ctrl(), out, save_voxel_pickle=True)
    assert (out / "manifest.json").is_file()
    assert (out / "graph.json").is_file()
    assert (out / VOXEL_PICKLE_FILENAME).is_file()


def test_apply_se2_to_memory_updates_bounds():
    mem = _make_memory()
    mem.add_observation(
        _rgb(),
        np.array([1.0, 1.0, 0.5]),
        ["box"],
        extent_half=np.array([0.1, 0.1, 0.1]),
    )
    from dataclasses import replace

    for i, n in enumerate(mem._nodes):
        if n.labels and n.labels[0] == "box":
            mem._nodes[i] = replace(
                n,
                bounds_3d={
                    "min": [0.9, 0.9, 0.4],
                    "max": [1.1, 1.1, 0.6],
                    "center": [1.0, 1.0, 0.5],
                    "size": [0.2, 0.2, 0.2],
                },
            )
    t = se2_matrix(0.5, 0.0, 0.0)
    apply_se2_to_memory(graph_memory=mem, voxel_map=None, transform=t)
    box = next(n for n in mem.get_nodes() if n.labels and n.labels[0] == "box")
    np.testing.assert_allclose(box.xyz[:2], [1.5, 1.0], atol=1e-6)
    assert box.bounds_3d is not None
    np.testing.assert_allclose(box.bounds_3d["min"][:2], [1.4, 0.9], atol=1e-6)


def test_apply_se2_transforms_belief_history():
    """Main added position_history / change_events; refine must move those xyz too."""
    from dataclasses import replace

    mem = _make_memory()
    mem.add_observation(_rgb(), np.array([1.0, 0.0, 0.5]), ["mug"])
    for i, n in enumerate(mem._nodes):
        if n.labels and n.labels[0] == "mug":
            mem._nodes[i] = replace(
                n,
                position_history=[{"step": 1, "xyz": [1.0, 0.0, 0.5], "confidence": 0.6}],
                change_events=[
                    {
                        "type": "position_contradiction",
                        "from_xyz": [0.8, 0.0, 0.5],
                        "to_xyz": [1.0, 0.0, 0.5],
                    }
                ],
                position_covariance=np.diag([0.04, 0.01, 0.0]),
            )
            break
    mem._change_events = [
        {"type": "expected_object_missing", "last_xyz": [1.0, 0.0, 0.5]},
    ]
    t = se2_matrix(0.25, -0.1, 0.0)
    apply_se2_to_graph(mem, t)
    mug = next(n for n in mem.get_nodes() if n.labels and n.labels[0] == "mug")
    np.testing.assert_allclose(mug.position_history[0]["xyz"][:2], [1.25, -0.1], atol=1e-6)
    np.testing.assert_allclose(mug.change_events[0]["from_xyz"][:2], [1.05, -0.1], atol=1e-6)
    np.testing.assert_allclose(mug.change_events[0]["to_xyz"][:2], [1.25, -0.1], atol=1e-6)
    # Pure translation: covariance unchanged.
    np.testing.assert_allclose(mug.position_covariance, np.diag([0.04, 0.01, 0.0]), atol=1e-8)
    np.testing.assert_allclose(mem._change_events[0]["last_xyz"][:2], [1.25, -0.1], atol=1e-6)


def test_apply_se2_to_namedtuple_voxel_frames():
    """SparseVoxelMap stores immutable Frame namedtuples — refine must use _replace."""
    from collections import namedtuple

    from emet.memory.lifelong import apply_se2_to_voxel_map

    Frame = namedtuple(
        "Frame",
        ["camera_pose", "base_pose", "xyz", "full_world_xyz"],
    )
    cam = np.eye(4, dtype=np.float64)
    cam[:3, 3] = [1.0, 2.0, 0.5]
    fr = Frame(
        camera_pose=cam,
        base_pose=np.array([1.0, 2.0, 0.1], dtype=np.float64),
        xyz=np.array([[1.0, 2.0, 0.5]], dtype=np.float64),
        full_world_xyz=np.array([[1.1, 2.1, 0.5]], dtype=np.float64),
    )

    class _VM:
        def __init__(self):
            self.observations = [fr]
            self.semantic_memory = None
            self.voxel_pcd = None

    vm = _VM()
    t = se2_matrix(0.5, 0.0, 0.0)
    assert apply_se2_to_voxel_map(vm, t) is True
    out = vm.observations[0]
    np.testing.assert_allclose(out.camera_pose[:2, 3], [1.5, 2.0], atol=1e-6)
    np.testing.assert_allclose(out.base_pose[:2], [1.5, 2.0], atol=1e-6)
    np.testing.assert_allclose(out.xyz[0, :2], [1.5, 2.0], atol=1e-6)


def test_refresh_rerun_after_memory_load_calls_force():
    from emet.memory.lifelong import refresh_rerun_after_memory_load

    calls: list[str] = []

    class _Viz:
        enabled = True

        def update_voxel_map(self, *, space, robot_base_xy=None, force=False):
            calls.append(f"voxel force={force}")

        def log_custom_pointcloud(self, *args, **kwargs):
            calls.append("semantic")

        def log_dynagraph_state(self, gm, *, ground_truth_mode=False, force=False):
            calls.append(f"graph force={force}")

    class _Pts:
        def __init__(self, n):
            self._points = np.zeros((n, 3))
            self._rgb = np.zeros((n, 3))

        @property
        def shape(self):
            return self._points.shape

    class _SM(_Pts):
        pass

    class _VM:
        def __init__(self):
            self.voxel_pcd = _Pts(10)
            self.semantic_memory = _SM(12)

    class _Ctrl:
        def __init__(self):
            self.rerun_visualizer = _Viz()
            self.space = object()
            self.voxel_map = _VM()
            self.graph_memory = object()
            self.robot = None

        def get_voxel_map(self):
            return self.voxel_map

    out = refresh_rerun_after_memory_load(_Ctrl())
    assert out["ok"] is True
    assert out["voxel"] is True
    assert out["graph"] is True
    assert "voxel force=True" in calls
    assert "graph force=True" in calls


def test_lifelong_saves_and_loads_open_vocab_scene_graph(tmp_path):
    from emet.mapping.scene_graph.open_vocab_scene_graph import OpenVocabSceneGraph, SceneGraphNode
    from emet.memory.format import OPEN_VOCAB_SCENE_GRAPH_DIR
    from emet.memory.lifelong import load_lifelong_checkpoint, save_lifelong_checkpoint

    sg = OpenVocabSceneGraph()
    node = SceneGraphNode(node_id=1, labels=["aerosol"], label_counts={"aerosol": 3}, observation_count=3)
    node.center = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    sg.nodes[1] = node
    sg._next_id = 2

    class _Proc:
        scene_graph = sg

    class _VM:
        def __init__(self):
            self._proc = _Proc()

        def get_scene_graph(self):
            return self._proc.scene_graph

        def set_scene_graph_processor(self, proc):
            self._proc = proc

    class _Ctrl:
        def __init__(self):
            self.graph_memory = _make_memory()
            self.graph_memory.set_graph_timestep(3)
            self.graph_memory.add_observation(_rgb(), np.array([0.5, 0.5, 0.2]), ["mug"])
            self.obs_count = 3
            self.voxel_map = _VM()
            self._open_vocab_sg_processor = _Proc()

        def get_voxel_map(self):
            return self.voxel_map

    path = tmp_path / "both"
    save_lifelong_checkpoint(_Ctrl(), path, save_voxel_pickle=False)
    assert (path / OPEN_VOCAB_SCENE_GRAPH_DIR / "scene_graph.json").is_file()
    assert (path / "graph.json").is_file()
    man = (path / "manifest.json").read_text(encoding="utf-8")
    assert "has_open_vocab_scene_graph" in man

    class _EmptyProc:
        def __init__(self):
            self.scene_graph = OpenVocabSceneGraph()

    class _CtrlLoad:
        def __init__(self):
            self.graph_memory = _make_memory()
            self.obs_count = 0
            self.voxel_map = _VM()
            self.voxel_map._proc = _EmptyProc()
            self._open_vocab_sg_processor = self.voxel_map._proc

        def get_voxel_map(self):
            return self.voxel_map

    ctrl = _CtrlLoad()
    info = load_lifelong_checkpoint(ctrl, path, refine_start=False)
    assert info["graph_loaded"] is True
    assert info["open_vocab_loaded"] is True
    restored = ctrl.get_voxel_map().get_scene_graph()
    assert restored.num_objects == 1
    assert "aerosol" in restored.nodes[1].labels
    np.testing.assert_allclose(restored.nodes[1].center, [1.0, 2.0, 0.5], atol=1e-6)


def test_apply_se2_moves_open_vocab_centers():
    from emet.mapping.scene_graph.open_vocab_scene_graph import OpenVocabSceneGraph, SceneGraphNode
    from emet.memory.lifelong import apply_se2_to_open_vocab_scene_graph

    sg = OpenVocabSceneGraph()
    node = SceneGraphNode(node_id=0, labels=["box"], label_counts={"box": 1}, observation_count=1)
    node.center = np.array([1.0, 0.0, 0.2], dtype=np.float64)
    sg.nodes[0] = node
    n = apply_se2_to_open_vocab_scene_graph(sg, se2_matrix(0.5, -0.25, 0.0))
    assert n == 1
    np.testing.assert_allclose(node.center[:2], [1.5, -0.25], atol=1e-6)
