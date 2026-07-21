# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from emet.eval.dynamic_exploration_runner import (
    _invalidate_checkpoint_nodes_near_moves,
    count_object_nodes,
    count_object_nodes_near_xy,
    run_logged_subprocess,
)


@dataclass
class _FakeNode:
    labels: list[str] = field(default_factory=list)
    is_viewpoint: bool = False
    xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    last_seen: int = 0
    is_frontier: bool = False


class _FakeMemory:
    def __init__(self, nodes: list[_FakeNode]) -> None:
        self._nodes = nodes
        self.staleness_horizon = 256

    def get_nodes(self) -> list[_FakeNode]:
        return list(self._nodes)


def test_count_object_nodes_no_hint_excludes_viewpoints():
    mem = _FakeMemory(
        [
            _FakeNode(labels=["mug"]),
            _FakeNode(labels=["apple"], is_viewpoint=True),
            _FakeNode(labels=["vase"]),
        ]
    )
    assert count_object_nodes(mem) == 2


def test_count_object_nodes_label_hint_matches_any_label():
    mem = _FakeMemory(
        [
            _FakeNode(labels=["red_mug", "kitchen"]),
            _FakeNode(labels=["apple"]),
            _FakeNode(labels=["obj_main_variant"]),
        ]
    )
    assert count_object_nodes(mem, label_hint="mug") == 1
    assert count_object_nodes(mem, label_hint="obj_main") == 1
    assert count_object_nodes(mem, label_hint="OBJ_MAIN") == 1


def test_count_object_nodes_label_hint_no_match():
    mem = _FakeMemory([_FakeNode(labels=["apple"]), _FakeNode(labels=["vase"])])
    assert count_object_nodes(mem, label_hint="mug") == 0


def test_count_object_nodes_none_memory():
    assert count_object_nodes(None) == 0


def test_count_object_nodes_near_xy():
    mem = _FakeMemory(
        [
            _FakeNode(labels=["mug"], xyz=[0.0, 0.0, 0.9]),
            _FakeNode(labels=["far"], xyz=[5.0, 5.0, 0.9]),
            _FakeNode(labels=["vp"], xyz=[0.1, 0.1, 0.9], is_viewpoint=True),
        ]
    )
    assert count_object_nodes_near_xy(mem, [0.0, 0.0], radius_m=0.75) == 1
    assert count_object_nodes_near_xy(mem, [5.0, 5.0], radius_m=0.75) == 1
    assert count_object_nodes_near_xy(None, [0.0, 0.0]) == 0


def test_invalidate_checkpoint_nodes_near_moves(tmp_path: Path):
    graph = {
        "final_step": 100,
        "nodes": [
            {"node_id": 1, "xyz": [1.0, 2.0, 0.9], "last_seen": 100, "is_viewpoint": False},
            {"node_id": 2, "xyz": [9.0, 9.0, 0.9], "last_seen": 100, "is_viewpoint": False},
        ],
    }
    ckpt = tmp_path / "cycle_0"
    ckpt.mkdir()
    (ckpt / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    aged = _invalidate_checkpoint_nodes_near_moves(
        ckpt,
        [{"old_pos": [1.0, 2.0, 0.9], "pos": [3.0, 4.0, 0.9], "target": "obj_main"}],
        staleness_horizon=256,
    )
    assert aged == 1
    data = json.loads((ckpt / "graph.json").read_text(encoding="utf-8"))
    by_id = {n["node_id"]: n for n in data["nodes"]}
    assert by_id[1]["last_seen"] == 0  # max(0, 100 - 256 - 1)
    assert by_id[2]["last_seen"] == 100


def test_run_logged_subprocess_writes_progress(tmp_path: Path):
    log_path = tmp_path / "child.log"
    progress = tmp_path / "progress.jsonl"
    proc = run_logged_subprocess(
        [sys.executable, "-c", "print('hello-dyna'); import sys; sys.stdout.flush()"],
        cwd=tmp_path,
        env=None,
        log_path=log_path,
        timeout_s=30.0,
        label="unit_test",
        progress_path=progress,
        heartbeat_s=60.0,
        stale_log_s=3600.0,
    )
    assert proc.returncode == 0
    assert "hello-dyna" in log_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert events[0]["event"] == "subprocess_start"
    assert events[-1]["event"] == "subprocess_end"
    assert events[-1]["returncode"] == 0


def test_run_logged_subprocess_timeout(tmp_path: Path):
    log_path = tmp_path / "slow.log"
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=None,
            log_path=log_path,
            timeout_s=1.0,
            label="timeout_test",
            progress_path=None,
            heartbeat_s=0.5,
            stale_log_s=3600.0,
        )


def test_run_logged_subprocess_timeout_kills_process_group(tmp_path: Path):
    """Timeout must kill the whole session so grandchild workers do not orphan."""
    import time as time_mod

    log_path = tmp_path / "pg.log"
    marker = tmp_path / "child_alive"
    # Parent spawns a grandchild via subprocess that touches marker until killed.
    script = (
        "import subprocess, sys, time, pathlib\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-c',\n"
        "    'import pathlib,time; m=pathlib.Path(' + repr(str(m)) + ');\\n'\n"
        "    'while True:\\n'\n"
        "    ' m.write_text(str(time.time())); time.sleep(0.2)'\n"
        "])\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged_subprocess(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=None,
            log_path=log_path,
            timeout_s=1.5,
            label="pg_kill_test",
            progress_path=None,
            heartbeat_s=0.5,
            stale_log_s=3600.0,
        )
    time_mod.sleep(0.8)
    assert marker.is_file()
    t0 = float(marker.read_text())
    time_mod.sleep(0.6)
    t1 = float(marker.read_text())
    assert t1 == t0, "grandchild kept writing after timeout — process group not killed"
