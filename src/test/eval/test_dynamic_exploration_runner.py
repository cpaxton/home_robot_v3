# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from emet.eval.dynamic_exploration_runner import count_object_nodes, run_logged_subprocess


@dataclass
class _FakeNode:
    labels: list[str] = field(default_factory=list)
    is_viewpoint: bool = False


class _FakeMemory:
    def __init__(self, nodes: list[_FakeNode]) -> None:
        self._nodes = nodes

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
