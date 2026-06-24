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

"""Tests for MolmoBot H5 reader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from emet.datasets.molmobot.reader import MolmoBotBatchReader, iter_molmobot_episodes, summarize_dataset


def _pad_json(obj: dict) -> np.ndarray:
    raw = json.dumps(obj).encode("utf-8")
    buf = raw + b"\x00" * max(0, 256 - len(raw))
    return np.frombuffer(buf[:256], dtype="S256")


@pytest.fixture()
def synthetic_h5(tmp_path: Path) -> Path:
    path = tmp_path / "trajectories_batch_0_of_1.h5"
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("traj_0")
        grp.create_dataset("obs_scene", data=_pad_json({"robot": "rby1"}))
        agent = grp.create_group("obs/agent")
        qpos = agent.create_group("qpos")
        qpos.create_dataset("0", data=_pad_json({"j1": 0.0}))
        qpos.create_dataset("1", data=_pad_json({"j1": 0.1}))
        qpos.create_dataset("2", data=_pad_json({"j1": 0.2}))
        actions = grp.create_group("actions")
        jp = actions.create_group("joint_pos")
        jp.create_dataset("0", data=_pad_json({"j1": 0.0}))
        jp.create_dataset("1", data=_pad_json({"j1": 0.05}))
        jp.create_dataset("2", data=_pad_json({"j1": 0.15}))
        grp.create_dataset("rewards", data=np.zeros(3))
        grp.create_dataset("success", data=np.zeros(3, dtype=bool))
    return path


def test_read_synthetic_episode(synthetic_h5: Path):
    ep = MolmoBotBatchReader(synthetic_h5).read_episode("traj_0")
    assert ep.length == 3
    assert ep.steps[1].qpos["j1"] == pytest.approx(0.1)
    trimmed = ep.trimmed_actions()
    assert len(trimmed) == 1


def test_summarize_dataset(synthetic_h5: Path):
    summary = summarize_dataset(synthetic_h5.parent)
    assert summary["episodes"] == 1
    assert summary["length_max"] == 3


def test_iter_molmobot_episodes(synthetic_h5: Path):
    eps = list(iter_molmobot_episodes(synthetic_h5))
    assert len(eps) == 1


def test_export_lerobot_tree(tmp_path: Path, synthetic_h5: Path):
    from emet.datasets.molmobot.lerobot_export import export_lerobot_tree

    out = tmp_path / "out"
    n = export_lerobot_tree(synthetic_h5, out, max_episodes=1)
    assert n == 1
    assert (out / "episode_0000" / "metadata.jsonl").is_file()
