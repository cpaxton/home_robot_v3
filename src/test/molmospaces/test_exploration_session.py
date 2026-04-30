# Copyright (c) Hello Robot, Inc. All rights reserved.

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from emet.core.interfaces import Observations
from emet.molmospaces.episode_writer import MolmoEpisodeWriter
from emet.molmospaces.exploration import MolmoExploreSession


def _obs() -> Observations:
    return Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((16, 24, 3), dtype=np.uint8),
        depth=np.ones((16, 24), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=np.eye(4),
        seq_id=1,
    )


def test_molmo_explore_session_records_frames(tmp_path: Path) -> None:
    robot = MagicMock()
    robot.get_observation.return_value = _obs()
    robot.get_base_pose.return_value = np.array([0.1, 0.2, 0.0])
    robot.move_base_to = MagicMock()
    robot.switch_to_navigation_mode = MagicMock()
    robot.move_to_nav_posture = MagicMock()
    robot.set_velocity = MagicMock()

    writer = MolmoEpisodeWriter(tmp_path, episode_fields={"robot": "mock"}, save_depth=False)
    session = MolmoExploreSession(robot, writer, navigate_every=1000)
    session.run(steps=3, capture_hz=100.0)
    writer.finalize()

    assert writer.frame_count == 3
    prog = (tmp_path / "explore_progress.txt").read_text(encoding="utf-8")
    assert "explore started" in prog
    assert "first frame saved" in prog
    assert "explore loop finished" in prog
    lines = (tmp_path / "metadata.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    assert robot.get_observation.call_count == 3
