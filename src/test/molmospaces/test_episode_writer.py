# Copyright (c) Hello Robot, Inc. All rights reserved.

import json
from pathlib import Path

import numpy as np
import pytest

from emet.core.interfaces import Observations
from emet.molmospaces.episode_writer import (
    MolmoEpisodeWriter,
    export_nerfstudio_transforms,
    write_episode_rgb_mp4,
)


def _minimal_obs(rgb_shape=(32, 48, 3), with_pose: bool = True) -> Observations:
    h, w = rgb_shape[0], rgb_shape[1]
    pose = np.eye(4, dtype=np.float64) if with_pose else None
    K = np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1.0]], dtype=np.float64)
    return Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros(rgb_shape, dtype=np.uint8),
        depth=np.ones((h, w), dtype=np.float32) * 1.5,
        camera_K=K,
        camera_pose=pose,
        seq_id=0,
    )


def test_molmo_episode_writer_writes_images_jsonl_episode(tmp_path: Path) -> None:
    writer = MolmoEpisodeWriter(
        tmp_path,
        episode_fields={"molmospaces_scene": "ithor", "robot": "rby1"},
        save_depth=True,
    )
    obs = _minimal_obs()
    obs.rgb[:, :, 0] = 200
    writer.write_frame(obs, 0)
    writer.finalize()
    assert (tmp_path / "images" / "frame_000000.png").is_file()
    assert (tmp_path / "depths" / "frame_000000.npy").is_file()
    lines = (tmp_path / "metadata.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["frame_idx"] == 0
    assert row["image"] == "images/frame_000000.png"
    assert row["depth"] == "depths/frame_000000.npy"
    assert row["camera_pose"] is not None
    ep = json.loads((tmp_path / "episode.json").read_text())
    assert ep["num_frames"] == 1
    assert ep["molmospaces_scene"] == "ithor"


def test_export_nerfstudio_transforms_roundtrip(tmp_path: Path) -> None:
    writer = MolmoEpisodeWriter(tmp_path, episode_fields={"robot": "rby1"}, save_depth=False)
    obs = _minimal_obs()
    writer.write_frame(obs)
    writer.finalize()
    out = export_nerfstudio_transforms(tmp_path)
    assert out.name == "transforms.json"
    data = json.loads(out.read_text())
    assert "camera_angle_x" in data
    assert len(data["frames"]) == 1
    assert data["frames"][0]["file_path"] == "images/frame_000000.png"
    assert len(data["frames"][0]["transform_matrix"]) == 4


def test_export_skips_rows_without_camera_pose(tmp_path: Path) -> None:
    writer = MolmoEpisodeWriter(tmp_path, episode_fields={}, save_depth=False)
    obs = _minimal_obs(with_pose=False)
    writer.write_frame(obs)
    writer.finalize()
    out = export_nerfstudio_transforms(tmp_path)
    data = json.loads(out.read_text())
    assert data["frames"] == []


def test_write_episode_rgb_mp4_produces_nonempty_file(tmp_path: Path) -> None:
    writer = MolmoEpisodeWriter(tmp_path, episode_fields={"robot": "test"}, save_depth=False)
    for i in range(4):
        obs = _minimal_obs()
        obs.rgb[:] = (i * 40) % 256
        writer.write_frame(obs, i)
    writer.finalize()
    try:
        out = write_episode_rgb_mp4(tmp_path, fps=5.0)
    except RuntimeError as e:
        if "VideoWriter" in str(e) or "opencv" in str(e).lower():
            pytest.skip(f"OpenCV VideoWriter unavailable in this environment: {e}")
        raise
    assert out.name == "episode_rgb.mp4"
    assert out.stat().st_size > 200
    ep = json.loads((tmp_path / "episode.json").read_text())
    assert ep.get("rgb_mp4") == "episode_rgb.mp4"


def test_writer_no_depth_when_disabled(tmp_path: Path) -> None:
    writer = MolmoEpisodeWriter(tmp_path, episode_fields={}, save_depth=False)
    obs = _minimal_obs()
    writer.write_frame(obs)
    writer.finalize()
    assert not (tmp_path / "depths").exists() or not list((tmp_path / "depths").glob("*.npy"))
    row = json.loads((tmp_path / "metadata.jsonl").read_text().strip())
    assert row["depth"] is None
