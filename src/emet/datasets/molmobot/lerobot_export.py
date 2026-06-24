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

"""Export MolmoBot episodes to a LeRobot-friendly JSONL + folder layout."""

from __future__ import annotations

import json
from pathlib import Path

from emet.datasets.molmobot.reader import iter_molmobot_episodes


def export_episode_jsonl(episode, out_dir: Path, *, task: str = "molmobot") -> Path:
    """Write one episode as ``metadata.jsonl`` with qpos + action fields (no video decode)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"
    trimmed = episode.trimmed_actions()
    with meta_path.open("w") as f:
        for step in trimmed:
            row = {
                "task": task,
                "traj_key": episode.traj_key,
                "step": step.index,
                "observation.state": step.qpos,
                "action": step.action_joint_pos or {},
                "cameras": step.camera_video_paths,
                "obs_scene": episode.obs_scene,
            }
            f.write(json.dumps(row) + "\n")
    episode_meta = {
        "traj_key": episode.traj_key,
        "h5_path": episode.h5_path,
        "length": episode.length,
        "trimmed_length": len(trimmed),
        "task": task,
    }
    (out_dir / "episode.json").write_text(json.dumps(episode_meta, indent=2))
    return meta_path


def export_lerobot_tree(
    src: Path | str,
    out: Path | str,
    *,
    task: str = "molmobot",
    max_episodes: int | None = None,
) -> int:
    """Export up to *max_episodes* trajectories under *out*/episode_NNN/."""
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for ep in iter_molmobot_episodes(src):
        ep_dir = out_root / f"episode_{count:04d}"
        export_episode_jsonl(ep, ep_dir, task=task)
        count += 1
        if max_episodes is not None and count >= max_episodes:
            break
    (out_root / "dataset_info.json").write_text(
        json.dumps({"task": task, "episodes": count, "format": "emet-molmobot-jsonl"}, indent=2)
    )
    return count
