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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_depth_rmse_and_trajectory_metrics(tmp_path: Path) -> None:
    from emet.perception.depth.lingbot_eval import (
        depth_rmse,
        estimate_depth_scale,
        load_episode,
        trajectory_ate_rmse,
    )

    ep = tmp_path / "ep"
    (ep / "images").mkdir(parents=True)
    (ep / "depths").mkdir(parents=True)
    h, w = 48, 64
    for i in range(4):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        np.save(ep / "depths" / f"frame_{i:06d}.npy", np.full((h, w), 1.0 + 0.1 * i, dtype=np.float32))
        from PIL import Image

        Image.fromarray(rgb).save(ep / "images" / f"frame_{i:06d}.png")
        pose = np.eye(4)
        pose[0, 3] = float(i) * 0.5
        row = {
            "frame_idx": i,
            "image": f"images/frame_{i:06d}.png",
            "depth": f"depths/frame_{i:06d}.npy",
            "camera_pose": pose.tolist(),
            "camera_K": [[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]],
        }
        with (ep / "metadata.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")

    episode = load_episode(ep)
    assert len(episode.frames) == 4

    pred = np.full((h, w), 2.0, dtype=np.float32)
    gt = np.full((h, w), 1.0, dtype=np.float32)
    sc = estimate_depth_scale(pred, gt)
    assert abs(sc - 0.5) < 0.05
    rmse, n = depth_rmse(pred, gt, scale=sc)
    assert n == h * w
    assert rmse < 1e-3

    gt_poses = np.stack([f.camera_pose for f in episode.frames if f.camera_pose is not None], axis=0)
    pred_poses = gt_poses.copy()
    pred_poses[:, :3, 3] *= 2.0
    ate, s = trajectory_ate_rmse(pred_poses, gt_poses)
    assert ate < 0.05
    assert abs(s - 0.5) < 0.1
