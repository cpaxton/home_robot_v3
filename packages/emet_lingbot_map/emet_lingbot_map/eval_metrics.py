# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Evaluation metrics for LingBot vs sim ground truth (used from main emet venv)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _valid_depth_mask(depth: np.ndarray, *, z_min: float = 0.05, z_max: float = 12.0) -> np.ndarray:
    return np.isfinite(depth) & (depth > z_min) & (depth < z_max)


def estimate_depth_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    """Median ratio gt/pred on overlapping valid pixels."""
    m = _valid_depth_mask(pred) & _valid_depth_mask(gt)
    if int(m.sum()) < 100:
        return 1.0
    ratios = gt[m] / np.maximum(pred[m], 1e-6)
    return float(np.median(ratios))


def depth_rmse(pred: np.ndarray, gt: np.ndarray, scale: float = 1.0) -> tuple[float, int]:
    m = _valid_depth_mask(pred * scale) & _valid_depth_mask(gt)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    err = gt[m] - pred[m] * scale
    return float(np.sqrt(np.mean(err * err))), n


def camera_centers(poses: np.ndarray) -> np.ndarray:
    """poses: Nx4x4 cam-to-world -> Nx3 centers."""
    return poses[:, :3, 3]


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Align src to dst (Nx3). Returns scale, R, t such that dst ~ s R src + t."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / max(n, 1)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.trace(np.diag(s) @ np.eye(3)) / max(var_src, 1e-9)) if var_src > 1e-9 else 1.0
    scale = float(np.sum(s) / max(var_src, 1e-9))
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def trajectory_ate_rmse(pred_poses: np.ndarray, gt_poses: np.ndarray) -> tuple[float, float]:
    """Sim(3) align pred camera centers to GT; return (ATE RMSE meters, scale)."""
    pred_c = camera_centers(pred_poses)
    gt_c = camera_centers(gt_poses)
    if pred_c.shape[0] != gt_c.shape[0] or pred_c.shape[0] < 2:
        return float("nan"), 1.0
    s, r, t = _umeyama_sim3(pred_c, gt_c)
    aligned = (s * (pred_c @ r.T)) + t
    err = aligned - gt_c
    ate = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    return ate, s


def load_lingbot_predictions(lingbot_dir: Path) -> dict[int, dict]:
    meta = lingbot_dir / "lingbot_predictions.jsonl"
    if not meta.is_file():
        raise FileNotFoundError(meta)
    out: dict[int, dict] = {}
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = int(row["frame_idx"])
        depth_path = lingbot_dir / row["depth"]
        pose = np.asarray(row["camera_pose"], dtype=np.float64).reshape(4, 4)
        K = np.asarray(row["camera_K"], dtype=np.float64).reshape(3, 3)
        depth = np.load(depth_path).astype(np.float32)
        out[idx] = {"depth": depth, "camera_pose": pose, "camera_K": K}
    return out


def evaluate_episode(
    episode_dir: Path,
    lingbot_dir: Path,
    *,
    resize_depth_to_rgb: bool = True,
) -> dict[str, float | int]:
    """Compare lingbot outputs to episode GT depths and camera_pose."""
    import cv2

    from emet_lingbot_map.episode_loader import load_depth_meters, load_episode, load_rgb

    episode = load_episode(episode_dir)
    preds = load_lingbot_predictions(lingbot_dir)

    depth_errors: list[float] = []
    depth_counts: list[int] = []
    scales: list[float] = []
    pred_poses: list[np.ndarray] = []
    gt_poses: list[np.ndarray] = []

    for fr in episode.frames:
        if fr.frame_idx not in preds:
            continue
        gt_d = load_depth_meters(fr)
        if gt_d is None:
            continue
        pred_d = preds[fr.frame_idx]["depth"]
        if resize_depth_to_rgb:
            rgb = load_rgb(fr)
            h, w = rgb.shape[:2]
            if pred_d.shape[:2] != (h, w):
                pred_d = cv2.resize(pred_d, (w, h), interpolation=cv2.INTER_LINEAR)
            if gt_d.shape[:2] != (h, w):
                gt_d = cv2.resize(gt_d, (w, h), interpolation=cv2.INTER_LINEAR)
        sc = estimate_depth_scale(pred_d, gt_d)
        scales.append(sc)
        rmse, n = depth_rmse(pred_d, gt_d, scale=sc)
        if n > 0:
            depth_errors.append(rmse)
            depth_counts.append(n)
        if fr.camera_pose is not None:
            pred_poses.append(preds[fr.frame_idx]["camera_pose"])
            gt_poses.append(fr.camera_pose)

    pred_arr = np.stack(pred_poses, axis=0) if pred_poses else np.zeros((0, 4, 4))
    gt_arr = np.stack(gt_poses, axis=0) if gt_poses else np.zeros((0, 4, 4))
    ate, traj_scale = trajectory_ate_rmse(pred_arr, gt_arr)

    return {
        "num_frames_compared": len(depth_errors),
        "depth_rmse_mean_m": float(np.mean(depth_errors)) if depth_errors else float("nan"),
        "depth_scale_median": float(np.median(scales)) if scales else float("nan"),
        "trajectory_ate_rmse_m": ate,
        "trajectory_sim3_scale": traj_scale,
    }
