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

"""Episode I/O and metrics for LingBot-Map eval (main emet venv; no lingbot_map import)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class EpisodeFrame:
    frame_idx: int
    rgb_path: Path
    depth_path: Path | None
    camera_K: np.ndarray | None
    camera_pose: np.ndarray | None


@dataclass
class Episode:
    root: Path
    frames: list[EpisodeFrame]


def load_episode(episode_dir: Path | str) -> Episode:
    root = Path(episode_dir)
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    frames: list[EpisodeFrame] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = int(row.get("frame_idx", len(frames)))
        rel_img = row.get("image")
        if not rel_img:
            continue
        depth_rel = row.get("depth")
        depth_path = root / str(depth_rel) if depth_rel else None
        K = row.get("camera_K")
        pose = row.get("camera_pose")
        frames.append(
            EpisodeFrame(
                frame_idx=idx,
                rgb_path=root / str(rel_img),
                depth_path=depth_path if depth_path and depth_path.is_file() else None,
                camera_K=np.asarray(K, dtype=np.float64).reshape(3, 3) if K is not None else None,
                camera_pose=np.asarray(pose, dtype=np.float64).reshape(4, 4) if pose is not None else None,
            )
        )
    frames.sort(key=lambda f: f.frame_idx)
    if not frames:
        raise ValueError(f"No frames in {meta_path}")
    return Episode(root=root, frames=frames)


def load_rgb(frame: EpisodeFrame) -> np.ndarray:
    with Image.open(frame.rgb_path) as im:
        return np.asarray(im.convert("RGB"))


def load_depth_meters(frame: EpisodeFrame) -> np.ndarray | None:
    if frame.depth_path is None:
        return None
    d = np.load(frame.depth_path).astype(np.float32)
    finite = d[np.isfinite(d) & (d > 0)]
    if finite.size and float(np.median(finite)) > 100.0:
        d = d / 1000.0
    return d


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
        depth = np.load(lingbot_dir / row["depth"]).astype(np.float32)
        pose = np.asarray(row["camera_pose"], dtype=np.float64).reshape(4, 4)
        K = np.asarray(row["camera_K"], dtype=np.float64).reshape(3, 3)
        out[idx] = {"depth": depth, "camera_pose": pose, "camera_K": K}
    return out


def _valid_depth_mask(depth: np.ndarray, *, z_min: float = 0.05, z_max: float = 12.0) -> np.ndarray:
    return np.isfinite(depth) & (depth > z_min) & (depth < z_max)


def estimate_depth_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    m = _valid_depth_mask(pred) & _valid_depth_mask(gt)
    if int(m.sum()) < 100:
        return 1.0
    return float(np.median(gt[m] / np.maximum(pred[m], 1e-6)))


def depth_rmse(pred: np.ndarray, gt: np.ndarray, scale: float = 1.0) -> tuple[float, int]:
    m = _valid_depth_mask(pred * scale) & _valid_depth_mask(gt)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    err = gt[m] - pred[m] * scale
    return float(np.sqrt(np.mean(err * err))), n


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / max(src.shape[0], 1)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(s) / max(var_src, 1e-9))
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def trajectory_ate_rmse(pred_poses: np.ndarray, gt_poses: np.ndarray) -> tuple[float, float]:
    pred_c = pred_poses[:, :3, 3]
    gt_c = gt_poses[:, :3, 3]
    if pred_c.shape[0] != gt_c.shape[0] or pred_c.shape[0] < 2:
        return float("nan"), 1.0
    s, r, t = _umeyama_sim3(pred_c, gt_c)
    aligned = (s * (pred_c @ r.T)) + t
    err = aligned - gt_c
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1)))), s


def evaluate_lingbot_vs_gt(episode_dir: Path, lingbot_dir: Path) -> dict[str, float | int]:
    episode = load_episode(episode_dir)
    preds = load_lingbot_predictions(lingbot_dir)
    depth_errors: list[float] = []
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


def evaluate_da3_depth_rmse(
    episode_dir: Path,
    *,
    model_id: str = "depth-anything/DA3-SMALL",
    process_res: int = 378,
    clip_max_m: float = 4.0,
    device: str = "cuda",
) -> dict[str, float | int]:
    """Per-frame DA3 depth vs sensor GT on recorded episode."""
    from emet.perception.depth.da3_estimator import DA3DepthEstimator

    episode = load_episode(episode_dir)
    est = DA3DepthEstimator(
        model_id=model_id,
        device=device,
        process_res=process_res,
        clip_output_max_m=clip_max_m,
    )
    errors: list[float] = []
    scales: list[float] = []
    for fr in episode.frames:
        gt_d = load_depth_meters(fr)
        if gt_d is None:
            continue
        rgb = load_rgb(fr)
        pred_d = est.infer(rgb)
        if pred_d is None:
            continue
        h, w = rgb.shape[:2]
        if pred_d.shape[:2] != (h, w):
            pred_d = cv2.resize(pred_d, (w, h), interpolation=cv2.INTER_LINEAR)
        if gt_d.shape[:2] != (h, w):
            gt_d = cv2.resize(gt_d, (w, h), interpolation=cv2.INTER_LINEAR)
        sc = estimate_depth_scale(pred_d, gt_d)
        scales.append(sc)
        rmse, n = depth_rmse(pred_d, gt_d, scale=sc)
        if n > 0:
            errors.append(rmse)
    return {
        "num_frames_compared": len(errors),
        "depth_rmse_mean_m": float(np.mean(errors)) if errors else float("nan"),
        "depth_scale_median": float(np.median(scales)) if scales else float("nan"),
    }
