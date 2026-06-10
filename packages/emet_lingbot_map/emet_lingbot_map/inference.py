# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Run LingBot-Map streaming inference on an image sequence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from emet_lingbot_map.episode_loader import Episode, load_episode


@dataclass
class InferenceConfig:
    checkpoint: Path
    image_size: int = 518
    patch_size: int = 14
    num_scale_frames: int = 8
    keyframe_interval: int | None = None
    kv_cache_sliding_window: int = 64
    max_frame_num: int = 1024
    camera_num_iterations: int = 4
    use_sdpa: bool = False
    offload_to_cpu: bool = True
    mode: str = "streaming"  # streaming | windowed
    window_size: int = 64
    overlap_keyframes: int | None = None


@dataclass
class FramePrediction:
    frame_idx: int
    depth: np.ndarray  # HxW float32 meters (model units, may need scale align)
    camera_pose: np.ndarray  # 4x4 cam-to-world
    camera_K: np.ndarray  # 3x3 from model intrinsics when available


def _load_model(cfg: InferenceConfig, device: torch.device):
    if cfg.mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        enable_3d_rope=True,
        max_frame_num=cfg.max_frame_num,
        kv_cache_sliding_window=cfg.kv_cache_sliding_window,
        kv_cache_scale_frames=cfg.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=cfg.use_sdpa,
        camera_num_iterations=cfg.camera_num_iterations,
    )
    ckpt = torch.load(str(cfg.checkpoint), map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def _preprocess_paths(image_paths: list[Path], *, image_size: int, patch_size: int) -> torch.Tensor:
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    paths = [str(p) for p in image_paths]
    return load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )


def _postprocess_predictions(predictions: dict[str, Any], images: torch.Tensor) -> dict[str, Any]:
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    predictions["extrinsic"] = extrinsic_4x4[..., :3, :4]
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    if "images" in predictions:
        predictions.pop("images", None)
    for k, v in list(predictions.items()):
        if isinstance(v, torch.Tensor):
            predictions[k] = v.detach().cpu()
    return predictions


def _depth_tensor_to_numpy(depth_t: torch.Tensor, frame_i: int) -> np.ndarray:
    """Extract HxW depth for frame index from batched depth output."""
    d = depth_t
    if d.ndim == 5:
        d = d[0, frame_i, 0]
    elif d.ndim == 4:
        d = d[frame_i, 0]
    elif d.ndim == 3:
        d = d[frame_i]
    return np.asarray(d, dtype=np.float32)


def _pose_tensor_to_4x4(extrinsic_t: torch.Tensor, frame_i: int) -> np.ndarray:
    ext = extrinsic_t
    if ext.ndim == 4 and ext.shape[0] == 1:
        ext = ext[0]
    row = ext[frame_i]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :4] = np.asarray(row[:3, :4], dtype=np.float64)
    return mat


def _intrinsic_tensor_to_3x3(intrinsic_t: torch.Tensor, frame_i: int) -> np.ndarray:
    K = intrinsic_t
    if K.ndim == 4 and K.shape[0] == 1:
        K = K[0]
    return np.asarray(K[frame_i], dtype=np.float64).reshape(3, 3)


def run_inference_on_episode(episode: Episode, cfg: InferenceConfig) -> list[FramePrediction]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("LingBot-Map streaming inference requires CUDA.")

    image_paths = [f.rgb_path for f in episode.frames]
    images = _preprocess_paths(image_paths, image_size=cfg.image_size, patch_size=cfg.patch_size)
    num_frames = int(images.shape[0])

    kf = cfg.keyframe_interval
    if kf is None:
        kf = 1 if num_frames <= 320 else (num_frames + 319) // 320

    model = _load_model(cfg, device)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    output_device = torch.device("cpu") if cfg.offload_to_cpu else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if cfg.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=cfg.num_scale_frames,
                keyframe_interval=kf,
                output_device=output_device,
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=cfg.window_size,
                overlap_size=16,
                overlap_keyframes=cfg.overlap_keyframes,
                num_scale_frames=cfg.num_scale_frames,
                keyframe_interval=kf,
                output_device=output_device,
            )

    images_cpu = images.cpu() if not cfg.offload_to_cpu else predictions.get("images", images.cpu())
    predictions = _postprocess_predictions(predictions, images_cpu)

    depth_t = predictions["depth"]
    extr_t = predictions["extrinsic"]
    intr_t = predictions["intrinsic"]

    out: list[FramePrediction] = []
    for i, fr in enumerate(episode.frames):
        depth = _depth_tensor_to_numpy(depth_t, i)
        pose = _pose_tensor_to_4x4(extr_t, i)
        K = _intrinsic_tensor_to_3x3(intr_t, i)
        out.append(FramePrediction(frame_idx=fr.frame_idx, depth=depth, camera_pose=pose, camera_K=K))
    return out


def save_predictions(
    predictions: list[FramePrediction],
    output_dir: Path | str,
    *,
    episode: Episode | None = None,
) -> Path:
    out_root = Path(output_dir)
    depth_dir = out_root / "depths"
    pose_dir = out_root / "poses"
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict[str, Any]] = []
    for pred in predictions:
        stem = f"frame_{pred.frame_idx:06d}"
        depth_path = depth_dir / f"{stem}.npy"
        pose_path = pose_dir / f"{stem}.npy"
        np.save(depth_path, pred.depth.astype(np.float32))
        np.save(pose_path, pred.camera_pose.astype(np.float64))
        meta_rows.append(
            {
                "frame_idx": pred.frame_idx,
                "depth": str(depth_path.relative_to(out_root)),
                "camera_pose": pred.camera_pose.tolist(),
                "camera_K": pred.camera_K.tolist(),
            }
        )

    meta_path = out_root / "lingbot_predictions.jsonl"
    with meta_path.open("w", encoding="utf-8") as fp:
        for row in meta_rows:
            fp.write(json.dumps(row) + "\n")

    summary = {
        "num_frames": len(predictions),
        "episode_root": str(episode.root) if episode is not None else None,
    }
    (out_root / "lingbot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_root


def infer_episode_to_disk(
    episode_dir: Path | str,
    cfg: InferenceConfig,
    output_dir: Path | str,
    *,
    first_k: int | None = None,
) -> Path:
    episode = load_episode(episode_dir)
    if first_k is not None and first_k > 0:
        episode.frames = episode.frames[: int(first_k)]
    preds = run_inference_on_episode(episode, cfg)
    return save_predictions(preds, output_dir, episode=episode)
