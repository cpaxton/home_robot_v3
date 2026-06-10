# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Streaming single-frame server for emet debug / Dynamem subprocess bridge."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from emet_lingbot_map.inference import InferenceConfig, _load_model, _postprocess_predictions, _preprocess_paths


class LingBotStreamSession:
    """Stateful streaming session (one frame at a time)."""

    def __init__(self, cfg: InferenceConfig) -> None:
        import torch

        self.cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type != "cuda":
            raise RuntimeError("LingBot streaming requires CUDA")
        self._model = _load_model(cfg, self._device)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        if dtype != torch.float32 and getattr(self._model, "aggregator", None) is not None:
            self._model.aggregator = self._model.aggregator.to(dtype=dtype)
        self._dtype = dtype
        self._frame_count = 0
        self._scale_done = False
        self._scale_frames = cfg.num_scale_frames
        self._keyframe_interval = max(1, int(cfg.keyframe_interval or 1))
        self._image_paths: list[Path] = []
        self._temp_dir = Path(tempfile.mkdtemp(prefix="lingbot_stream_"))

    def reset(self) -> None:
        import torch

        self._model.clean_kv_cache()
        self._frame_count = 0
        self._scale_done = False
        self._image_paths.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _append_rgb_path(self, rgb: np.ndarray) -> Path:
        path = self._temp_dir / f"frame_{self._frame_count:06d}.jpg"
        Image.fromarray(np.asarray(rgb[:, :, :3], dtype=np.uint8), mode="RGB").save(path, quality=95)
        self._image_paths.append(path)
        return path

    def process_frame(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (depth HxW, camera_pose 4x4, camera_K 3x3) for one RGB frame."""
        import torch

        path = self._append_rgb_path(rgb)
        images = _preprocess_paths([path], image_size=self.cfg.image_size, patch_size=self.cfg.patch_size)
        images = images.to(self._device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._dtype):
            if not self._scale_done:
                n_scale = min(self._scale_frames, max(1, images.shape[0]))
                scale_batch = images[:n_scale].unsqueeze(0)
                self._model.forward(
                    scale_batch,
                    num_frame_for_scale=n_scale,
                    num_frame_per_block=n_scale,
                    causal_inference=True,
                )
                self._scale_done = True
                self._frame_count += 1
                # First frame after scale init: run as streaming frame 0
                stream_batch = images.unsqueeze(0)
            else:
                is_keyframe = self._keyframe_interval <= 1 or (self._frame_count % self._keyframe_interval == 0)
                if not is_keyframe:
                    self._model._set_skip_append(True)
                stream_batch = images.unsqueeze(0)
                self._model.forward(
                    stream_batch,
                    num_frame_for_scale=self._scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
                if not is_keyframe:
                    self._model._set_skip_append(False)
                self._frame_count += 1

        # Read last prediction from model cache / heads — use single-frame streaming API
        # Fallback: run full streaming on accumulated temp images (slow but correct for v1)
        all_images = _preprocess_paths(
            self._image_paths,
            image_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
        ).to(self._device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._dtype):
            preds = self._model.inference_streaming(
                all_images,
                num_scale_frames=self._scale_frames,
                keyframe_interval=self._keyframe_interval,
                output_device=torch.device("cpu"),
            )
        preds = _postprocess_predictions(preds, all_images.cpu())
        fi = len(self._image_paths) - 1
        depth_t = preds["depth"]
        if depth_t.ndim == 5:
            depth = np.asarray(depth_t[0, fi, 0], dtype=np.float32)
        elif depth_t.ndim == 4:
            depth = np.asarray(depth_t[fi, 0], dtype=np.float32)
        else:
            depth = np.asarray(depth_t[fi], dtype=np.float32)
        ext = preds["extrinsic"]
        intr = preds["intrinsic"]
        if ext.ndim == 4 and ext.shape[0] == 1:
            ext = ext[0]
        if intr.ndim == 4 and intr.shape[0] == 1:
            intr = intr[0]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :4] = np.asarray(ext[fi, :3, :4], dtype=np.float64)
        K = np.asarray(intr[fi], dtype=np.float64).reshape(3, 3)
        return depth, pose, K


def run_stream_server(checkpoint: Path, *, use_sdpa: bool = False) -> None:
    """JSON-lines protocol on stdin/stdout for live frames.

    Input lines:
      {"cmd": "reset"}
      {"cmd": "frame", "rgb_path": "/path/to.jpg"}
      {"cmd": "frame", "rgb_b64": "..."}  # optional

    Output lines:
      {"ok": true, "depth_shape": [H,W], "depth_b64": "...", "camera_pose": [[...]], "camera_K": [[...]]}
      {"ok": false, "error": "..."}
    """
    cfg = InferenceConfig(
        checkpoint=checkpoint,
        use_sdpa=use_sdpa,
        keyframe_interval=2,
        offload_to_cpu=True,
    )
    session = LingBotStreamSession(cfg)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "reset":
                session.reset()
                sys.stdout.write(json.dumps({"ok": True, "reset": True}) + "\n")
                sys.stdout.flush()
                continue
            if cmd == "frame":
                rgb_path = msg.get("rgb_path")
                if rgb_path:
                    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
                else:
                    import base64
                    import io

                    raw = base64.b64decode(msg["rgb_b64"])
                    rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                depth, pose, K = session.process_frame(rgb)
                import base64

                depth_bytes = depth.astype(np.float32).tobytes()
                out = {
                    "ok": True,
                    "depth_shape": list(depth.shape),
                    "depth_b64": base64.b64encode(depth_bytes).decode("ascii"),
                    "camera_pose": pose.tolist(),
                    "camera_K": K.tolist(),
                }
                sys.stdout.write(json.dumps(out) + "\n")
                sys.stdout.flush()
                continue
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown cmd {cmd!r}"}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    import os

    ckpt = Path(os.environ.get("LINGBOT_MAP_CHECKPOINT", ""))
    if not ckpt.is_file():
        print(json.dumps({"ok": False, "error": f"LINGBOT_MAP_CHECKPOINT missing: {ckpt}"}), file=sys.stderr)
        sys.exit(1)
    run_stream_server(ckpt, use_sdpa=os.environ.get("LINGBOT_MAP_USE_SDPA", "") == "1")
