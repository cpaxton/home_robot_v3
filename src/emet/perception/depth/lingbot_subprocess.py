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

"""Subprocess bridge to .venv-lingbot-map for batch streaming inference."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from emet.utils.logger import Logger

logger = Logger(__name__)


def default_lingbot_venv_python(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[4]
    return Path(os.environ.get("LINGBOT_MAP_VENV", root / ".venv-lingbot-map")) / "bin" / "python"


def default_checkpoint() -> Path:
    return Path(os.environ.get("LINGBOT_MAP_CHECKPOINT", ""))


class LingBotSubprocessClient:
    """Run LingBot-Map inference in isolated venv via ``python -m emet_lingbot_map infer``."""

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        *,
        venv_python: Path | str | None = None,
        keyframe_interval: int | None = 2,
        use_sdpa: bool = True,
        repo_root: Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else default_checkpoint()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"LingBot checkpoint not found: {self.checkpoint} (set LINGBOT_MAP_CHECKPOINT)")
        self.venv_python = Path(venv_python) if venv_python else default_lingbot_venv_python(repo_root)
        if not self.venv_python.is_file():
            raise FileNotFoundError(
                f"LingBot venv python not found: {self.venv_python} (run ./scripts/install_lingbot_map.sh)"
            )
        self.keyframe_interval = keyframe_interval
        self.use_sdpa = use_sdpa

    def infer_episode(self, episode_dir: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.venv_python),
            "-m",
            "emet_lingbot_map",
            "infer",
            "--episode",
            str(episode_dir),
            "--output",
            str(output_dir),
            "--checkpoint",
            str(self.checkpoint),
        ]
        if self.keyframe_interval is not None:
            cmd.extend(["--keyframe-interval", str(int(self.keyframe_interval))])
        if self.use_sdpa:
            cmd.append("--use-sdpa")
        logger.info("LingBot subprocess: %s", " ".join(cmd))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"LingBot infer failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        return output_dir

    def load_latest_prediction(self, lingbot_output: Path, frame_idx: int) -> dict | None:
        meta = lingbot_output / "lingbot_predictions.jsonl"
        if not meta.is_file():
            return None
        row = None
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("frame_idx", -1)) == frame_idx:
                row = r
        if row is None:
            return None
        depth = np.load(lingbot_output / row["depth"]).astype(np.float32)
        pose = np.asarray(row["camera_pose"], dtype=np.float64).reshape(4, 4)
        K = np.asarray(row["camera_K"], dtype=np.float64).reshape(3, 3)
        return {"depth": depth, "camera_pose": pose, "camera_K": K}


class LingBotRollingBuffer:
    """Accumulate RGB frames in a temp episode; batch-infer every ``infer_every_n`` frames."""

    def __init__(
        self,
        client: LingBotSubprocessClient,
        *,
        infer_every_n: int = 2,
        max_buffer_frames: int = 400,
    ) -> None:
        self.client = client
        self.infer_every_n = max(1, int(infer_every_n))
        self.max_buffer_frames = max(8, int(max_buffer_frames))
        self._tmpdir = Path(tempfile.mkdtemp(prefix="lingbot_roll_"))
        self._images = self._tmpdir / "images"
        self._images.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._tmpdir / "metadata.jsonl"
        self._meta_fp = self._meta_path.open("w", encoding="utf-8")
        self._lingbot_out = self._tmpdir / "lingbot"
        self._frame_count = 0
        self._last_pred: dict | None = None
        self._last_pose: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._last_K: np.ndarray | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._meta_fp:
            self._meta_fp.close()
            self._meta_fp = None  # type: ignore[assignment]

    def cleanup(self) -> None:
        self.close()
        if self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_frame(self, rgb: np.ndarray, *, camera_K: np.ndarray | None, camera_pose: np.ndarray | None) -> int:
        idx = self._frame_count
        name = f"frame_{idx:06d}.png"
        rel = f"images/{name}"
        Image.fromarray(np.asarray(rgb[:, :, :3], dtype=np.uint8), mode="RGB").save(self._images / name)
        row = {
            "frame_idx": idx,
            "image": rel,
            "depth": None,
            "camera_K": camera_K.tolist() if camera_K is not None else None,
            "camera_pose": camera_pose.tolist() if camera_pose is not None else None,
        }
        self._meta_fp.write(json.dumps(row) + "\n")
        self._meta_fp.flush()
        self._frame_count += 1
        return idx

    def update(
        self,
        rgb: np.ndarray,
        *,
        camera_K: np.ndarray | None = None,
        camera_pose: np.ndarray | None = None,
        force: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Append frame; maybe run LingBot infer. Returns (depth, camera_pose, K) if updated."""
        idx = self._write_frame(rgb, camera_K=camera_K, camera_pose=camera_pose)
        run = force or (self._frame_count % self.infer_every_n == 0)
        if not run:
            return self._last_depth, self._last_pose, self._last_K

        if self._frame_count > self.max_buffer_frames:
            logger.warning("LingBot rolling buffer exceeded max frames; truncating is not implemented yet.")

        if self._lingbot_out.exists():
            shutil.rmtree(self._lingbot_out, ignore_errors=True)
        self.client.infer_episode(self._tmpdir, self._lingbot_out)
        pred = self.client.load_latest_prediction(self._lingbot_out, idx)
        if pred is None:
            return self._last_depth, self._last_pose, self._last_K
        self._last_depth = pred["depth"]
        self._last_pose = pred["camera_pose"]
        self._last_K = pred["camera_K"]
        return self._last_depth, self._last_pose, self._last_K
