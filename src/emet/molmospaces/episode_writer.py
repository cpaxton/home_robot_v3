# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Per-episode layout for exploration / NeRF-oriented datasets (RGB + pose + intrinsics).

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from emet.core.interfaces import Observations


def _np_to_jsonable(x: np.ndarray | None) -> Any:
    if x is None:
        return None
    return np.asarray(x, dtype=float).tolist()


def export_nerfstudio_transforms(
    episode_dir: Path | str,
    *,
    output_name: str = "transforms.json",
    camera_angle_x: float | None = None,
) -> Path:
    """
    Build a NERFStudio-style ``transforms.json`` from ``metadata.jsonl``.

    Uses ``camera_pose`` from each line as **camera-to-world** (4x4, same convention
    as ``Observations.camera_pose`` in emet: camera frame expressed in world / world
    from camera as used for projecting rays — see docs/molmospaces.md).

    If ``camera_angle_x`` is omitted, it is derived from ``camera_K`` and image width
    when present; otherwise a default is used and a warning is implied by missing K.
    """
    root = Path(episode_dir)
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")

    lines = meta_path.read_text().splitlines()
    frames: list[dict[str, Any]] = []
    w = h = None
    angle_x = camera_angle_x
    first_rel: str | None = None
    first_K: list[list[float]] | None = None

    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        rel = row.get("image")
        if not rel:
            continue
        pose = row.get("camera_pose")
        if pose is None:
            continue
        c2w = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        if first_rel is None:
            first_rel = rel
            first_K = row.get("camera_K")

        frames.append(
            {
                "file_path": rel.replace("\\", "/"),
                "transform_matrix": c2w.tolist(),
            }
        )

    if first_rel:
        img_path = root / first_rel
        if img_path.is_file():
            with Image.open(img_path) as im:
                w, h = im.size

    if angle_x is None and w is not None and first_K is not None:
        Kk = np.asarray(first_K, dtype=np.float64).reshape(3, 3)
        fx = float(Kk[0, 0])
        if fx > 1e-6:
            angle_x = float(2.0 * np.arctan(0.5 * float(w) / fx))

    if angle_x is None:
        angle_x = float(np.radians(60.0))

    out: dict[str, Any] = {
        "camera_angle_x": angle_x,
        "frames": frames,
    }
    if w is not None and h is not None:
        out["w"] = w
        out["h"] = h

    out_path = root / output_name
    out_path.write_text(json.dumps(out, indent=2))
    return out_path


def write_episode_rgb_mp4(
    episode_dir: Path | str,
    *,
    fps: float = 10.0,
    filename: str = "episode_rgb.mp4",
) -> Path:
    """
    Encode all ``images/*.png`` listed in ``metadata.jsonl`` (ordered by ``frame_idx``)
    into a single HxW MP4 under the episode directory (``mp4v`` fourcc via OpenCV).

    Updates ``episode.json`` with key ``rgb_mp4`` when that file already exists.
    """
    import cv2

    root = Path(episode_dir)
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")

    rows: list[dict[str, Any]] = []
    for line in meta_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("frame_idx", 0)))
    if not rows:
        raise ValueError("metadata.jsonl has no frames; cannot write MP4")

    paths = [root / str(r["image"]) for r in rows if r.get("image")]
    if not paths or not paths[0].is_file():
        raise FileNotFoundError(f"No image files under {root}")

    first_rgb = np.asarray(Image.open(paths[0]).convert("RGB"))
    h, w = first_rgb.shape[:2]
    out_path = root / filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, max(0.5, float(fps)), (w, h))
    if not vw.isOpened():
        raise RuntimeError(
            "cv2.VideoWriter could not open for MP4 (mp4v). "
            "Check OpenCV build / codecs; on Debian try: apt install ffmpeg libopencv-videoio-dev"
        )

    for p in paths:
        if not p.is_file():
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = np.asarray(Image.fromarray(rgb).resize((w, h)))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()

    if not out_path.is_file() or out_path.stat().st_size < 32:
        raise RuntimeError(f"MP4 missing or too small: {out_path}")

    ep = root / "episode.json"
    if ep.is_file():
        meta = json.loads(ep.read_text(encoding="utf-8"))
        meta["rgb_mp4"] = filename
        ep.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return out_path


class MolmoEpisodeWriter:
    """
    Write ``images/``, optional ``depths/``, ``metadata.jsonl``, and finalize ``episode.json``.

    Each JSONL row: frame index, relative image path, optional depth, ``camera_pose``,
    ``camera_K``, gps/compass, seq_id.
    """

    def __init__(
        self,
        output_dir: Path | str,
        *,
        episode_fields: dict[str, Any],
        save_depth: bool = True,
    ) -> None:
        self.root = Path(output_dir)
        self.images_dir = self.root / "images"
        self.depths_dir = self.root / "depths"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if save_depth:
            self.depths_dir.mkdir(parents=True, exist_ok=True)
        self.save_depth = save_depth
        self._episode_fields = dict(episode_fields)
        self._meta_path = self.root / "metadata.jsonl"
        self._meta_fp = self._meta_path.open("w", encoding="utf-8")
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    def close(self) -> None:
        if getattr(self, "_meta_fp", None) is not None:
            self._meta_fp.close()
            self._meta_fp = None  # type: ignore[assignment]

    def write_frame(self, obs: Observations, frame_idx: int | None = None) -> int:
        """Save RGB (and optionally depth), append one JSONL record. Returns frame index used."""
        idx = int(self._frame_count if frame_idx is None else frame_idx)
        rgb = np.asarray(obs.rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 rgb, got {rgb.shape}")

        img_name = f"frame_{idx:06d}.png"
        rel_img = f"images/{img_name}"
        Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB").save(self.images_dir / img_name)

        depth_rel = None
        if self.save_depth and obs.depth is not None:
            d = np.asarray(obs.depth, dtype=np.float32)
            stem = self.depths_dir / f"frame_{idx:06d}"
            np.save(stem, d)
            depth_rel = f"depths/{stem.name}.npy"

        row: dict[str, Any] = {
            "frame_idx": idx,
            "image": rel_img,
            "depth": depth_rel,
            "camera_pose": _np_to_jsonable(obs.camera_pose),
            "camera_K": _np_to_jsonable(obs.camera_K),
            "gps": obs.gps.tolist() if obs.gps is not None else None,
            "compass": obs.compass.tolist() if obs.compass is not None else None,
            "seq_id": int(obs.seq_id),
        }
        self._meta_fp.write(json.dumps(row) + "\n")
        self._meta_fp.flush()
        self._frame_count = idx + 1
        return idx

    def finalize(self, *, extra: dict[str, Any] | None = None) -> None:
        """Write ``episode.json`` and close JSONL."""
        self.close()
        meta: dict[str, Any] = {
            **self._episode_fields,
            "num_frames": int(self._frame_count),
        }
        if extra:
            meta.update(extra)
        try:
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            meta["git_commit"] = rev
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            meta["git_commit"] = None
        (self.root / "episode.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
