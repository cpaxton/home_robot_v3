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

"""Read MolmoBot-Data ``trajectories_batch_*.h5`` files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from emet.datasets.molmobot.schema import MolmoBotEpisode, MolmoBotStep


def _require_h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for MolmoBot datasets. Install with: uv sync --extra sim or pip install h5py"
        ) from e
    return h5py


def decode_json_field(raw: bytes | np.ndarray | str | None) -> dict | list | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip("\x00").strip()
        return json.loads(text) if text else None
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in ("S", "O"):
            raw = raw.tobytes() if raw.dtype.kind == "S" else raw.item()
        else:
            raw = bytes(raw)
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", errors="ignore").strip("\x00").strip()
        return json.loads(text) if text else None
    return None


def _read_json_list(group, key: str) -> list[dict | None]:
    h5py = _require_h5py()
    if key not in group:
        return []
    node = group[key]
    if hasattr(node, "shape") and len(getattr(node, "shape", ())) == 0:
        one = decode_json_field(node[()])
        return [one]
    out: list[dict | None] = []
    if isinstance(node, h5py.Group):
        keys = sorted(node.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        for k in keys:
            child = node[k]
            out.append(decode_json_field(child[()] if hasattr(child, "__getitem__") else child))
        return out
    for i in range(len(node)):
        out.append(decode_json_field(node[i][()]))
    return out


def _video_path_for_step(sensor_data_group, camera: str, step_idx: int, h5_dir: Path) -> str:
    if sensor_data_group is None or camera not in sensor_data_group:
        return ""
    node = sensor_data_group[camera]
    try:
        raw = node[step_idx][()]
    except (KeyError, IndexError, TypeError):
        try:
            raw = node[()]
        except Exception:
            return ""
    if isinstance(raw, bytes):
        name = raw.decode("utf-8", errors="ignore").strip("\x00").strip()
    elif isinstance(raw, np.ndarray) and raw.dtype.kind == "S":
        name = raw.tobytes().decode("utf-8", errors="ignore").strip("\x00").strip()
    else:
        name = str(raw).strip()
    if not name:
        return ""
    return str((h5_dir / name).resolve())


class MolmoBotBatchReader:
    """Iterate trajectories inside one ``trajectories_batch_*.h5`` file."""

    def __init__(self, h5_path: Path | str):
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(self.h5_path)

    def traj_keys(self) -> list[str]:
        h5py = _require_h5py()
        with h5py.File(self.h5_path, "r") as h5:
            return sorted(k for k in h5.keys() if str(k).startswith("traj_"))

    def read_episode(self, traj_key: str) -> MolmoBotEpisode:
        h5py = _require_h5py()
        h5_dir = self.h5_path.parent
        with h5py.File(self.h5_path, "r") as h5:
            if traj_key not in h5:
                raise KeyError(f"{traj_key!r} not in {self.h5_path}")
            grp = h5[traj_key]
            obs_scene = decode_json_field(grp["obs_scene"][()]) if "obs_scene" in grp else {}
            if not isinstance(obs_scene, dict):
                obs_scene = {}

            qpos_list = _read_json_list(grp["obs/agent"], "qpos") if "obs/agent" in grp else []
            joint_pos_actions = _read_json_list(grp["actions"], "joint_pos") if "actions" in grp else []
            commanded = _read_json_list(grp["actions"], "commanded_action") if "actions" in grp else []
            sensor_data = grp["obs/sensor_data"] if "obs/sensor_data" in grp else None
            cameras: list[str] = []
            if sensor_data is not None:
                cameras = list(sensor_data.keys())

            n = max(len(qpos_list), len(joint_pos_actions), 1)
            steps: list[MolmoBotStep] = []
            for i in range(n):
                qpos_raw = qpos_list[i] if i < len(qpos_list) else None
                qpos = qpos_raw if isinstance(qpos_raw, dict) else {}
                act_raw = joint_pos_actions[i] if i < len(joint_pos_actions) else None
                cmd_raw = commanded[i] if i < len(commanded) else None
                cams = {cam: _video_path_for_step(sensor_data, cam, i, h5_dir) for cam in cameras}
                steps.append(
                    MolmoBotStep(
                        index=i,
                        qpos={str(k): float(v) for k, v in qpos.items()},
                        action_joint_pos=(
                            {str(k): float(v) for k, v in act_raw.items()} if isinstance(act_raw, dict) else None
                        ),
                        commanded_action=cmd_raw if isinstance(cmd_raw, dict) else None,
                        camera_video_paths={k: v for k, v in cams.items() if v},
                    )
                )

            rewards = grp["rewards"][()].tolist() if "rewards" in grp else []
            success = grp["success"][()].tolist() if "success" in grp else []

        return MolmoBotEpisode(
            traj_key=traj_key,
            h5_path=str(self.h5_path),
            obs_scene=obs_scene,
            steps=steps,
            rewards=[float(x) for x in rewards],
            success=[bool(x) for x in success],
        )

    def __iter__(self) -> Iterator[MolmoBotEpisode]:
        for key in self.traj_keys():
            yield self.read_episode(key)


def iter_molmobot_episodes(root: Path | str) -> Iterator[MolmoBotEpisode]:
    """Walk *root* for ``trajectories_batch_*.h5`` and yield all trajectories."""
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix == ".h5":
        yield from MolmoBotBatchReader(root_path)
        return
    for h5_path in sorted(root_path.rglob("trajectories_batch_*.h5")):
        yield from MolmoBotBatchReader(h5_path)


def summarize_dataset(root: Path | str) -> dict[str, object]:
    """Aggregate stats for ``emet dataset molmobot inspect``."""
    lengths: list[int] = []
    qpos_dims: set[int] = set()
    cameras: set[str] = set()
    n_episodes = 0
    for ep in iter_molmobot_episodes(root):
        n_episodes += 1
        lengths.append(ep.length)
        if ep.steps and ep.steps[0].qpos:
            qpos_dims.add(len(ep.steps[0].qpos))
        for step in ep.steps:
            cameras.update(step.camera_video_paths.keys())
    return {
        "episodes": n_episodes,
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "qpos_dims": sorted(qpos_dims),
        "cameras": sorted(cameras),
    }
