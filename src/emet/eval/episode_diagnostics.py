# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Sim-agnostic episode diagnostics: top-down maps, trajectories, RGB frames, MP4."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DIAGNOSTICS_MANIFEST = "diagnostics_manifest.json"


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        if name == "EMET_EVAL_EXPORT_MAP":
            return _env_truthy("HABITAT_EQA_EXPORT_MAP", default)
        if name == "EMET_EVAL_EXPORT_VIDEO":
            return _env_truthy("HABITAT_EQA_EXPORT_VIDEO", default)
        if name == "EMET_EVAL_EXPORT_GRAPH":
            return _env_truthy("HABITAT_EQA_EXPORT_GRAPH", default)
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raw = os.environ.get("HABITAT_EQA_MAP_STRIDE", "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class EpisodeDiagnosticsConfig:
    export_map: bool = True
    export_map_stride: int = 0
    export_obstacle_grids: bool = True
    export_trajectory: bool = True
    export_rgb_frames: bool = True
    export_video: bool = True
    export_object_crops: bool = True
    export_full_graph: bool = False
    max_map_side: int = 640
    video_fps: float = 6.0

    @classmethod
    def from_env(cls, **overrides: Any) -> EpisodeDiagnosticsConfig:
        cfg = cls(
            export_map=_env_truthy("EMET_EVAL_EXPORT_MAP", True),
            export_map_stride=_env_int("EMET_EVAL_MAP_STRIDE", 0),
            export_obstacle_grids=_env_truthy("EMET_EVAL_EXPORT_OBSTACLE_GRIDS", True),
            export_trajectory=_env_truthy("EMET_EVAL_EXPORT_TRAJECTORY", True),
            export_rgb_frames=_env_truthy("EMET_EVAL_EXPORT_FRAMES", True),
            export_video=_env_truthy("EMET_EVAL_EXPORT_VIDEO", True),
            export_object_crops=_env_truthy("EMET_EVAL_EXPORT_OBJECT_CROPS", True),
            export_full_graph=_env_truthy("EMET_EVAL_EXPORT_GRAPH", False),
        )
        for key, val in overrides.items():
            if val is not None and hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg


@dataclass
class _RecordedFrame:
    step_idx: int
    rgb: np.ndarray | None = None
    pose: tuple[float, float, float] | None = None


@dataclass
class EpisodeDiagnosticsRecorder:
    """Buffer RGB frames, poses, and optional stride map snapshots during an episode."""

    cfg: EpisodeDiagnosticsConfig = field(default_factory=EpisodeDiagnosticsConfig.from_env)
    _frames: list[_RecordedFrame] = field(default_factory=list, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def record_from_agent(self, agent: Any) -> None:
        rgb = None
        pose = robot_xy_from_agent(agent)
        robot = getattr(agent, "robot", None)
        if robot is not None and hasattr(robot, "get_observation"):
            try:
                obs = robot.get_observation()
                if obs is not None and getattr(obs, "rgb", None) is not None:
                    rgb = np.asarray(obs.rgb)
            except Exception:
                pass
        self.record_step(rgb=rgb, pose=pose, agent=agent, step_idx=self._step)
        self._step += 1

    def record_step(
        self,
        *,
        rgb: np.ndarray | None,
        pose: tuple[float, float, float] | None,
        agent: Any,
        step_idx: int | None = None,
    ) -> None:
        idx = int(self._step if step_idx is None else step_idx)
        if idx >= self._step:
            self._step = idx + 1
        self._frames.append(_RecordedFrame(step_idx=idx, rgb=rgb, pose=pose))
        stride = int(self.cfg.export_map_stride or 0)
        if self.cfg.export_map and stride > 0 and idx % stride == 0:
            self._maybe_snapshot_map(agent, idx, intermediate=True)

    def flush(self, episode_dir: Path | str, agent: Any | None = None) -> dict[str, Any]:
        root = Path(episode_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"episode_dir": str(root.resolve())}

        if self.cfg.export_trajectory and self._frames:
            traj_path = root / "trajectory.jsonl"
            with traj_path.open("w", encoding="utf-8") as fh:
                for fr in self._frames:
                    row = {"step": fr.step_idx, "pose_xyt": list(fr.pose) if fr.pose else None}
                    fh.write(json.dumps(row) + "\n")
            manifest["trajectory"] = str(traj_path)

        if self.cfg.export_rgb_frames and self._frames:
            frames_dir = root / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            meta_path = root / "metadata.jsonl"
            with meta_path.open("w", encoding="utf-8") as fh:
                for fr in self._frames:
                    if fr.rgb is None:
                        continue
                    rel = f"frames/rgb_{fr.step_idx:04d}.png"
                    _save_rgb_png(frames_dir / f"rgb_{fr.step_idx:04d}.png", fr.rgb)
                    fh.write(
                        json.dumps(
                            {
                                "frame_idx": fr.step_idx,
                                "image": rel,
                                "pose_xyt": list(fr.pose) if fr.pose else None,
                            }
                        )
                        + "\n"
                    )
            manifest["metadata_jsonl"] = str(meta_path)

        if agent is not None:
            if self.cfg.export_map:
                map_path = self._maybe_snapshot_map(agent, step_idx=-1, intermediate=False, root=root)
                if map_path:
                    manifest["topdown_map"] = str(map_path)
            if self.cfg.export_obstacle_grids:
                manifest.update(_save_obstacle_grids(agent, root))
            if self.cfg.export_object_crops:
                crop_path = _export_object_crops(agent, root)
                if crop_path:
                    manifest["object_crops_mosaic"] = str(crop_path)
            if self.cfg.export_full_graph:
                ckpt = _export_full_graph(agent, root)
                if ckpt:
                    manifest["graph_checkpoint"] = str(ckpt)
            floor_path = _write_floor_metrics(agent, root)
            if floor_path:
                manifest["floor_metrics"] = str(floor_path)

        if self.cfg.export_video:
            mp4 = _write_episode_mp4(root, fps=self.cfg.video_fps)
            if mp4:
                manifest["episode_rgb_mp4"] = str(mp4)

        manifest_path = root / DIAGNOSTICS_MANIFEST
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest["diagnostics_manifest"] = str(manifest_path)
        return manifest

    def _maybe_snapshot_map(
        self,
        agent: Any,
        step_idx: int,
        *,
        intermediate: bool,
        root: Path | None = None,
    ) -> Path | None:
        vm = getattr(agent, "voxel_map", None)
        if vm is None:
            return None
        from emet.visualization.map_snapshot import snapshot_eval_from_voxel_map

        xy = robot_xy_from_agent(agent)
        img, _ = snapshot_eval_from_voxel_map(vm, xy, max_side=self.cfg.max_map_side)
        if img is None:
            return None
        if intermediate:
            maps_dir = Path(".") / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            out = maps_dir / f"step_{step_idx:04d}.png"
        else:
            if root is None:
                return None
            out = root / "topdown_map.png"
        _save_rgb_png(out, img)
        return out


def robot_xy_from_agent(agent: Any) -> tuple[float, float, float] | None:
    robot = getattr(agent, "robot", None)
    if robot is None:
        return None
    try:
        if hasattr(robot, "get_base_pose"):
            xyt = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
            if xyt.size >= 2:
                theta = float(xyt[2]) if xyt.size >= 3 else 0.0
                return (float(xyt[0]), float(xyt[1]), theta)
    except Exception:
        pass
    return None


def attach_diagnostics_recorder(agent: Any, recorder: EpisodeDiagnosticsRecorder) -> None:
    if getattr(agent, "_diag_update_wrapped", False):
        return
    original = agent.update

    def wrapped_update(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        recorder.record_from_agent(agent)
        return result

    agent.update = wrapped_update  # type: ignore[method-assign]
    agent._diag_update_wrapped = True


def flush_episode_diagnostics(
    episode_dir: Path | str,
    agent: Any | None,
    recorder: EpisodeDiagnosticsRecorder | None = None,
    *,
    cfg: EpisodeDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    if recorder is None:
        recorder = EpisodeDiagnosticsRecorder(cfg=cfg or EpisodeDiagnosticsConfig.from_env())
    if not any(
        (
            recorder.cfg.export_map,
            recorder.cfg.export_video,
            recorder.cfg.export_rgb_frames,
            recorder.cfg.export_trajectory,
            recorder.cfg.export_obstacle_grids,
            recorder.cfg.export_object_crops,
            recorder.cfg.export_full_graph,
        )
    ):
        return {}
    return recorder.flush(episode_dir, agent=agent)


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.ndim != 3:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB").save(path)


def _save_obstacle_grids(agent: Any, root: Path) -> dict[str, str]:
    vm = getattr(agent, "voxel_map", None)
    if vm is None or not hasattr(vm, "get_2d_map"):
        return {}
    obstacles, explored = vm.get_2d_map()
    obs = _to_numpy_bool(obstacles)
    exp = _to_numpy_bool(explored)
    np.save(root / "obstacles_2d.npy", obs)
    np.save(root / "explored_2d.npy", exp)
    go = getattr(vm, "grid_origin", None)
    if go is not None and hasattr(go, "cpu"):
        go = go.cpu().numpy()
    meta = {
        "grid_resolution": float(getattr(vm, "grid_resolution", 0.1) or 0.1),
        "grid_origin": np.asarray(go).reshape(-1).tolist() if go is not None else [0.0, 0.0],
        "shape_hw": [int(obs.shape[0]), int(obs.shape[1])],
    }
    grid_meta = root / "grid_meta.json"
    grid_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {
        "obstacles_2d": str(root / "obstacles_2d.npy"),
        "explored_2d": str(root / "explored_2d.npy"),
        "grid_meta": str(grid_meta),
    }


def _to_numpy_bool(arr: Any) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    return np.asarray(arr).astype(bool)


def _export_object_crops(agent: Any, root: Path) -> Path | None:
    gm = getattr(agent, "graph_memory", None)
    if gm is None:
        return None
    try:
        from emet.memory.headless_export import export_dynagraph_visual_assets

        export_dynagraph_visual_assets(gm, root)
        mosaic = root / "dynagraph" / "crops_mosaic.png"
        return mosaic if mosaic.is_file() else None
    except Exception:
        return None


def _export_full_graph(agent: Any, root: Path) -> Path | None:
    gm = getattr(agent, "graph_memory", None)
    vm = getattr(agent, "voxel_map", None)
    if gm is None or vm is None:
        return None
    try:
        from emet.memory.headless_export import export_graph_eqa_dir

        ckpt = root / "graph_checkpoint"
        export_graph_eqa_dir(gm, vm, str(ckpt), title="episode checkpoint")
        return ckpt
    except Exception:
        return None


def _write_floor_metrics(agent: Any, root: Path) -> Path | None:
    vm = getattr(agent, "voxel_map", None)
    if vm is None:
        return None
    try:
        from emet.memory.floor_metrics import compute_explored_floor_metrics, write_floor_metrics_json

        metrics = compute_explored_floor_metrics(vm)
        return write_floor_metrics_json(root, metrics)
    except Exception:
        return None


def _write_episode_mp4(root: Path, *, fps: float) -> Path | None:
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        return None
    try:
        from emet.molmospaces.episode_writer import write_episode_rgb_mp4

        images_dir = root / "images"
        frames = root / "frames"
        if not images_dir.is_dir() and frames.is_dir():
            images_dir.mkdir(exist_ok=True)
            rows_out: list[str] = []
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                img = str(row.get("image", ""))
                if img.startswith("frames/"):
                    src = root / img
                    dst_name = Path(img).name.replace("rgb_", "frame_")
                    dst = images_dir / dst_name
                    if src.is_file() and not dst.exists():
                        import shutil

                        shutil.copy2(src, dst)
                    row = dict(row)
                    row["image"] = f"images/{dst_name}"
                rows_out.append(json.dumps(row))
            meta_path.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
        return write_episode_rgb_mp4(root, fps=fps, filename="episode_rgb.mp4")
    except Exception:
        return None
