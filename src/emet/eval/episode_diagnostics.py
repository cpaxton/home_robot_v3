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

from emet.utils.logger import Logger
from emet.eval.episode_video import normalize_yaw_delta

logger = Logger(__name__)

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


def _env_truthy_or_none(name: str) -> bool | None:
    """Return None when the env var is unset (for config precedence)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        if name == "EMET_EVAL_EXPORT_MAP":
            return _env_truthy_or_none("HABITAT_EQA_EXPORT_MAP")
        if name == "EMET_EVAL_EXPORT_VIDEO":
            return _env_truthy_or_none("HABITAT_EQA_EXPORT_VIDEO")
        if name == "EMET_EVAL_EXPORT_GRAPH":
            return _env_truthy_or_none("HABITAT_EQA_EXPORT_GRAPH")
        return None
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


def _env_map_max_side(default: int = 1280) -> int:
    raw = os.environ.get("EMET_EVAL_MAP_MAX_SIDE", "").strip()
    if not raw:
        raw = os.environ.get("HABITAT_EQA_MAP_MAX_SIDE", "").strip()
    if not raw:
        return default
    try:
        return max(256, int(raw))
    except ValueError:
        return default


def _env_map_min_side(default: int = 1024) -> int:
    raw = os.environ.get("EMET_EVAL_MAP_MIN_SIDE", "").strip()
    if not raw:
        return default
    try:
        return max(128, int(raw))
    except ValueError:
        return default


def _env_map_video_stride(default: int = 5) -> int:
    raw = os.environ.get("EMET_EVAL_MAP_VIDEO_STRIDE", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
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
    export_voxel_history: bool = False
    export_voxel_pickle: bool = False
    max_map_side: int = 1280
    min_map_side: int = 1024
    filter_map_islands: bool = True
    export_gt_navmesh_map: bool = True
    export_map_overlay: bool = True
    export_map_video: bool = True
    map_video_stride: int = 5
    video_fps: float = 6.0
    export_video_substeps: bool = True
    video_motion_paced: bool = True
    video_meters_per_frame: float = 0.25
    video_radians_per_frame: float = 0.1745329252  # 10 deg
    video_crossfade_teleport_m: float = 1.5

    @classmethod
    def from_env(cls, parameters: Any = None, **overrides: Any) -> EpisodeDiagnosticsConfig:
        from emet.config.eval_config import resolve_episode_diagnostics_config

        return resolve_episode_diagnostics_config(parameters, **overrides)


@dataclass
class _RecordedFrame:
    step_idx: int
    rgb: np.ndarray | None = None
    pose: tuple[float, float, float] | None = None


@dataclass
class EpisodeDiagnosticsRecorder:
    """Buffer RGB frames, poses, and optional stride map snapshots during an episode."""

    cfg: EpisodeDiagnosticsConfig = field(default_factory=EpisodeDiagnosticsConfig.from_env)
    spawn_record: dict[str, Any] | None = None
    habitat_pathfinder: Any | None = None
    habitat_floor_y: float | None = None
    _frames: list[_RecordedFrame] = field(default_factory=list, init=False, repr=False)
    _stride_snapshots: list[tuple[int, np.ndarray]] = field(default_factory=list, init=False, repr=False)
    _stride_overlay_snapshots: list[tuple[int, np.ndarray]] = field(
        default_factory=list, init=False, repr=False
    )
    _nav_attempts: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _planning_step: int = field(default=0, init=False, repr=False)
    _frame_seq: int = field(default=0, init=False, repr=False)
    _habitat_substep_hook: Any | None = field(default=None, init=False, repr=False)

    def append_nav_attempt(self, row: dict[str, Any]) -> None:
        self._nav_attempts.append(dict(row))

    def record_from_agent(self, agent: Any) -> None:
        rgb = None
        pose = robot_xy_from_agent(agent)
        robot = getattr(agent, "robot", None)
        if robot is not None and hasattr(robot, "get_observation"):
            try:
                obs = robot.get_observation()
                if obs is not None and getattr(obs, "rgb", None) is not None:
                    rgb = np.asarray(obs.rgb)
            except Exception as exc:
                logger.warning(f"diagnostics RGB fetch failed: {exc}")
        self.record_step(rgb=rgb, pose=pose, agent=agent, capture_map_stride=True)
        self._planning_step += 1

    def record_habitat_substep(self, *, rgb: np.ndarray | None, pose: tuple[float, float, float] | None) -> None:
        """Append one RGB frame per Habitat ``sim.step`` (nav / rotate substeps)."""
        if rgb is None and pose is None:
            return
        if self._frames and pose is not None and rgb is not None:
            last = self._frames[-1]
            if last.pose is not None and last.rgb is not None:
                same_pose = (
                    abs(last.pose[0] - pose[0]) < 1e-4
                    and abs(last.pose[1] - pose[1]) < 1e-4
                    and abs(normalize_yaw_delta(last.pose[2], pose[2])) < 1e-3
                )
                if same_pose and np.array_equal(last.rgb, rgb):
                    return
        self.record_step(rgb=rgb, pose=pose, agent=None, capture_map_stride=False)

    def record_step(
        self,
        *,
        rgb: np.ndarray | None,
        pose: tuple[float, float, float] | None,
        agent: Any,
        step_idx: int | None = None,
        capture_map_stride: bool = True,
    ) -> None:
        idx = int(self._frame_seq if step_idx is None else step_idx)
        if step_idx is None:
            self._frame_seq += 1
        elif idx >= self._frame_seq:
            self._frame_seq = idx + 1
        self._frames.append(_RecordedFrame(step_idx=idx, rgb=rgb, pose=pose))
        if not capture_map_stride or agent is None:
            return
        stride = self._effective_map_stride()
        stride_step = self._planning_step if step_idx is None else idx
        if stride > 0 and stride_step % stride == 0:
            self._capture_stride_snapshot(agent, stride_step)

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

        if self._stride_snapshots or self._stride_overlay_snapshots:
            maps_dir = root / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            for step_idx, img in self._stride_snapshots:
                out = maps_dir / f"step_{step_idx:04d}.png"
                _save_rgb_png(out, img)
            for step_idx, img in self._stride_overlay_snapshots:
                out = maps_dir / f"overlay_step_{step_idx:04d}.png"
                _save_rgb_png(out, img)
            manifest["map_stride_dir"] = str(maps_dir)

        if self.cfg.export_map_video:
            map_mp4 = _write_map_exploration_mp4(
                root,
                fps=self.cfg.video_fps,
                prefer_overlay=bool(self._stride_overlay_snapshots),
            )
            if map_mp4:
                manifest["topdown_exploration_mp4"] = str(map_mp4)

        if agent is not None:
            if self.cfg.export_map or self.cfg.export_gt_navmesh_map or self.cfg.export_map_overlay:
                map_paths = self._maybe_snapshot_maps(agent, root=root)
                manifest.update(map_paths)
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
            if self.cfg.export_voxel_history or self.cfg.export_voxel_pickle:
                spawn = self.spawn_record if isinstance(self.spawn_record, dict) else None
                manifest.update(
                    export_voxel_observation_history(
                        agent,
                        root,
                        spawn_record=spawn,
                        export_jsonl=self.cfg.export_voxel_history,
                        export_pickle=self.cfg.export_voxel_pickle,
                    )
                )

        if self._nav_attempts:
            nav_path = root / "nav_attempts.jsonl"
            with nav_path.open("w", encoding="utf-8") as fh:
                for row in self._nav_attempts:
                    fh.write(json.dumps(row) + "\n")
            manifest["nav_attempts_jsonl"] = str(nav_path)
        else:
            nav_path = root / "nav_attempts.jsonl"
            if nav_path.is_file():
                nav_path.unlink()

        if self.cfg.export_video:
            mp4 = _write_episode_mp4(root, cfg=self.cfg)
            if mp4:
                manifest["episode_rgb_mp4"] = str(mp4)
                manifest["head_camera_mp4"] = str(mp4)

        manifest_path = root / DIAGNOSTICS_MANIFEST
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest["diagnostics_manifest"] = str(manifest_path)
        return manifest

    def _effective_map_stride(self) -> int:
        stride = int(self.cfg.export_map_stride or 0)
        if stride > 0:
            return stride
        if self.cfg.export_map_video:
            return max(1, int(self.cfg.map_video_stride or 5))
        return 0

    def _capture_stride_snapshot(self, agent: Any, step_idx: int) -> None:
        img = self._render_eval_map_rgb(agent, include_trajectory=True)
        if img is not None:
            self._stride_snapshots.append((int(step_idx), img))
        if self.cfg.export_map_overlay or self.cfg.export_map_video:
            overlay = self._render_overlay_map_rgb(agent, include_trajectory=True)
            if overlay is not None:
                self._stride_overlay_snapshots.append((int(step_idx), overlay))

    def _trajectory_poses(self) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for fr in self._frames:
            if fr.pose is None:
                continue
            p = fr.pose
            theta = float(p[2]) if len(p) >= 3 else 0.0
            out.append((float(p[0]), float(p[1]), theta))
        return out

    def _render_eval_map_rgb(self, agent: Any, *, include_trajectory: bool = False) -> np.ndarray | None:
        vm = getattr(agent, "voxel_map", None)
        if vm is None:
            return None
        from emet.visualization.map_snapshot import snapshot_eval_from_voxel_map

        xy = robot_xy_from_agent(agent)
        traj = self._trajectory_poses() if (self.cfg.export_trajectory or include_trajectory) else None
        img, _ = snapshot_eval_from_voxel_map(
            vm,
            xy,
            max_side=self.cfg.max_map_side,
            min_map_side=self.cfg.min_map_side,
            trajectory_xyt=traj,
            filter_islands=self.cfg.filter_map_islands,
        )
        return img

    def _render_overlay_map_rgb(self, agent: Any, *, include_trajectory: bool = False) -> np.ndarray | None:
        vm = getattr(agent, "voxel_map", None)
        if vm is None:
            return None
        from emet.visualization.map_snapshot import snapshot_eval_overlay_from_voxel_map

        xy = robot_xy_from_agent(agent)
        traj = self._trajectory_poses() if (self.cfg.export_trajectory or include_trajectory) else None
        gt_nav = self._habitat_gt_navigable(agent)
        return snapshot_eval_overlay_from_voxel_map(
            vm,
            xy,
            max_side=self.cfg.max_map_side,
            min_map_side=self.cfg.min_map_side,
            trajectory_xyt=traj,
            gt_navigable=gt_nav,
            filter_islands=self.cfg.filter_map_islands,
        )

    def _habitat_gt_navigable(self, agent: Any) -> np.ndarray | None:
        pf = self.habitat_pathfinder
        if pf is None or not getattr(pf, "is_loaded", False):
            return None
        vm = getattr(agent, "voxel_map", None)
        if vm is None or not hasattr(vm, "get_2d_map"):
            return None
        obstacles, explored = vm.get_2d_map()
        from emet.habitat.navmesh_topdown import rasterize_habitat_navmesh_grid
        from emet.visualization.map_snapshot import _grid_origin_xy

        go = _grid_origin_xy(getattr(vm, "grid_origin", None))
        res = float(getattr(vm, "grid_resolution", 0.1) or 0.1)
        floor_y = self.habitat_floor_y
        if floor_y is None and isinstance(self.spawn_record, dict):
            snapped = (self.spawn_record.get("init_pose_snapped") or {})
            if isinstance(snapped, dict) and "y" in snapped:
                floor_y = float(snapped["y"])
        if floor_y is None:
            floor_y = 0.0
        shape = (int(np.asarray(explored).shape[0]), int(np.asarray(explored).shape[1]))
        return rasterize_habitat_navmesh_grid(
            pf, shape, go, res, floor_y=float(floor_y)
        )

    def _maybe_snapshot_maps(self, agent: Any, *, root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        vm = getattr(agent, "voxel_map", None)
        if vm is None:
            return out
        xy = robot_xy_from_agent(agent)
        traj = self._trajectory_poses() if self.cfg.export_trajectory else None
        gt_nav = None
        if self.cfg.export_gt_navmesh_map or self.cfg.export_map_overlay:
            gt_nav = self._habitat_gt_navigable(agent)

        if self.cfg.export_map:
            img = self._render_eval_map_rgb(agent)
            if img is not None:
                map_path = root / "topdown_map.png"
                _save_rgb_png(map_path, img)
                out["topdown_map"] = str(map_path)

        if self.cfg.export_gt_navmesh_map and gt_nav is not None:
            from emet.habitat.navmesh_topdown import habitat_gt_topdown_cropped
            from emet.visualization.map_snapshot import _grid_origin_xy

            obstacles, explored = vm.get_2d_map()
            go = _grid_origin_xy(getattr(vm, "grid_origin", None))
            res = float(getattr(vm, "grid_resolution", 0.1) or 0.1)
            floor_y = float(self.habitat_floor_y or 0.0)
            _nav_full, gt_rgb = habitat_gt_topdown_cropped(
                self.habitat_pathfinder,
                obstacles,
                explored,
                go,
                res,
                xy,
                floor_y=floor_y,
                margin_cells=8,
                max_side=self.cfg.max_map_side,
                min_map_side=self.cfg.min_map_side,
                trajectory_xyt=traj,
                filter_islands=self.cfg.filter_map_islands,
            )
            gt_path = root / "topdown_gt_navmesh.png"
            _save_rgb_png(gt_path, gt_rgb)
            out["topdown_gt_navmesh"] = str(gt_path)

        if self.cfg.export_map_overlay:
            from emet.visualization.map_snapshot import snapshot_eval_overlay_from_voxel_map

            overlay = snapshot_eval_overlay_from_voxel_map(
                vm,
                xy,
                max_side=self.cfg.max_map_side,
                min_map_side=self.cfg.min_map_side,
                trajectory_xyt=traj,
                gt_navigable=gt_nav,
                filter_islands=self.cfg.filter_map_islands,
            )
            if overlay is not None:
                overlay_path = root / "topdown_map_overlay.png"
                _save_rgb_png(overlay_path, overlay)
                out["topdown_map_overlay"] = str(overlay_path)
        return out

    def _maybe_snapshot_map(self, agent: Any, *, root: Path) -> Path | None:
        img = self._render_eval_map_rgb(agent)
        if img is None:
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


def _is_habitat_robot_client(robot: Any) -> bool:
    return robot is not None and robot.__class__.__name__ == "HabitatRobotClient"


def _make_habitat_substep_hook(agent: Any, recorder: EpisodeDiagnosticsRecorder):
    from emet_habitat.observations import habitat_rgb_depth_to_observations

    def hook(robot: Any, frame: Any) -> None:
        if not recorder.cfg.export_video_substeps:
            return
        try:
            obs = habitat_rgb_depth_to_observations(
                rgb=frame.rgb,
                depth=frame.depth,
                agent_state=frame.agent_state,
                intrinsics=frame.intrinsics,
                semantic=getattr(frame, "semantic", None),
                **robot._observation_kwargs(),
            )
        except Exception as exc:
            logger.warning(f"habitat substep observation failed: {exc}")
            return
        if obs is None or obs.rgb is None:
            return
        pose = (float(obs.gps[0]), float(obs.gps[1]), float(obs.compass[0]))
        recorder.record_habitat_substep(rgb=np.asarray(obs.rgb), pose=pose)

    return hook


def bind_diagnostics_recorder(
    agent: Any,
    recorder: EpisodeDiagnosticsRecorder,
    *,
    spawn_record: dict[str, Any] | None = None,
    habitat_pathfinder: Any | None = None,
    habitat_floor_y: float | None = None,
) -> None:
    """Register recorder on agent step callbacks (invoked from DynamemController.update)."""
    if spawn_record is not None:
        recorder.spawn_record = spawn_record
    if habitat_pathfinder is not None:
        recorder.habitat_pathfinder = habitat_pathfinder
    if habitat_floor_y is not None:
        recorder.habitat_floor_y = habitat_floor_y
    agent._episode_diagnostics_recorder = recorder
    callbacks = list(getattr(agent, "_on_step_callbacks", None) or [])
    cb = recorder.record_from_agent
    if cb not in callbacks:
        callbacks.append(cb)
    agent._on_step_callbacks = callbacks

    robot = getattr(agent, "robot", None)
    if recorder.cfg.export_video_substeps and _is_habitat_robot_client(robot):
        hook = _make_habitat_substep_hook(agent, recorder)
        recorder._habitat_substep_hook = hook
        add_hook = getattr(robot, "add_post_step_hook", None)
        if callable(add_hook):
            add_hook(hook)


def unbind_diagnostics_recorder(agent: Any, recorder: EpisodeDiagnosticsRecorder) -> None:
    callbacks = list(getattr(agent, "_on_step_callbacks", None) or [])
    cb = recorder.record_from_agent
    if cb in callbacks:
        callbacks.remove(cb)
    agent._on_step_callbacks = callbacks
    if getattr(agent, "_episode_diagnostics_recorder", None) is recorder:
        agent._episode_diagnostics_recorder = None
    robot = getattr(agent, "robot", None)
    hook = recorder._habitat_substep_hook
    if hook is not None and _is_habitat_robot_client(robot):
        remove_hook = getattr(robot, "remove_post_step_hook", None)
        if callable(remove_hook):
            remove_hook(hook)
    recorder._habitat_substep_hook = None


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
            recorder.cfg.export_map_video,
            recorder.cfg.export_video,
            recorder.cfg.export_rgb_frames,
            recorder.cfg.export_trajectory,
            recorder.cfg.export_obstacle_grids,
            recorder.cfg.export_object_crops,
            recorder.cfg.export_full_graph,
            recorder.cfg.export_voxel_history,
            recorder.cfg.export_voxel_pickle,
        )
    ):
        return {}
    return recorder.flush(episode_dir, agent=agent)

def habitat_export_voxel_history_default() -> bool:
    """Habitat runners enable voxel history unless explicitly disabled."""
    raw = os.environ.get("EMET_EVAL_EXPORT_VOXEL_HISTORY", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


def export_voxel_observation_history(
    agent: Any,
    root: Path,
    *,
    spawn_record: dict[str, Any] | None = None,
    export_jsonl: bool = True,
    export_pickle: bool = False,
) -> dict[str, str]:
    """Export slim per-observation voxel debug history already held in memory."""
    vm = getattr(agent, "voxel_map", None)
    if vm is None:
        return {}
    out: dict[str, str] = {}
    observations = list(getattr(vm, "observations", []) or [])
    if export_jsonl and observations:
        from emet.visualization.map_snapshot import _grid_origin_xy, world_xy_to_grid_ij

        obstacles, explored = vm.get_2d_map()
        obs_grid = _to_numpy_bool(obstacles)
        shape_hw = (int(obs_grid.shape[0]), int(obs_grid.shape[1]))
        res = float(getattr(vm, "grid_resolution", 0.1) or 0.1)
        origin_xy = _grid_origin_xy(getattr(vm, "grid_origin", None))

        header: dict[str, Any] = {
            "type": "header",
            "n_observations": len(observations),
            "grid_resolution": res,
            "grid_origin_xy": origin_xy.tolist(),
            "shape_hw": list(shape_hw),
        }
        if spawn_record:
            header["spawn_record"] = spawn_record

        path = root / "observations_history.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(header) + "\n")
            for obs_idx, frame in enumerate(observations):
                row = _observation_history_row(
                    obs_idx,
                    frame,
                    origin_xy=origin_xy,
                    grid_resolution=res,
                    shape_hw=shape_hw,
                    world_xy_to_grid_ij=world_xy_to_grid_ij,
                )
                fh.write(json.dumps(row) + "\n")
        out["observations_history"] = str(path)

    if export_pickle and hasattr(vm, "write_to_pickle"):
        pkl_path = root / "voxel_debug.pkl"
        vm.write_to_pickle(str(pkl_path))
        if pkl_path.is_file():
            out["voxel_debug_pickle"] = str(pkl_path)

    if spawn_record and "spawn_record" not in out:
        spawn_path = root / "spawn_record.json"
        spawn_path.write_text(json.dumps(spawn_record, indent=2) + "\n", encoding="utf-8")
        out["spawn_record"] = str(spawn_path)
    return out


def _observation_history_row(
    obs_idx: int,
    frame: Any,
    *,
    origin_xy: np.ndarray,
    grid_resolution: float,
    shape_hw: tuple[int, int],
    world_xy_to_grid_ij: Any,
) -> dict[str, Any]:
    camera_t = _pose_translation_xyz(frame.camera_pose)
    base_pose = _to_numpy_f64(frame.base_pose)
    base_xyt: list[float] | None = None
    gps_grid_ij: list[int] | None = None
    if base_pose is not None and base_pose.size >= 2:
        theta = float(base_pose[2]) if base_pose.size >= 3 else 0.0
        base_xyt = [float(base_pose[0]), float(base_pose[1]), theta]
        gi, gj = world_xy_to_grid_ij(base_xyt[:2], origin_xy, grid_resolution, shape_hw)
        gps_grid_ij = [gi, gj]

    cam_xy = (float(camera_t[0]), float(camera_t[1]))
    # Voxel-map world uses planar (Habitat X, Habitat Z) on axes 0/1; axis 2 is floor-relative height.
    cam_planar = cam_xy
    cam_ij_xy = list(world_xy_to_grid_ij(cam_xy, origin_xy, grid_resolution, shape_hw))
    cam_ij_xz = list(world_xy_to_grid_ij(cam_planar, origin_xy, grid_resolution, shape_hw))

    pcd = _to_numpy_f64(frame.full_world_xyz)
    pcd_centroid_xy: list[float] | None = None
    pcd_centroid_xz: list[float] | None = None
    n_world_points = 0
    if pcd is not None and pcd.ndim == 2 and pcd.shape[0] > 0 and pcd.shape[1] >= 3:
        n_world_points = int(pcd.shape[0])
        pcd_centroid_xy = [float(pcd[:, 0].mean()), float(pcd[:, 1].mean())]
        pcd_centroid_xz = [float(pcd[:, 0].mean()), float(pcd[:, 1].mean())]

    depth_stats = _depth_summary(frame.depth)
    return {
        "type": "observation",
        "obs_idx": obs_idx,
        "camera_pose_t": camera_t,
        "base_pose_xyt": base_xyt,
        "gps_grid_ij": gps_grid_ij,
        "camera_grid_ij": cam_ij_xy,
        "camera_grid_ij_xz": cam_ij_xz,
        "pcd_centroid_xy": pcd_centroid_xy,
        "pcd_centroid_xz": pcd_centroid_xz,
        "depth_valid_frac": depth_stats["valid_frac"],
        "depth_median_m": depth_stats["median_m"],
        "n_world_points": n_world_points,
        "gps_camera_grid_delta_ij": (
            [cam_ij_xy[0] - gps_grid_ij[0], cam_ij_xy[1] - gps_grid_ij[1]]
            if gps_grid_ij is not None
            else None
        ),
    }


def _pose_translation_xyz(pose: Any) -> list[float]:
    arr = _to_numpy_f64(pose)
    if arr is None:
        return [0.0, 0.0, 0.0]
    if arr.shape == (4, 4):
        t = arr[:3, 3]
    elif arr.size >= 3:
        t = arr.reshape(-1)[:3]
    else:
        t = np.zeros(3, dtype=np.float64)
    return [float(t[0]), float(t[1]), float(t[2])]


def _to_numpy_f64(arr: Any) -> np.ndarray | None:
    if arr is None:
        return None
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    out = np.asarray(arr, dtype=np.float64)
    if out.size == 0:
        return None
    return out


def _depth_summary(depth: Any) -> dict[str, float | None]:
    arr = _to_numpy_f64(depth)
    if arr is None:
        return {"valid_frac": None, "median_m": None}
    flat = arr.reshape(-1)
    finite = np.isfinite(flat)
    if not finite.any():
        return {"valid_frac": 0.0, "median_m": None}
    valid = flat[finite]
    positive = valid[valid > 0.0]
    valid_frac = float(positive.size / max(1, flat.size))
    median_m = float(np.median(positive)) if positive.size else None
    return {"valid_frac": valid_frac, "median_m": median_m}



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
    except Exception as exc:
        logger.warning(f"object crop export failed: {exc}")
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
    except Exception as exc:
        logger.warning(f"graph checkpoint export failed: {exc}")
        return None


def _write_floor_metrics(agent: Any, root: Path) -> Path | None:
    vm = getattr(agent, "voxel_map", None)
    if vm is None:
        return None
    try:
        from emet.memory.floor_metrics import compute_explored_floor_metrics, write_floor_metrics_json

        metrics = compute_explored_floor_metrics(vm)
        return write_floor_metrics_json(root, metrics)
    except Exception as exc:
        logger.warning(f"floor metrics export failed: {exc}")
        return None


def _write_map_exploration_mp4(
    root: Path,
    *,
    fps: float,
    prefer_overlay: bool = True,
) -> Path | None:
    """Encode ``maps/overlay_step_*.png`` or ``maps/step_*.png`` to ``topdown_exploration.mp4``."""
    maps_dir = root / "maps"
    if not maps_dir.is_dir():
        return None
    pattern = "overlay_step_*.png" if prefer_overlay else "step_*.png"
    paths = sorted(maps_dir.glob(pattern))
    if not paths and prefer_overlay:
        paths = sorted(maps_dir.glob("step_*.png"))
    if len(paths) < 2:
        return None
    try:
        from emet.eval.episode_video import write_png_sequence_mp4

        out = root / "topdown_exploration.mp4"
        return write_png_sequence_mp4(paths, out, fps=fps)
    except Exception as exc:
        logger.warning(f"map exploration MP4 export failed: {exc}")
        return None


def _write_episode_mp4(root: Path, *, cfg: EpisodeDiagnosticsConfig) -> Path | None:
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        return None
    try:
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

        from emet.eval.episode_video import write_episode_mp4_from_metadata

        return write_episode_mp4_from_metadata(
            root,
            fps=cfg.video_fps,
            filename="episode_rgb.mp4",
            motion_paced=cfg.video_motion_paced,
            meters_per_repeat=cfg.video_meters_per_frame,
            radians_per_repeat=cfg.video_radians_per_frame,
            crossfade_teleport_m=cfg.video_crossfade_teleport_m,
        )
    except Exception as exc:
        logger.warning(f"episode MP4 export failed: {exc}")
        return None
