# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Common memory save/load format: directory layout, MemoryState schema,
# and save_memory / load_memory for use by Dynamem, SVM, GraphEQA and read_map.

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

MEMORY_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
POINT_CLOUD_FILENAME = "point_cloud.npz"
FRAMES_FILENAME = "frames.pkl"
FRAMES_DIR = "frames"
FRAME_RGB_FILENAME = "rgb.png"  # legacy per-subdirectory name
FRAME_DEPTH_FILENAME = "depth.npy"  # legacy per-subdirectory name
FRAME_POSE_FILENAME = "pose.npz"  # legacy per-subdirectory name
GRAPH_FILENAME = "graph.json"
SCENE_GRAPH_REPORT_TXT = "scene_graph_report.txt"
DESCRIPTIONS_FILENAME = "descriptions.json"
USER_MESSAGES_FILENAME = "user_messages.json"


@dataclass
class PointCloudBlob:
    """Point cloud data: xyz, rgb; optional feats, weights, obs_id (DynaMem)."""

    xyz: np.ndarray  # (N, 3)
    rgb: Optional[np.ndarray] = None  # (N, 3) or (N,) for scalar
    feats: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    obs_id: Optional[np.ndarray] = None  # per-point observation id (DynaMem)


@dataclass
class FrameBlob:
    """One observation frame: pose, image, optional depth/feats/instance."""

    camera_pose: np.ndarray  # (4, 4)
    base_pose: Optional[np.ndarray] = None  # (3,) xyt
    camera_K: Optional[np.ndarray] = None  # (3, 3)
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    feats: Optional[np.ndarray] = None
    world_xyz: Optional[np.ndarray] = None
    instance: Optional[np.ndarray] = None
    instance_classes: Optional[np.ndarray] = None
    instance_scores: Optional[np.ndarray] = None
    info: Optional[Dict[str, Any]] = None


@dataclass
class GraphNodeView:
    """Serializable graph node (labels, xyz, obs_id)."""

    node_id: int
    labels: List[str]
    xyz: List[float]  # [x, y, z]
    obs_id: int


@dataclass
class GraphEdgeView:
    """Serializable graph edge."""

    id1: int
    id2: int
    relation: str


@dataclass
class GraphBlob:
    """Graph: nodes and edges (GraphEQA)."""

    nodes: List[GraphNodeView]
    edges: List[GraphEdgeView]


@dataclass
class UserMessageBlob:
    """One user text message with context (identity, robot location, timestamp)."""

    text: str
    user_identity: Optional[str] = None  # e.g. "user", "operator", user id
    robot_location: Optional[List[float]] = None  # (x, y) or (x, y, theta) when received
    timestamp: Optional[str] = None  # ISO8601 or unix time string
    extra: Optional[Dict[str, Any]] = None


@dataclass
class MemoryManifest:
    """Manifest for the memory directory."""

    version: int = MEMORY_FORMAT_VERSION
    backend: str = "unified"  # dynamem | svm | graph_eqa | unified
    created_at: Optional[str] = None
    description: Optional[str] = None
    has_point_cloud: bool = True
    has_frames: bool = True
    has_graph: bool = False
    has_text_descriptions: bool = False
    has_user_messages: bool = False
    frames_inline: bool = True  # True = frames in frames.pkl; False = frames/0.npz etc.
    compressed: bool = False  # for rgb/depth in frames


@dataclass
class MemoryState:
    """In-memory representation of saved memory. Used by Rerun backend and loaders.

    Directory layout (on disk):
      <path>/
        manifest.json           -- version, backend, which blobs exist
        point_cloud.npz         -- combined_xyz, combined_rgb, optional feats/weights/obs_id
        frames/rgb_0001.png     -- RGB image per frame (zero-padded index)
        frames/depth_0001.npy   -- depth per frame
        frames/pose_0001.npz    -- camera_pose, base_pose, camera_K per frame
        graph.json              -- optional; nodes + edges
        descriptions.json       -- optional; text per observation
        user_messages.json      -- optional; user text messages (identity, location, timestamp)
    """

    point_cloud: Optional[PointCloudBlob] = None
    frames: List[FrameBlob] = field(default_factory=list)
    graph: Optional[GraphBlob] = None
    text_descriptions: Optional[List[str]] = None
    user_messages: List[UserMessageBlob] = field(default_factory=list)
    # For 2D map derivation (obstacles, explored) when loading for display
    grid_origin: Optional[np.ndarray] = None  # (3,) or (2,)
    grid_resolution: float = 0.05
    grid_size: Optional[Tuple[int, int]] = None
    # Precomputed 2D maps if available (from voxel_map.get_2d_map())
    obstacles_2d: Optional[np.ndarray] = None
    explored_2d: Optional[np.ndarray] = None
    manifest: Optional[MemoryManifest] = None


def _to_native(obj: Any) -> Any:
    """Convert numpy/torch to native for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    return obj


def _load_frame(rgb_file: Optional[Path], depth_file: Optional[Path], pose_file: Path) -> FrameBlob:
    """Load a single frame from its component files."""
    pose_npz = np.load(pose_file, allow_pickle=True)
    camera_pose = pose_npz["camera_pose"]
    base_pose = pose_npz["base_pose"] if "base_pose" in pose_npz.files else None
    camera_K = pose_npz["camera_K"] if "camera_K" in pose_npz.files else None
    rgb = None
    if rgb_file is not None and rgb_file.exists():
        bgr = cv2.imread(str(rgb_file))
        rgb = bgr[:, :, ::-1].copy() if bgr is not None else None
    depth = None
    if depth_file is not None and depth_file.exists():
        depth = np.load(depth_file)
    return FrameBlob(
        camera_pose=camera_pose,
        base_pose=base_pose,
        camera_K=camera_K,
        rgb=rgb,
        depth=depth,
    )


def _load_frames_dir(frames_path: Path) -> List[FrameBlob]:
    """Load frames from frames/ directory. Supports two layouts:

    Flat (current):  frames/rgb_0001.png, frames/depth_0001.npy, frames/pose_0001.npz
    Legacy:          frames/0/rgb.png,    frames/0/depth.npy,    frames/0/pose.npz
    """
    frames: List[FrameBlob] = []

    # Detect flat layout: look for pose_NNNN.npz files
    flat_poses = sorted(frames_path.glob("pose_*.npz"))
    if flat_poses:
        for pose_file in flat_poses:
            tag = pose_file.stem.replace("pose_", "")  # e.g. "0001"
            rgb_file = frames_path / f"rgb_{tag}.png"
            depth_file = frames_path / f"depth_{tag}.npy"
            frames.append(_load_frame(rgb_file, depth_file, pose_file))
        return frames

    # Legacy layout: numbered subdirectories
    frame_dirs = sorted(
        [d for d in frames_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    for frame_dir in frame_dirs:
        pose_file = frame_dir / FRAME_POSE_FILENAME
        if not pose_file.exists():
            continue
        rgb_file = frame_dir / FRAME_RGB_FILENAME
        depth_file = frame_dir / FRAME_DEPTH_FILENAME
        frames.append(_load_frame(rgb_file, depth_file, pose_file))

    return frames


def save_memory(state: MemoryState, path: str) -> None:
    """Write MemoryState to a directory. Creates path if needed.

    Layout: manifest.json, point_cloud.npz, frames/rgb_NNNN.png + depth_NNNN.npy + pose_NNNN.npz,
    optional graph.json, optional descriptions.json.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    manifest = state.manifest or MemoryManifest()
    if manifest.created_at is None:
        manifest.created_at = datetime.utcnow().isoformat() + "Z"
    manifest.has_point_cloud = state.point_cloud is not None
    manifest.has_frames = len(state.frames) > 0
    manifest.has_graph = state.graph is not None and (
        len(state.graph.nodes) > 0 or len(state.graph.edges) > 0
    )
    manifest.has_text_descriptions = (
        state.text_descriptions is not None and len(state.text_descriptions) > 0
    )
    manifest.has_user_messages = len(state.user_messages) > 0
    manifest.frames_inline = False  # we use frames/<i>/ layout with PNG

    with open(path / MANIFEST_FILENAME, "w") as f:
        json.dump(asdict(manifest), f, indent=2)

    if state.point_cloud is not None:
        pc = state.point_cloud
        save_dict = {"xyz": pc.xyz}
        if pc.rgb is not None:
            save_dict["rgb"] = pc.rgb
        if pc.feats is not None:
            save_dict["feats"] = pc.feats
        if pc.weights is not None:
            save_dict["weights"] = pc.weights
        if pc.obs_id is not None:
            save_dict["obs_id"] = pc.obs_id
        np.savez_compressed(path / POINT_CLOUD_FILENAME, **save_dict)

    if state.frames:
        frames_path = path / FRAMES_DIR
        frames_path.mkdir(exist_ok=True)
        for i, fr in enumerate(state.frames):
            tag = f"{i:04d}"
            if fr.rgb is not None:
                rgb = np.asarray(fr.rgb)
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else np.clip(rgb, 0, 255).astype(np.uint8)
                bgr = rgb[:, :, ::-1].copy()
                cv2.imwrite(str(frames_path / f"rgb_{tag}.png"), bgr)
            if fr.depth is not None:
                np.save(frames_path / f"depth_{tag}.npy", np.asarray(fr.depth))
            pose_dict = {"camera_pose": fr.camera_pose}
            if fr.base_pose is not None:
                pose_dict["base_pose"] = fr.base_pose
            if fr.camera_K is not None:
                pose_dict["camera_K"] = fr.camera_K
            np.savez(frames_path / f"pose_{tag}.npz", **pose_dict)

    if state.graph is not None and (
        len(state.graph.nodes) > 0 or len(state.graph.edges) > 0
    ):
        graph_data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "labels": n.labels,
                    "xyz": n.xyz,
                    "obs_id": n.obs_id,
                }
                for n in state.graph.nodes
            ],
            "edges": [
                {"id1": e.id1, "id2": e.id2, "relation": e.relation}
                for e in state.graph.edges
            ],
        }
        with open(path / GRAPH_FILENAME, "w") as f:
            json.dump(graph_data, f, indent=2)

    if state.text_descriptions:
        with open(path / DESCRIPTIONS_FILENAME, "w") as f:
            json.dump(state.text_descriptions, f, indent=2)

    if state.user_messages:
        messages_data = [
            {
                "text": m.text,
                "user_identity": m.user_identity,
                "robot_location": m.robot_location,
                "timestamp": m.timestamp,
                "extra": m.extra,
            }
            for m in state.user_messages
        ]
        with open(path / USER_MESSAGES_FILENAME, "w") as f:
            json.dump(messages_data, f, indent=2)


def load_memory(path: str) -> MemoryState:
    """Load a memory directory into MemoryState. Raises if path is not a directory or manifest missing."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Memory path is not a directory: {path}")

    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        m = json.load(f)
    manifest = MemoryManifest(**m)

    state = MemoryState(manifest=manifest)

    if manifest.has_point_cloud and (path / POINT_CLOUD_FILENAME).exists():
        npz = np.load(path / POINT_CLOUD_FILENAME, allow_pickle=True)
        xyz = npz["xyz"]
        rgb = npz["rgb"] if "rgb" in npz.files else None
        feats = npz["feats"] if "feats" in npz.files else None
        weights = npz["weights"] if "weights" in npz.files else None
        obs_id = npz["obs_id"] if "obs_id" in npz.files else None
        state.point_cloud = PointCloudBlob(
            xyz=xyz,
            rgb=rgb,
            feats=feats,
            weights=weights,
            obs_id=obs_id,
        )

    if manifest.has_frames:
        frames_path = path / FRAMES_DIR
        if frames_path.is_dir():
            state.frames = _load_frames_dir(frames_path)
        elif (path / FRAMES_FILENAME).exists():
            with open(path / FRAMES_FILENAME, "rb") as f:
                frames_data = pickle.load(f)
            for d in frames_data:
                state.frames.append(
                    FrameBlob(
                        camera_pose=d["camera_pose"],
                        base_pose=d.get("base_pose"),
                        camera_K=d.get("camera_K"),
                        rgb=d.get("rgb"),
                        depth=d.get("depth"),
                        feats=d.get("feats"),
                        world_xyz=d.get("world_xyz"),
                        instance=d.get("instance"),
                        instance_classes=d.get("instance_classes"),
                        instance_scores=d.get("instance_scores"),
                        info=d.get("info"),
                    )
                )

    if manifest.has_graph and (path / GRAPH_FILENAME).exists():
        with open(path / GRAPH_FILENAME) as f:
            graph_data = json.load(f)
        state.graph = GraphBlob(
            nodes=[GraphNodeView(**n) for n in graph_data.get("nodes", [])],
            edges=[GraphEdgeView(**e) for e in graph_data.get("edges", [])],
        )

    if manifest.has_text_descriptions and (path / DESCRIPTIONS_FILENAME).exists():
        with open(path / DESCRIPTIONS_FILENAME) as f:
            state.text_descriptions = json.load(f)

    if (path / USER_MESSAGES_FILENAME).exists():
        with open(path / USER_MESSAGES_FILENAME) as f:
            raw = json.load(f)
        for d in raw:
            state.user_messages.append(
                UserMessageBlob(
                    text=d["text"],
                    user_identity=d.get("user_identity"),
                    robot_location=d.get("robot_location"),
                    timestamp=d.get("timestamp"),
                    extra=d.get("extra"),
                )
            )

    return state


def is_memory_directory(path: str) -> bool:
    """Return True if path is a directory containing manifest.json."""
    p = Path(path)
    return p.is_dir() and (p / MANIFEST_FILENAME).exists()
