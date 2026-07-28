# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Common memory save/load format: directory layout, MemoryState schema,
# and save_memory / load_memory for use by Dynamem, SVM, GraphEQA and read_map.

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MEMORY_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
POINT_CLOUD_FILENAME = "point_cloud.npz"
VOXEL_PICKLE_FILENAME = "voxel_map.pkl"
FRAMES_FILENAME = "frames.pkl"
FRAMES_DIR = "frames"
FRAME_RGB_FILENAME = "rgb.png"  # legacy per-subdirectory name
FRAME_DEPTH_FILENAME = "depth.npy"  # legacy per-subdirectory name
FRAME_POSE_FILENAME = "pose.npz"  # legacy per-subdirectory name
GRAPH_FILENAME = "graph.json"
SCENE_GRAPH_REPORT_TXT = "scene_graph_report.txt"
# Open-vocab SceneGraphProcessor dump (``memory_backend=open_vocab`` lifelong dirs).
OPEN_VOCAB_SCENE_GRAPH_DIR = "open_vocab_scene_graph"
SIM_GT_PLACEMENTS_FILENAME = "sim_object_placements.json"
GT_ALIGNMENT_REPORT_TXT = "gt_alignment_report.txt"
DESCRIPTIONS_FILENAME = "descriptions.json"
USER_MESSAGES_FILENAME = "user_messages.json"


@dataclass
class PointCloudBlob:
    """Point cloud data: xyz, rgb; optional feats, weights, obs_id (DynaMem)."""

    xyz: np.ndarray  # (N, 3)
    rgb: np.ndarray | None = None  # (N, 3) or (N,) for scalar
    feats: np.ndarray | None = None
    weights: np.ndarray | None = None
    obs_id: np.ndarray | None = None  # per-point observation id (DynaMem)


@dataclass
class FrameBlob:
    """One observation frame: pose, image, optional depth/feats/instance."""

    camera_pose: np.ndarray  # (4, 4)
    base_pose: np.ndarray | None = None  # (3,) xyt
    camera_K: np.ndarray | None = None  # (3, 3)
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    feats: np.ndarray | None = None
    world_xyz: np.ndarray | None = None
    instance: np.ndarray | None = None
    instance_classes: np.ndarray | None = None
    instance_scores: np.ndarray | None = None
    # Optional structured detections (YoloE + centroids); saved as detections_NNNN.json
    detections: list[dict[str, Any]] | None = None
    gt_associations: list[dict[str, Any]] | None = None
    info: dict[str, Any] | None = None


@dataclass
class GraphNodeView:
    """Serializable graph node (labels, xyz, obs_id, optional description)."""

    node_id: int
    labels: list[str]
    xyz: list[float]  # [x, y, z]
    obs_id: int
    description: str | None = None
    body_key: str | None = None
    extent_half: list[float] | None = None
    bounds: list[list[float]] | None = None
    detection_label: str | None = None
    last_seen: int | None = None  # graph timestep when last observed (staleness)
    support_count: int | None = None
    is_viewpoint: bool = False
    is_frontier: bool = False
    belief_confidence: float | None = None
    position_covariance: list[list[float]] | None = None
    position_history: list[dict[str, Any]] | None = None
    identity_key: str | None = None
    change_events: list[dict[str, Any]] | None = None
    expected_absence_count: int = 0
    last_absence_step: int = -1


@dataclass
class GraphEdgeView:
    """Serializable graph edge."""

    id1: int
    id2: int
    relation: str
    confidence: float | None = None
    last_evidence_step: int | None = None
    contradiction_count: int = 0


@dataclass
class GraphBlob:
    """Graph: nodes and edges (GraphEQA)."""

    nodes: list[GraphNodeView]
    edges: list[GraphEdgeView]


@dataclass
class UserMessageBlob:
    """One user text message with context (identity, robot location, timestamp)."""

    text: str
    user_identity: str | None = None  # e.g. "user", "operator", user id
    robot_location: list[float] | None = None  # (x, y) or (x, y, theta) when received
    timestamp: str | None = None  # ISO8601 or unix time string
    extra: dict[str, Any] | None = None


@dataclass
class MemoryManifest:
    """Manifest for the memory directory."""

    version: int = MEMORY_FORMAT_VERSION
    backend: str = "unified"  # dynamem | svm | graph_eqa | unified
    created_at: str | None = None
    description: str | None = None
    has_point_cloud: bool = True
    has_frames: bool = True
    has_graph: bool = False
    has_text_descriptions: bool = False
    has_user_messages: bool = False
    frames_inline: bool = True  # True = frames in frames.pkl; False = frames/0.npz etc.
    compressed: bool = False  # for rgb/depth in frames
    has_instance_masks: bool = False  # frames/instance_*.npy (+ classes/scores) when True
    has_detection_json: bool = False  # frames/detections_*.json per frame when True
    has_world_xyz_maps: bool = False  # optional full HxWx3 world_xyz_*.npy per frame
    ground_truth_mode: bool = False
    has_sim_gt: bool = False
    sim_gt_placements_file: str | None = None
    has_gt_associations: bool = False
    # Controller observation step at save time; restored as obs_count / graph timestep so
    # staleness pruning does not drop reloaded nodes (lifelong checkpoint resume).
    final_step: int | None = None
    has_voxel_pickle: bool = False  # voxel_map.pkl alongside (SparseVoxelMapDynamem.write_to_pickle)
    has_open_vocab_scene_graph: bool = False  # open_vocab_scene_graph/ sidecar


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
        graph.json              -- optional; GraphEQA nodes + edges
        open_vocab_scene_graph/ -- optional; OpenVocabSceneGraph (scene_graph.json, crops, …)
        descriptions.json       -- optional; text per observation
        user_messages.json      -- optional; user text messages (identity, location, timestamp)
    """

    point_cloud: PointCloudBlob | None = None
    frames: list[FrameBlob] = field(default_factory=list)
    graph: GraphBlob | None = None
    text_descriptions: list[str] | None = None
    user_messages: list[UserMessageBlob] = field(default_factory=list)
    # For 2D map derivation (obstacles, explored) when loading for display
    grid_origin: np.ndarray | None = None  # (3,) or (2,)
    grid_resolution: float = 0.05
    grid_size: tuple[int, int] | None = None
    # Precomputed 2D maps if available (from voxel_map.get_2d_map())
    obstacles_2d: np.ndarray | None = None
    explored_2d: np.ndarray | None = None
    manifest: MemoryManifest | None = None


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


def _load_frame(rgb_file: Path | None, depth_file: Path | None, pose_file: Path) -> FrameBlob:
    """Load a single frame from its component files."""
    pose_npz = np.load(pose_file, allow_pickle=True)
    camera_pose = pose_npz["camera_pose"]
    base_pose = pose_npz["base_pose"] if "base_pose" in pose_npz.files else None
    camera_K = pose_npz["camera_K"] if "camera_K" in pose_npz.files else None
    world_xyz = pose_npz["world_xyz"] if "world_xyz" in pose_npz.files else None
    info = None
    if "labels" in pose_npz.files:
        labels_arr = pose_npz["labels"]
        info = {"labels": list(labels_arr.flat)} if labels_arr.size else {"labels": []}
        if "description" in pose_npz.files:
            desc_arr = pose_npz["description"]
            if desc_arr.size and desc_arr.flat[0] is not None:
                info["description"] = str(desc_arr.flat[0])
    rgb = None
    if rgb_file is not None and rgb_file.exists():
        bgr = cv2.imread(str(rgb_file))
        rgb = bgr[:, :, ::-1].copy() if bgr is not None else None
    depth = None
    if depth_file is not None and depth_file.exists():
        depth = np.load(depth_file)

    frames_dir = pose_file.parent
    tag = pose_file.stem.replace("pose_", "")
    instance = None
    instance_classes = None
    instance_scores = None
    p_i = frames_dir / f"instance_{tag}.npy"
    if p_i.exists():
        instance = np.load(p_i)
    p_ic = frames_dir / f"instance_classes_{tag}.npy"
    if p_ic.exists():
        instance_classes = np.load(p_ic)
    p_is = frames_dir / f"instance_scores_{tag}.npy"
    if p_is.exists():
        instance_scores = np.load(p_is)
    p_wmap = frames_dir / f"world_xyz_map_{tag}.npy"
    if p_wmap.exists():
        world_xyz = np.load(p_wmap)
    detections = None
    p_det = frames_dir / f"detections_{tag}.json"
    if p_det.exists():
        with open(p_det, encoding="utf-8") as f:
            detections = json.load(f)
    gt_associations = None
    p_gt = frames_dir / f"gt_assoc_{tag}.json"
    if p_gt.exists():
        with open(p_gt, encoding="utf-8") as f:
            gt_associations = json.load(f)

    return FrameBlob(
        camera_pose=camera_pose,
        base_pose=base_pose,
        camera_K=camera_K,
        rgb=rgb,
        depth=depth,
        world_xyz=world_xyz,
        instance=instance,
        instance_classes=instance_classes,
        instance_scores=instance_scores,
        detections=detections,
        gt_associations=gt_associations,
        info=info,
    )


def _load_frames_dir(frames_path: Path) -> list[FrameBlob]:
    """Load frames from frames/ directory. Supports two layouts:

    Flat (current):  frames/rgb_0001.png, frames/depth_0001.npy, frames/pose_0001.npz
    Legacy:          frames/0/rgb.png,    frames/0/depth.npy,    frames/0/pose.npz
    """
    frames: list[FrameBlob] = []

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
    manifest.has_graph = state.graph is not None and (len(state.graph.nodes) > 0 or len(state.graph.edges) > 0)
    manifest.has_text_descriptions = state.text_descriptions is not None and len(state.text_descriptions) > 0
    manifest.has_user_messages = len(state.user_messages) > 0
    manifest.frames_inline = False  # we use frames/<i>/ layout with PNG
    manifest.has_instance_masks = any(fr.instance is not None for fr in state.frames)
    manifest.has_detection_json = any(bool(fr.detections) for fr in state.frames)
    manifest.has_gt_associations = any(bool(fr.gt_associations) for fr in state.frames)
    manifest.has_world_xyz_maps = any(
        fr.world_xyz is not None and np.asarray(fr.world_xyz).ndim == 3 for fr in state.frames
    )

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
                    rgb = (
                        (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                        if rgb.max() <= 1.0
                        else np.clip(rgb, 0, 255).astype(np.uint8)
                    )
                bgr = rgb[:, :, ::-1].copy()
                cv2.imwrite(str(frames_path / f"rgb_{tag}.png"), bgr)
            if fr.depth is not None:
                np.save(frames_path / f"depth_{tag}.npy", np.asarray(fr.depth))
            pose_dict = {"camera_pose": fr.camera_pose}
            if fr.base_pose is not None:
                pose_dict["base_pose"] = fr.base_pose
            if fr.camera_K is not None:
                pose_dict["camera_K"] = fr.camera_K
            if fr.world_xyz is not None and fr.world_xyz.size >= 3:
                pose_dict["world_xyz"] = np.asarray(fr.world_xyz).reshape(-1, 3)[:1]
            if fr.info is not None and "labels" in fr.info:
                pose_dict["labels"] = np.array(fr.info["labels"], dtype=object)
            if fr.info is not None and fr.info.get("description"):
                pose_dict["description"] = np.array(fr.info["description"], dtype=object)
            np.savez(frames_path / f"pose_{tag}.npz", **pose_dict)

            if fr.instance is not None:
                np.save(frames_path / f"instance_{tag}.npy", np.asarray(fr.instance, dtype=np.int64))
            if fr.instance_classes is not None:
                np.save(frames_path / f"instance_classes_{tag}.npy", np.asarray(fr.instance_classes))
            if fr.instance_scores is not None:
                np.save(frames_path / f"instance_scores_{tag}.npy", np.asarray(fr.instance_scores))
            wx = fr.world_xyz
            if wx is not None and np.asarray(wx).ndim == 3:
                np.save(frames_path / f"world_xyz_map_{tag}.npy", np.asarray(wx, dtype=np.float32))
            if fr.detections:
                det_path = frames_path / f"detections_{tag}.json"
                with open(det_path, "w", encoding="utf-8") as df:
                    json.dump(_to_native(fr.detections), df, indent=2)
            if fr.gt_associations:
                gt_path = frames_path / f"gt_assoc_{tag}.json"
                with open(gt_path, "w", encoding="utf-8") as gf:
                    json.dump(_to_native(fr.gt_associations), gf, indent=2)

    if state.graph is not None and (len(state.graph.nodes) > 0 or len(state.graph.edges) > 0):
        graph_data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "labels": n.labels,
                    "xyz": n.xyz,
                    "obs_id": n.obs_id,
                    **({"description": n.description} if getattr(n, "description", None) else {}),
                    **({"body_key": n.body_key} if getattr(n, "body_key", None) else {}),
                    **({"extent_half": n.extent_half} if getattr(n, "extent_half", None) else {}),
                    **({"bounds": n.bounds} if getattr(n, "bounds", None) else {}),
                    **({"detection_label": n.detection_label} if getattr(n, "detection_label", None) else {}),
                    **({"last_seen": int(n.last_seen)} if getattr(n, "last_seen", None) is not None else {}),
                    **(
                        {"support_count": int(n.support_count)}
                        if getattr(n, "support_count", None) is not None
                        else {}
                    ),
                    **({"is_viewpoint": True} if getattr(n, "is_viewpoint", False) else {}),
                    **({"is_frontier": True} if getattr(n, "is_frontier", False) else {}),
                    **(
                        {"belief_confidence": float(n.belief_confidence)}
                        if getattr(n, "belief_confidence", None) is not None
                        else {}
                    ),
                    **(
                        {"position_covariance": n.position_covariance}
                        if getattr(n, "position_covariance", None) is not None
                        else {}
                    ),
                    **(
                        {"position_history": n.position_history}
                        if getattr(n, "position_history", None)
                        else {}
                    ),
                    **(
                        {"identity_key": n.identity_key}
                        if getattr(n, "identity_key", None)
                        else {}
                    ),
                    **(
                        {"change_events": n.change_events}
                        if getattr(n, "change_events", None)
                        else {}
                    ),
                    **(
                        {"expected_absence_count": int(n.expected_absence_count)}
                        if getattr(n, "expected_absence_count", 0)
                        else {}
                    ),
                    **(
                        {"last_absence_step": int(n.last_absence_step)}
                        if getattr(n, "last_absence_step", -1) >= 0
                        else {}
                    ),
                }
                for n in state.graph.nodes
            ],
            "edges": [
                {
                    "id1": e.id1,
                    "id2": e.id2,
                    "relation": e.relation,
                    **(
                        {"confidence": float(e.confidence)}
                        if getattr(e, "confidence", None) is not None
                        else {}
                    ),
                    **(
                        {"last_evidence_step": int(e.last_evidence_step)}
                        if getattr(e, "last_evidence_step", None) is not None
                        else {}
                    ),
                    **(
                        {"contradiction_count": int(e.contradiction_count)}
                        if getattr(e, "contradiction_count", 0)
                        else {}
                    ),
                }
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
    _manifest_field_names = {f.name for f in fields(MemoryManifest)}
    manifest = MemoryManifest(**{k: v for k, v in m.items() if k in _manifest_field_names})

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
                        detections=d.get("detections"),
                        gt_associations=d.get("gt_associations"),
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
