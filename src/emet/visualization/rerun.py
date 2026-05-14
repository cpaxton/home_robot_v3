#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
import socket
import time
import timeit
from dataclasses import replace
from typing import Any

import numpy as np
import rerun as rr

from emet.memory.format import MemoryState, PointCloudBlob


# Rerun's native viewer requires a display; use spawn=False when headless
def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET"))


import rerun.blueprint as rrb
import torch

from emet.core.interfaces import Observations
from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY, read_emet_session
from emet.mapping.scene_graph import SceneGraph
from emet.mapping.voxel.voxel_map import SparseVoxelMapNavigationSpace
from emet.motion import HelloStretchIdx
from emet.perception.wrapper import OvmmPerception
from emet.utils.geometry import nav_xyt_to_world_xyt
from emet.utils.logger import Logger
from emet.visualization import urdf_visualizer

logger = Logger(__name__)

# Canonical Rerun entity paths (use these for consistent logging across live and memory view):
#   world/point_cloud       Points3D
#   world/obstacles        Points3D (2D map obstacles)
#   world/explored         Points3D (2D map explored)
#   world/frames/<i>       Transform3D
#   world/frames/<i>/rgb   Image
#   world/frames/<i>/depth DepthImage
#   world/frames/current   (at each frame time) Transform3D, rgb, depth — scrub "frame" timeline for playback
#   world/graph/nodes      Points3D (graph nodes)
#   world/dynagraph/nodes  Points3D (Dynagraph graph nodes)
#   world/dynagraph/summary TextDocument (tree view)
#   world/memory/text      TextDocument
#   world/head_camera      Transform3D (optional); world/head_camera/rgb Image, world/head_camera/depth
#   world/ee_camera        same for end-effector camera
#   world/robot            Transform3D (base pose)
#   world/ee               Transform3D (end-effector pose)
#   world/xyz              Arrows3D (axes, static)
#   world/map_box          Boxes3D (static)


def decompose_homogeneous_matrix(homogeneous_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a 4x4 homogeneous transformation matrix into its rotation matrix and translation vector components.

    Args:
        homogeneous_matrix (numpy.ndarray): A 4x4 matrix representing a homogeneous transformation.

    Returns:
        tuple: A tuple containing:
            - rotation_matrix : A 3x3 matrix representing the rotation component.
            - translation_vector : A 1D array of length 3 representing the translation component.
    """
    if homogeneous_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")
    rotation_matrix = homogeneous_matrix[:3, :3]
    translation_vector = homogeneous_matrix[:3, 3]
    return rotation_matrix, translation_vector


def occupancy_map_to_indices(occupancy_map):
    """
    Convert a 2D occupancy map to an Nx3 array of float indices of occupied cells.

    Args:
    occupancy_map (np.ndarray): 2D boolean array where True represents occupied cells.

    Returns:
    np.ndarray: Nx3 float array where each row is [x, y, 0] of an occupied cell.
    """
    # Find the indices of occupied cells
    occupied_indices = np.where(occupancy_map)

    # Create the Nx3 array
    num_points = len(occupied_indices[0])
    xyz_array = np.zeros((num_points, 3), dtype=float)

    # Fill in x and y coordinates
    xyz_array[:, 0] = occupied_indices[0]  # x coordinates
    xyz_array[:, 1] = occupied_indices[1]  # y coordinates
    # z coordinates are already 0

    return xyz_array


def occupancy_map_to_3d_points(
    occupancy_map: np.ndarray,
    grid_center: np.ndarray | torch.Tensor,
    grid_resolution: float,
    offset: np.ndarray | None = np.zeros(3),
) -> np.ndarray:
    """
    Converts a 2D occupancy map to a list of 3D points.
    Args:
        occupancy_map: A 2D array boolean map
        grid_center: The (x, y, z) coordinates of the center of the grid map
        grid_resolution: The resolution of the grid map
        offset: The (x, y, z) offset to be added to the points

    Returns:
        np.ndarray: A array of 3D points representing the occupied cells in the world frame.
    """
    points = []
    rows, cols = occupancy_map.shape
    center_row, center_col, _ = grid_center

    if isinstance(grid_center, torch.Tensor):
        grid_center = grid_center.cpu().numpy()

    indices = occupancy_map_to_indices(occupancy_map)
    points = (indices - grid_center) * grid_resolution + offset
    return points


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to uint8 for Rerun; accept float [0,1] or [0,255] or int."""
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size > 0 and np.nanmax(arr) <= 1.0:
            arr = (arr * 255).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
    return arr.astype(np.uint8)


def log_to_rerun(topic_name, data, **kwargs):
    """
    Log data to rerun
    Args:
        topic_name (str): Topic name
        data (object): Data to log
    """
    rr.log(topic_name, rr.Clear(recursive=True))
    rr.log(topic_name, data, **kwargs)


class StretchURDFLogger(urdf_visualizer.URDFVisualizer):
    link_names = []
    link_poses = []

    def load_robot_mesh(self, cfg: dict = None, use_collision: bool = False):
        """
        Load robot mesh using urdf visualizer to rerun
        This is to be run once at the beginning of the rerun
        Args:
            cfg (dict): Configuration of the robot
            use_collision (bool): use collision mesh
        """
        trimesh_list = self.get_tri_meshes(cfg=cfg, use_collision=use_collision)
        self.link_names = trimesh_list["link"]
        self.link_poses = trimesh_list["pose"]
        for i in range(len(trimesh_list["link"])):
            rr.log(
                f"world/robot/mesh/{trimesh_list['link'][i]}",
                rr.Mesh3D(
                    vertex_positions=trimesh_list["mesh"][i].vertices,
                    triangle_indices=trimesh_list["mesh"][i].faces,
                    vertex_normals=trimesh_list["mesh"][i].vertex_normals,
                ),
                static=True,
            )

    def log_transforms(self, obs, debug: bool = False):
        """
        Log robot mesh using urdf visualizer to rerun
        Args:
            obs (dict): Observation dataclass
            use_collision (bool): use collision mesh
        """
        state = obs["joint"]
        cfg = {}
        for k in HelloStretchIdx.name_to_idx:
            cfg[k] = state[HelloStretchIdx.name_to_idx[k]]
        lk_cfg = {
            "joint_wrist_yaw": cfg["wrist_yaw"],
            "joint_wrist_pitch": cfg["wrist_pitch"],
            "joint_wrist_roll": cfg["wrist_roll"],
            "joint_lift": cfg["lift"],
            "joint_arm_l0": cfg["arm"] / 4,
            "joint_arm_l1": cfg["arm"] / 4,
            "joint_arm_l2": cfg["arm"] / 4,
            "joint_arm_l3": cfg["arm"] / 4,
            "joint_head_pan": cfg["head_pan"],
            "joint_head_tilt": cfg["head_tilt"],
        }
        if "gripper" in cfg.keys():
            lk_cfg["joint_gripper_finger_left"] = cfg["gripper"]
            lk_cfg["joint_gripper_finger_right"] = cfg["gripper"]
        t0 = timeit.default_timer()
        tms = self.get_tri_meshes(cfg=lk_cfg, use_collision=False)
        t1 = timeit.default_timer()
        self.link_poses = tms["pose"]
        self.link_names = tms["link"]
        for link in self.link_names:
            idx = self.link_names.index(link)
            rr.set_time_seconds("realtime", time.time())
            rr.log(
                f"world/robot/mesh/{link}",
                rr.Transform3D(
                    translation=self.link_poses[idx][:3, 3],
                    mat3x3=self.link_poses[idx][:3, :3],
                    axis_length=0.0,
                ),
            )
        t2 = timeit.default_timer()
        if debug:
            print("Time to get tri meshes (ms): ", 1000 * (t1 - t0))
            print("Time to log robot transforms (ms): ", 1000 * (t2 - t1))
            print("Total time to log robot transforms (ms): ", 1000 * (t2 - t0))


class NullVisualizer:
    """Drop-in replacement for RerunVisualizer that silently ignores all calls.

    Used when rerun is disabled (no display, headless without server, etc.) so
    that callers don't need null-checks at every call site.
    """

    enabled = False

    def __getattr__(self, name):
        return _null_noop


def _null_noop(*args, **kwargs):
    return None


def _sim_step_counter(obs: Any) -> int:
    """Simulation step from a ZMQ observation dict (``step``) or ``Observations.seq_id``."""
    if isinstance(obs, dict):
        s = obs.get("step")
        try:
            return int(s) if s is not None else -1
        except (TypeError, ValueError):
            return -1
    if isinstance(obs, Observations):
        sid = getattr(obs, "seq_id", -1)
        return int(sid) if sid >= 0 else -1
    return -1


def _pick_rerun_head_cam(obs: Any, servo: Observations | None) -> Observations | None:
    """Choose head RGB for Rerun from full ``obs`` vs low-latency ``servo``.

    Stretch: decoded servo ``Observations`` typically have no ``seq_id``; we keep preferring ``servo``
    for the higher-rate preview.

    Generic ZMQ: both full obs and servo carry ``step`` / ``seq_id``. Prefer the **full** observation
    whenever it is at least as new as servo (same step → full-res RGB). Use servo only when it is
    strictly newer (lower latency between socket arrivals).
    """
    obs_head: Observations | None = None
    if isinstance(obs, dict) and obs.get("rgb") is not None:
        obs_head = Observations.from_dict(obs)
    elif isinstance(obs, Observations) and obs.rgb is not None:
        obs_head = obs

    servo_ok = servo is not None and getattr(servo, "rgb", None) is not None
    if obs_head is None:
        return servo if servo_ok else None
    if not servo_ok:
        return obs_head

    obs_s = _sim_step_counter(obs)
    servo_s = int(servo.seq_id) if getattr(servo, "seq_id", -1) >= 0 else -1

    # Stretch-style: servo stream has no step metadata — prefer it when present.
    if servo_s < 0:
        return servo

    if obs_s < 0:
        return servo

    if servo_s > obs_s:
        return servo
    return obs_head


class RerunVisualizer:
    """Live Rerun logging for Stretch (mesh + joints) and generic ZMQ robots (cameras + base pose)."""

    enabled = True
    camera_point_radius = 0.01
    max_displayed_points_per_camera: int = 10000

    def __init__(
        self,
        display_robot_mesh: bool = True,
        spawn_gui: bool = False,
        open_browser: bool = True,
        headless: bool = False,
        server_memory_limit: str = "4GB",
        collapse_panels: bool = True,
        show_cameras_in_3d_view: bool = False,
        show_camera_point_clouds: bool = True,
        output_path=None,
        *,
        memory_view: bool = False,
        num_frames: int = 0,
        mjcf_robot: tuple[str, tuple[str, ...], int, str] | None = None,
        rerun_native_viewer: bool = False,
    ):
        """Rerun visualizer class
        Args:
            display_robot_mesh (bool): Display robot mesh (Stretch URDF) or MJCF skeleton when *mjcf_robot* is set
            spawn_gui (bool): If True, native Rerun desktop viewer (TCP). Default False (browser via ``rr.serve``).
            open_browser (bool): When using the web server, open a browser tab if a display exists.
            headless (bool): If True, no native viewer and no auto-open browser; use the :9090 URL manually.
            rerun_native_viewer (bool): Same as env ``RERUN_NATIVE_VIEWER=1``: use the native app, not the browser.
            server_memory_limit (str): Server memory limit E.g. 2GB or 20%
            collapse_panels (bool): Set to false to have customizable rerun panels
        """
        # RERUN_HEADLESS=1 forces no native viewer and no auto-open browser (web server only).
        if os.environ.get("RERUN_HEADLESS", "").lower() in ("1", "true", "yes"):
            headless = True
        want_native = bool(rerun_native_viewer) or (
            os.environ.get("RERUN_NATIVE_VIEWER", "").lower() in ("1", "true", "yes")
        )
        # RERUN_BIND_ALL=1 makes the server listen on 0.0.0.0 for remote viewing (Tailscale, etc.)
        if os.environ.get("RERUN_BIND_ALL", "").lower() in ("1", "true", "yes"):
            os.environ["RERUN_SERVER_HOST"] = "0.0.0.0"
            os.environ["RERUN_SERVER_WS_HOST"] = "0.0.0.0"

        if headless:
            spawn_gui = False
            open_browser = False
        elif want_native:
            if has_display():
                spawn_gui = True
                open_browser = False
            else:
                logger.warning(
                    "Native Rerun viewer requested but no DISPLAY/WAYLAND; starting web server only "
                    "(open http://<this-host>:9090?url=ws://<this-host>:9877 manually)."
                )
                spawn_gui = False
                open_browser = False
        elif spawn_gui or open_browser:
            if "DOCKER" in os.environ:
                spawn_gui = False
                open_browser = True
                logger.warning("Docker environment detected. Using web Rerun viewer.")
            if not has_display():
                spawn_gui = False
                open_browser = False
                logger.warning(
                    "No DISPLAY/WAYLAND set. Rerun web server only; open :9090 manually (or use SSH port forwarding)."
                )

        self.open_browser = open_browser
        rr.init("Stretch_robot", spawn=spawn_gui)

        if output_path is not None:
            rr.save(output_path / "rerun_log.rrd")
        # ``rr.serve`` streams over WebSockets (web UI on :9090). ``init(spawn=True)`` streams to the native
        # viewer over TCP (default :9876). Doing both routes logs only to the WebSocket sink, so the native
        # window stays empty while the browser looks fine. Serve only when we are not using a spawned native viewer.
        use_spawned_native = bool(spawn_gui) and not bool(headless)
        if not use_spawned_native:
            rr.serve(
                open_browser=open_browser and has_display() and not headless,
                server_memory_limit=server_memory_limit,
            )
            # Always print: logging is often WARNING+, and we want the URL even when the browser auto-opens.
            local_url = "http://127.0.0.1:9090?url=ws://127.0.0.1:9877"
            print(f"Rerun web viewer: {local_url}", flush=True)
            if os.environ.get("RERUN_BIND_ALL", "").lower() in ("1", "true", "yes"):
                hn = socket.gethostname()
                print(f"Rerun web viewer (LAN/remote): http://{hn}:9090?url=ws://{hn}:9877", flush=True)
        else:
            print(
                "Rerun: native desktop viewer (TCP). For the web UI omit --rerun-native / RERUN_NATIVE_VIEWER "
                "or set RERUN_HEADLESS=1, then use http://127.0.0.1:9090?url=ws://127.0.0.1:9877",
                flush=True,
            )

        self.display_robot_mesh = display_robot_mesh
        self.show_cameras_in_3d_view = show_cameras_in_3d_view
        self.show_camera_point_clouds = show_camera_point_clouds

        self.mjcf_skeleton = None
        self.urdf_logger = None
        if mjcf_robot is not None and display_robot_mesh:
            mjcf_path, joint_names, dof, base_link = mjcf_robot
            try:
                from emet.visualization.mjcf_rerun_robot import MjcfBodySkeletonLogger

                self.mjcf_skeleton = MjcfBodySkeletonLogger(mjcf_path, joint_names, dof, base_link)
            except Exception as e:
                logger.warning("MJCF Rerun robot skeleton disabled (%s).", e)
        if self.mjcf_skeleton is None and display_robot_mesh and mjcf_robot is None:
            self.urdf_logger = StretchURDFLogger()
            self.urdf_logger.load_robot_mesh(use_collision=False)

        # Create environment Box place holder
        rr.log(
            "world/map_box",
            rr.Boxes3D(half_sizes=[10, 10, 3], centers=[0, 0, 2], colors=[255, 255, 255, 255]),
            static=True,
        )
        # World Origin
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )
        # Seed realtime timeline so the viewer has a valid time when it opens (live SVM/Dynamem).
        if not memory_view:
            rr.set_time_seconds("realtime", time.time())

        self.bbox_colors_memory = {}
        self.step_delay_s = 0.3
        self.collapse_panels = collapse_panels
        self._memory_view = memory_view
        if memory_view:
            self.setup_memory_blueprint(collapse_panels, num_frames)
        else:
            self.setup_blueprint(collapse_panels)

    def setup_memory_blueprint(self, collapse_panels: bool, num_frames: int) -> None:
        """Blueprint for viewing saved memory: 3D, current-frame video (scrub timeline), frame strip, text."""
        # "Current frame" shows world/frames/current — log at each frame time for video-like scrub.
        frame_views = [
            rrb.Spatial2DView(name="current_frame", origin="world/frames/current"),
        ] + [
            rrb.Spatial2DView(name=f"frame_{i}", origin=f"world/frames/{i}")
            for i in range(min(num_frames, 16))  # cap at 16 panels
        ]
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                *frame_views,
                rrb.TextDocumentView(name="memory/text", origin="world/memory/text"),
            ),
            column_shares=[3, 1],
        )
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=collapse_panels,
        )
        self._memory_blueprint = my_blueprint
        # Defer sending so show_memory can log data first, then call send_memory_blueprint().
        if not self._memory_view:
            rr.send_blueprint(my_blueprint)

    def send_memory_blueprint(self) -> None:
        """Send the memory layout blueprint. Call after log_memory_state() when using memory_view."""
        if getattr(self, "_memory_blueprint", None) is not None:
            rr.send_blueprint(self._memory_blueprint)

    def setup_blueprint(self, collapse_panels: bool):
        """Setup the blueprint for the visualizer (memory view: 3D, 2D images, optional text).
        Args:
            collapse_panels (bool): fully hides the blueprint/selection panels,
                                    and shows the simplified time panel
        """
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="world/head_camera"),
                rrb.Spatial2DView(name="ee_rgb", origin="world/ee_camera"),
                rrb.TextDocumentView(name="memory/text", origin="world/memory/text"),
            ),
            column_shares=[3, 1],
        )
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=collapse_panels,
        )
        rr.send_blueprint(my_blueprint)

    def clear_identity(self, identity_name: str):
        """Clear existing rerun identity.

        This is useful if you want to clear a rerun identity and leave a blank there.
        Args:
            identity_name (str): rerun identity name
        """
        rr.log(identity_name, rr.Clear(recursive=True))

    def log_custom_2d_image(self, identity_name: str, img: np.ndarray | torch.Tensor):
        """Log custom 2d image

        Args:
            identity_name (str): rerun identity name
            img (2D or 3D array): the 2d image you want to log into rerun
        """
        if not self._memory_view:
            rr.set_time_seconds("realtime", time.time())
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img)
        log_to_rerun(identity_name, rr.Image(img))

    def log_text(self, identity_name: str, text: str):
        """Log a custom markdown text

        Args:
            identity_name (str): rerun identity name
            text (str): Markdown codes you want to log in rerun
        """
        if not self._memory_view:
            rr.set_time_seconds("realtime", time.time())
        rr.log(identity_name, rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))

    def log_arrow3D(
        self,
        identity_name: str,
        origins: list | list[list] | np.ndarray | torch.Tensor,
        vectors: list | list[list] | np.ndarray | torch.Tensor,
        colors: list | list[list] | np.ndarray | torch.Tensor,
        radii: float,
    ):
        """Log custom 3D arrows

        Args:
            identity_name (str): rerun identity name
            origins (a N x 3 array): origins of all 3D arrows
            vectors (a N x 3 array): directions and lengths of all 3D arrows
            colors (a N x 3 array): RGB colors of all 3D arrows
            radii (float): size of the arrows
        """
        # rr.init("Stretch_robot", spawn=(not self.open_browser))
        rr.log(
            identity_name,
            rr.Arrows3D(origins=origins, vectors=vectors, colors=colors, radii=radii),
        )

    def log_custom_pointcloud(
        self,
        identity_name: str,
        points: list | list[list] | np.ndarray | torch.Tensor,
        colors: list | list[list] | np.ndarray | torch.Tensor,
        radii: float,
    ):
        """Log custom 3D pointcloud

        Args:
            identity_name (str): rerun identity name
            points (a N x 3 array): xyz coordinates of all 3D points
            colors (a N x 3 array): RGB colors of all 3D points
            radii (float): size of the arrows
        """
        # rr.init("Stretch_robot", spawn=(not self.open_browser))
        log_to_rerun(
            identity_name,
            rr.Points3D(
                points,
                colors=colors,
                radii=radii,
            ),
        )

    def log_dynagraph_state(self, graph_memory: Any) -> None:
        """Log ``GraphEQAMemory`` nodes and tree summary under ``world/dynagraph/``."""
        if getattr(self, "_memory_view", False):
            return
        rr.set_time_seconds("realtime", time.time())
        log_to_rerun(
            "world/dynagraph/summary",
            rr.TextDocument(graph_memory.print_memory(), media_type=rr.MediaType.MARKDOWN),
        )
        nodes = graph_memory.get_nodes()
        if not nodes:
            self.clear_identity("world/dynagraph/nodes")
            return
        xyz = np.array([n.xyz for n in nodes], dtype=np.float64)
        labels: list[str] = []
        for n in nodes:
            lb = ", ".join(n.labels) if n.labels else str(n.node_id)
            sc = int(getattr(n, "support_count", 1))
            if sc != 1:
                lb = f"{lb} (x{sc})"
            labels.append(lb)
        log_to_rerun(
            "world/dynagraph/nodes",
            rr.Points3D(positions=xyz, radii=0.06, labels=labels),
        )

    def log_head_camera(self, obs: Observations, *, mapping_depth: np.ndarray | None = None):
        """Log head camera pose and images.

        Args:
            obs: Observation dataclass.
            mapping_depth: Optional H×W depth (meters), same shape as ``obs.rgb`` rows/cols. When provided,
                the head point cloud (and optional depth panel) uses this buffer so Rerun matches the
                depth map fused into DynaMem (e.g. DA3). Otherwise ZMQ ``obs.depth`` is used.
        """
        if obs is None or getattr(obs, "rgb", None) is None:
            return
        rr.set_time_seconds("realtime", time.time())
        rgb = np.ascontiguousarray(_rgb_to_uint8(obs.rgb))
        log_to_rerun("world/head_camera/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))

        cam = obs
        if mapping_depth is not None and rgb.ndim == 3 and rgb.shape[2] == 3:
            md = np.asarray(mapping_depth, dtype=np.float32)
            if md.ndim == 2 and md.shape == tuple(rgb.shape[:2]):
                cam = replace(obs, depth=md, xyz=None)

        if self.show_camera_point_clouds:
            head_xyz = cam.get_xyz_in_world_frame()
            if head_xyz is not None:
                head_xyz = head_xyz.reshape(-1, 3)
                head_rgb = rgb.reshape(-1, 3)
                log_to_rerun(
                    "world/head_camera/points",
                    rr.Points3D(
                        positions=head_xyz,
                        radii=np.ones(head_xyz.shape[:2]) * self.camera_point_radius,
                        colors=np.int64(head_rgb),
                    ),
                )
        else:
            dvis = cam.depth if cam.depth is not None else obs.depth
            if dvis is not None:
                log_to_rerun("world/head_camera/depth", rr.depthimage(dvis))

        if self.show_cameras_in_3d_view and getattr(obs, "camera_pose", None) is not None:
            rot, trans = decompose_homogeneous_matrix(obs.camera_pose)
            log_to_rerun("world/head_camera", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3))
            log_to_rerun(
                "world/head_camera",
                rr.Pinhole(
                    resolution=[rgb.shape[1], rgb.shape[0]],
                    image_from_camera=obs.camera_K,
                    image_plane_distance=0.15,
                ),
            )

    def log_robot_xyt(self, obs: Observations):
        """Log robot base pose in **MuJoCo / map world** when ``navigation_origin_xyt`` is in ``emet_session``.

        Otherwise uses raw ``gps``/``compass`` (episode-relative on sim servers, or client-local otherwise).
        """
        xy = np.asarray(obs["gps"], dtype=float).reshape(-1)[:2]
        comp = np.asarray(obs["compass"], dtype=float).ravel()
        theta = float(comp[0]) if comp.size else 0.0
        sess = read_emet_session(obs)
        wxyt = nav_xyt_to_world_xyt(np.array([float(xy[0]), float(xy[1]), theta], dtype=np.float64), sess)
        xy_w = np.asarray(wxyt[:2], dtype=float).reshape(-1)
        theta_w = float(wxyt[2])
        # Live streaming: static=True pins the entity to a single value in the viewer timeline.
        static_pose = bool(getattr(self, "_memory_view", False))
        rb_arrow = rr.Arrows3D(
            origins=[0, 0, 0],
            vectors=[0.4, 0, 0],
            radii=0.02,
            labels="robot",
            colors=[255, 0, 0, 255],
        )
        rr.log("world/robot/arrow", rb_arrow, static=static_pose)
        rr.log(
            "world/robot/blob",
            rr.Points3D([0, 0, 0], colors=[255, 0, 0, 255], radii=0.13),
            static=static_pose,
        )
        rr.log(
            "world/robot",
            rr.Transform3D(
                translation=[float(xy_w[0]), float(xy_w[1]), 0],
                rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=theta_w),
                axis_length=0.7,
            ),
            static=static_pose,
        )

    def log_ee_frame(self, obs):
        """log end effector pose
        Args:
            obs (Observations): Observation dataclass
        """
        # rr.set_time_seconds("realtime", time.time())
        # EE Frame
        if "ee_pose" not in obs:
            return
        if obs["ee_pose"] is None:
            return
        rot, trans = decompose_homogeneous_matrix(obs["ee_pose"])
        rr.Arrows3D(origins=[0, 0, 0], vectors=[0.2, 0, 0], radii=0.02, labels="ee", colors=[0, 255, 0, 255])
        # log_to_rerun("world/ee/arrow", ee_arrow)
        rr.log("world/ee", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3))

    def log_ee_camera(self, servo):
        """Log end effector camera pose and images
        Args:
            servo (Servo): Servo observation dataclass
        """
        if servo is None:
            return
        if getattr(servo, "ee_rgb", None) is None or getattr(servo, "ee_depth", None) is None:
            return
        rr.set_time_seconds("realtime", time.time())

        # EE Camera
        ee_rgb = np.ascontiguousarray(_rgb_to_uint8(servo.ee_rgb))
        log_to_rerun("world/ee_camera/rgb", rr.Image(ee_rgb, color_model=rr.ColorModel.RGB))

        if self.show_camera_point_clouds:
            ee_xyz = servo.get_ee_xyz_in_world_frame().reshape(-1, 3)
            ee_rgb = servo.ee_rgb.reshape(-1, 3)
            # Remove points below z = 0
            # and where distance from camera > 2 meters
            idx_depth = servo.ee_depth.reshape(-1) < 2
            idx_z = np.where(ee_xyz[:, 2] > 0)
            idx = np.intersect1d(idx_depth, idx_z)
            ee_xyz = ee_xyz[idx]
            ee_rgb = ee_rgb[idx]
            if self.max_displayed_points_per_camera > 0:
                idx = np.arange(ee_xyz.shape[0])
                np.random.shuffle(idx)
                ee_xyz = ee_xyz[idx[: self.max_displayed_points_per_camera]]
                ee_rgb = ee_rgb[idx[: self.max_displayed_points_per_camera]]
            log_to_rerun(
                "world/ee_camera/points",
                rr.Points3D(
                    positions=ee_xyz,
                    radii=np.ones(ee_xyz.shape[:2]) * self.camera_point_radius,
                    colors=np.int64(ee_rgb),
                ),
            )
        else:
            log_to_rerun("world/ee_camera/depth", rr.depthimage(servo.ee_depth))

        if self.show_cameras_in_3d_view:
            rot, trans = decompose_homogeneous_matrix(servo.ee_camera_pose)
            log_to_rerun("world/ee_camera", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3))
            log_to_rerun(
                "world/ee_camera",
                rr.Pinhole(
                    resolution=[servo.ee_rgb.shape[1], servo.ee_rgb.shape[0]],
                    image_from_camera=servo.ee_camera_K,
                    image_plane_distance=0.15,
                ),
            )

    def log_robot_state(self, obs):
        """Log robot joint states"""
        rr.set_time_seconds("realtime", time.time())
        state = obs["joint"]
        for k in HelloStretchIdx.name_to_idx:
            rr.log(
                f"robot_state/joint_pose/{k}",
                rr.Scalar(state[HelloStretchIdx.name_to_idx[k]]),
                static=True,
            )

    def log_robot_transforms(self, obs):
        """
        Log robot mesh transforms using urdf visualizer"""
        self.urdf_logger.log_transforms(obs)

    def log_memory_state(
        self,
        state: MemoryState,
        *,
        explored_radius: float = 0.025,
        obstacle_radius: float = 0.05,
        world_radius: float = 0.03,
        static: bool = False,
    ) -> None:
        """Log a MemoryState to Rerun (point cloud, 2D maps, frames, graph, text).

        Single representation for both live agents and read_map loaded memory.
        Use static=True when viewing a saved memory so data shows regardless of timeline.
        """
        if not static:
            rr.set_time_seconds("realtime", time.time())
        log_kw = {"static": True} if static else {}

        # Clear 2D map entities if we won't log them, so loaded state without 2D doesn't show stale data
        if state.grid_origin is None or state.obstacles_2d is None:
            self.clear_identity("world/obstacles")
        if state.grid_origin is None or state.explored_2d is None:
            self.clear_identity("world/explored")

        if state.point_cloud is not None:
            pc = state.point_cloud
            xyz = pc.xyz
            if hasattr(xyz, "cpu"):
                xyz = xyz.cpu().numpy()
            xyz = np.asarray(xyz, dtype=np.float64)
            rgb = pc.rgb
            if rgb is not None and hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            if rgb is None:
                rgb = np.ones((xyz.shape[0], 3), dtype=np.uint8) * 128
            else:
                rgb = _rgb_to_uint8(rgb)
            n = xyz.shape[0]
            log_to_rerun(
                "world/point_cloud",
                rr.Points3D(
                    positions=xyz,
                    radii=np.ones(n) * world_radius,
                    colors=rgb,
                ),
                **log_kw,
            )

        grid_origin = state.grid_origin
        grid_resolution = state.grid_resolution
        obstacles_2d = state.obstacles_2d
        explored_2d = state.explored_2d
        if grid_origin is not None and obstacles_2d is not None:
            if hasattr(grid_origin, "cpu"):
                grid_origin = grid_origin.cpu().numpy()
            obs_points = np.array(occupancy_map_to_3d_points(obstacles_2d, grid_origin, grid_resolution))
            obs_points[:, 2] += 0.01
            n_obs = obs_points.shape[0]
            rr.log(
                "world/obstacles",
                rr.Points3D(
                    positions=obs_points,
                    radii=np.ones(n_obs) * obstacle_radius,
                    colors=[255, 0, 0],
                ),
                **log_kw,
            )
        if grid_origin is not None and explored_2d is not None:
            if hasattr(grid_origin, "cpu"):
                grid_origin = grid_origin.cpu().numpy()
            explored_points = np.array(occupancy_map_to_3d_points(explored_2d, grid_origin, grid_resolution))
            explored_points[:, 2] -= 0.01
            n_exp = explored_points.shape[0]
            rr.log(
                "world/explored",
                rr.Points3D(
                    positions=explored_points,
                    radii=np.ones(n_exp) * explored_radius,
                    colors=[255, 255, 255],
                ),
                **log_kw,
            )

        for i, frame in enumerate(state.frames):
            rot, trans = decompose_homogeneous_matrix(frame.camera_pose)
            log_to_rerun(
                f"world/frames/{i}",
                rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.2),
                **log_kw,
            )
            if frame.rgb is not None:
                rgb = np.asarray(frame.rgb)
                if rgb.ndim == 3 and rgb.shape[2] == 3:
                    rgb = _rgb_to_uint8(rgb)
                log_to_rerun(f"world/frames/{i}/rgb", rr.Image(rgb), **log_kw)
            if frame.depth is not None:
                depth = np.asarray(frame.depth)
                if depth.ndim == 2:
                    log_to_rerun(f"world/frames/{i}/depth", rr.DepthImage(depth), **log_kw)

        if state.graph is not None and state.graph.nodes:
            nodes = state.graph.nodes
            xyz = np.array([n.xyz for n in nodes], dtype=np.float64)
            labels = [", ".join(n.labels) if n.labels else str(n.node_id) for n in nodes]
            rr.log(
                "world/graph/nodes",
                rr.Points3D(positions=xyz, radii=0.05, labels=labels),
                **log_kw,
            )

        parts = []
        if state.text_descriptions:
            parts.append("\n\n".join(state.text_descriptions))
        if state.user_messages:
            from emet.memory.format import UserMessageBlob

            lines = ["## User messages"]
            for m in state.user_messages:
                if not isinstance(m, UserMessageBlob):
                    lines.append(f"- {m}")
                    continue
                meta = []
                if m.timestamp:
                    meta.append(m.timestamp)
                if m.user_identity:
                    meta.append(f"**{m.user_identity}**")
                if m.robot_location:
                    loc = m.robot_location
                    if len(loc) >= 3:
                        meta.append(f"robot (x={loc[0]:.2f}, y={loc[1]:.2f}, θ={loc[2]:.2f})")
                    elif len(loc) >= 2:
                        meta.append(f"robot (x={loc[0]:.2f}, y={loc[1]:.2f})")
                head = " | ".join(meta) if meta else ""
                lines.append(f"- {head}\n  {m.text}" if head else f"- {m.text}")
            parts.append("\n".join(lines))
        # Robot state over time (base pose per frame); scrub the 'frame' timeline in the viewer to see it.
        if state.frames:
            state_lines = [
                "## Robot state over time",
                "Scrub the **frame** timeline to see the robot move and the current-frame image.",
                "",
            ]
            for i, fr in enumerate(state.frames):
                xyt = fr.base_pose
                if xyt is not None:
                    xyt = np.asarray(xyt).ravel()
                    if len(xyt) >= 3:
                        state_lines.append(f"- **Frame {i}**: base (x={xyt[0]:.2f}, y={xyt[1]:.2f}, θ={xyt[2]:.2f})")
                    elif len(xyt) >= 2:
                        state_lines.append(f"- **Frame {i}**: base (x={xyt[0]:.2f}, y={xyt[1]:.2f})")
                else:
                    trans = fr.camera_pose[:3, 3] if fr.camera_pose is not None else None
                    if trans is not None:
                        state_lines.append(
                            f"- **Frame {i}**: camera at (x={trans[0]:.2f}, y={trans[1]:.2f}, z={trans[2]:.2f})"
                        )
            parts.append("\n".join(state_lines))
        if parts:
            rr.log(
                "world/memory/text",
                rr.TextDocument("\n\n---\n\n".join(parts), media_type=rr.MediaType.MARKDOWN),
                **log_kw,
            )
        else:
            rr.log(
                "world/memory/text",
                rr.TextDocument(
                    "*No text or commands recorded for this memory.*\n\n"
                    "To see commands and monologue here, save memory with `user_messages` and "
                    "`text_descriptions` populated (e.g. from the dynamem/agent run).",
                    media_type=rr.MediaType.MARKDOWN,
                ),
                **log_kw,
            )

        # Frame timeline: log robot pose and current-frame image at each time for scrub playback.
        if static and state.frames:
            for i, frame in enumerate(state.frames):
                rr.set_time_sequence("frame", i)
                x, y, theta = 0.0, 0.0, 0.0
                if frame.base_pose is not None:
                    xyt = np.asarray(frame.base_pose).ravel()
                    if len(xyt) >= 3:
                        x, y, theta = float(xyt[0]), float(xyt[1]), float(xyt[2])
                    elif len(xyt) >= 2:
                        x, y = float(xyt[0]), float(xyt[1])
                elif frame.camera_pose is not None:
                    trans = frame.camera_pose[:3, 3]
                    x, y = float(trans[0]), float(trans[1])
                rr.log(
                    "world/robot",
                    rr.Transform3D(
                        translation=[x, y, 0],
                        rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=theta),
                        axis_length=0.5,
                    ),
                )
                rr.log(
                    "world/frames/current",
                    rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)),
                )
                if frame.rgb is not None:
                    rgb = np.asarray(frame.rgb)
                    if rgb.ndim == 3 and rgb.shape[2] == 3:
                        rgb = _rgb_to_uint8(rgb)
                    rr.log("world/frames/current/rgb", rr.Image(rgb))
                if frame.depth is not None:
                    depth = np.asarray(frame.depth)
                    if depth.ndim == 2:
                        rr.log("world/frames/current/depth", rr.DepthImage(depth))

    def update_voxel_map(
        self,
        space: SparseVoxelMapNavigationSpace,
        debug: bool = False,
        explored_radius=0.025,
        obstacle_radius=0.05,
        world_radius=0.03,
        robot_base_xy: np.ndarray | tuple[float, float] | None = None,
    ):
        """Log voxel map and send it to Rerun visualizer.

        Builds a minimal MemoryState from space and calls log_memory_state.
        Also logs a top-down RGB to ``world/map_snapshot/topdown`` (same path as
        ``send_map_snapshot``) so the blueprint ``map_topdown`` view stays live.
        """
        rr.set_time_seconds("realtime", time.time())

        t0 = timeit.default_timer()
        points, _, _, rgb = space.voxel_map.voxel_pcd.get_pointcloud()
        if rgb is None:
            return

        grid_origin = space.voxel_map.grid_origin
        if hasattr(grid_origin, "cpu"):
            grid_origin = grid_origin.cpu().numpy()
        grid_resolution = float(space.voxel_map.grid_resolution)
        obstacles, explored = space.voxel_map.get_2d_map()
        if hasattr(obstacles, "cpu"):
            obstacles = obstacles.cpu().numpy()
        if hasattr(explored, "cpu"):
            explored = explored.cpu().numpy()
        if hasattr(points, "cpu"):
            points = points.cpu().numpy()
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()

        state = MemoryState(
            point_cloud=PointCloudBlob(xyz=points, rgb=rgb),
            grid_origin=grid_origin,
            grid_resolution=grid_resolution,
            obstacles_2d=obstacles,
            explored_2d=explored,
        )
        t1 = timeit.default_timer()
        self.log_memory_state(
            state,
            explored_radius=explored_radius,
            obstacle_radius=obstacle_radius,
            world_radius=world_radius,
        )
        t2 = timeit.default_timer()

        try:
            from emet.visualization.map_snapshot import snapshot_from_voxel_map

            img, _stats, _discord = snapshot_from_voxel_map(space.voxel_map, robot_base_xy, max_side=640)
            if img is not None and getattr(img, "size", 0) > 0:
                self.log_custom_2d_image("world/map_snapshot/topdown", img)
        except Exception as e:
            logger.debug("Rerun top-down map snapshot skipped: %s", e)

        if debug:
            print("Time to get voxel data: ", t1 - t0)
            print("Time to log memory state: ", t2 - t1)

    def _log_instance_boxes(
        self,
        centers: list,
        half_sizes: list[list[float]],
        labels: list[str],
        colors: list[np.ndarray],
        entity: str = "world/objects",
    ) -> None:
        """Log 3D boxes with labels (shared by update_scene_graph and optional MemoryState)."""
        log_to_rerun(
            entity,
            rr.Boxes3D(
                half_sizes=half_sizes,
                centers=centers,
                labels=labels,
                radii=0.01,
                colors=colors,
            ),
            static=True,
        )

    def update_scene_graph(
        self,
        scene_graph: SceneGraph,
        semantic_sensor: OvmmPerception | None = None,
        verbose: bool = False,
    ):
        """Log objects bounding boxes and relationships.
        Uses shared _log_instance_boxes so loaded memory with instance boxes could use the same path.
        """
        if not scene_graph.instances:
            return
        rr.set_time_seconds("realtime", time.time())
        centers = []
        labels = []
        bounds = []
        colors = []

        t0 = timeit.default_timer()
        for idx, instance in enumerate(scene_graph.instances):
            if semantic_sensor and semantic_sensor.is_semantic():
                name = semantic_sensor.get_class_name_for_id(instance.category_id)
            else:
                name = f"obj_{instance.global_id}"

            # Replace spaces with underscores
            name = name.replace(" ", "_") if name is not None else None

            # Create colors (key by name so same class gets same color)
            if name not in self.bbox_colors_memory:
                self.bbox_colors_memory[name] = np.random.randint(0, 255, 3)

            best_view = instance.get_best_view()
            bbox_bounds = best_view.bounds  # 3D Bounds
            point_cloud_rgb = instance.point_cloud
            pcd_rgb = instance.point_cloud_rgb
            if point_cloud_rgb is not None and pcd_rgb is not None:
                pos_np = (
                    point_cloud_rgb.cpu().numpy() if hasattr(point_cloud_rgb, "cpu") else np.asarray(point_cloud_rgb)
                )
                col_np = np.int64(pcd_rgb.cpu().numpy() if hasattr(pcd_rgb, "cpu") else pcd_rgb)
                log_to_rerun(
                    f"world/{instance.id}_{name}" if name is not None else f"world/{instance.id}",
                    rr.Points3D(positions=pos_np, colors=col_np),
                    static=True,
                )
            half_sizes = [(float(b[1]) - float(b[0])) / 2 for b in bbox_bounds]
            bounds.append(half_sizes)
            pose = scene_graph.get_ins_center_pos(idx)
            pose_np = pose.cpu().numpy() if hasattr(pose, "cpu") else np.asarray(pose)
            confidence = best_view.score if best_view.score is not None else 0.0
            centers.append(rr.components.PoseTranslation3D(pose_np))
            labels.append(f"{name} {confidence:.2f}")
            colors.append(self.bbox_colors_memory[name])
        self._log_instance_boxes(centers, bounds, labels, colors, entity="world/objects")
        t1 = timeit.default_timer()
        if verbose:
            print("Time to log scene graph objects: ", t1 - t0)

    def update_open_vocab_scene_graph(
        self,
        scene_graph,
        verbose: bool = False,
    ) -> None:
        """Log an OpenVocabSceneGraph to Rerun.

        Visualizes:
        - Per-object colored point clouds at world/scene_graph/<id>_<label>
        - 3D bounding boxes at world/scene_graph/objects
        - Spatial edges as line segments at world/scene_graph/edges
        - Node labels + observation counts at world/scene_graph/labels
        - Best crop images at world/scene_graph/crops/<id>
        - Text summary at world/scene_graph/summary
        """
        rr.set_time_seconds("realtime", time.time())

        if not scene_graph.nodes:
            return

        t0 = timeit.default_timer()

        centers = []
        half_sizes_list = []
        labels = []
        colors = []

        for nid, node in scene_graph.nodes.items():
            label = node.primary_label
            safe_label = label.replace(" ", "_")

            # Consistent color per label
            if safe_label not in self.bbox_colors_memory:
                self.bbox_colors_memory[safe_label] = np.random.randint(0, 255, 3)
            color = self.bbox_colors_memory[safe_label]

            # Per-object point cloud
            if node.point_cloud is not None and node.point_cloud.shape[0] > 0:
                pts = node.point_cloud.detach().cpu().numpy()
                if node.point_cloud_rgb is not None:
                    pcd_rgb = node.point_cloud_rgb.detach().cpu().numpy()
                    pcd_rgb = _rgb_to_uint8(pcd_rgb)
                else:
                    pcd_rgb = np.tile(color, (pts.shape[0], 1))
                log_to_rerun(
                    f"world/scene_graph/{nid}_{safe_label}",
                    rr.Points3D(positions=pts, colors=pcd_rgb, radii=0.01),
                )

            # Bounding box
            if node.bounds is not None and node.center is not None:
                mins = node.bounds[:, 0].cpu().numpy()
                maxs = node.bounds[:, 1].cpu().numpy()
                half_sizes = ((maxs - mins) / 2).tolist()
                center = node.center.tolist()
                obs_tag = f"x{node.observation_count}" if node.observation_count > 1 else ""
                stable_tag = " [stable]" if node.is_stable else ""
                box_label = f"{label} {obs_tag}{stable_tag}"

                centers.append(center)
                half_sizes_list.append(half_sizes)
                labels.append(box_label)
                colors.append(color.tolist())

            # Best crop image
            if node.best_crop is not None:
                try:
                    crop = node.best_crop
                    if crop.dtype != np.uint8:
                        crop = np.clip(crop, 0, 255).astype(np.uint8)
                    log_to_rerun(
                        f"world/scene_graph/crops/{nid}_{safe_label}",
                        rr.Image(crop),
                    )
                except Exception:
                    pass

        # Log all bounding boxes together
        if centers:
            log_to_rerun(
                "world/scene_graph/objects",
                rr.Boxes3D(
                    half_sizes=half_sizes_list,
                    centers=centers,
                    labels=labels,
                    radii=0.01,
                    colors=colors,
                ),
            )

        # Log edges as line segments
        scene_graph.update_edges()
        if scene_graph.edges:
            line_strips = []
            edge_colors = []
            edge_labels = []
            relation_colors = {
                "near": [200, 200, 200],
                "on": [0, 200, 100],
                "on_floor": [100, 100, 255],
            }
            for edge in scene_graph.edges:
                src_node = scene_graph.nodes.get(edge.source_id)
                tgt_node = scene_graph.nodes.get(edge.target_id)
                if src_node is None or src_node.center is None:
                    continue
                src_pos = src_node.center.tolist()
                if tgt_node is not None and tgt_node.center is not None:
                    tgt_pos = tgt_node.center.tolist()
                elif edge.target_id == -1:
                    # on_floor: draw line down to z=0
                    tgt_pos = [src_pos[0], src_pos[1], 0.0]
                else:
                    continue
                line_strips.append([src_pos, tgt_pos])
                edge_colors.append(relation_colors.get(edge.relation, [180, 180, 180]))
                src_lbl = src_node.primary_label if src_node else "?"
                tgt_lbl = tgt_node.primary_label if tgt_node else "floor"
                edge_labels.append(f"{src_lbl} --{edge.relation}--> {tgt_lbl}")

            if line_strips:
                log_to_rerun(
                    "world/scene_graph/edges",
                    rr.LineStrips3D(
                        line_strips,
                        colors=edge_colors,
                        radii=0.005,
                        labels=edge_labels,
                    ),
                )

        stable_count = len(scene_graph.stable_objects)
        header = f"## Scene Graph\n\n**{scene_graph.num_objects}** objects ({stable_count} stable)\n\n"
        # Object table
        table_lines = ["| ID | Label | Seen | Stable |", "|---|---|---|---|"]
        for node in scene_graph.nodes.values():
            table_lines.append(
                f"| {node.node_id} | {node.primary_label} | "
                f"{node.observation_count}x | "
                f"{'yes' if node.is_stable else 'no'} |"
            )
        table = "\n".join(table_lines)
        # Edge list
        edge_text = ""
        if scene_graph.edges:
            edge_lines = ["### Spatial Relations"]
            for edge in scene_graph.edges:
                src = scene_graph.nodes.get(edge.source_id)
                tgt = scene_graph.nodes.get(edge.target_id)
                src_lbl = src.primary_label if src else "?"
                tgt_lbl = tgt.primary_label if tgt else "floor"
                edge_lines.append(f"- {src_lbl} **{edge.relation}** {tgt_lbl}")
            edge_text = "\n".join(edge_lines)

        full_text = header + table + "\n\n" + edge_text
        log_to_rerun(
            "world/scene_graph/summary",
            rr.TextDocument(full_text, media_type=rr.MediaType.MARKDOWN),
        )

        t1 = timeit.default_timer()
        if verbose:
            print(f"Scene graph visualization: {t1 - t0:.3f}s")

    def update_nav_goal(self, goal, timeout=10):
        """Log navigation goal
        Args:
            goal (np.ndarray): Goal coordinates
        """
        ts = time.time()
        rr.set_time_seconds("realtime", ts)
        log_to_rerun("world/xyt_goal", rr.Points3D([0, 0, 0], colors=[0, 255, 0, 50], radii=0.1))
        log_to_rerun(
            "world/xyt_goal",
            rr.Transform3D(
                translation=[goal[0], goal[1], 0],
                rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=goal[2]),
                axis_length=0.5,
            ),
        )
        # rr.set_time_seconds("realtime", ts + timeout)
        # log_to_rerun("world/xyt_goal", rr.Clear(recursive=True))
        # rr.set_time_seconds("realtime", ts)

    def step(self, obs, servo, *, mapping_depth: np.ndarray | None = None):
        """Log streaming robot/sensor data.

        *obs* is typically the full ZMQ observation dict (Stretch / Generic) or an Observations instance.
        *servo* is optional low-res head/EE `Observations` (Stretch servo thread); if missing but *obs*
        contains ``rgb``, head camera is logged from *obs* instead.

        When the full-observation socket has no frame yet (or frames are skipped e.g. missing depth),
        *obs* may be ``None`` while *servo* still carries head RGB and base pose — log from *servo* in that case.

        *mapping_depth* is the H×W depth (meters) last used for DynaMem voxel fusion; when its shape matches
        head RGB, ``world/head_camera/points`` is built from it so Rerun matches ``world/point_cloud``.
        """
        head_cam = _pick_rerun_head_cam(obs, servo)
        if head_cam is None or getattr(head_cam, "rgb", None) is None:
            time.sleep(0.05)
            return

        if isinstance(obs, Observations):
            obs_pose: dict[str, Any] = {
                "gps": obs.gps,
                "compass": obs.compass,
                "ee_pose": obs.ee_pose,
                "joint": obs.joint,
            }
            if obs.emet_session is not None:
                obs_pose[EMET_ZMQ_SESSION_KEY] = obs.emet_session
        elif isinstance(obs, dict):
            obs_pose = obs
        else:
            obs_pose = {
                "gps": head_cam.gps,
                "compass": head_cam.compass,
                "ee_pose": head_cam.ee_pose,
                "joint": head_cam.joint,
            }
            if getattr(head_cam, "emet_session", None) is not None:
                obs_pose[EMET_ZMQ_SESSION_KEY] = head_cam.emet_session

        rr.set_time_seconds("realtime", time.time())
        try:
            t0 = timeit.default_timer()
            self.log_robot_xyt(obs_pose)
            self.log_ee_frame(obs_pose)

            self.log_head_camera(head_cam, mapping_depth=mapping_depth)
            self.log_ee_camera(head_cam)

            if self.display_robot_mesh and getattr(self, "mjcf_skeleton", None) is not None:
                self.mjcf_skeleton.apply_and_log(obs_pose)
            elif self.display_robot_mesh and getattr(self, "urdf_logger", None) is not None:
                self.log_robot_state(obs_pose)
                self.log_robot_transforms(obs_pose)
            t1 = timeit.default_timer()
            sleep_time = self.step_delay_s - (t1 - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(e)
            raise e
