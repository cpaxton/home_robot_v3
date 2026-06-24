# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Allen Institute for AI (MolmoSpaces). Apache-2.0.
# Vendored from molmo_spaces/utils/scene_maps.py (iTHORMap + from_mj_model_path).
# Two-pass occupancy (Pass 1: high/low visual geoms vs floor+1.5m; Pass 2: drop high bodies)
# matches upstream molmospaces main; merged MJCF skips robot subtree in Pass 1 (base_link).

from __future__ import annotations

import gc
import json

import cv2
import mujoco
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from emet.simulation.molmo_occupancy._geom_aabb import geom_aabb
from emet.simulation.molmo_occupancy._linalg import inverse_homogeneous_matrix
from emet.simulation.molmo_occupancy.mj_model_bindings import MjModelBindings
from emet.simulation.molmo_occupancy.opengl_rendering import MjOpenGLRenderer
from emet.simulation.molmo_occupancy.proc_thor_map import ProcTHORMap, circular_kernel


def _delete_blacklisted_bodies(spec: mujoco.MjSpec) -> int:
    """Upstream deletes problematic asset bodies; emet stub skips (no static blacklist)."""
    return 0


def _handle_compile_error_and_blacklist(_error: Exception) -> None:
    return


def _strip_spec_keyframes(spec: mujoco.MjSpec) -> None:
    """Remove keyframes before compile after ``spec.delete(body)`` (merged MJCF + robot keys duplicate)."""
    for key in list(spec.keys):
        spec.delete(key)


def _safe_model_data(spec: mujoco.MjSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile *spec* after blacklist cleanup; mirrors upstream ProcTHORMap.safe_model_data."""
    _delete_blacklisted_bodies(spec)
    try:
        model = spec.compile()
    except ValueError as e:
        _handle_compile_error_and_blacklist(e)
        raise
    finally:
        del spec
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _body_in_robot_subtree(model: mujoco.MjModel, body_id: int, robot_root_id: int) -> bool:
    """True if *body_id* is *robot_root_id* or a descendant (kinematic parent walk)."""
    if robot_root_id < 0:
        return False
    b = body_id
    for _ in range(model.nbody + 2):
        if b < 0:
            return False
        if b == robot_root_id:
            return True
        b = int(model.body_parentid[b])
    return False


def _canonical_ithor_top_level_body_name(model: mujoco.MjModel, body_id: int) -> str | None:
    raw = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if not raw:
        return None
    parts = raw.split("_")
    if len(parts) < 2:
        return raw
    parts[-2] = "0"
    return "_".join(parts)


class iTHORMap(ProcTHORMap):
    """iTHOR-style orthographic occupancy (MolmoSpaces iTHORMap vendored slice)."""

    def __init__(
        self,
        occupancy: np.ndarray,
        world_to_map: np.ndarray,
        map_to_world: np.ndarray,
        px_per_m: int,
    ) -> None:
        super().__init__(occupancy=occupancy, world_to_map=world_to_map, map_to_world=map_to_world, px_per_m=px_per_m)

    @classmethod
    def from_mj_model_path(
        cls,
        model_path: str,
        camera: str | None = None,
        agent_radius: float | None = None,
        px_per_m: int = 100,
        device_id: int | None = None,
        *,
        robot_root_body_name: str | None = "base_link",
    ) -> iTHORMap:
        # Two passes (upstream iTHORMap): (1) high vs low visual geoms; (2) occupancy with highs removed.

        # Pass 1
        spec = mujoco.MjSpec.from_file(model_path)
        model, mj_data = _safe_model_data(spec)

        floor_ids: list[int] = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "floor" in geom_name.lower():
                if int(model.geom(geom_id).contype) == 0:
                    floor_ids.append(geom_id)
        if len(floor_ids) == 0:
            raise RuntimeError("No floors found in the model (visual floor geoms with contype==0)")

        aabb_center, aabb_size = geom_aabb(model, mj_data, floor_ids, tight_mesh=False)
        z_threshold = 1.5 + (float(aabb_center[2]) + float(aabb_size[2]) / 2.0)

        robot_root_id = -1
        robot_root_name: str | None = None
        if robot_root_body_name:
            robot_root_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body_name))
            if robot_root_id >= 0:
                robot_root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_id)
        if robot_root_name is None:
            for candidate in ("chassis", "base_link", "base"):
                cid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate))
                if cid >= 0:
                    robot_root_id = cid
                    robot_root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cid)
                    break

        high_names: set[str] = set()
        low_names: set[str] = set()
        for geom_id in range(model.ngeom):
            if int(model.geom(geom_id).contype) != 0:
                continue
            body_id = int(model.geom_bodyid[geom_id])
            if _body_in_robot_subtree(model, body_id, robot_root_id):
                continue
            gaabb_c, gaabb_s = geom_aabb(model, mj_data, [geom_id], tight_mesh=False)
            min_z = float(gaabb_c[2]) - float(gaabb_s[2]) / 2.0
            top_level = _canonical_ithor_top_level_body_name(model, body_id)
            if top_level is None:
                continue
            if min_z > z_threshold:
                high_names.add(top_level)
            else:
                low_names.add(top_level)

        high_names -= low_names

        del mj_data, model

        # Pass 2 — strip robot + keyframes before scene body deletes (MjSpec duplicates included keys).
        spec = mujoco.MjSpec.from_file(model_path)
        if robot_root_name:
            for body in list(spec.worldbody.bodies):
                if body.name == robot_root_name:
                    spec.delete(body)
                    break
        _strip_spec_keyframes(spec)
        for body in list(spec.worldbody.bodies):
            body_name = body.name
            if body_name and "ceiling" in body_name.lower():
                spec.delete(body)
            elif body_name and "light" in body_name.lower():
                spec.delete(body)
            elif body_name in high_names:
                spec.delete(body)

        _strip_spec_keyframes(spec)
        model, mj_data = _safe_model_data(spec)

        floor_ids = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "floor" in geom_name.lower():
                if int(model.geom(geom_id).contype) == 0:
                    floor_ids.append(geom_id)
        if len(floor_ids) == 0:
            raise RuntimeError("No floors found in the model (visual floor geoms with contype==0)")

        if camera is None:
            aabb_center, aabb_size = geom_aabb(model, mj_data, floor_ids, tight_mesh=False)
            aabb_size = aabb_size + np.array([2, 2, 0])
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = aabb_center
            cam.distance = 5.0
            cam.azimuth = 0
            cam.elevation = -90
            cam.orthographic = 1
            h = int(round(px_per_m * aabb_size[0]))
            w = int(round(px_per_m * aabb_size[1]))
            px_per_m_eff = h / aabb_size[0]
            renderer = MjOpenGLRenderer(MjModelBindings(model), height=h, width=w, device_id=device_id)
            renderer.update(mj_data, cam)
            for glcam in renderer.scene.camera:
                glcam.orthographic = 1
                glcam.frustum_bottom = -aabb_size[0] / 2
                glcam.frustum_top = aabb_size[0] / 2
        else:
            cam_model = model.cam(camera)
            assert model.cam_orthographic[cam_model.id], "Camera must be orthographic"
            w, h = model.cam_resolution[cam_model.id]
            px_per_m_eff = h / cam_model.fovy.item()
            renderer = MjOpenGLRenderer(MjModelBindings(model), height=h, width=w, device_id=device_id)
            renderer.update(mj_data, camera)

        cam_to_world = np.eye(4)
        cam_to_world[:3, 3] = renderer.scene.camera[0].pos
        camera_x_ax = np.cross(renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
        cam_to_world[:3, :3] = np.column_stack(
            (camera_x_ax, renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
        )
        assert np.allclose(cam_to_world[:3, 2], [0, 0, 1]), "Camera must be pointing straight down"

        renderer.enable_segmentation_rendering()
        seg = renderer.render()[..., 0]
        renderer.close()

        occupancy = np.ones_like(seg, dtype=bool)
        for floor_id in floor_ids:
            occupancy &= seg != floor_id
        if agent_radius is not None:
            rad_px = int(agent_radius * px_per_m_eff)
            kernel = circular_kernel(rad_px)
            occupancy = cv2.dilate(occupancy.astype(np.uint8), kernel).astype(bool)

        cam_to_map = np.array([[0, -px_per_m_eff, 0, h / 2], [px_per_m_eff, 0, 0, w / 2]])
        world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)
        map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
        centered_to_cam = np.array([[1 / px_per_m_eff, 0, 0], [0, 1 / px_per_m_eff, 0], [0, 0, 1]])
        cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
        cam_to_world_floor[2, 2] = 0
        map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered
        occupancy = ~occupancy

        del model
        del mj_data
        gc.collect()

        return cls(occupancy, world_to_map, map_to_world, int(np.ceil(px_per_m_eff)))

    def save(self, path: str) -> None:
        if path.endswith(".png"):
            img = Image.fromarray(self.occupancy.astype(np.uint8) * 255)
            metadata = PngInfo()
            metadata.add_text("world_to_map", json.dumps(self.world_to_map.tolist()))
            metadata.add_text("map_to_world", json.dumps(self.map_to_world.tolist()))
            metadata.add_text("px_per_m", json.dumps(self.px_per_m))
            rid = getattr(self, "room_ids_to_name", None)
            if rid is not None:
                metadata.add_text("room_ids_to_name", json.dumps(rid))
            img.save(path, pnginfo=metadata)
        elif path.endswith(".npz"):
            np.savez(
                path,
                occupancy=self.occupancy,
                world_to_map=self.world_to_map,
                map_to_world=self.map_to_world,
                px_per_m=self.px_per_m,
            )
        else:
            raise ValueError(f"Unsupported file format: {path}")

    @classmethod
    def load(cls, path: str) -> iTHORMap:
        if path.endswith(".png"):
            img = Image.open(path)
            world_to_map = np.array(json.loads(img.info["world_to_map"]))
            map_to_world = np.array(json.loads(img.info["map_to_world"]))
            px_per_m = int(np.ceil(json.loads(img.info["px_per_m"])))
            occupancy = np.array(img) > 0
            return cls(occupancy=occupancy, world_to_map=world_to_map, map_to_world=map_to_world, px_per_m=px_per_m)
        if path.endswith(".npz"):
            data = np.load(path)
            return cls(
                occupancy=data["occupancy"],
                world_to_map=data["world_to_map"],
                map_to_world=data["map_to_world"],
                px_per_m=int(data["px_per_m"]),
            )
        raise ValueError(f"Unsupported file format: {path}")
