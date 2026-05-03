# Copyright (c) Allen Institute for AI (MolmoSpaces). Apache-2.0.
# Vendored from molmo_spaces/utils/scene_maps.py (ProcTHORMap + helpers).

from __future__ import annotations

import json

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from emet.simulation.molmo_occupancy._linalg import homogenize, single_or_batch


def circular_kernel(radius: int) -> np.ndarray:
    import cv2

    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel


class ProcTHORMap:
    """2D occupancy grid with world/map transforms (MolmoSpaces ProcTHORMap API subset)."""

    def __init__(
        self,
        occupancy: np.ndarray,
        world_to_map: np.ndarray,
        map_to_world: np.ndarray,
        px_per_m: int,
        room_map: np.ndarray | None = None,
        room_ids_to_name: dict | None = None,
    ) -> None:
        self.occupancy = occupancy
        self._room_map = room_map
        self.room_ids_to_name = room_ids_to_name
        if room_ids_to_name is not None:
            self.room_names_to_id = {v: k for k, v in room_ids_to_name.items()}
        else:
            self.room_names_to_id = None
        self.world_to_map = world_to_map
        self.map_to_world = map_to_world
        self._px_per_m = px_per_m

    @property
    def room_map(self) -> np.ndarray | None:
        return self._room_map

    @property
    def px_per_m(self) -> int:
        return int(self._px_per_m)

    def get_free_points(self) -> np.ndarray:
        free_points_px = np.argwhere(self.occupancy)
        return self.pos_px_to_m(free_points_px)

    def get_free_points_by_room(self, room_key: str) -> np.ndarray:
        if self.room_names_to_id is None or self._room_map is None:
            raise ValueError("room map not available")
        room_id = self.room_names_to_id[room_key]
        free_points_px = np.argwhere(self.occupancy)
        free_points_px = free_points_px[
            self._room_map[free_points_px[:, 0], free_points_px[:, 1]] == room_id
        ]
        return self.pos_px_to_m(free_points_px)

    @single_or_batch
    def pos_m_to_px(self, pos_m: np.ndarray) -> np.ndarray:
        assert pos_m.ndim == 2 and pos_m.shape[-1] == 3
        return np.round(homogenize(pos_m) @ self.world_to_map.T).astype(int)

    @single_or_batch
    def pos_px_to_m(self, pos_px: np.ndarray) -> np.ndarray:
        assert pos_px.ndim == 2 and pos_px.shape[-1] == 2
        return homogenize(pos_px) @ self.map_to_world.T

    @single_or_batch
    def check_collision(self, pos: np.ndarray) -> bool | np.ndarray:
        pos_px = self.pos_m_to_px(pos)
        in_range_mask = np.all((pos_px >= 0) & (pos_px < self.occupancy.shape), axis=1)
        ret = in_range_mask.copy()
        ret[in_range_mask] = self.occupancy[pos_px[in_range_mask, 0], pos_px[in_range_mask, 1]]
        return ret

    def save(self, path: str) -> None:
        if path.endswith(".png"):
            img = Image.fromarray(self.occupancy.astype(np.uint8) * 255)
            metadata = PngInfo()
            metadata.add_text("world_to_map", json.dumps(self.world_to_map.tolist()))
            metadata.add_text("map_to_world", json.dumps(self.map_to_world.tolist()))
            metadata.add_text("px_per_m", json.dumps(self.px_per_m))
            if self.room_ids_to_name is not None:
                metadata.add_text("room_ids_to_name", json.dumps(self.room_ids_to_name))
            img.save(path, pnginfo=metadata)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    @classmethod
    def load(cls, path: str) -> ProcTHORMap:
        if not path.endswith(".png"):
            raise ValueError("Only PNG load supported in emet vendored slice")
        img = Image.open(path)
        world_to_map = np.array(json.loads(img.info["world_to_map"]))
        map_to_world = np.array(json.loads(img.info["map_to_world"]))
        px_per_m = int(np.ceil(json.loads(img.info["px_per_m"])))
        occupancy = np.array(img) > 0
        room_ids_to_name = None
        if "room_ids_to_name" in img.info:
            room_ids_to_name = json.loads(img.info["room_ids_to_name"])
        return cls(
            occupancy=occupancy,
            world_to_map=world_to_map,
            map_to_world=map_to_world,
            px_per_m=px_per_m,
            room_ids_to_name=room_ids_to_name,
        )
