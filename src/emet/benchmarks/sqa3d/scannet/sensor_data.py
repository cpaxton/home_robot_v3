# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ScanNet ``.sens`` reader (Python 3 port of ScanNet SensReader)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
_COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


@dataclass
class RGBDFrame:
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_data: bytes
    depth_data: bytes
    color_compression_type: str
    depth_compression_type: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float

    def decompress_color(self) -> np.ndarray:
        if self.color_compression_type == "jpeg":
            arr = np.frombuffer(self.color_data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Failed to decode ScanNet JPEG color frame")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.color_compression_type == "png":
            arr = np.frombuffer(self.color_data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Failed to decode ScanNet PNG color frame")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Unsupported ScanNet color compression: {self.color_compression_type!r}")

    def decompress_depth(self) -> np.ndarray:
        if self.depth_compression_type == "zlib_ushort":
            raw = zlib.decompress(self.depth_data)
            depth = np.frombuffer(raw, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            return depth
        raise ValueError(f"Unsupported ScanNet depth compression: {self.depth_compression_type!r}")


@dataclass
class ScanNetSensorData:
    intrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    frames: list[RGBDFrame]

    @classmethod
    def load(cls, path: Path | str) -> ScanNetSensorData:
        path = Path(path)
        with path.open("rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            if version != 4:
                raise ValueError(f"Unsupported .sens version {version} in {path}")
            strlen = struct.unpack("Q", f.read(8))[0]
            f.read(strlen)  # sensor_name
            intrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            f.read(16 * 4)  # extrinsic_color
            intrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            f.read(16 * 4)  # extrinsic_depth
            color_compression_type = _COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            depth_compression_type = _COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            color_width = struct.unpack("I", f.read(4))[0]
            color_height = struct.unpack("I", f.read(4))[0]
            depth_width = struct.unpack("I", f.read(4))[0]
            depth_height = struct.unpack("I", f.read(4))[0]
            depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            frames: list[RGBDFrame] = []
            for _ in range(num_frames):
                camera_to_world = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
                timestamp_color = struct.unpack("Q", f.read(8))[0]
                timestamp_depth = struct.unpack("Q", f.read(8))[0]
                color_size_bytes = struct.unpack("Q", f.read(8))[0]
                depth_size_bytes = struct.unpack("Q", f.read(8))[0]
                color_data = f.read(color_size_bytes)
                depth_data = f.read(depth_size_bytes)
                frames.append(
                    RGBDFrame(
                        camera_to_world=camera_to_world,
                        timestamp_color=timestamp_color,
                        timestamp_depth=timestamp_depth,
                        color_data=color_data,
                        depth_data=depth_data,
                        color_compression_type=color_compression_type,
                        depth_compression_type=depth_compression_type,
                        color_width=color_width,
                        color_height=color_height,
                        depth_width=depth_width,
                        depth_height=depth_height,
                        depth_shift=depth_shift,
                    )
                )
        return cls(
            intrinsic_color=intrinsic_color,
            intrinsic_depth=intrinsic_depth,
            color_width=color_width,
            color_height=color_height,
            depth_width=depth_width,
            depth_height=depth_height,
            depth_shift=depth_shift,
            frames=frames,
        )
