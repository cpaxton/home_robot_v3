# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ROS2 ``sensor_msgs/LaserScan`` subscriber for Mars /scan."""

from __future__ import annotations

import threading

import numpy as np
import rclpy.time
from rclpy.time import Time
from sensor_msgs.msg import LaserScan


class RosLidar:
    """Last ``/scan`` as Nx2 base-frame points (same layout as Stretch bridge)."""

    _max_dist = 100.0

    def __init__(self, ros_client, name: str = "/scan", verbose: bool = False):
        self.name = name
        self._points: np.ndarray | None = None
        self.verbose = verbose
        self._lock = threading.Lock()
        self._t = Time()
        self._ros_client = ros_client
        self._subscriber = self._ros_client.create_subscription(
            LaserScan, self.name, self._lidar_scan_callback, 10
        )

    def _lidar_scan_callback(self, scan_msg: LaserScan) -> None:
        ranges = np.array(scan_msg.ranges, dtype=np.float64)
        ranges[~np.isfinite(ranges)] = self._max_dist
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        lidar_points = np.column_stack((xs, ys))
        if self.verbose:
            print(f"[LIDAR] {self.name}: {lidar_points.shape[0]} points")
        with self._lock:
            self._t = rclpy.time.Time.from_msg(scan_msg.header.stamp)
            self._points = lidar_points

    def get_time(self) -> Time:
        return self._t

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._points

    def wait_for_scan(self) -> None:
        rate = self._ros_client.create_rate(5)
        while rclpy.ok():
            with self._lock:
                if self._points is not None:
                    return
            rate.sleep()
