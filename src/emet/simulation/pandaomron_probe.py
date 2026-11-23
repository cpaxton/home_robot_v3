# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""PandaOmron simulation force probe. Call under the physics lock."""

import time
from collections import deque

import mujoco
import numpy as np

from emet.motion.guarded_placement_probe import (
    MAX_LATERAL_DRIFT_M,
    MAX_SPEED_M_S,
    MAX_TRAVEL_M,
    GuardedPlacementProbe,
)
from emet.robots.pandaomron import GRASP_BODY
from emet.simulation.pandaomron_environment import CONTROL_HZ

_TRACKING_WATCHDOG_S = 0.4
_TICK_M = MAX_SPEED_M_S / CONTROL_HZ


class PandaPlacementProbe:
    def __init__(self, model, robot, qpos_addresses, limits, calibration):
        self.model, self.robot = model, robot
        self.qpos_addresses, self.limits = qpos_addresses, limits
        self.calibration = calibration
        self.scratch = mujoco.MjData(model)
        sensor_names = robot.gripper["right"].important_sensors
        self.force, self.torque = (model.sensor(sensor_names[key]) for key in ("force_ee", "torque_ee"))
        if self.force.objid[0] != self.torque.objid[0] or self.force.dim[0] != 3 or self.torque.dim[0] != 3:
            raise ValueError("invalid_panda_wrist_wrench_sensor")
        self.guard = None
        self.probe_id = None
        self.spent = set()
        self.samples = deque(maxlen=8)
        self.waypoints = deque()

    @property
    def active(self):
        return self.guard is not None and self.guard.status in ("descending", "holding")

    def hold_goal(self, data):
        return np.clip(data.qpos[self.qpos_addresses], self.limits[:, 0], self.limits[:, 1])

    def _wrench(self, data):
        rotation = data.site_xmat[self.force.objid[0]].reshape(3, 3)
        return np.concatenate([
            rotation @ data.sensordata[sensor.adr[0] : sensor.adr[0] + 3] for sensor in (self.force, self.torque)
        ])

    def _refresh_tracking(self, capture_end, *, now):
        if not isinstance(capture_end, (int, float)) or not 0 <= now - capture_end <= _TRACKING_WATCHDOG_S:
            raise ValueError("probe_payload_tracking_lost")
        self.tracking_end = max(self.tracking_end, capture_end)

    def start(self, request, data):
        now = time.monotonic()
        probe_id = request["probe_id"]
        if (
            self.calibration is None or self.active or len(self.spent) >= 256
            or not isinstance(probe_id, str) or not 1 <= len(probe_id) <= 64
        ):
            raise ValueError("probe_unavailable")
        if probe_id in self.spent or request["calibration_id"] != self.calibration.calibration_id:
            raise ValueError("probe_identity_invalid")
        self.tracking_end = 0.0
        self._refresh_tracking(request["capture_end_s"], now=now)
        q = np.asarray(request["waypoints"], dtype=float)
        if (
            q.ndim != 2 or q.shape[1:] != self.qpos_addresses.shape or not 1 <= len(q) <= 40
            or not np.isfinite(q).all() or np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1])
        ):
            raise ValueError("invalid_probe_waypoints")
        start = data.body(GRASP_BODY).xpos.copy()
        rotation = data.body(GRASP_BODY).xmat.copy()
        self.scratch.qpos[:] = data.qpos
        previous = 0.0
        for waypoint in q:
            self.scratch.qpos[self.qpos_addresses] = waypoint
            mujoco.mj_kinematics(self.model, self.scratch)
            ee = self.scratch.body(GRASP_BODY)
            delta = ee.xpos - start
            travel = -float(delta[2])
            if (
                np.linalg.norm(delta[:2]) > MAX_LATERAL_DRIFT_M / 2
                or np.linalg.norm(ee.xmat - rotation) > 0.01
                or not 0 < travel - previous <= _TICK_M + 1e-8
                or travel > MAX_TRAVEL_M
            ):
                raise ValueError("invalid_probe_cartesian_path")
            previous = travel
        if len(self.samples) < 4 or now - self.samples[-1][0] > 0.125:
            raise ValueError("probe_baseline_unavailable")
        recent = tuple(self.samples)[-4:]
        joints = np.array([row[2] for row in recent])
        if np.max(np.ptp(joints, axis=0)) > 0.002 or np.max(np.abs(joints[-1] - data.qpos[self.qpos_addresses])) > 0.002:
            raise ValueError("probe_baseline_unsettled")
        baseline = np.median([row[1] for row in recent], axis=0)
        guard = GuardedPlacementProbe(
            self.calibration, start_xyz=start, baseline=baseline, started_at=now,
            contact_band_m=request["contact_band_m"],
        )
        if previous > guard.upper or previous < guard.upper - _TICK_M:
            raise ValueError("probe_waypoints_outside_contact_band")
        self.guard, self.probe_id = guard, probe_id
        self.last_sim_s = float(data.time)
        self.waypoints = deque(q)
        self.spent.add(probe_id)

    def keepalive(self, request):
        if request["probe_id"] != self.probe_id or self.guard is None:
            raise ValueError("probe_identity_invalid")
        if self.active:
            self._refresh_tracking(request["capture_end_s"], now=time.monotonic())

    def tick(self, data, goal):
        now = time.monotonic()
        wrench = self._wrench(data)
        measured = data.qpos[self.qpos_addresses].copy()
        self.samples.append((now, wrench, measured))
        if not self.active:
            return goal
        if data.time <= self.last_sim_s and self.guard.last_sample_at != self.guard.started_at:
            self.guard.stop("probe_feedback_stale")
            return self.hold_goal(data)
        self.last_sim_s = float(data.time)
        status = self.guard.update(
            now=now, sampled_at=now, xyz=data.body(GRASP_BODY).xpos, wrench=wrench,
            retention_ok=now - self.tracking_end <= _TRACKING_WATCHDOG_S,
        )
        if status == "descending":
            return self.waypoints.popleft() if self.waypoints else goal
        return self.hold_goal(data)

    def cancel(self, probe_id, data):
        if probe_id != self.probe_id or not self.active:
            raise ValueError("probe_identity_invalid")
        self.guard.stop("probe_cancelled")
        return self.hold_goal(data)

    def status(self):
        if self.guard is None:
            return None
        return {
            "probe_id": self.probe_id, "status": self.guard.status, "reason": self.guard.reason,
            "travel_m": self.guard.last_travel_m, "calibration_id": self.calibration.calibration_id,
            "wrench_delta_world": self.guard.last_wrench_delta.tolist(),
        }

    def telemetry(self):
        if not self.samples:
            return None
        stamp, wrench, _ = self.samples[-1]
        return {"sample_monotonic_s": stamp, "force_world_n": wrench[:3].tolist(),
                "torque_world_nm": wrench[3:].tolist()}
