# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Client helpers for PandaOmron's native joint-position bridge."""

import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from emet.controller.generic_zmq_client import GenericZmqClient
from emet.robots.pandaomron import ARM_JOINTS, CONTROL_JOINTS, PANDA_ARM_CHAIN, TORSO_JOINT


class PandaOmronClient(GenericZmqClient):
    def get_native_feedback(self):
        """Read atomic, fresh joint/base feedback without waiting for RGB-D."""
        from emet.controller.manipulation.native_inputs import JointStateEvidence, ObservationUnavailable

        with self._obs_lock:
            state = self._state
            received = getattr(self, "_state_received_monotonic", None)
            if not isinstance(state, dict) or received is None:
                raise ObservationUnavailable("joint_state_unknown")
            joints = JointStateEvidence(state.get("joint_positions"), received).require(max_age_s=.25)
            base = np.asarray(state.get("base_pose"), dtype=float)
            session = state.get("emet_session", getattr(self, "_emet_session_cache", None))
            if base.shape != (3,) or not np.isfinite(base).all() or not isinstance(session, dict):
                raise ObservationUnavailable("invalid_joint_feedback")
            return SimpleNamespace(joint=joints, gps=base[:2].copy(), compass=base[2:].copy(),
                                   emet_session=session)

    def hold_position(self):
        """Replace an outstanding arm goal with measured joint positions."""
        measured = self.get_joint_state_evidence()
        positions = measured.require(max_age_s=.25)
        session = self.get_emet_session() or {}
        limits = session.get("native_joint_goal_limits", {})
        goal = np.array([np.clip(positions[index], *limits[name])
                         for index, name in enumerate(CONTROL_JOINTS)])
        self.send_action({"joint": goal.tolist()}, timeout=.4)
        return True

    def open_gripper(self, gripper_name="right_gripper", amount=None, *, blocking=True):
        """Open the native fingers and confirm their measured motor positions."""
        target = np.array([value for _, value in PANDA_ARM_CHAIN.gripper_open_configuration])
        if amount is not None:
            value = float(amount)
            if not np.isfinite(value) or not 0 <= value <= 1.:
                raise ValueError("invalid_native_gripper_opening")
            target = target * value
        self.command_gripper_joints(target)
        if not blocking:
            return True
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            q, dq, _ = self.get_joint_state(timeout=.1)
            if q is not None and np.max(np.abs(q[-2:] - target)) < .001:
                if np.max(np.abs(dq[-2:])) < .01:
                    return True
            time.sleep(.02)
        return False

    def command_gripper_joints(self, targets) -> bool:
        self.send_action({"gripper_joint_targets": np.asarray(targets, dtype=float).tolist()})
        return True

    def guarded_place_probe(self, waypoints, *, contact_band_m, calibration_id, retention_capture):
        """Submit a checked descent and keep the bridge informed of live retention."""
        capabilities = (self.get_emet_session() or {}).get("capabilities", {})
        if capabilities.get("guarded_placement_probe") is not True:
            raise RuntimeError("probe_unavailable")
        probe_id = uuid4().hex
        capture = retention_capture()
        sent = False
        finished = False
        try:
            sent = True
            self.send_action({"placement_probe": {
                "probe_id": probe_id, "calibration_id": calibration_id, "capture_end_s": capture,
                "contact_band_m": list(contact_band_m), "waypoints": np.asarray(waypoints).tolist(),
            }}, timeout=0.4)
            deadline = time.monotonic() + 3.4
            while time.monotonic() < deadline:
                status = (self._state or {}).get("placement_probe_status")
                if status and status["probe_id"] == probe_id and status["status"] in ("contact", "stopped"):
                    finished = True
                    return status
                fresh = retention_capture()
                if fresh > capture:
                    self.send_action({"placement_probe_keepalive": {
                        "probe_id": probe_id, "capture_end_s": fresh,
                    }}, timeout=0.2)
                    capture = fresh
                time.sleep(0.025)
            raise TimeoutError("probe_stop_unconfirmed")
        finally:
            if sent and not finished:
                try:
                    self.send_action({"placement_probe_cancel": probe_id}, timeout=0.2)
                except (RuntimeError, TimeoutError):
                    # The independent bridge watchdog also holds when progress
                    # goes unacknowledged. Without a receipt callers keep grip.
                    pass

    def arm_to(self, joint_angles=None, gripper=None, head=None, blocking=True, **kwargs) -> bool:
        """Command seven Panda angles while holding the current torso height."""
        if head is not None:
            raise ValueError("PandaOmron has no independently actuated head")
        arm = np.asarray(joint_angles, dtype=float)
        if arm.shape != (len(ARM_JOINTS),) or not np.isfinite(arm).all():
            raise ValueError("expected seven finite Panda arm joint angles")
        timeout = float(kwargs.get("timeout", 5.0))
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        measured = self.get_joint_positions(timeout=timeout)
        if measured is None:
            return False
        session = self.get_emet_session() or {}
        limits = (session.get("native_joint_goal_limits") or {}).get(TORSO_JOINT)
        if limits is None:
            raise RuntimeError("native torso joint limits are unavailable")
        goal = np.r_[arm, np.clip(measured[len(ARM_JOINTS)], *limits)]
        action = {"joint": goal.tolist()}
        if gripper is not None:
            action["gripper"] = float(gripper)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        self.send_action(action, timeout=remaining)
        if not blocking:
            return True
        while time.monotonic() < deadline:
            q, dq, _ = self.get_joint_state(timeout=min(0.1, deadline - time.monotonic()))
            if q is not None and np.max(np.abs(q[: len(CONTROL_JOINTS)] - goal)) < 0.006:
                if np.max(np.abs(dq[: len(CONTROL_JOINTS)])) < 0.02:
                    return True
            time.sleep(0.02)
        return False
