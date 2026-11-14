# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Opt-in native policy and verified primitive handoff, shared by agent and eval."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.motion.release import ReleaseLimits

FINE_GRASP_YAWS = (0.0, -0.175, 0.175, -0.35, 0.35, -0.7, 0.7, -1.57, 1.57)
NATIVE_POLICIES = ("legacy", "bounded_release")


def require_native_policy(name):
    if name not in NATIVE_POLICIES:
        raise ValueError("invalid_native_tamp_policy")
    return name


@dataclass(frozen=True)
class NativePolicy:
    name: str
    payload_binding: str = "preserve"
    release_limits: ReleaseLimits | None = None
    pregrasp_fallback_offset_m: tuple[float, float, float] | None = None
    arm_extension_margin_m: float = 0.0


def native_policy(name):
    require_native_policy(name)
    return (
        NativePolicy(name)
        if name == "legacy"
        else NativePolicy(
            name,
            payload_binding="verified_lift",
            release_limits=ReleaseLimits(0.04, 0.04, open_settle_timeout_s=12.0),
            pregrasp_fallback_offset_m=(0.0, 0.0, 0.08),
            arm_extension_margin_m=0.0005,
        )
    )


def bind_loaded_payload(adapter, plan, *, rebind=True):
    """One measured primitive handoff; never renew a violated/creeping bound.

    The caller already verified the lift and one-second bilateral hold.
    Check the original bound throughout an additional 0.3 s interval, with
    <=2 mm / 0.01 rad change relative to its first sample. Placement still
    replans all segments using the resulting measured transform.
    """
    executor = plan.executor
    collision = executor.collision
    if (
        getattr(executor, "_loaded_binding_done", False)
        or executor.payload_body != plan.object_body
        or collision.payload_body != plan.object_body
        or collision.payload_parent != executor.ee_body
    ):
        return False
    scratch = mujoco.MjData(adapter.model)
    adapter.synchronize(adapter.data)
    active = executor._settling_window(2.0)
    anchor, started = None, float(adapter.data.time)
    while active():
        adapter.synchronize(adapter.data)
        if not executor.payload_retained():
            return False
        mujoco.mj_copyData(scratch, adapter.model, adapter.data)
        mujoco.mj_kinematics(adapter.model, scratch)
        mujoco.mj_collision(adapter.model, scratch)
        obj, ee = scratch.body(plan.object_body), scratch.body(executor.ee_body)
        touched = {
            int(b)
            for c in scratch.contact
            if obj.id in adapter.model.geom_bodyid[c.geom]
            for b in adapter.model.geom_bodyid[c.geom]
            if b != obj.id
        }
        tips = {adapter.model.body(n).id for n in ("rubber_tip_left", "rubber_tip_right")}
        relative = np.eye(4)
        relative[:3, :3] = ee.xmat.reshape(3, 3).T @ obj.xmat.reshape(3, 3)
        relative[:3, 3] = ee.xmat.reshape(3, 3).T @ (obj.xpos - ee.xpos)
        if anchor is None:
            anchor = relative.copy()
            started = float(adapter.data.time)
        drift = float(np.linalg.norm(relative[:3, 3] - anchor[:3, 3]))
        angle = float(np.arccos(np.clip((np.sum(relative[:3, :3] * anchor[:3, :3]) - 1) / 2, -1, 1)))
        stable = bool(
            touched == tips
            and adapter.model.body(plan.support_body).id not in touched
            and drift <= 0.002
            and angle <= 0.01
        )
        executor.event(
            phase="loaded_grasp_check", accepted=stable, interval_translation_m=drift, interval_rotation_rad=angle
        )
        if not stable:
            return False
        if adapter.data.time - started >= 0.3:
            previous = collision.payload_transform.copy()
            if rebind:
                collision.set_payload(adapter.model, adapter.data, plan.object_body, executor.ee_body)
            executor._loaded_binding_done = True
            executor.event(
                phase="loaded_grasp_binding",
                accepted=True,
                previous_reference=previous.tolist(),
                reference=collision.payload_transform.tolist(),
                reference_changed=rebind,
                placement_replan_required=True,
            )
            return True
        time.sleep(0.05)
    return False
