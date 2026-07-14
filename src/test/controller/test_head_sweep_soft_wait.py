"""Head-sweep soft wait: exit on near-goal / settled creep, not full joint settle."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from emet.controller.controller_dynamem import (
    DYNAMEM_HEAD_SWEEP_MAX_WAIT_S,
    DynamemController,
)
from emet.motion.kinematics import HelloStretchIdx


def test_head_to_sweep_stops_when_motion_stops_even_if_far_from_goal():
    """Once the head is stationary, proceed — do not wait for exact joint target."""
    agent = DynamemController.__new__(DynamemController)
    calls: list[dict] = []
    jp = np.zeros(20)
    jp[HelloStretchIdx.HEAD_PAN] = 0.2  # far from 1.5
    jp[HelloStretchIdx.HEAD_TILT] = -0.5
    jv = np.zeros(20)  # not moving

    def head_to(pan, tilt, **kwargs):
        calls.append({"pan": pan, "tilt": tilt, **kwargs})

    agent.robot = SimpleNamespace(
        head_to=head_to,
        get_joint_state=lambda: (jp, jv, None),
    )
    t0 = time.time()
    agent._head_to_sweep(1.5, -0.5)
    dt = time.time() - t0
    assert calls and calls[0]["blocking"] is False
    assert calls[0].get("reliable") is False
    assert dt < 0.45
    assert dt >= 0.05


def test_head_to_sweep_exits_early_when_near_goal_even_if_creeping():
    """Slow residual crawl should not burn the full max wait once near the target."""
    agent = DynamemController.__new__(DynamemController)
    jp = np.zeros(20)
    jp[HelloStretchIdx.HEAD_PAN] = 0.55  # within pan tol of 0.6
    jp[HelloStretchIdx.HEAD_TILT] = -0.52
    jv = np.zeros(20)
    jv[HelloStretchIdx.HEAD_PAN] = 0.08  # slow creep below SPEED_TOL=0.20
    n = {"i": 0}

    def get_js():
        n["i"] += 1
        # Tiny pos jitter that used to reset "stopped" under the old 0.012 delta tol.
        jp[HelloStretchIdx.HEAD_PAN] = 0.55 + 0.01 * ((n["i"] % 3) - 1)
        return jp.copy(), jv.copy(), None

    agent.robot = SimpleNamespace(
        head_to=lambda *a, **k: None,
        get_joint_state=get_js,
    )
    t0 = time.time()
    agent._head_to_sweep(0.6, -0.5)
    dt = time.time() - t0
    assert dt < 0.35


def test_head_to_sweep_waits_while_moving_then_exits_on_stop():
    agent = DynamemController.__new__(DynamemController)
    jp = np.zeros(20)
    jp[HelloStretchIdx.HEAD_PAN] = 0.0
    jp[HelloStretchIdx.HEAD_TILT] = -0.5
    state = {"n": 0}

    def get_js():
        state["n"] += 1
        jv = np.zeros(20)
        if state["n"] < 6:
            jv[HelloStretchIdx.HEAD_PAN] = 0.5  # clearly moving
            jp[HelloStretchIdx.HEAD_PAN] = 0.08 * state["n"]
        else:
            jv[HelloStretchIdx.HEAD_PAN] = 0.0
            jp[HelloStretchIdx.HEAD_PAN] = 0.4
        return jp.copy(), jv.copy(), None

    agent.robot = SimpleNamespace(
        head_to=lambda *a, **k: None,
        get_joint_state=get_js,
    )
    t0 = time.time()
    agent._head_to_sweep(1.5, -0.5)
    dt = time.time() - t0
    assert state["n"] >= 6
    assert dt < DYNAMEM_HEAD_SWEEP_MAX_WAIT_S


def test_head_to_sweep_caps_wait_if_never_stops():
    agent = DynamemController.__new__(DynamemController)
    jp = np.zeros(20)
    jv = np.ones(20) * 0.5
    n = {"i": 0}

    def get_js():
        n["i"] += 1
        jp[HelloStretchIdx.HEAD_PAN] = 0.1 * n["i"]  # always changing a lot
        return jp.copy(), jv.copy(), None

    agent.robot = SimpleNamespace(
        head_to=lambda *a, **k: None,
        get_joint_state=get_js,
    )
    t0 = time.time()
    agent._head_to_sweep(1.5, -0.5)
    dt = time.time() - t0
    assert dt >= DYNAMEM_HEAD_SWEEP_MAX_WAIT_S - 0.12
    assert dt < DYNAMEM_HEAD_SWEEP_MAX_WAIT_S + 0.35
