# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import pytest

from emet.motion.algo.rrt import RRT
from emet.motion.algo.rrt_connect import RRTConnect
from emet.motion.algo.shortcut import Shortcut
from emet.motion.utils.simple_env import SimpleEnv


def _run_simple_env(planner, env, start, goal, visualize: bool = False):
    """Helper function to run planner and start/goal"""
    print("--------------")
    print("Planner =", planner)
    print("Start =", start)
    print("Goal =", goal)
    random.seed(0)
    np.random.seed(0)
    res = planner.plan(start, goal)
    print("Success:", res.success)
    if res.success:
        print("Plan =", [n.state for n in res.trajectory])
    assert res.success, f"Planning failed with {planner}"
    if visualize:
        if res.success:
            env.show([n.state for n in res.trajectory])
        else:
            env.show([start, goal])
    return res


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_rrt_simple_env(start, goal, obs, visualize: bool = False):
    """Test just pure RRT stuff"""
    env = SimpleEnv(obs)
    planner = RRT(env.get_space(), env.validate)
    _run_simple_env(planner, env, start, goal, visualize)


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_shortcut_rrt_simple_env(start, goal, obs, visualize: bool = False):
    """Test just pure RRT stuff"""
    env = SimpleEnv(obs)
    planner0 = RRT(env.get_space(), env.validate)
    planner1 = Shortcut(planner0)
    res0 = _run_simple_env(planner0, env, start, goal, False)
    res1 = _run_simple_env(planner1, env, start, goal, visualize)
    assert len(res0.trajectory) >= len(res1.trajectory), "Shortcut should not make plans longer"


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_shortcut_rrt_connect_simple_env(start, goal, obs, visualize: bool = False):
    """Test the connect code"""
    env = SimpleEnv(obs)
    planner0 = RRTConnect(env.get_space(), env.validate)
    planner1 = Shortcut(planner0)
    res0 = _run_simple_env(planner0, env, start, goal, False)
    res1 = _run_simple_env(planner1, env, start, goal, visualize)
    assert len(res0.trajectory) >= len(res1.trajectory), "Shortcut should not make plans longer"


def test_configuration_space_extend_yields_first_midpoint():
    """extend must yield the first mid-config (regression: old code skipped it)."""
    from emet.motion.base import ConfigurationSpace

    space = ConfigurationSpace(2, np.zeros(2), np.ones(2) * 10.0, step_size=1.0)
    pts = list(space.extend(np.array([0.0, 0.0]), np.array([3.0, 0.0])))
    assert len(pts) >= 2
    assert pts[-1][0] == pytest.approx(3.0)
    # First yielded point must be between start and goal (not jump to near-goal).
    assert pts[0][0] == pytest.approx(1.0, abs=0.2)


def test_shortcut_never_accepts_segment_through_obstacle():
    """Even with many shortcut iters, mid-configs that hit the obstacle stay invalid."""
    # Box obstacle covers (5,5)→(8,8) exclusive of the lower-left corner (strict >).
    env = SimpleEnv(np.array([5.0, 5.0]), obstacle_size=3.0)
    space = env.get_space()
    from emet.motion.algo.node import TreeNode
    from emet.motion.base import Planner, PlanResult

    class _FixedPlanner(Planner):
        def __init__(self):
            super().__init__(space, env.validate)
            self.nodes = []

        def reset(self):
            self.nodes = []

        def plan(self, start, goal, verbose: bool = False, **kwargs):
            wps = [
                np.array([1.0, 1.0]),
                np.array([1.0, 8.5]),
                np.array([4.0, 8.5]),
                np.array([8.5, 8.5]),
                np.array([9.0, 9.0]),
            ]
            nodes = []
            parent = None
            for w in wps:
                n = TreeNode(w, parent=parent)
                nodes.append(n)
                parent = n
            self.nodes = nodes
            return PlanResult(True, nodes, planner=self)

    random.seed(0)
    np.random.seed(0)
    short = Shortcut(_FixedPlanner(), shortcut_iter=200)
    res = short.plan(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert res.success
    for n in res.trajectory:
        assert env.validate(n.state), f"shortcut path hit obstacle at {n.state}"
    # Interior of the obstacle box must be invalid.
    assert not env.validate(np.array([6.5, 6.5]))
    for i in range(len(res.trajectory) - 1):
        a = res.trajectory[i].state
        b = res.trajectory[i + 1].state
        for qi in space.extend(a, b):
            assert env.validate(qi), f"unsafe shortcut segment {a} -> {b} at {qi}"
    # A direct start→goal chord must have at least one invalid mid sample.
    direct_hit = False
    for qi in space.extend(np.array([1.0, 1.0]), np.array([9.0, 9.0])):
        if not env.validate(qi):
            direct_hit = True
            break
    assert direct_hit, "test setup: diagonal should clip the obstacle"


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_rrt_connect_simple_env(start, goal, obs, visualize: bool = False):
    """Test the connect code"""
    env = SimpleEnv(obs)
    planner = RRTConnect(env.get_space(), env.validate)
    _run_simple_env(planner, env, start, goal, visualize)


if __name__ == "__main__":
    # Run a simple test here
    start = np.array([1, 1])
    goal = np.array([9, 9])
    obs = np.array([0, 9])
    # TODO: enable for debugging
    test_rrt_simple_env(start, goal, obs, visualize=True)
    test_shortcut_rrt_simple_env(start, goal, obs, visualize=True)
    test_rrt_connect_simple_env(start, goal, obs, visualize=True)
    test_shortcut_rrt_connect_simple_env(start, goal, obs, visualize=True)

    start = np.array([1, 4])
    goal = np.array([9, 9])
    obs = np.array([1, 5])
    # TODO: enable if you want to debug this
    # test_rrt_simple_env(start, goal, obs, visualize=True)
    # test_shortcut_rrt_simple_env(start, goal, obs, visualize=True)
    test_rrt_connect_simple_env(start, goal, obs, visualize=True)
    test_shortcut_rrt_connect_simple_env(start, goal, obs, visualize=True)
