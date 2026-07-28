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


import numpy as np

from emet.motion.algo.rrt import TreeNode
from emet.motion.base import Planner, PlanResult


class Shortcut(Planner):
    """Define RRT planning problem and parameters. Holds two different trees and tries to connect them with some probabability."""

    def __init__(
        self,
        planner: Planner,
        shortcut_iter: int = 100,
    ):
        self.planner = planner
        super().__init__(self.planner.space, self.planner.validate)
        self.shortcut_iter = shortcut_iter
        self.reset()

    def reset(self):
        self.nodes = None

    def plan(self, start, goal, verbose: bool = False, **kwargs) -> PlanResult:
        """Do shortcutting"""
        self.planner.reset()
        if verbose:
            print("Call internal planner")
        res = self.planner.plan(start, goal, verbose=verbose, **kwargs)
        self.nodes = self.planner.nodes
        if not res.success or len(res.trajectory) < 4:
            # Planning failed so nothing to do here
            return res
        # Now try to shorten things
        # print("Plan =")
        # for i, pt in enumerate(res.trajectory):
        #     print(i, pt.state)
        for _i in range(self.shortcut_iter):
            # Sample two indices
            idx0 = np.random.randint(len(res.trajectory) - 3)
            idx1 = np.random.randint(idx0 + 1, len(res.trajectory))
            node_a = res.trajectory[idx0]
            node_b = res.trajectory[idx1]
            # Extend between them — every mid-config must validate (prefer no
            # shortcut over an unsafe chord through collision).
            previous_node = node_a
            success = False
            goal = np.asarray(node_b.state, dtype=np.float64).reshape(-1)
            for qi in self.space.extend(node_a.state, node_b.state):
                qi_arr = np.asarray(qi, dtype=np.float64).reshape(-1)
                if not self.validate(qi_arr):
                    success = False
                    break
                if np.allclose(qi_arr, goal, atol=1e-9, rtol=0.0):
                    success = True
                    break
                self.nodes.append(TreeNode(qi_arr, parent=previous_node))
                previous_node = self.nodes[-1]
            else:
                # Generator finished without yielding an exact goal — only accept
                # if the final extend point reached the goal under tolerance.
                success = False
            if success:
                node_b.parent = previous_node
        new_trajectory = res.trajectory[-1].backup()
        return PlanResult(True, new_trajectory, planner=self)
