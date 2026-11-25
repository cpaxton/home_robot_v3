"""Shared measured-start waypoint conversion for single-hand WBC."""

import numpy as np

from emet.core.pose_stream import FORMAT as POSE_STREAM_FORMAT
from emet.core.pose_stream import pose_stream_goal
from emet.rl.manipulation.tracking import TaskPoseTrajectory


class SingleHandPoseStream:
    def __init__(self):
        self._state = None

    @staticmethod
    def capabilities():
        return {"format": POSE_STREAM_FORMAT, "base": True, "head": False, "arms": ["arm"], "free_base": True}

    @staticmethod
    def is_finish(goal):
        return goal == {"pose_stream": {"format": POSE_STREAM_FORMAT, "finish": True}}

    def validate(self, goal):
        if self.is_finish(goal):
            return None
        record = goal["pose_stream"]
        if (
            set(goal) != {"pose_stream"}
            or not isinstance(record, dict)
            or set(record) != {"format", "frame", "free_base", "duration_s", "waypoint"}
            or record["format"] != POSE_STREAM_FORMAT
        ):
            raise ValueError("invalid pose-stream command")
        return pose_stream_goal(
            record["waypoint"],
            duration_s=record["duration_s"],
            frame=record["frame"],
            free_base=record["free_base"],
            capabilities=self.capabilities(),
        )["pose_stream"]

    def trajectory(self, control, goal, *, new):
        from scipy.spatial.transform import Rotation

        record = self.validate(goal)
        c = control
        base, hand = c.base_pose(), c.ee_pose()
        mode = (record["frame"], record["free_base"])
        if new:
            self._state = (
                mode,
                np.r_[base, c.data.xpos[c.base, 2]],
                base.copy(),
                hand.copy(),
                not record["free_base"],
            )
        previous_mode, anchor, base_goal, hand_goal, base_active = self._state
        if mode != previous_mode:
            raise ValueError("pose stream cannot change frame or base mode")
        waypoint = record["waypoint"]
        local = mode[0] == "base_link"
        rotation = Rotation.from_euler("z", anchor[2])
        if "base" in waypoint:
            base_goal = np.array(waypoint["base"])
            if local:
                base_goal[:2] = anchor[:2] + rotation.apply([*base_goal[:2], 0])[:2]
                base_goal[2] += anchor[2]
            base_active = True
        if "arms" in waypoint:
            pose = waypoint["arms"]["arm"]
            orient = Rotation.from_quat(pose[3:])
            xyz = np.array(pose[:3])
            if local:
                xyz = np.r_[anchor[:2], anchor[3]] + rotation.apply(xyz)
                orient = rotation * orient
            hand_goal = np.r_[xyz, orient.as_matrix()[:, :2].ravel()]
        self._state = (mode, anchor, base_goal, hand_goal, base_active)
        return TaskPoseTrajectory(
            [0, record["duration_s"]],
            [base, base_goal],
            [hand, hand_goal],
            [c.grip_reference] * 2,
            base_mask=[base_active] * 3,
        )
