"""Client for a driver-guarded, isolated Mars arm joint action."""

import threading

ACTION_NAME = "/mars/arm/recover_joint"
STOP_UNCONFIRMED = -100


class RecoveryGoal:
    def __init__(self, pending):
        self.pending = pending
        self.result = None
        self._cancel_requested = False
        self._cancel_sent = False
        self._lock = threading.Lock()
        pending.add_done_callback(self._accepted)

    def _accepted(self, pending):
        with self._lock:
            handle = pending.result()
            if handle.accepted:
                self.result = handle.get_result_async()
                self._cancel_if_requested(handle)

    def _cancel_if_requested(self, handle):
        finished = self.result is not None and self.result.done()
        if self._cancel_requested and not self._cancel_sent and handle.accepted and not finished:
            self._cancel_sent = True
            handle.cancel_goal_async()

    def cancel(self):
        with self._lock:
            self._cancel_requested = True
            if self.pending.done():
                self._cancel_if_requested(self.pending.result())

    def outcome(self):
        from action_msgs.msg import GoalStatus

        if not self.pending.done():
            return None
        if not self.pending.result().accepted:
            return "rejected"
        if self.result is None or not self.result.done():
            return None
        result = self.result.result()
        if result.result.error_code == STOP_UNCONFIRMED:
            return "stop_unconfirmed"
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            return "succeeded" if result.result.error_code == result.result.SUCCESSFUL else "failed"
        if result.status == GoalStatus.STATUS_CANCELED:
            return "cancelled"
        if result.status == GoalStatus.STATUS_ABORTED:
            return "aborted"
        raise RuntimeError(f"Unknown joint recovery action terminal status: {result.status}")


class IsolatedJointDriver:
    def __init__(self, node):
        self.client = None
        try:
            from control_msgs.action import FollowJointTrajectory
        except ImportError:
            return

        from rclpy.action import ActionClient

        self.action_type = FollowJointTrajectory
        self.client = ActionClient(node, FollowJointTrajectory, ACTION_NAME)

    def ready(self):
        return self.client is not None and self.client.server_is_ready()

    def start(self, joint, position, target, duration_s):
        from rclpy.duration import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = self.action_type.Goal()
        goal.trajectory.joint_names = [joint]
        goal.trajectory.points = [
            JointTrajectoryPoint(positions=[position]),
            JointTrajectoryPoint(
                positions=[target],
                time_from_start=Duration(seconds=max(0.02, duration_s)).to_msg(),
            ),
        ]
        return RecoveryGoal(self.client.send_goal_async(goal))
