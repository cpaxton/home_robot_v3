"""Client for the Innate scheduled trajectory in Emet's adopted arm session."""

import time

from std_srvs.srv import Trigger

PREFIX = "/emet/mars/arm"


class NativeArmGoal:
    def __init__(self, trajectory, release):
        self.trajectory = trajectory
        self.release = release
        self.stop = None

    def cancel(self):
        if self.stop is None:
            self.stop = self.release.call_async(Trigger.Request())

    def outcome(self):
        future = self.stop if self.stop is not None else self.trajectory
        if not future.done():
            return None
        result = future.result()
        if self.stop is not None:
            return "cancelled" if result.success else "stop_unconfirmed"
        return "succeeded" if result.success else "aborted"


class NativeArmDriver:
    def __init__(self, node):
        self.trajectory = None
        try:
            from mars_msgs.srv import GotoJS
        except ModuleNotFoundError as exc:
            if exc.name != "mars_msgs":
                raise
            return
        self.request_type = GotoJS
        self.trajectory = node.create_client(GotoJS, PREFIX + "/goto_js_v2")
        self.acquire = node.create_client(Trigger, PREFIX + "/acquire")
        self.release = node.create_client(Trigger, PREFIX + "/release")

    def ready(self):
        return self.trajectory is not None and all(
            client.service_is_ready() for client in (self.trajectory, self.acquire, self.release)
        )

    def start(self, joint, position, target, duration_s, *, start_arm):
        # ROS responses are handled on the bridge's separate spin thread.
        lease = self.acquire.call_async(Trigger.Request())
        deadline = time.monotonic() + 1.0
        while not lease.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not lease.done() or not lease.result().success:
            raise RuntimeError("Innate native arm ownership could not be confirmed")
        try:
            request = self.request_type.Request()
            request.time = min(2.0, max(1.0, duration_s))
            targets = list(start_arm)
            targets[int(joint.removeprefix("joint")) - 1] = target
            request.data.data = targets
            return NativeArmGoal(self.trajectory.call_async(request), self.release)
        except Exception:
            self.release.call_async(Trigger.Request())
            raise
