"""Robot-independent pose-stream messages over the existing command protocol."""

import math
from numbers import Real

from emet.core.command_client import (
    cancel_reactive_goal,
    command_receipt,
    send_command,
    update_reactive_goal,
    wait_reactive_goal,
)

FORMAT = "pose_stream_v1"


def _vector(values, size):
    values = list(values)
    if len(values) != size or not all(
        isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x) for x in values
    ):
        raise ValueError(f"expected {size} finite pose coordinates")
    return list(map(float, values))


def pose_stream_goal(waypoint, *, duration_s, frame, free_base, capabilities):
    """Validate a waypoint against the connected robot's advertised channels."""
    if frame not in ("odom", "base_link") or type(free_base) is not bool:
        raise ValueError("invalid pose-stream frame or free-base mode")
    if (
        not isinstance(duration_s, Real)
        or isinstance(duration_s, bool)
        or not math.isfinite(duration_s)
        or duration_s <= 0
    ):
        raise ValueError("waypoint duration must be positive and finite")
    if not isinstance(waypoint, dict) or not waypoint or set(waypoint) - {"base", "head", "arms"}:
        raise ValueError("waypoint must supply base, head or named arms")
    if free_base and capabilities.get("free_base") is not True:
        raise ValueError("peer does not support free-base coordination")
    clean = {}
    for channel, size in (("base", 3), ("head", 2)):
        if channel in waypoint:
            if capabilities.get(channel) is not True:
                raise ValueError(f"peer does not support {channel} pose streaming")
            clean[channel] = _vector(waypoint[channel], size)
    if "arms" in waypoint:
        arms = waypoint["arms"]
        if not isinstance(arms, dict) or not arms or set(arms) - set(capabilities.get("arms", ())):
            raise ValueError("unsupported or empty named arm selection")
        clean["arms"] = {name: _vector(pose, 7) for name, pose in arms.items()}
        if any(abs(math.hypot(*p[3:]) - 1.0) > 1e-4 for p in clean["arms"].values()):
            raise ValueError("arm pose requires a unit xyzw quaternion")
    return {
        "pose_stream": {
            "format": FORMAT,
            "frame": frame,
            "free_base": free_base,
            "duration_s": float(duration_s),
            "waypoint": clean,
        }
    }


class PoseStream:
    """One reactive command identity, updated by subsequent waypoints."""

    def __init__(self, client, *, frame="odom", free_base=False):
        self.client, self.frame, self.free_base = client, frame, free_base
        self.action = None
        self.closed = False
        self.finishing = False

    def send(self, waypoint, *, duration_s):
        if self.closed or self.finishing:
            raise RuntimeError("pose stream is closed or finishing")
        caps = getattr(self.client, "_state", {}).get("pose_stream")
        if not isinstance(caps, dict) or caps.get("format") != FORMAT:
            raise RuntimeError("peer does not advertise a pose-stream executor")
        goal = pose_stream_goal(
            waypoint, duration_s=duration_s, frame=self.frame, free_base=self.free_base, capabilities=caps
        )
        if self.action is None:
            self.action = send_command(
                self.client, {"reactive_goal": goal, "reactive_timeout_s": max(3.0, duration_s + 2)}
            )
        else:
            update_reactive_goal(self.client, self.action, goal)
        return self.action

    def finish(self):
        if self.action is None or self.closed:
            raise RuntimeError("pose stream has no active waypoint")
        if not self.finishing:
            update_reactive_goal(self.client, self.action, {"pose_stream": {"format": FORMAT, "finish": True}})
            self.finishing = True

    def wait(self, timeout_s):
        if self.action is None or self.closed:
            raise RuntimeError("pose stream has no active waypoint")
        self.finish()
        result = wait_reactive_goal(self.client, self.action, timeout=timeout_s)
        receipt = command_receipt(self.client, self.action)
        if not receipt or not receipt.get("result", {}).get("stop_confirmed"):
            raise RuntimeError("pose-stream completion did not confirm a stop")
        self.closed = True
        return result

    def cancel(self):
        if self.closed or self.action is None:
            return True
        confirmed = cancel_reactive_goal(self.client, self.action)
        if confirmed:
            self.closed = True
        return confirmed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.cancel():
            raise RuntimeError("pose-stream stop unconfirmed")


class PoseStreamClientMixin:
    def open_pose_stream(self, *, frame="odom", free_base=False):
        return PoseStream(self, frame=frame, free_base=free_base)
