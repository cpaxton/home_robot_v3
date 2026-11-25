import pytest

from emet.core.pose_stream import pose_stream_goal


def test_pose_stream_encodes_dual_arm_names_and_omitted_channels():
    target = [0.3, 0.2, 0.7, 0, 0, 0, 1]
    caps = {"arms": ["left", "right"], "base": True, "head": False}
    waypoint = {"arms": {"left": target, "right": target}}
    record = pose_stream_goal(waypoint, duration_s=1, frame="odom", free_base=False, capabilities=caps)
    assert record["pose_stream"]["waypoint"] == waypoint
    with pytest.raises(ValueError, match="free-base"):
        pose_stream_goal(waypoint, duration_s=1, frame="odom", free_base=True, capabilities=caps)
