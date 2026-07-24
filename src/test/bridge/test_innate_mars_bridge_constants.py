# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from innate_mars_bridge.constants import (
    ARM_STATE_TOPIC,
    EE_IMAGE_TOPIC,
    EXPECTED_TF_FRAMES,
    EXPECTED_TOPICS,
    HEAD_LEFT_IMAGE_TOPIC,
    INNATE_OS_GIT_REF,
    INNATE_OS_REPO,
    NAVIGATE_TO_POSE_ACTION,
    ODOM_TOPIC,
    SPIN_ACTION,
)


def test_expected_topics_cover_bridge_subscriptions():
    assert ARM_STATE_TOPIC in EXPECTED_TOPICS
    assert ODOM_TOPIC in EXPECTED_TOPICS
    assert HEAD_LEFT_IMAGE_TOPIC in EXPECTED_TOPICS
    assert EE_IMAGE_TOPIC in EXPECTED_TOPICS


def test_innate_os_pin_documented():
    assert INNATE_OS_REPO.startswith("https://")
    assert INNATE_OS_GIT_REF


def test_nav2_action_name():
    assert NAVIGATE_TO_POSE_ACTION == "navigate_to_pose"
    assert SPIN_ACTION == "spin"


def test_tf_frames_include_cameras():
    assert "camera_optical_frame" in EXPECTED_TF_FRAMES
    assert "ee_link" in EXPECTED_TF_FRAMES
