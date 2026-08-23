# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from emet.mars import (
    MarsCameraHealth,
    parse_bridge_status_output,
    parse_camera_health_output,
    print_bridge_status,
)


def test_parse_bridge_status_output_running():
    raw = """10105 /usr/bin/python3 /home/jetson1/innate-os/ros2_ws/install/innate_mars_bridge/lib/innate_mars_bridge/server --ros-args -r __node:=innate_mars_zmq_server
---
LISTEN 0      100          0.0.0.0:4404       0.0.0.0:*
LISTEN 0      100          0.0.0.0:4401       0.0.0.0:*
LISTEN 0      100          0.0.0.0:4403       0.0.0.0:*
---
[server-1] [INFO] [1782397281.975882818] [innate_mars_zmq_server]: InnateMarsRosInterface: waiting for cameras...
"""
    status = parse_bridge_status_output("herman", "jetson1", raw)
    assert status.pid == 10105
    assert status.process_running
    assert status.listening_ports >= {4401, 4403, 4404}
    assert status.ready_for_stream
    assert status.headline_log() is not None
    assert "waiting for cameras" in (status.headline_log() or "").lower()


def test_parse_bridge_status_output_not_running():
    raw = """---
---
---tmux-unavailable---
"""
    status = parse_bridge_status_output("herman", "jetson1", raw)
    assert not status.process_running
    assert not status.listening_ports
    assert not status.ready_for_stream


def test_parse_bridge_status_ignores_pgrep_self_match_with_echo_separators():
    """pgrep -af echoes the remote zsh -c probe, which embeds echo '---' substrings."""
    raw = (
        "7461 /usr/bin/python3 /home/jetson1/innate-os/ros2_ws/install/innate_mars_bridge/"
        "lib/innate_mars_bridge/server --ros-args -r __node:=innate_mars_zmq_server\n"
        "11541 zsh -c pgrep -af 'innate_mars_zmq_server' 2>/dev/null || true; "
        "echo '---'; ss -tlnH 2>/dev/null | grep -E ':440[1-4] ' || true; "
        "echo '---'; tmux has-session -t ros_nodes 2>/dev/null && "
        "tmux capture-pane -t ros_nodes:emet-bridge -p 2>/dev/null | tail -12 "
        "|| echo '---tmux-unavailable---'\n"
        "---\n"
        "LISTEN 0      100          0.0.0.0:4401  0.0.0.0:*\n"
        "LISTEN 0      100          0.0.0.0:4402  0.0.0.0:*\n"
        "LISTEN 0      100          0.0.0.0:4403  0.0.0.0:*\n"
        "LISTEN 0      100          0.0.0.0:4404  0.0.0.0:*\n"
        "---\n"
        "[server-1] Running all...\n"
        "[server-1] Warning: wrist camera (/mars/arm/image_raw) is all black\n"
    )
    status = parse_bridge_status_output("herman", "jetson1", raw)
    assert status.pid == 7461
    assert status.listening_ports == {4401, 4402, 4403, 4404}
    assert status.ready_for_stream


def test_print_bridge_status_smoke(capsys):
    status = parse_bridge_status_output(
        "herman",
        "jetson1",
        "9999 python3 innate_mars_zmq_server\n---\nLISTEN 0 100 0.0.0.0:4401 0.0.0.0:*\n---\n",
    )
    print_bridge_status(status, profile="herman", show_next_steps=False)
    out = capsys.readouterr().out
    assert "herman" in out
    assert "1/4" in out
    assert "pid 9999" in out


def test_parse_camera_health_missing_arducam():
    health = parse_camera_health_output("arducam=0\narm_pubs=0\nhead_pubs=1\nby_id=usb-3D_USB_Camera_…-video-index0,\n")
    assert health.arducam_symlink is False
    assert health.arm_publishers == 0
    assert health.head_left_publishers == 1
    assert health.head_ok
    assert not health.wrist_ok
    assert "Arducam" in health.note


def test_print_bridge_status_includes_camera_line(capsys):
    status = parse_bridge_status_output(
        "herman",
        "jetson1",
        "9999 python3 innate_mars_zmq_server\n---\n"
        "LISTEN 0 100 0.0.0.0:4401 0.0.0.0:*\n"
        "LISTEN 0 100 0.0.0.0:4403 0.0.0.0:*\n---\n",
    )
    status.camera = MarsCameraHealth(
        arm_publishers=0,
        head_left_publishers=1,
        arducam_symlink=False,
        note="wrist down: no Arducam V4L symlink under /dev/v4l/by-id/",
    )
    print_bridge_status(status, profile="herman", show_next_steps=False)
    out = capsys.readouterr().out
    assert "cameras:" in out
    assert "wrist down" in out
    assert "Arducam" in out
