"""The live operator view cannot acquire motor authority or disguise stale state."""

import ast
import base64
import http.client
import json
import math
import threading
from pathlib import Path

import pytest

from emet.robots.innate_mars.passive_ssh import relay_command
from emet.visualization.operator_console import ConsoleState, make_server


def packet(sequence=1):
    timing = {"source_age_s": 0.01, "source_stamp_ns": 10**9}
    return {
        "session": "robot-session",
        "sequence": sequence,
        "kind": "state",
        "samples": {
            "arm": {
                "names": [f"joint{i}" for i in range(1, 7)],
                "position": [0] * 6,
                "velocity": [0] * 6,
                "effort_proxy": [0] * 6,
                **timing,
            },
            "odom": {
                "position": [1, 2, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
                "twist_child": [0] * 6,
                "frame_id": "odom",
                "child_frame_id": "base_link",
                **timing,
            },
            "hand": {"position": [1.2, 2, 0.5], "quaternion_xyzw": [0, 0, 0, 1], "frame_id": "odom", **timing},
        },
    }


@pytest.fixture
def state():
    value = ConsoleState()
    value.ingest(packet())
    return value


def edit(state, **command):
    return state.edit({"revision": state.snapshot()["revision"], **command})


def test_target_contract_is_masked_se2_se3_with_readonly_grip_and_no_authority(state):
    data = state.snapshot()
    assert not data["motion_enabled"] and data["commands_sent"] == 0
    assert data["targets"]["base"] == {"pose": [1, 2, 0], "mask": [True] * 3}
    assert not data["targets"]["hands"][0]["active"]
    data = edit(state, op="mode", mode=3)
    assert data["targets"]["base"]["mask"] == [False] * 3
    assert data["targets"]["hands"][0]["active"]
    assert data["targets"]["hands"][0]["gripper"]["mask"] == [False]
    edit(state, op="nudge", translation=[0.01, 0.01, 0.01], rotation=[0.05, 0.04, 0.03])
    hand = state.snapshot()["targets"]["hands"][0]["pose"]
    assert hand["position"] == [1.21, 2.01, 0.51]
    assert math.isclose(sum(x * x for x in hand["quaternion_xyzw"]), 1)
    assert not state.snapshot()["motion_enabled"]


def test_base_preview_is_nonholonomic_and_limits_are_atomic(state):
    edit(state, op="nudge", translation=[0.01, 0, 0], rotation=[0, 0, 0.1])
    edit(state, op="nudge", translation=[0.01, 0, 0], rotation=[0, 0, 0])
    base = state.snapshot()["targets"]["base"]["pose"]
    assert base[:2] == pytest.approx([1.01 + 0.01 * math.cos(0.1), 2 + 0.01 * math.sin(0.1)])
    frozen = state.snapshot()["targets"]
    with pytest.raises(ValueError, match="forward travel"):
        edit(state, op="nudge", translation=[0, 0.01, 0], rotation=[0, 0, 0])
    assert state.snapshot()["targets"] == frozen
    for _ in range(6):
        edit(state, op="nudge", translation=[0, 0, 0], rotation=[0, 0, 0.1])
    frozen = state.snapshot()["targets"]
    with pytest.raises(ValueError, match="envelope"):
        edit(state, op="nudge", translation=[0, 0, 0], rotation=[0, 0, 0.1])
    assert state.snapshot()["targets"] == frozen


def test_stale_sources_and_local_disconnect_disable_previews_without_synthetic_state():
    clock = [1.0]
    state = ConsoleState(clock=lambda: clock[0])
    state.ingest(packet())
    before = state.snapshot()["targets"]
    clock[0] += 2
    assert not state.snapshot()["connected"]
    with pytest.raises(ValueError, match="fresh odometry"):
        edit(state, op="nudge", translation=[0.01, 0, 0], rotation=[0, 0, 0])
    assert state.snapshot()["targets"] == before
    old = packet(2)
    old["samples"]["odom"]["source_age_s"] = 10
    state.ingest(old)
    assert state.snapshot()["connected"] and not state.snapshot()["samples"]["odom"]["fresh"]


def test_reordered_and_cross_frame_edits_are_rejected(state):
    revision = state.snapshot()["revision"]
    edit(state, op="mode", mode=2)
    with pytest.raises(ValueError, match="stale preview"):
        state.edit({"revision": revision, "op": "mode", "mode": 1})
    bad = packet(2)
    bad["samples"]["hand"]["frame_id"] = "base_link"
    with pytest.raises(ValueError, match="share odom"):
        state.ingest(bad)


def test_preview_reset_uses_measurement_without_recentering_envelope(state):
    origin = state.snapshot()["origin"].copy()
    edit(state, op="nudge", translation=[0.01, 0, 0], rotation=[0, 0, 0.1])
    measured = packet(2)
    measured["samples"]["odom"]["position"][0] += 0.1
    state.ingest(measured)
    data = edit(state, op="reset")
    assert data["targets"]["base"]["pose"] == [1.1, 2, 0]
    assert data["origin"] == origin and not data["motion_enabled"]
    outside = packet(3)
    outside["samples"]["odom"]["position"][0] += 0.5
    state.ingest(outside)
    with pytest.raises(ValueError, match="not recentered"):
        edit(state, op="reset")
    assert state.snapshot()["origin"] == origin


def test_public_snapshot_cannot_recenter_internal_preview_limits(state):
    snapshot = state.snapshot()
    snapshot["origin"][0] = 99
    snapshot["blockers"].clear()
    snapshot["targets"]["base"]["pose"][0] = 99
    actual = state.snapshot()
    assert actual["origin"] == [1, 2, 0]
    assert actual["targets"]["base"]["pose"] == [1, 2, 0]
    assert actual["blockers"]


@pytest.mark.parametrize("position", [[0] * 5, [0] * 5 + [float("nan")], [True] * 6])
def test_malformed_robot_state_is_not_accepted(position):
    state = ConsoleState()
    data = packet()
    data["samples"]["arm"]["position"] = position
    with pytest.raises(ValueError, match="finite"):
        state.ingest(data)
    assert not state.snapshot()["connected"]


def test_camera_expiry_unknown_stamp_and_duplicate_packets_do_not_look_live():
    clock = [1.0]
    state = ConsoleState(clock=lambda: clock[0])
    image = {
        "session": "r",
        "sequence": 1,
        "kind": "camera",
        "name": "wrist",
        "jpeg": base64.b64encode(b"\xff\xd8image").decode(),
        "source_age_s": 0.01,
    }
    state.ingest(image)
    assert state.frame("wrist")
    clock[0] += 2
    state.ingest(image)
    assert state.frame("wrist") is None
    image.update(sequence=2, source_age_s=None)
    state.ingest(image)
    assert state.frame("wrist") is None


@pytest.fixture
def server(state):
    http = make_server(state, port=0)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield http
    http.shutdown()
    http.server_close()
    thread.join(timeout=2)


def request(server, method, path, payload=None, **headers):
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    client.request(method, path, None if payload is None else json.dumps(payload), headers=headers)
    response = client.getresponse()
    result = (response.status, response.read(), dict(response.headers))
    client.close()
    return result


def test_http_preview_cannot_publish_or_enable_motion_and_cross_origin_is_blocked(server, state):
    status, raw, headers = request(server, "GET", "/api/state")
    assert status == 200 and not json.loads(raw)["motion_enabled"]
    assert headers["Cache-Control"] == "no-store"
    for path in ("/api/command", "/api/enable", "/api/stop"):
        assert request(server, "POST", path, {"op": "drive"})[0] == 404
    assert request(server, "POST", "/api/preview", {}, Origin="https://evil.example")[0] == 403
    assert request(server, "GET", "/api/state", Host="evil.example")[0] == 403
    assert request(server, "POST", "/api/preview", [], **{"Content-Type": "application/json"})[0] == 409
    status, body, _ = request(
        server,
        "POST",
        "/api/preview",
        {"revision": state.snapshot()["revision"], "op": "mode", "mode": 2},
        **{"Content-Type": "application/json"},
    )
    assert status == 200 and json.loads(body)["targets"]["preview_only"]


def test_html_exposes_lost_focus_deadman_and_readonly_motion_state(server):
    status, raw, _ = request(server, "GET", "/")
    assert status == 200
    page = raw.decode()
    assert "visibilitychange" in page and "pointercancel" in page and "keys.has(' ')" in page
    assert "Robot stop · unavailable" in page and "no robot commands" in page


@pytest.mark.parametrize("target", ["-oProxyCommand=evil", "user@host;touch /tmp/evil", "host", "user@host\ncommand"])
def test_relay_host_cannot_inject_shell_or_ssh_options(target):
    with pytest.raises(ValueError, match="destination"):
        relay_command(target)


def test_remote_program_contains_only_passive_ros_interactions():
    from emet.robots.innate_mars import passive_relay

    tree = ast.parse(Path(passive_relay.__file__).read_text())
    methods = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "create_subscription" in methods
    assert not methods & {"create_publisher", "create_client", "call_async", "command_velocity", "publish"}
    cmd = relay_command("jetson1@192.168.1.43")
    assert "StrictHostKeyChecking=yes" in cmd and "BatchMode=yes" in cmd
