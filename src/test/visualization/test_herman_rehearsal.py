"""Offline playback has no actuator endpoints and rejects unknown assets/views."""

import ast
import http.client
import json
import threading
from pathlib import Path

import pytest

from emet.visualization import herman_rehearsal as viewer


class FakeReplay:
    def state(self):
        return {"hardware_motion_enabled": False}

    def image(self, index, frame, camera, *orbit):
        if index or frame or camera not in {"orbit", "head_left", "camera_arm"}:
            raise ValueError("unknown replay/frame/camera")
        return b"jpeg"

    def read(self, name):
        assert name in {"head_left.jpg", "head_right.jpg"}
        return b"capture"


@pytest.fixture
def server():
    value = viewer.make_server(FakeReplay(), port=0)
    thread = threading.Thread(target=value.serve_forever, daemon=True)
    thread.start()
    yield value
    value.shutdown()
    value.server_close()
    thread.join(timeout=2)


def request(server, method, path, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read(), dict(response.headers)
    finally:
        connection.close()


def test_local_replay_and_image_endpoints(server):
    status, body, headers = request(server, "GET", "/api/state")
    assert status == 200 and json.loads(body) == {"hardware_motion_enabled": False}
    assert headers["Cache-Control"] == "no-store"
    assert request(server, "GET", "/frame.jpg?camera=head_left")[1] == b"jpeg"
    assert request(server, "GET", "/head_left.jpg")[1] == b"capture"
    assert request(server, "GET", "/")[0] == 200


@pytest.mark.parametrize("path", ["/frame.jpg?replay=-1", "/frame.jpg?frame=bad", "/frame.jpg?camera=unknown"])
def test_replay_frame_camera_validation(server, path):
    assert request(server, "GET", path)[0] == 400


@pytest.mark.parametrize("headers", [{"Host": "outside.example"}, {"Origin": "https://outside.example"}])
def test_cross_origin_requests_are_rejected(server, headers):
    assert request(server, "GET", "/api/state", headers)[0] == 403


def test_no_motion_or_arbitrary_asset_endpoints(server):
    assert request(server, "POST", "/api/control")[0] == 501
    assert request(server, "GET", "/../robot_description.urdf")[0] == 404
    source = Path(viewer.__file__).read_text()
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"mj_step", "create_publisher", "create_client", "send_command", "predict"}
