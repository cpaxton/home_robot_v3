"""Offline playback has no actuator endpoints and rejects unknown assets/views."""

import ast
import http.client
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from emet.visualization import herman_rehearsal as viewer


class FakeReplay:
    def __init__(self):
        self.render_threads = []
        self.closed_thread = None

    def state(self):
        return {"hardware_motion_enabled": False}

    def image(self, index, frame, camera, *orbit):
        self.render_threads.append(threading.get_ident())
        if index or frame or camera not in {"orbit", "head_left", "camera_arm"}:
            raise ValueError("unknown replay/frame/camera")
        return b"jpeg"

    def read(self, name):
        assert name in {"head_left.jpg", "head_right.jpg"}
        return b"capture"

    def close(self):
        self.closed_thread = threading.get_ident()


@pytest.fixture
def server(tmp_path):
    video = tmp_path / "example.mp4"
    video.write_bytes(b"0123456789")
    value = viewer.make_server(FakeReplay(), port=0, video=video)
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
    assert status == 200 and json.loads(body) == {
        "hardware_motion_enabled": False,
        "video_available": True,
        "video_webm_available": False,
    }
    assert headers["Cache-Control"] == "no-store"
    assert request(server, "GET", "/frame.jpg?camera=head_left")[1] == b"jpeg"
    assert request(server, "GET", "/head_left.jpg")[1] == b"capture"
    assert request(server, "GET", "/")[0] == 200


@pytest.mark.parametrize(
    "header, expected",
    [
        (None, (200, b"0123456789")),
        ("bytes=2-5", (206, b"2345")),
        ("bytes=7-", (206, b"789")),
        ("bytes=-3", (206, b"789")),
    ],
)
def test_video_can_be_played_and_seeked_with_http_byte_ranges(server, header, expected):
    status, body, headers = request(server, "GET", "/video.mp4", {"Range": header} if header else {})
    assert (status, body) == expected
    assert headers["Content-Type"] == "video/mp4" and headers["Accept-Ranges"] == "bytes"
    assert int(headers["Content-Length"]) == len(body)
    if header:
        assert headers["Content-Range"].endswith("/10")


@pytest.mark.parametrize("header", ["bytes=20-", "bytes=8-3", "bytes=0-1,4-5", "other=0-1", "bytes=-0"])
def test_invalid_video_ranges_fail_without_exposing_other_files(server, header):
    status, body, headers = request(server, "GET", "/video.mp4", {"Range": header})
    assert status == 416 and body == b"" and headers["Content-Range"] == "bytes */10"


def test_optional_webm_fallback_has_correct_mime_and_range_support(server, tmp_path):
    assert request(server, "GET", "/video.webm")[0] == 404
    (tmp_path / "example.webm").write_bytes(b"abcdefghij")
    status, body, headers = request(server, "GET", "/video.webm", {"Range": "bytes=2-4"})
    assert status == 206 and body == b"cde" and headers["Content-Type"] == "video/webm"
    assert json.loads(request(server, "GET", "/api/state")[1])["video_webm_available"]


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


def test_idle_browser_preconnection_cannot_stall_playback(server):
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=2):
        assert request(server, "GET", "/api/state")[0] == 200
        assert request(server, "GET", "/frame.jpg")[0] == 200


def test_concurrent_http_keeps_rendering_and_teardown_on_one_graphics_thread():
    replay = FakeReplay()
    server = viewer.make_server(replay, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as clients:
            results = list(clients.map(lambda _: request(server, "GET", "/frame.jpg"), range(8)))
        assert all(status == 200 for status, _, _ in results)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(set(replay.render_threads)) == 1
    assert replay.closed_thread == replay.render_threads[0]
