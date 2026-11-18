"""Offline Herman replay: captured images beside calibrated static MuJoCo renders.

This process loads recorded qpos only. It never steps physics, loads an actor,
opens a robot connection, or offers a motion endpoint.
"""

import argparse
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import mujoco
import numpy as np

from emet.rl.manipulation.herman_rehearsal import sha, verify_artifacts


class Replay:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.inventory = verify_artifacts(self.directory)["files"]
        self.report = json.loads(self.read("report.json"))
        if self.report.get("format") != "herman-rehearsal-v1" or not self.report.get("complete"):
            raise ValueError("complete offline rehearsal required")
        self.model = mujoco.MjModel.from_binary_path(str(self.directory / "scene.mjb"))
        # Display illumination only; recorded dynamics and camera calibration
        # remain those of the checksummed artifact. This is no RGB photometric fit.
        self.model.vis.headlight.active = 1
        self.model.vis.headlight.ambient[:] = 0.4
        self.model.vis.headlight.diffuse[:] = 0.7
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.lock = threading.Lock()
        self.entries = [
            {"key": "captured", "label": "Captured posture · static FK only", "replay": "captured-replay.npz"}
        ]
        for name, rows in self.report["results"].items():
            for row in rows:
                if row.get("replay"):
                    self.entries.append(
                        {
                            "key": name + ":" + row["id"],
                            "label": name
                            + " · "
                            + row["id"].rsplit(":", 1)[-1]
                            + " · "
                            + ("pass" if row["success"] else row["code"]),
                            "replay": row["replay"],
                            "metrics": row["metrics"],
                        }
                    )
        self.loaded = {}
        self.targets = {}
        self.anchors = {}

    def read(self, name):
        if name not in self.inventory or Path(name).name != name:
            raise ValueError("unknown replay asset")
        path = self.directory / name
        if path.is_symlink() or sha(path) != self.inventory[name]:
            raise ValueError("replay asset changed")
        return path.read_bytes()

    def frames(self, index):
        if not 0 <= index < len(self.entries):
            raise ValueError("unknown replay")
        if index not in self.loaded:
            with np.load(io.BytesIO(self.read(self.entries[index]["replay"])), allow_pickle=False) as archive:
                qpos, times = archive["qpos"].copy(), archive["time_s"].copy()
                targets = archive["targets"].copy()
            if (
                qpos.ndim != 2
                or qpos.shape[1] != self.model.nq
                or not 0 < len(qpos) <= 10000
                or times.shape != (len(qpos),)
                or not np.isfinite(qpos).all()
                or not np.isfinite(times).all()
                or np.any(np.diff(times) < 0)
                or targets.shape != (len(qpos), 12)
                or not np.isfinite(targets).all()
            ):
                raise ValueError("malformed finite replay")
            self.loaded[index] = (qpos, times)
            self.targets[index] = targets
        return self.loaded[index]

    def state(self):
        entries = []
        with self.lock:
            for index, entry in enumerate(self.entries):
                _, times = self.frames(index)
                entries.append({**entry, "frames": len(times), "duration_s": float(times[-1])})
        return {
            "entries": entries,
            "audit": self.report["audit"],
            "summary": self.report.get("summary", {}),
            "depth": self.report["depth"],
            "proposed_start": self.report.get("proposed_start"),
            "hardware_motion_enabled": False,
            "message": "Offline recorded replay. Unobserved geometry is unknown. Wrist optics unverified.",
        }

    def render(self, index, frame, camera, azimuth=225, elevation=-15):
        if camera not in {"orbit", "hand", "head_left", "camera_arm"}:
            raise ValueError("unknown camera")
        if not np.isfinite([azimuth, elevation]).all() or not -89 <= elevation <= -5 or abs(azimuth) > 3600:
            raise ValueError("bounded orbit required")
        with self.lock:
            qpos, times = self.frames(index)
            if not 0 <= frame < len(qpos):
                raise ValueError("unknown frame")
            if index not in self.anchors:
                self.data.qpos[:] = qpos[0]
                mujoco.mj_forward(self.model, self.data)
                self.anchors[index] = self.data.body("ee_link").xpos.copy()
            self.data.qpos[:] = qpos[frame]
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            view = camera
            if camera in {"orbit", "hand"}:
                view = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(view)
                view.lookat[:] = self.anchors[index] if camera == "hand" else [0.03, 0, 0.18]
                view.distance = 0.28 if camera == "hand" else 0.65
                view.azimuth, view.elevation = azimuth, elevation
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0  # Hide conservative transparent collision proxies visually only.
            self.renderer.update_scene(self.data, camera=view, scene_option=option)
            if camera in {"orbit", "hand"}:
                body = self.data.body("ee_link")
                self._axes(body.xpos.copy(), body.xmat.copy().reshape(3, 3), [1, 0.5, 0.1, 1], 0.045)
                if index:
                    target = self.targets[index][frame, 3:]
                    columns = target[3:].reshape(3, 2)
                    rotation = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
                    self._axes(target[:3], rotation, [0.2, 0.9, 1, 0.8], 0.055)
            rgb = self.renderer.render().copy()
            return rgb

    def _axes(self, position, rotation, color, length):
        scene = self.renderer.scene
        for axis in range(3):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("replay marker budget exhausted")
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), position, np.eye(3).ravel(), color)
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.0015, position, position + rotation[:, axis] * length
            )
            scene.ngeom += 1

    def image(self, index, frame, camera, azimuth=225, elevation=-15):
        rgb = self.render(index, frame, camera, azimuth, elevation)
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("render encoding failed")
        return data.tobytes()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def byte_range(header, size):
    """One HTTP byte range for a browser's seekable local MP4."""
    if header is None:
        return 0, size - 1
    import re

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("invalid byte range")
    low, high = match.groups()
    if not low:
        suffix = int(high)
        if suffix <= 0:
            raise ValueError("invalid byte range")
        return max(0, size - suffix), size - 1
    start = int(low)
    end = min(int(high), size - 1) if high else size - 1
    if not 0 <= start <= end < size:
        raise ValueError("unsatisfiable byte range")
    return start, end


def make_server(replay, port=8771, video=None):
    video = Path(video) if video is not None else None
    render_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="herman-render")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body, content_type="application/json", code=200):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            for key, value in {
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; frame-ancestors 'none'",
            }.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            hosts = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts or self.headers.get("Origin") not in (
                None,
                *["http://" + h for h in hosts],
            ):
                self.reply({"error": "local access only"}, code=403)
                return
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    self.reply(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self.reply(
                        {
                            **replay.state(),
                            "video_available": bool(video and video.is_file()),
                            "video_webm_available": bool(video and video.with_suffix(".webm").is_file()),
                        }
                    )
                elif url.path == "/api/frames":
                    index = int(query.get("replay", ["0"])[0])
                    with replay.lock:
                        _, times = replay.frames(index)
                        self.reply({"time_s": times.tolist(), "frames": len(times)})
                elif url.path == "/frame.jpg":
                    self.reply(
                        render_worker.submit(
                            replay.image,
                            int(query.get("replay", ["0"])[0]),
                            int(query.get("frame", ["0"])[0]),
                            query.get("camera", ["head_left"])[0],
                            float(query.get("azimuth", ["225"])[0]),
                            float(query.get("elevation", ["-15"])[0]),
                        ).result(timeout=15),
                        "image/jpeg",
                    )
                elif url.path in {"/head_left.jpg", "/head_right.jpg"}:
                    self.reply(replay.read(url.path[1:]), "image/jpeg")
                elif url.path in {"/video.mp4", "/video.webm"} and video:
                    asset = video.with_suffix(".webm") if url.path.endswith(".webm") else video
                    if not asset.is_file():
                        self.reply({"error": "video not available"}, code=404)
                        return
                    size = asset.stat().st_size
                    try:
                        start, end = byte_range(self.headers.get("Range"), size)
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206 if self.headers.get("Range") else 200)
                    self.send_header("Content-Type", "video/webm" if url.path.endswith(".webm") else "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(end - start + 1))
                    if self.headers.get("Range"):
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    with asset.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = stream.read(min(65536, remaining))
                            if not chunk:
                                raise OSError("video changed during transfer")
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                else:
                    self.reply({"error": "not found"}, code=404)
            except (BrokenPipeError, ConnectionResetError):
                return
            except (ValueError, KeyError, OSError) as exc:
                self.reply({"error": str(exc)}, code=400)

    class Server(ThreadingHTTPServer):
        # Browsers pre-open sockets without sending a request. Concurrent HTTP
        # handling prevents those sockets from blocking frames/video entirely.
        # All graphics creation, rendering and teardown stay on one worker.
        def server_close(self):
            super().server_close()
            try:
                close = getattr(replay, "close", None)
                if close is not None:
                    render_worker.submit(close).result(timeout=15)
            finally:
                render_worker.shutdown(wait=True, cancel_futures=True)

    return Server(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--video", type=Path, help="optional local MP4 to serve at /video.mp4")
    args = parser.parse_args()
    replay = Replay(args.rehearsal)
    video = args.video or args.rehearsal.with_name(args.rehearsal.name + "-comparison.mp4")
    server = make_server(replay, args.port, video)
    print(f"Offline Herman rehearsal: http://localhost:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
