"""Offline Herman replay: captured images beside calibrated static MuJoCo renders.

This process loads recorded qpos only. It never steps physics, loads an actor,
opens a robot connection, or offers a motion endpoint.
"""

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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
            if (
                qpos.ndim != 2
                or qpos.shape[1] != self.model.nq
                or not 0 < len(qpos) <= 10000
                or times.shape != (len(qpos),)
                or not np.isfinite(qpos).all()
                or not np.isfinite(times).all()
                or np.any(np.diff(times) < 0)
            ):
                raise ValueError("malformed finite replay")
            self.loaded[index] = (qpos, times)
        return self.loaded[index]

    def state(self):
        return {
            "entries": self.entries,
            "audit": self.report["audit"],
            "summary": self.report.get("summary", {}),
            "depth": self.report["depth"],
            "proposed_start": self.report.get("proposed_start"),
            "hardware_motion_enabled": False,
            "message": "Offline recorded replay. Unobserved geometry is unknown. Wrist optics unverified.",
        }

    def image(self, index, frame, camera, azimuth=135, elevation=-25):
        if camera not in {"orbit", "head_left", "camera_arm"}:
            raise ValueError("unknown camera")
        if not np.isfinite([azimuth, elevation]).all() or not -89 <= elevation <= -5 or abs(azimuth) > 3600:
            raise ValueError("bounded orbit required")
        with self.lock:
            qpos, times = self.frames(index)
            if not 0 <= frame < len(qpos):
                raise ValueError("unknown frame")
            self.data.qpos[:] = qpos[frame]
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            view = camera
            if camera == "orbit":
                view = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(view)
                view.lookat[:] = [0.1, 0, 0.25]
                view.distance, view.azimuth, view.elevation = 0.85, azimuth, elevation
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0  # Hide conservative transparent collision proxies visually only.
            self.renderer.update_scene(self.data, camera=view, scene_option=option)
            rgb = self.renderer.render().copy()
            ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("render encoding failed")
            return data.tobytes()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def make_server(replay, port=8771):
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
                    self.reply(replay.state())
                elif url.path == "/api/frames":
                    index = int(query.get("replay", ["0"])[0])
                    with replay.lock:
                        _, times = replay.frames(index)
                        self.reply({"time_s": times.tolist(), "frames": len(times)})
                elif url.path == "/frame.jpg":
                    self.reply(
                        replay.image(
                            int(query.get("replay", ["0"])[0]),
                            int(query.get("frame", ["0"])[0]),
                            query.get("camera", ["head_left"])[0],
                            float(query.get("azimuth", ["135"])[0]),
                            float(query.get("elevation", ["-25"])[0]),
                        ),
                        "image/jpeg",
                    )
                elif url.path in {"/head_left.jpg", "/head_right.jpg"}:
                    self.reply(replay.read(url.path[1:]), "image/jpeg")
                else:
                    self.reply({"error": "not found"}, code=404)
            except (ValueError, KeyError, OSError) as exc:
                self.reply({"error": str(exc)}, code=400)

    # Single-threaded HTTP keeps the EGL context on its creating thread. Frames
    # are recorded at 20 Hz; playback never advances simulation or a robot.
    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    replay = Replay(args.rehearsal)
    server = make_server(replay, args.port)
    print(f"Offline Herman rehearsal: http://localhost:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        replay.close()


if __name__ == "__main__":
    main()
