"""Local Mars camera/telemetry console with explicitly non-executing pose previews.

No robot command client or motor endpoint exists in this console. A future
verified executor must be attached through GuardedControlSession, not this UI.
"""

import base64
import copy
import json
import math
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import click

from emet.robots.innate_mars.passive_ssh import relay_command

CAMERAS = {"head_left", "head_right", "wrist"}
BLOCKERS = [
    "Robot feedback units and read validity need verification",
    "Whole-body motor driver and measured stop are not connected",
    "Robot command-loss watchdog needs verification",
    "Policy performance in the first hardware envelope is not qualified",
]


def vector(value, size):
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"finite {size}-component vector required")
    return [float(v) for v in value]


def quaternion(value):
    q = vector(value, 4)
    if abs(sum(x * x for x in q) - 1) > 1e-3:
        raise ValueError("unit quaternion required")
    return q


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotate(q, increments):
    def multiply(a, b):
        x, y, z, w = a
        X, Y, Z, W = b
        return [
            w * X + x * W + y * Z - z * Y,
            w * Y - x * Z + y * W + z * X,
            w * Z + x * Y - y * X + z * W,
            w * W - x * X - y * Y - z * Z,
        ]

    for axis, angle in enumerate(increments):
        delta = [0.0, 0.0, 0.0, math.cos(angle / 2)]
        delta[axis] = math.sin(angle / 2)
        q = multiply(delta, q)
    norm = math.sqrt(sum(x * x for x in q))
    return [x / norm for x in q]


class ConsoleState:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.session = None
        self.sequence = -1
        self.received = None
        self.samples = {}
        self.cameras = {}
        self.error = None
        self.targets = None
        self.origin = None
        self.hand_origin = None
        self.revision = 0
        self.mode = 1

    def ingest(self, packet):
        with self.lock:
            session, sequence = packet.get("session"), packet.get("sequence")
            if not isinstance(session, str) or not session or type(sequence) is not int:
                raise ValueError("identified relay packet required")
            if session != self.session:
                self.session, self.sequence = session, -1
                self.samples, self.cameras = {}, {}
                self.targets = self.origin = self.hand_origin = None
                self.revision += 1
            if sequence <= self.sequence:
                return
            self.sequence = sequence
            now = self.clock()
            kind = packet.get("kind")
            if kind == "state":
                samples = copy.deepcopy(packet["samples"])
                for name, sample in samples.items():
                    if name == "odom":
                        vector(sample["position"], 3)
                        quaternion(sample["quaternion_xyzw"])
                        vector(sample["twist_child"], 6)
                        if sample.get("frame_id") != "odom":
                            raise ValueError("preview requires odom frame")
                    elif name == "hand":
                        vector(sample["position"], 3)
                        quaternion(sample["quaternion_xyzw"])
                        if sample.get("frame_id") != "odom":
                            raise ValueError("hand and base must share odom frame")
                    elif name == "arm":
                        vector(sample["position"], 6)
                        vector(sample["velocity"], 6)
                        if sample.get("names") != [f"joint{i}" for i in range(1, 7)]:
                            raise ValueError("explicit arm joint order required")
                    else:
                        raise ValueError("unexpected state channel")
                    self._source_age(sample)
                self.samples = samples
                self.received = now
            elif kind == "camera":
                if (
                    packet.get("name") not in CAMERAS
                    or not isinstance(packet.get("jpeg"), str)
                    or len(packet["jpeg"]) > 2_000_000
                ):
                    raise ValueError("bounded named camera frame required")
                data = base64.b64decode(packet["jpeg"], validate=True)
                if not data.startswith(b"\xff\xd8"):
                    raise ValueError("JPEG camera frame required")
                self._source_age(packet)
                self.cameras[packet["name"]] = {
                    "data": data,
                    "received": now,
                    "source_age_s": packet.get("source_age_s"),
                    "source_stamp_ns": packet.get("source_stamp_ns"),
                }
            elif kind == "camera_error":
                self.cameras.pop(packet.get("name"), None)
            elif kind != "hello":
                raise ValueError("unknown passive packet")
            self.error = None

    @staticmethod
    def _source_age(sample):
        age = sample.get("source_age_s")
        if age is not None and (type(age) not in (int, float) or not math.isfinite(age)):
            raise ValueError("finite source age or explicit unknown required")
        return age

    def _timed(self, sample, received, *, max_age=0.5):
        local_age = max(0.0, self.clock() - received) if received is not None else None
        source_age = self._source_age(sample)
        age = source_age + local_age if source_age is not None and local_age is not None else None
        return {
            "source_age_s": age,
            "arrival_age_s": local_age,
            "fresh": bool(source_age is not None and source_age >= 0 and age <= max_age and local_age <= max_age),
        }

    def _snapshot(self):
        samples = {k: {**copy.deepcopy(v), **self._timed(v, self.received)} for k, v in self.samples.items()}
        cameras = {k: self._timed(v, v["received"], max_age=1.0) for k, v in self.cameras.items()}
        connected = self.received is not None and self.clock() - self.received <= 1
        odom = samples.get("odom", {})
        if self.targets is None and odom.get("fresh"):
            p, q = odom["position"], odom["quaternion_xyzw"]
            self.origin = [*p[:2], yaw(q)]
            self.targets = {"base": self.origin.copy(), "hand": None}
            self.revision += 1
        hand = samples.get("hand", {})
        if self.targets is not None and self.targets["hand"] is None and hand.get("fresh"):
            self.targets["hand"] = {k: copy.deepcopy(hand[k]) for k in ("position", "quaternion_xyzw")}
            self.hand_origin = hand["position"].copy()
            self.revision += 1
        contract = None
        if self.targets is not None:
            arm = samples.get("arm", {})
            grip = arm.get("position", [None] * 6)[5]
            contract = {
                "frame": "odom",
                "base": {"pose": self.targets["base"].copy(), "mask": [self.mode != 3] * 3},
                "hands": [
                    {
                        "pose": copy.deepcopy(self.targets["hand"]),
                        "active": self.mode != 1,
                        "gripper": {"position": [grip], "mask": [False]},
                    }
                ],
                "preview_only": True,
            }
        return {
            "connected": connected,
            "mode": self.mode,
            "revision": self.revision,
            "samples": samples,
            "cameras": cameras,
            "targets": contract,
            "origin": None if self.origin is None else self.origin.copy(),
            "motion_enabled": False,
            "commands_sent": 0,
            "blockers": BLOCKERS.copy(),
            "feedback_units_verified": False,
            "error": self.error,
            "age_note": "reported capture age plus local waiting time; SSH transit delay is not measured",
        }

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def edit(self, command):
        with self.lock:
            state = self._snapshot()
            if type(command.get("revision")) is not int or command["revision"] != self.revision:
                raise ValueError("stale preview revision; refresh before editing")
            if not state["samples"].get("odom", {}).get("fresh") or self.targets is None:
                raise ValueError("fresh odometry required to preview goals")
            op = command.get("op")
            if op == "mode":
                if type(command.get("mode")) is not int or command["mode"] not in (1, 2, 3):
                    raise ValueError("select base, hand/base-held or hand/base-free")
                if command["mode"] != 1 and not state["samples"].get("hand", {}).get("fresh"):
                    raise ValueError("fresh hand transform required")
                self.mode = command["mode"]
            elif op == "nudge":
                translation = vector(command.get("translation"), 3)
                rotation = vector(command.get("rotation"), 3)
                if max(map(abs, translation)) > 0.02 or max(map(abs, rotation)) > 0.1:
                    raise ValueError("preview increments must be <=2cm and <=0.1rad")
                candidate = copy.deepcopy(self.targets)
                if self.mode == 1:
                    if translation[1:] != [0.0, 0.0] or rotation[:2] != [0.0, 0.0]:
                        raise ValueError("base preview uses forward travel and yaw")
                    base = candidate["base"]
                    base[0] += translation[0] * math.cos(base[2])
                    base[1] += translation[0] * math.sin(base[2])
                    base[2] = math.remainder(base[2] + rotation[2], 2 * math.pi)
                    if (
                        math.dist(base[:2], self.origin[:2]) > 0.25
                        or abs(math.remainder(base[2] - self.origin[2], 2 * math.pi)) > math.pi / 4
                    ):
                        raise ValueError("preview outside 25cm / 45degree envelope")
                else:
                    if not state["samples"].get("hand", {}).get("fresh") or candidate["hand"] is None:
                        raise ValueError("fresh hand transform required")
                    hand = candidate["hand"]
                    hand["position"] = [a + b for a, b in zip(hand["position"], translation, strict=True)]
                    hand["quaternion_xyzw"] = rotate(hand["quaternion_xyzw"], rotation)
                    if math.dist(hand["position"], self.hand_origin) > 0.15:
                        raise ValueError("hand preview outside 15cm neighborhood")
                self.targets = candidate
            elif op == "reset":
                # Reset only staged targets, never the latched preview origin
                # (and never robot state, faults or command ownership).
                odom = state["samples"]["odom"]
                candidate = copy.deepcopy(self.targets)
                candidate["base"] = [*odom["position"][:2], yaw(odom["quaternion_xyzw"])]
                hand = state["samples"].get("hand", {})
                if candidate["hand"] is not None:
                    if not hand.get("fresh"):
                        raise ValueError("fresh hand transform required to reset hand preview")
                    candidate["hand"] = {k: copy.deepcopy(hand[k]) for k in ("position", "quaternion_xyzw")}
                base = candidate["base"]
                if (
                    math.dist(base[:2], self.origin[:2]) > 0.25
                    or abs(math.remainder(base[2] - self.origin[2], 2 * math.pi)) > math.pi / 4
                    or (
                        candidate["hand"] is not None
                        and math.dist(candidate["hand"]["position"], self.hand_origin) > 0.15
                    )
                ):
                    raise ValueError("measured pose outside original preview envelope; origin is not recentered")
                self.targets = candidate
            else:
                raise ValueError("only local preview edits are supported; no motion endpoint")
            self.revision += 1
            return self._snapshot()

    def frame(self, camera):
        with self.lock:
            sample = self.cameras.get(camera)
            return sample["data"] if sample and self._timed(sample, sample["received"], max_age=1.0)["fresh"] else None


class PassiveConnection:
    def __init__(self, state, target, *, seconds=3600, fps=5):
        self.state = state
        self.command = relay_command(target, seconds=seconds, fps=fps)
        self.process = None
        self.thread = None
        self.closed = threading.Event()

    def start(self):
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        from emet.robots.innate_mars.passive_relay import PREFIX

        try:
            while not self.closed.is_set():
                line = self.process.stdout.readline(2_100_000)
                if not line:
                    break
                if len(line) >= 2_100_000:
                    raise ValueError("oversized relay line")
                if not line.startswith(PREFIX.encode()):
                    continue
                self.state.ingest(json.loads(line[len(PREFIX) :]))
        except (ValueError, KeyError, TypeError) as exc:
            with self.state.lock:
                self.state.error = f"passive relay: {exc}"
        finally:
            if not self.closed.is_set():
                with self.state.lock:
                    self.state.error = self.state.error or "Passive relay disconnected; restart console to reconnect"
            if self.process.poll() is None:
                self.process.terminate()

    def close(self):
        self.closed.set()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)


def make_server(state, *, port=8770, training_run=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def allowed(self):
            hosts = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
            return self.headers.get("Host") in hosts and self.headers.get("Origin") in (
                None,
                *[f"http://{h}" for h in hosts],
            )

        def reply(self, body, *, code=200, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.allowed():
                self.reply({"error": "local access only"}, code=403)
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.reply(Path(__file__).with_suffix(".html").read_bytes(), content_type="text/html; charset=utf-8")
            elif path == "/api/state":
                data = state.snapshot()
                if training_run and Path(training_run).is_file():
                    try:
                        run = json.loads(Path(training_run).read_text())
                        data["training"] = {
                            k: run.get(k)
                            for k in ("phase", "decisions", "control_ticks", "complete", "mixture", "general_mixture")
                        }
                    except (OSError, ValueError):
                        data["training"] = {"phase": "temporarily unavailable"}
                self.reply(data)
            elif path.startswith("/camera/") and path.endswith(".jpg"):
                name = path.removeprefix("/camera/").removesuffix(".jpg")
                image = state.frame(name) if name in CAMERAS else None
                self.reply(image or b"", code=200 if image else 503, content_type="image/jpeg")
            else:
                self.reply({"error": "not found"}, code=404)

        def do_POST(self):
            if not self.allowed():
                self.reply({"error": "local access only"}, code=403)
                return
            if self.path != "/api/preview":
                self.reply({"error": "no robot motion endpoint"}, code=404)
                return
            try:
                self.connection.settimeout(2)
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 2048 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("small JSON preview edit required")
                command = json.loads(self.rfile.read(size))
                if not isinstance(command, dict):
                    raise ValueError("preview edit object required")
                self.reply(state.edit(command))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply({"error": str(exc)}, code=409)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


@click.command()
@click.option("--ssh", "target", default="jetson1@192.168.1.43", show_default=True)
@click.option("--port", type=click.IntRange(1024, 65535), default=8770, show_default=True)
@click.option("--seconds", type=click.IntRange(1, 21600), default=3600, show_default=True)
@click.option("--fps", type=click.FloatRange(min=0.1, max=10), default=5.0, show_default=True)
@click.option(
    "--training-run",
    type=click.Path(path_type=Path),
    default="logs/manipulation/goal-change-curriculum-v1/curriculum/run.json",
)
def main(target, port, seconds, fps, training_run):
    """Live Mars cameras + feedback + local goal previews. Sends no motor commands."""
    state = ConsoleState()
    connection = PassiveConnection(state, target, seconds=seconds, fps=fps)
    server = make_server(state, port=port, training_run=training_run)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        connection.start()
        click.echo(f"Herman passive console: http://localhost:{port} (goal previews only; no motor commands)")
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        connection.close()
        server.server_close()


if __name__ == "__main__":
    main()
