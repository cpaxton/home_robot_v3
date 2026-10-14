"""Local, simulation-only policy test drive with an interactive MuJoCo camera.

Run: MUJOCO_GL=egl python -m emet.rl.test_drive --port 8765
No robot client, transport, connection profile, or hardware command endpoint.
"""

import argparse
import io
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

MODES = ("stretch-arm", "mars-base")
SCENARIOS = {
    "stretch-arm": ("reach", "pregrasp", "carry", "place", "disturbance"),
    "mars-base": ("standard", "retarget", "push", "contact_retreat"),
}


def checkpoints(roots, mode):
    result = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            continue
        for manifest in root.rglob("manifest.json"):
            try:
                data = json.loads(manifest.read_text())
                arm = data.get("format") == "emet-manipulation-v1" and data.get("robot") == "stretch"
                base = data.get("robot") == "innate_mars" and data.get("task") == "base"
                if not ((mode == "stretch-arm" and arm) or (mode == "mars-base" and base)):
                    continue
                path = manifest.parent.resolve()
                if not path.is_relative_to(root) or not (path / "policy.zip").exists():
                    continue
                result[str(path)] = {
                    "id": str(path),
                    "label": str(path.relative_to(root)),
                    "root": root.name,
                    "steps": data.get("steps"),
                    "reward": data.get("training_reward", "earlier/diagnostic"),
                }
            except (OSError, ValueError):
                continue  # An export may still be in progress; retry on refresh.
    return list(result.values())


def training_status(roots):
    rows = []
    for root in roots:
        for path in sorted(Path(root).glob("seed-*/run.json")):
            try:
                run = json.loads(path.read_text())
                metrics_file = path.parent / "metrics.jsonl"
                metrics = {}
                if metrics_file.exists():
                    with metrics_file.open("rb") as stream:
                        stream.seek(max(0, metrics_file.stat().st_size - 65536))
                        lines = stream.read().splitlines()
                    for line in reversed(lines):
                        try:
                            metrics = json.loads(line)
                            break
                        except ValueError:
                            continue
                tracking = path.parent / "wandb-run.json"
                steps = int(metrics.get("timesteps", 0))
                initial = int(run["initialization_steps"])
                rows.append(
                    {
                        "seed": run["seed"],
                        "steps": steps,
                        "ppo_steps": max(0, steps - initial),
                        "total": initial + int(run["steps"]),
                        "phase": "complete"
                        if (path.parent / "complete.json").exists()
                        else "PPO"
                        if steps > initial
                        else "initializing",
                        "url": json.loads(tracking.read_text()).get("url") if tracking.exists() else None,
                    }
                )
            except (OSError, ValueError, KeyError):
                continue
    return rows


class Simulation:
    """Owns only a local Gym/MuJoCo environment, driven by one worker thread."""

    def __init__(self, mode="stretch-arm", scenario=None, seed=600001):
        if mode not in MODES:
            raise ValueError("only local simulation modes are supported")
        scenario = scenario or SCENARIOS[mode][0]
        if scenario not in SCENARIOS[mode]:
            raise ValueError("invalid simulation scenario")
        if not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2^31)")
        self.mode, self.scenario, self.seed = mode, scenario, seed
        if mode == "stretch-arm":
            from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv

            self.env = ManipulationEnv(ManipulationConfig(scenario=scenario, horizon=30))
        else:
            from emet.rl.base import MarsBaseEnv
            from emet.rl.env import EnvConfig

            self.env = MarsBaseEnv(
                EnvConfig(task="base", contact_training=True, base_scenario=scenario, horizon_seconds=30)
            )
        self.obs, self.info = self.env.reset(seed=seed)
        self.done = False
        self.policy = None
        self.policy_name = "Feedback controller"
        self.push_ticks = 0
        self.interventions = []

    def load_policy(self, path, roots):
        if path == "feedback":
            self.policy, self.policy_name = None, "Feedback controller"
            return
        allowed = {row["id"] for row in checkpoints(roots, self.mode)}
        if path not in allowed:
            raise ValueError("select a complete checkpoint from the configured local roots")
        if self.mode == "stretch-arm":
            from emet.rl.manipulation.policy import ManipulationPolicy

            candidate = ManipulationPolicy(path, robot="stretch", fingerprint=self.env.fingerprint)
        else:
            from emet.rl.policy import PPOPolicy

            candidate = PPOPolicy(path)
            if candidate.task != "base":
                raise ValueError("not a base policy")
        self.policy, self.policy_name = candidate, path

    def goal(self, values):
        if self.done:
            raise ValueError("reset the finished episode before changing its goal")
        target = np.asarray(values, dtype=float)
        if target.shape != ((5,) if self.mode == "stretch-arm" else (3,)) or not np.isfinite(target).all():
            raise ValueError("invalid goal")
        if self.mode == "stretch-arm":
            self.env.control.set_reference(target, phase=self.env.control.phase)
            self.env._previous_cost = float(np.linalg.norm(target - self.env.control.positions()))
            self.env.settle_time = 0
            self.obs = self.env.control.observe()
        else:
            if np.any(abs(target[:2]) > 2) or abs(target[2]) > np.pi:
                raise ValueError("interactive base goals are limited to ±2 m and ±pi rad")
            self.env.set_goal({"base": target.tolist()})
            self.obs, _ = self.env.control.observe()
        self.interventions.append({"time": float(self.env.data.time), "goal": target.tolist()})

    def push(self):
        if self.done:
            raise ValueError("reset the finished episode first")
        self.push_ticks = 5
        self.interventions.append(
            {"time": float(self.env.data.time), "disturbance": "positive world X force for 0.25 s"}
        )

    def step(self):
        if self.done:
            return
        env = self.env
        body = env.payload if self.mode == "stretch-arm" else env.control.base_body
        env.data.xfrc_applied[body, 0] = (1.0 if self.mode == "stretch-arm" else 3.0) if self.push_ticks else 0
        self.push_ticks = max(0, self.push_ticks - 1)
        action = (
            self.policy(self.obs)
            if self.policy
            else (np.zeros(6) if self.mode == "stretch-arm" else env.expert_action())
        )
        self.obs, reward, done, truncated, self.info = env.step(action)
        self.done = bool(done or truncated)
        self.info["reward"] = reward
        if self.done or not self.push_ticks:
            env.data.xfrc_applied[body, 0] = 0

    def state(self):
        env = self.env
        if self.mode == "stretch-arm":
            current = env.control.positions().tolist()
            target = env.control.reference.tolist()
            limits = env.control.limits.tolist()
            fields = ["Lift (m)", "Extension (m)", "Wrist yaw (rad)", "Wrist pitch (rad)", "Wrist roll (rad)"]
        else:
            current, target = env.control.base_pose().tolist(), env.control.goal.base.tolist()
            limits = [[-2, 2], [-2, 2], [-np.pi, np.pi]]
            fields = ["X (m)", "Y (m)", "Yaw (rad)"]
        keys = (
            "is_success",
            "code",
            "time",
            "joint_error",
            "xy_error",
            "yaw_error",
            "peak_contact_force",
            "peak_measured_jerk",
            "retention_lost",
            "max_payload_translation",
            "max_payload_rotation",
            "wheel_lift",
            "unsafe_contact",
            "reward",
            "initialization_failed",
        )
        metrics = {key: self.info[key] for key in keys if key in self.info}
        return {
            "mode": self.mode,
            "scenario": self.scenario,
            "seed": self.seed,
            "done": self.done,
            "simulation_only": True,
            "policy": self.policy_name,
            "fields": fields,
            "current": current,
            "goal": target,
            "limits": limits,
            "metrics": metrics,
            "interventions": len(self.interventions),
            "wheel_support": "evaluated in simulator"
            if self.mode == "stretch-arm"
            else "unsupported: planar Mars model",
            "evaluation_scope": "interactive experiment, not acceptance evidence",
        }

    def close(self):
        self.env.close()


class TestDrive:
    __test__ = False

    def __init__(self, roots, mode):
        self.roots = roots
        self.commands = queue.Queue(maxsize=32)
        self.lock = threading.Lock()
        self.snapshot = {"status": "initializing", "simulation_only": True, "mode": mode}
        self.frame = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(mode,), daemon=True)
        self.thread.start()

    def _run(self, mode):
        sim = renderer = None
        running, single_steps, error = False, 0, None
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 135, -25, 2.8
        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        try:
            sim = Simulation(mode)
            while not self.closed.is_set():
                started = time.monotonic()
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    try:
                        op = command["op"]
                        if op == "reset":
                            running, single_steps = False, 0
                            new_mode = command.get("mode", sim.mode)
                            # Validate a new scene before replacing the current one.
                            with self.lock:
                                self.snapshot["status"] = "resetting"
                            replacement = Simulation(new_mode, command.get("scenario"), command.get("seed", sim.seed))
                            if renderer:
                                renderer.close()
                                renderer = None
                            old = sim
                            sim = replacement
                            if new_mode == old.mode:
                                sim.policy, sim.policy_name = old.policy, old.policy_name
                            old.close()
                        elif op == "run":
                            if sim.done:
                                raise ValueError("episode finished; reset before running")
                            running = True
                        elif op == "pause":
                            running, single_steps = False, 0
                        elif op == "step":
                            single_steps, running = 1, False
                        elif op == "goal":
                            sim.goal(command["values"])
                        elif op == "push":
                            sim.push()
                        elif op == "policy":
                            running = False
                            sim.load_policy(command["id"], self.roots)
                        elif op == "camera":
                            values = np.asarray(command["values"], dtype=float)
                            if values.shape != (3,) or not np.isfinite(values).all():
                                raise ValueError("invalid camera")
                            camera.azimuth, camera.elevation, camera.distance = (
                                values[0] % 360,
                                np.clip(values[1], -85, -5),
                                np.clip(values[2], 0.4, 6),
                            )
                        else:
                            raise ValueError("unknown simulation command")
                        error = None
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        running = False
                        error = str(exc)
                if (running or single_steps) and not sim.done:
                    sim.step()
                    single_steps = max(0, single_steps - 1)
                if sim.done:
                    running = False
                if renderer is None:
                    sim.env.model.vis.global_.offwidth = 960
                    sim.env.model.vis.global_.offheight = 640
                    renderer = mujoco.Renderer(sim.env.model, height=540, width=960)
                    sim.env.model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
                    camera.lookat[:] = sim.env.data.body("base_link").xpos + [
                        0,
                        0,
                        0.7 if sim.mode == "stretch-arm" else 0.25,
                    ]
                renderer.update_scene(sim.env.data, camera=camera, scene_option=option)
                state = sim.state()
                # Goal markers are visualization only and never collision geometry.
                if sim.mode == "mars-base":
                    point = [*state["goal"][:2], 0.04]
                else:
                    scratch = mujoco.MjData(sim.env.model)
                    mujoco.mj_copyData(scratch, sim.env.model, sim.env.data)
                    for group, value in zip(sim.env.control.q, state["goal"], strict=True):
                        scratch.qpos[group] = value / len(group)
                    mujoco.mj_kinematics(sim.env.model, scratch)
                    point = scratch.xpos[sim.env.control.ee]
                scene = renderer.scene
                if scene.ngeom < scene.maxgeom:
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.018] * 3),
                        np.asarray(point),
                        np.eye(3).ravel(),
                        np.array([0.2, 1, 0.7, 0.8]),
                    )
                    scene.ngeom += 1
                buffer = io.BytesIO()
                Image.fromarray(renderer.render()).save(buffer, format="JPEG", quality=85)
                with self.lock:
                    self.frame = buffer.getvalue()
                    self.snapshot = {
                        **state,
                        "status": "finished" if sim.done else "running" if running else "paused",
                        "error": error,
                        "wall_time": time.time(),
                    }
                self.closed.wait(max(0, 0.05 - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.snapshot.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if renderer:
                renderer.close()
            if sim:
                sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=MODES, default="stretch-arm")
    parser.add_argument("--policy-root", type=Path, action="append")
    args = parser.parse_args()
    roots = args.policy_root or [Path("logs/manipulation/physical-v2"), Path("logs/mars-base-extended")]
    app = TestDrive(roots, args.mode)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, content_type="application/json", code=200):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
            return self.headers.get("Host") in hosts and self.headers.get("Origin") in (
                None,
                *[f"http://{h}" for h in hosts],
            )

        def do_GET(self):
            if not self.allowed():
                self.reply({"error": "local access only"}, code=403)
            elif self.path == "/":
                self.reply(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            elif self.path.split("?")[0] == "/frame.jpg":
                with app.lock:
                    frame = app.frame
                self.reply(frame or b"", "image/jpeg", 200 if frame else 503)
            elif self.path == "/api/state":
                with app.lock:
                    snapshot = dict(app.snapshot)
                self.reply(snapshot)
            elif self.path == "/api/training":
                self.reply(training_status(roots))
            elif self.path == "/api/checkpoints":
                with app.lock:
                    mode = app.snapshot["mode"]
                self.reply(checkpoints(roots, mode))
            else:
                self.reply({"error": "not found"}, code=404)

        def do_POST(self):
            if not self.allowed() or self.path != "/api/command":
                self.reply({"error": "local simulation endpoint only"}, code=403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("small JSON command required")
                command = json.loads(self.rfile.read(size))
                if not isinstance(command, dict) or command.get("op") not in {
                    "reset",
                    "run",
                    "pause",
                    "step",
                    "goal",
                    "push",
                    "policy",
                    "camera",
                }:
                    raise ValueError("unsupported simulation command")
                app.commands.put_nowait(command)
                self.reply({"accepted": True}, code=202)
            except (ValueError, queue.Full) as exc:
                self.reply({"error": str(exc)}, code=400)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Simulation-only test drive: http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        app.closed.set()
        app.thread.join(timeout=15)
        server.server_close()


if __name__ == "__main__":
    main()
