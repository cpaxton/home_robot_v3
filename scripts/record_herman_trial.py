"""Record a bounded Herman test command with robot observations and Astra RGB."""

import argparse
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

from emet.robots.innate_mars.passive_ssh import relay_command


def stop_recorder(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def record(output, ssh, device, command, seconds):
    output.mkdir(parents=True)
    summary = {"format": "herman-trial-v1", "command": command, "command_started": False, "frame_sync_verified": False}

    def event(kind, **data):
        with (output / "events.jsonl").open("a") as log:
            log.write(json.dumps({"kind": kind, "host_ns": time.time_ns(), **data}) + "\n")

    camera_ready, robot_ready = threading.Event(), threading.Event()
    with (
        (output / "robot.jsonl").open("w") as observations,
        (output / "astra.log").open("w") as camera_log,
        (output / "relay.log").open("w") as relay_log,
        (output / "command.log").open("w") as command_log,
    ):
        astra = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-progress",
                "pipe:1",
                "-f",
                "v4l2",
                "-input_format",
                "mjpeg",
                "-framerate",
                "15",
                "-video_size",
                "640x480",
                "-i",
                str(device),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(output / "astra.mp4"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=camera_log,
        )
        relay = None
        readers = []
        try:
            relay = subprocess.Popen(
                relay_command(ssh, seconds=seconds + 25, fps=8),
                text=True,
                stdout=subprocess.PIPE,
                stderr=relay_log,
            )

            def read_camera():
                for line in astra.stdout:
                    key, _, value = line.strip().partition("=")
                    if key == "frame" and value.isdecimal() and int(value) > 0:
                        camera_ready.set()

            def read_robot():
                cameras = set()
                state = False
                for line in relay.stdout:
                    if not line.startswith("EMET_PASSIVE "):
                        continue
                    row = json.loads(line.removeprefix("EMET_PASSIVE "))
                    row["host_received_ns"] = time.time_ns()
                    observations.write(json.dumps(row, separators=(",", ":")) + "\n")
                    if row["kind"] == "state":
                        state = {"arm", "odom", "hand"} <= row["samples"].keys()
                    if row["kind"] == "camera":
                        cameras.add(row["name"])
                    if state and {"head_left", "head_right"} <= cameras:
                        robot_ready.set()

            for read in (read_camera, read_robot):
                reader = threading.Thread(target=read, daemon=True)
                reader.start()
                readers.append(reader)
            event("recorders_started")
            if (
                not camera_ready.wait(8)
                or not robot_ready.wait(12)
                or any(process.poll() is not None for process in (astra, relay))
            ):
                raise RuntimeError("Astra and robot camera/telemetry capture did not become ready")
            event("command_start")
            summary["command_started"] = True
            result = subprocess.run(command, stdout=command_log, stderr=subprocess.STDOUT, check=False)
            summary["command_returncode"] = result.returncode
            event("command_end", returncode=result.returncode)
            time.sleep(1)
            summary["recorders_alive_at_end"] = all(process.poll() is None for process in (astra, relay))
            return result.returncode if summary["recorders_alive_at_end"] else 2
        finally:
            for process in (astra, relay):
                if process is not None:
                    stop_recorder(process)
            for reader in readers:
                reader.join(timeout=2)
            event("recorders_stopped")
            (output / "trial.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssh", default="jetson1@192.168.1.43")
    parser.add_argument("--astra-device", type=Path, default=Path("/dev/video0"))
    parser.add_argument("--expected-seconds", type=float, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.expected_seconds <= 300:
        parser.error("provide a test command after -- and a 1–300 second expected recording duration")
    return record(args.output, args.ssh, args.astra_device, command, args.expected_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
