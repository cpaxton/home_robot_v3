"""Record a USB camera before, during, and after a robot test command.

Example:
    python scripts/record_robot_test.py --device /dev/video0 --output test.mp4 -- python run_test.py
"""

import argparse
import signal
import subprocess
import threading
import time
from pathlib import Path


def frame_readiness(progress, ready):
    for line in progress:
        key, _, value = line.strip().partition("=")
        if key == "frame" and value.isdecimal() and int(value) > 0:
            ready.set()


def stop_recording(recorder):
    if recorder.poll() is None:
        recorder.send_signal(signal.SIGINT)
        try:
            recorder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            recorder.kill()
            recorder.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="V4L2 camera, preferably a stable /dev/v4l/by-id path")
    parser.add_argument("--output", required=True, type=Path, help="New MP4 path")
    parser.add_argument("--size", default="640x480")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--input-format", default="mjpeg")
    parser.add_argument("--lead-in", default=1.0, type=float, help="Seconds to record before running the command")
    parser.add_argument("--tail", default=1.0, type=float, help="Seconds to record after the command exits")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.fps <= 0 or args.lead_in < 0 or args.tail < 0:
        parser.error("Supply a test command after -- and nonnegative recording timings")
    if args.output.suffix.lower() != ".mp4" or args.output.exists():
        parser.error("--output must be a new .mp4 file")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-progress",
        "pipe:1",
        "-stats_period",
        "0.25",
        "-f",
        "v4l2",
        "-input_format",
        args.input_format,
        "-video_size",
        args.size,
        "-framerate",
        str(args.fps),
        "-i",
        args.device,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    ready = threading.Event()
    with subprocess.Popen(ffmpeg, stdout=subprocess.PIPE, text=True, start_new_session=True) as recorder:
        reader = threading.Thread(target=frame_readiness, args=(recorder.stdout, ready), daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + 8.0
            while not ready.wait(timeout=0.1):
                if recorder.poll() is not None or time.monotonic() >= deadline:
                    raise SystemExit("Camera did not produce video. The test command was skipped.")
            time.sleep(args.lead_in)
            if recorder.poll() is not None:
                raise SystemExit("Camera stopped before the test. The test command was skipped.")
            print(f"Recording {args.output}", flush=True)
            result = subprocess.run(command, check=False)
            if recorder.poll() is not None:
                raise SystemExit("Camera stopped before the recording completed.")
            time.sleep(args.tail)
        finally:
            stop_recording(recorder)
            reader.join(timeout=1)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
