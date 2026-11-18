# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Owned, isolated native GraphEQA worker; no model or renderer assets mounted."""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from emet.eval.grapheqa_sensor import validate_packet


class NativeSensorGraph:
    def __init__(self, *, fork, contract, labels, inputs, output, image):
        self.inputs, self.output = Path(inputs), Path(output)
        self.inputs.mkdir()
        self.output.mkdir()
        self.output.chmod(0o775)  # Capability-free container writes as the host group.
        self.name = "emet-sensor-graph-" + uuid.uuid4().hex
        self.count = 0
        self.previous_stamp = 0
        self.log = (self.output / "worker.log").open("w")
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            self.name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--user",
            f"0:{os.getgid()}",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--cpuset-cpus",
            "12",
            "--memory",
            "6g",
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m",
        ]
        for key, value in {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "PYTHONPATH": "/reference:/work",
        }.items():
            command += ["-e", f"{key}={value}"]
        for source, target, readonly in [
            (fork / "scripts/serve_sensor_graph.py", "/work/serve.py", True),
            (contract, "/work/grapheqa_sensor.py", True),
            (fork / "graph_eqa", "/reference/graph_eqa", True),
            (labels, "/work/labels.yaml", True),
            (self.inputs, "/inputs", True),
            (self.output, "/output", False),
        ]:
            command += [
                "--mount",
                f"type=bind,source={source.resolve()},target={target}" + (",readonly" if readonly else ""),
            ]
        command += [
            "--entrypoint",
            "/bin/bash",
            image,
            "-c",
            "umask 0002; source /opt/ros/noetic/setup.bash; source /catkin_ws/devel/setup.bash; "
            "exec /root/miniforge3/envs/grapheqa/bin/python /work/serve.py "
            "--inputs /inputs --output /output --labels /work/labels.yaml",
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT, text=True
        )

    def step(self, packet, frontiers, *, num_labels, timeout_s=90):
        validate_packet(packet, num_labels=num_labels)
        stamp = int(packet["timestamp_ns"])
        if stamp <= self.previous_stamp:
            raise ValueError("Native graph requires a fresh packet")
        self.previous_stamp = stamp
        self.count += 1
        name = f"frame-{self.count:05d}.npz"
        temp = self.inputs / "pending.npz"
        np.savez_compressed(temp, **packet)
        temp.replace(self.inputs / name)
        self.process.stdin.write(json.dumps({"frame": name, "frontiers": frontiers}) + "\n")
        self.process.stdin.flush()
        response = self.output / (Path(name).stem + ".json")
        deadline = time.monotonic() + timeout_s
        while not response.exists():
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Native worker exited with {self.process.returncode}; see {self.output / 'worker.log'}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(f"Native graph did not acknowledge {name}")
            time.sleep(0.05)
        result = json.loads(response.read_text())
        if result["frames"] != self.count or result["timestamp_ns"] != stamp:
            raise RuntimeError("Native response belongs to a different capture")
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Docker clients do not guarantee that killing the CLI
                    # terminates its container. Remove only this owned name.
                    subprocess.run(
                        ["docker", "rm", "-f", self.name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=20,
                        check=False,
                    )
                    self.process.wait(timeout=10)
            if self.process.returncode != 0:
                raise RuntimeError(f"Native graph shutdown failed: {self.process.returncode}; see worker.log")
        finally:
            self.log.close()
