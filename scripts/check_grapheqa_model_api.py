#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Run GraphEQA qualification through an owned local emet VLM server.

Use emet jobs --gpu-exclusive. Default mode checks frozen-context API pairing;
--active-question-ids qualifies sensor-only reference interaction separately.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from run_shared_room_pairs import assert_frozen_source, git

from emet.utils.job_registry import validated_gpu_lock_fd


def run(args):
    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Run inside emet jobs --gpu-exclusive")
    root = Path(__file__).resolve().parents[1]
    revision, fork_revision = git(root, "rev-parse", "HEAD"), git(args.fork, "rev-parse", "HEAD")
    assert_frozen_source(root, revision)
    assert_frozen_source(args.fork, fork_revision)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    active_ids = getattr(args, "active_question_ids", None)
    manifest = {
        "mode": "live_active_reference_qualification" if active_ids else "live_model_api_frozen_context_pair",
        "status": "running",
        "source_commit": revision,
        "fork_commit": fork_revision,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest() if args.cases else None,
        "active_question_ids": active_ids,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = args.output / "interop.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    server_code = (
        "import sys; from emet.llms.openai_server import serve_openai_llm; "
        "serve_openai_llm(llm='qwen3-vl-eqa', host='127.0.0.1', port=int(sys.argv[1]), "
        "device='cuda', max_tokens=2048, multimodal=True)"
    )
    base = f"http://127.0.0.1:{args.port}"
    headers = {}
    if os.environ.get("EMET_LLM_SERVE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["EMET_LLM_SERVE_API_KEY"]

    def fetch(endpoint):
        request = urllib.request.Request(base + endpoint, headers=headers)
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-c", server_code, str(args.port)],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(9,),
        )
        try:
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Owned model server exited: {server.returncode}; see server.log")
                try:
                    if fetch("/health").get("ready"):
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model server startup exceeded 300s")
                time.sleep(1)
            model = fetch("/v1/models")["data"][0]
            manifest["served_model"] = model
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            if model.get("emet", {}).get("hf_model_id") != "Qwen/Qwen3-VL-8B-Instruct":
                raise RuntimeError("Model server did not load the declared Qwen3-VL-8B checkpoint")
            profile = args.output / "model.yaml"
            profile.write_text(
                yaml.safe_dump(
                    {
                        "vlm": {
                            "provider": "openai_compatible",
                            "name": model["id"],
                            "base_url": base + "/v1",
                            "api_key_env": "EMET_LLM_SERVE_API_KEY",
                            "response_mode": "text",
                            "max_tokens": 2048,
                            "timeout_s": 240,
                            "max_retries": 0,
                        }
                    }
                )
            )
            env = {**os.environ, "PYTHONPATH": str(args.fork)}
            command = [
                sys.executable,
                str(args.fork / "scripts/run_paired_context_eval.py"),
                "--cases",
                str(args.cases),
                "--model-config",
                f"local-qwen={profile}",
                "--output",
                str(args.output / "paired"),
            ]
            if args.answers:
                command += ["--answers", str(args.answers)]
            if active_ids:
                habitat = Path("/home/cpaxton/src/home_robot_v3/.venv-habitat")
                command = [
                    str(habitat / "bin/python"),
                    str(root / "scripts/run_grapheqa_active.py"),
                    "--fork",
                    str(args.fork),
                    "--model-config",
                    str(profile),
                    "--output",
                    str(args.output / "active"),
                    "--max-decisions",
                    str(args.max_decisions),
                    "--question-ids",
                    *map(str, active_ids),
                ]
                env.update(PYTHONPATH=str(root / "src"), LD_LIBRARY_PATH=str(habitat / "lib"))
            result = subprocess.run(
                command,
                cwd=root if active_ids else args.fork,
                env=env,
                check=False,
                timeout=1800 if active_ids else 600,
                pass_fds=(9,),
            )
            manifest.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode)
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
            manifest["owned_server_exit_code"] = server.returncode
            path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cases", type=Path)
    mode.add_argument("--active-question-ids", nargs="+", type=int)
    parser.add_argument("--max-decisions", type=int, default=4)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18653)
    raise SystemExit(run(parser.parse_args()))
