#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Compare three answer paths on identical frozen PNG packets, serially.

This evaluates answer/context modules, not full active agents. The emet arm has
an empty graph and uses production keyword extraction and model-final answering.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import secrets
import shutil
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from run_shared_room_pairs import assert_frozen_source, git

from emet.eval.frozen_eqa import FrozenServeState, png_hashes, run_emet_packet, semantic_choice
from emet.llms.openai_server import _make_handler
from emet.utils.job_registry import validated_gpu_lock_fd

ARMS = ("images_only", "reference_graph", "emet_answer_packet")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(fork):
    sys.path.insert(0, str(fork))
    spec = importlib.util.spec_from_file_location("reference_paired", fork / "scripts/run_paired_context_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trial_order(case_index, seed_index):
    offset = (case_index + seed_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def score(rows, cases, answers_path):
    # This function is called only after every model call and server shutdown.
    answers = json.loads(answers_path.read_text())
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        if answers.get(case["id"]) not in [chr(65 + i) for i in range(len(case["choices"]))]:
            raise ValueError(f"Missing scoring label for {case['id']}")
    for row in rows:
        row["predicted_letter"] = semantic_choice(row.get("answer", ""), by_id[row["case_id"]]["choices"])
        row["correct_if_stopped_here"] = row["predicted_letter"] == answers[row["case_id"]] if row["complete"] else None


def run(args):
    import torch
    from omegaconf import OmegaConf

    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Run through emet jobs --gpu-exclusive")
    root = Path(__file__).resolve().parents[1]
    revisions = {"emet": git(root, "rev-parse", "HEAD"), "reference": git(args.fork, "rev-parse", "HEAD")}
    assert_frozen_source(root, revisions["emet"])
    assert_frozen_source(args.fork, revisions["reference"])
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Use unique nonnegative trial seeds")
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise RuntimeError("Keep the shared 20 GiB disk reserve")
    reference = load_reference(args.fork)
    cases = reference.read_cases(args.cases)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": "frozen_answer_paths_not_active_agents",
        "source_commits": revisions,
        "driver_sha256": sha256(__file__),
        "adapter_sha256": sha256(root / "src/emet/eval/frozen_eqa.py"),
        "cases_sha256": sha256(args.cases),
        "images_sha256": {case["id"]: png_hashes(case["images"]) for case in cases},
        "arm_seeds": args.seeds,
        "seed_policy": "torch RNG reset once at each arm start; no HTTP seed parameter",
        "condition_order": "Latin rotation by case index plus seed index",
        "output_allowance_per_arm": 2048,
        "emet_budget_split": {"keyword": 64, "answer": 1984},
        "emet_memory": "empty graph; production keyword and model_final query_answer on supplied packet",
        "planned_rows": len(cases) * len(args.seeds) * len(ARMS),
        "runtime": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "pydantic", "openai"]},
        "status": "running",
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    key_name = "EMET_FROZEN_EQA_API_KEY"
    previous_key = os.environ.get(key_name)
    key = secrets.token_urlsafe(24)
    os.environ[key_name] = key
    state = FrozenServeState(llm_key="qwen3-vl-eqa", device="cuda", max_tokens=2048, multimodal=True, api_key=key)
    server, thread, rows = None, None, []
    try:
        start = time.monotonic()
        state.load()
        model = {
            "hf_model_id": state.client._resolved_hf_model_id,
            "quantization": state.client._quantization,
            "image_max_side": state.client.image_max_side,
            "image_max_pixels": state.client.image_max_pixels,
        }
        if model != {
            "hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "quantization": "int4",
            "image_max_side": 512,
            "image_max_pixels": 0,
        }:
            raise ValueError("Loaded model differs from the predeclared Qwen profile")
        manifest.update(
            model=model,
            checkpoint_generation_config=state.client.model.generation_config.to_dict(),
            model_load_s=time.monotonic() - start,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
        cfg = OmegaConf.create(
            {
                "vlm": {
                    "name": "qwen3-vl-eqa",
                    "provider": "openai_compatible",
                    "base_url": endpoint,
                    "api_key_env": key_name,
                    "response_mode": "text",
                    "max_tokens": 2048,
                    "timeout_s": 240,
                    "max_retries": 0,
                    "use_image": True,
                    "add_history": True,
                }
            }
        )
        for case_index, case in enumerate(cases):
            for seed_index, seed in enumerate(args.seeds):
                for arm in trial_order(case_index, seed_index):
                    assert_frozen_source(root, revisions["emet"])
                    assert_frozen_source(args.fork, revisions["reference"])
                    output = args.output / f"trial-{len(rows):04d}"
                    output.mkdir()
                    state.begin_arm(output, case["images"], 2048)
                    torch.manual_seed(seed)
                    row = {"case_id": case["id"], "arm": arm, "seed": seed, "interaction_steps": 0, "complete": False}
                    started = time.monotonic()
                    try:
                        if shutil.disk_usage(args.output).free < 20 * 1024**3 + 8 * 1024**2:
                            raise OSError("Stop inference recording before consuming the shared 20 GiB disk reserve")
                        if arm == "emet_answer_packet":
                            result = run_emet_packet(case, base_url=endpoint, model="qwen3-vl-eqa", api_key=key)
                        else:
                            condition = "model_only" if arm == "images_only" else "reference_graph"
                            result = reference.run_one(case, cfg, condition, output)
                        state.validate_arm()
                        row.update(result, complete=True)
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    row.update(
                        wall_s=time.monotonic() - started,
                        requested_output_tokens=state.requested_tokens,
                        model_calls=len(state.records),
                        usage=[record.get("response", {}).get("usage") for record in state.records],
                    )
                    rows.append(row)
                    with (args.output / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(json.dumps({key: value for key, value in row.items() if key != "raw"}), flush=True)
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join(timeout=10)
        if previous_key is None:
            os.environ.pop(key_name, None)
        else:
            os.environ[key_name] = previous_key
        manifest.update(
            completed_rows=len(rows),
            failed_rows=sum(not row["complete"] for row in rows),
            status="inference_complete" if len(rows) == manifest["planned_rows"] else "incomplete",
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    score(rows, cases, args.answers)
    (args.output / "scored_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest.update(
        answers_sha256=sha256(args.answers),
        scoring_complete=True,
        status="complete_with_failures" if manifest["failed_rows"] else "complete",
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return int(bool(manifest["failed_rows"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    raise SystemExit(run(parser.parse_args()))
