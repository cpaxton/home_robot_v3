#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Replay original vs annotated planner schemas on identical recorded requests.

Uses the real local model-server inference/message adapter in process, without
HTTP transport. This diagnoses the planning contract, not active task success.
No simulator, mapper, scorer or answer labels are read. Runtime policy is unchanged.
"""

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

from run_shared_room_pairs import assert_frozen_source, git

from emet.eval.planner_contract import (
    answer_before_action_schema,
    clarify_planner_schema,
    diagnostic_decoding,
    schema_aligned_history,
)
from emet.utils.job_registry import validated_gpu_lock_fd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_variant(request, variant):
    result = copy.deepcopy(request)
    content = result["messages"][0]["content"]
    prefix, raw = content.split("\n", 1)
    schema = json.loads(raw)
    if variant == "clarified":
        result["messages"][0]["content"] = prefix + "\n" + json.dumps(clarify_planner_schema(schema))
    elif variant == "answer_first":
        result["messages"][0]["content"] = prefix + "\n" + json.dumps(answer_before_action_schema(schema))
    elif variant == "history_contract":
        states = [
            message
            for message in result["messages"]
            if message["role"] == "user"
            and isinstance(message["content"], str)
            and message["content"].startswith("CURRENT STATE:")
        ]
        if len(states) != 1:
            raise ValueError("Expected exactly one reference planner state message")
        states[0]["content"] = schema_aligned_history(states[0]["content"])
    elif variant not in {"original", "greedy"}:
        raise ValueError("Unknown diagnostic variant")
    return result, schema


def read_requests(directory):
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    selected = []
    for index, row in enumerate(rows):
        if row["condition"] != "reference_graph":
            continue
        path = directory / f"request-{index:04d}/model_calls.jsonl"
        traces = [json.loads(line) for line in path.read_text().splitlines()]
        if len(traces) != 1:
            raise ValueError("Require exactly one recorded request per case, including failed replies")
        selected.append((row["case_id"], traces[0]["request"], path))
    if not selected or len({name for name, _, _ in selected}) != len(selected):
        raise ValueError("Require one reference-model profile and unique case IDs")
    return selected


def run(args):
    import torch
    from jsonschema import Draft202012Validator

    from emet.llms.openai_server import LLMServeState

    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Run through emet jobs --gpu-exclusive")
    root = Path(__file__).resolve().parents[1]
    revision = git(root, "rev-parse", "HEAD")
    assert_frozen_source(root, revision)
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise RuntimeError("Keep the shared 20 GiB disk reserve")
    selected = read_requests(args.recorded)
    if any(request["model"] != "qwen3-vl-eqa" or request.get("max_tokens") != 2048 for _, request, _ in selected):
        raise ValueError("This local replay requires the previously qualified Qwen model and 2,048-token cap")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": "frozen_planner_contract_not_active_eqa",
        "transport": "real LLMServeState.complete in process; no HTTP",
        "source_commit": revision,
        "driver_sha256": sha256(Path(__file__)),
        "annotation_sha256": sha256(root / "src/emet/eval/planner_contract.py"),
        "input_manifest_sha256": sha256(args.recorded / "manifest.json"),
        "request_files_sha256": {name: sha256(path) for name, _, path in selected},
        "comparison_variant": args.variant,
        "variant_order": f"original/{args.variant} on even cases; {args.variant}/original on odd cases",
        "seeds": args.seeds,
        "planned_rows": len(selected) * 2 * len(args.seeds or [None]),
        "status": "running",
        "scoring_labels_read": False,
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    state = LLMServeState(llm_key="qwen3-vl-eqa", device="cuda", max_tokens=2048, multimodal=True)
    rows = []
    try:
        state.load()
        model = {
            "hf_model_id": state.client._resolved_hf_model_id,
            "quantization": state.client._quantization,
            "image_max_side": state.client.image_max_side,
            "image_max_pixels": state.client.image_max_pixels,
        }
        if (
            model["hf_model_id"] != "Qwen/Qwen3-VL-8B-Instruct"
            or model["quantization"] != "int4"
            or model["image_max_side"] != 512
            or model["image_max_pixels"] != 0
        ):
            raise RuntimeError("Loaded model differs from the frozen request comparison")
        manifest["model"] = model
        manifest["checkpoint_generation_config"] = state.client.model.generation_config.to_dict()
        manifest["client_num_beams"] = state.client.num_beams
        # Reset sampling RNG before each paired request. Fixed seeds are a
        # diagnostic control, not a promise of bitwise GPU reproducibility.
        trials = [(item, seed) for item in selected for seed in (args.seeds or [None])]
        for index, ((name, original, _), seed) in enumerate(trials):
            order = ("original", args.variant) if index % 2 == 0 else (args.variant, "original")
            for variant in order:
                request, schema = request_variant(original, variant)
                row = {"case_id": name, "variant": variant, "seed": seed, "complete": False}
                trace = {"case_id": name, "variant": variant, "seed": seed, "request": request}
                started = time.monotonic()
                try:
                    with diagnostic_decoding(state.client, greedy=variant == "greedy") as config:
                        trace["generation_config"] = config.to_dict()
                        if seed is not None:
                            torch.manual_seed(seed)
                        response = state.complete(request)
                    trace["response"] = response
                    choice = response["choices"][0]
                    if choice["finish_reason"] != "stop":
                        raise ValueError(f"Incomplete planner response: {choice['finish_reason']}")
                    answer = json.loads(choice["message"]["content"])
                    Draft202012Validator(schema).validate(answer)
                    row.update(complete=True, answer=answer["answer"], steps=answer["steps"], usage=response["usage"])
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    trace["error"] = row["error"]
                row["wall_s"] = time.monotonic() - started
                rows.append(row)
                with (args.output / "requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(trace) + "\n")
                with (args.output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest.update(completed_rows=len(rows), failed_rows=sum(not row["complete"] for row in rows))
        manifest["status"] = (
            "complete"
            if len(rows) == manifest["planned_rows"] and not manifest["failed_rows"]
            else "incomplete_or_failed"
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded", type=Path, required=True, help="Earlier paired/ directory, with every request retained"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["clarified", "answer_first", "greedy", "history_contract"], default="clarified"
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit per-call sampling seeds; no HTTP seed claim")
    raise SystemExit(run(parser.parse_args()))
