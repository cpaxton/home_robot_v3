#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tiny recorded-evidence diagnostic via the production VLM conversation client.

No robot movement, manipulation, active exploration or official benchmark score.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from emet.eval.agentic_vlm_assess import _parse_json_object
from emet.eval.embodied_qa_datasets import EvaluationQuestion
from emet.eval.embodied_qa_evidence import bind_recorded_evidence
from emet.eval.eqa_decision_trace import record_vlm_request
from emet.eval.harness_capability_fixtures import load_development_cases
from emet.llms.conversation import VLMConversation
from emet.llms.openai_vllm_client import OpenaiVLLMClient

SYSTEM = (
    "Answer using only the supplied recorded evidence and conversation. Frames arrive in temporal order; "
    "they are recorded views, not a live robot. Do not invent actions or observations. "
    "If evidence is insufficient or the referent is ambiguous, abstain. "
    'On the final question return JSON {"answer": "short answer", "abstain": true or false}.'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/harness_capability_dev.json"))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    cases = [c for c in load_development_cases(args.config) if c["family"] in ("recall", "search_act", "insufficient")]
    manifest_path = args.evidence / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    qualification = json.loads((args.evidence / "qualification.json").read_text())
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if qualification["config_sha256"] != config_hash:
        raise ValueError("Evidence generated for a different fixture population")
    packets = {}
    for c in cases:
        q = EvaluationQuestion(
            "harness-capability-dev",
            c["id"],
            "development",
            c["question"],
            c["family"],
            "controlled-development",
            c["expected"],
            (0, 1),
            c["id"],
        )
        packets[c["id"]] = bind_recorded_evidence(q, manifest, root=args.evidence)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": config_hash,
        "evidence_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_url": args.base_url,
        "model": args.model,
        "quantization": "not established by endpoint metadata",
        "system_prompt": SYSTEM,
        "arms": ["full_history", "current_only"],
        "scope": "Recorded evidence diagnostic only; not active agent or physical task success; development cases",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    client = OpenaiVLLMClient(base_url=args.base_url, model=args.model, max_tokens=128)
    results = []
    for index, c in enumerate(cases):
        packet = packets[c["id"]]
        for arm in protocol["arms"] if index % 2 == 0 else list(reversed(protocol["arms"])):
            conversation = VLMConversation()
            calls = 0
            case_dir = args.output / f"{c['id']}-{arm}"

            def call(text, images):
                nonlocal calls
                calls += 1
                record_vlm_request(
                    case_dir / f"call-{calls}",
                    system_prompt=SYSTEM,
                    user_text=text,
                    images=images,
                    metadata={"arm": arm, "episode_id": c["id"], "scope": "recorded_evidence"},
                    history=conversation.messages,
                )
                return conversation.generate(client, [text, *images], system_prompt=SYSTEM)

            started = time.monotonic()
            row = {"episode_id": c["id"], "family": c["family"], "arm": arm, "reference_answer": c["expected"]}
            try:
                with client.preserve_conversation():
                    if arm == "full_history":
                        call(
                            "Earlier recorded view. Observe it; a question will follow. Acknowledge briefly.",
                            [packet.frames[0].load_rgb()],
                        )
                    for turn in c.get("delay_turns", []):
                        call(turn, [])
                    raw = call("Later recorded view. Final question: " + packet.question, [packet.frames[1].load_rgb()])
                parsed = _parse_json_object(str(raw))
                answer, abstain = parsed.get("answer"), parsed.get("abstain")
                valid = isinstance(answer, str) and type(abstain) is bool
                correct = valid and (
                    abstain
                    if c["expected"] == "abstain"
                    else not abstain and answer.strip().lower().rstrip(".") == c["expected"]
                )
                row.update(raw=raw, parsed=parsed, valid=valid, correct=bool(correct), error=None)
            except Exception as error:
                row.update(correct=False, valid=False, error=f"{type(error).__name__}: {error}")
            row.update(calls=calls, wall_s=time.monotonic() - started)
            results.append(row)
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    summary = {
        arm: {
            "correct": sum(r["correct"] for r in results if r["arm"] == arm),
            "n": sum(r["arm"] == arm for r in results),
            "errors": sum(bool(r.get("error")) for r in results if r["arm"] == arm),
        }
        for arm in protocol["arms"]
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
