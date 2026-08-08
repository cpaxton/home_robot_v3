# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""VLM-only Caliban A/B for the trailing ``Answer:`` cue JSON-collapse bug.

Reproduces the 2026-08-05 probe: Arm A uses the real assistant prefill
``{"reasoning":``, Arm B sends no prefill. Both arms then run through the real
``GraphEQAMemory.parse_answer`` path (JSON repair + labeled scrape + terse
fallback) to verify the letter survives the format collapse.

Usage:
    EMET_VL_ENDPOINT=openai@http://caliban:8000/v1 uv run python scripts/caliban_eqa_ab.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from emet.habitat.metrics import parse_mcq_choices_from_question
from emet.llms.openai_vllm_client import OpenaiVLLMClient, parse_openai_endpoint_spec
from emet.llms.prompts.hmeqa_eqa_prompt import HMEQA_EQA_PROMPT
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

QUESTION = "Did I leave the fruit bowl next to the microwave?"
CHOICES = [
    "Yes, I left the fruit bowl there",
    "No, it is not there",
    "I am not sure",
    "(Do not choose this option)",
]

PREFILL = '{"reasoning":'
OUT_PATH = Path("/tmp/caliban_ab_results.json")


def build_question_line(question: str, choices: list[str]) -> str:
    opts = " ".join(f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))
    return f"Question: {question} {opts}. Answer:"


def main() -> int:
    endpoint = os.environ.get("EMET_VL_ENDPOINT", "")
    if not endpoint:
        print("ERROR: set EMET_VL_ENDPOINT=openai@http://caliban:8000/v1", file=sys.stderr)
        return 2
    base_url, model_from_spec = parse_openai_endpoint_spec(endpoint)

    client = OpenaiVLLMClient(
        HMEQA_EQA_PROMPT,
        base_url=base_url,
        model=model_from_spec or "emet-vl",
        max_tokens=256,
        image_max_side=512,
    )
    mem = GraphEQAMemory(eqa_client=lambda _x: "", image_description_client=lambda _x: "")
    qline = build_question_line(QUESTION, CHOICES)
    print(f"question_line={qline!r}\n")

    results: dict[str, dict] = {}
    for name, prefill in (("armA_prefill", PREFILL), ("armB_noprefill", None)):
        raw = client.generate_multimodal(
            qline,
            max_new_tokens=256,
            assistant_prefill=prefill,
        )
        # parse twice exactly like query_answer: JSON-first (with prefill repair),
        # then labeled scrape if the answer came back empty.
        _, answer, confidence, action, _ = mem.parse_answer(
            raw or "",
            prefer_json=True,
            json_prefill=prefill,
        )
        if not (answer or "").strip():
            _, answer, confidence, action, _ = mem.parse_answer(raw or "", prefer_json=False)
        results[name] = {"raw": raw, "answer": answer, "confidence": confidence, "action": action}
        print(f"=== {name} (prefill={prefill!r}) ===")
        print(f"raw: {raw!r}")
        print(f"parsed: answer={answer!r} confidence={confidence} action={action!r}\n")

    print("--- summary ---")
    for name, r in results.items():
        ok = r["answer"].upper() in {"A", "B", "C", "D"}
        print(
            f"{name}: raw={r['raw']!r} -> answer={r['answer']!r} "
            f"confidence={r['confidence']} recoverable={ok}"
        )
    print("--- round-trip check: choices parse ---")
    print(parse_mcq_choices_from_question(qline))
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
