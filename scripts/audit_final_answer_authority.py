# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Counterfactual audit of saved final responses; no inference or policy tuning.

Hold the recorded final model response fixed and compare it with the scored
answer. This isolates answer replacement, not trajectory/context/stopping effects.
Gold answers are used only for reporting correctness, never selecting an answer.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from emet.memory.graph_eqa.eqa.mcq_debias import answer_is_unknownish, match_freeform_to_choice
from emet.utils.json_parse import first_json_dict_lenient


def audit_record(record):
    metrics = record.get("metrics") or {}
    if not record.get("complete"):
        return {"status": "infrastructure", "arm": record.get("arm")}
    raw = str(metrics.get("raw_eqa_output") or "").split("\n[agentic_submit]", 1)[0]
    response = first_json_dict_lenient(raw)
    if not isinstance(response, dict) or "answer" not in response:
        return {"status": "unparsed", "arm": record.get("arm")}
    answer = str(response.get("answer") or "").strip()
    choices = metrics.get("choices") or []
    index = None if answer_is_unknownish(answer, choices) else match_freeform_to_choice(answer, choices)
    letter = chr(ord("A") + index) if index is not None else ""
    return {
        "status": "parsed",
        "arm": record.get("arm"),
        "question_id": record.get("question_id"),
        "seed": record.get("seed"),
        "model_answer": answer,
        "model_confident": response.get("confidence"),
        "model_letter": letter,
        "scored_letter": metrics.get("predicted_answer"),
        "scored_correct": metrics.get("correct"),
        "model_correct": bool(letter and letter == metrics.get("gold_answer_letter")),
        "changed": letter != (metrics.get("predicted_answer") or ""),
    }


def audit(path):
    rows = [audit_record(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("arm")].append(row)
    summary = {}
    for arm, records in groups.items():
        parsed = [r for r in records if r["status"] == "parsed"]
        summary[arm] = {
            "recorded": len(records),
            "parsed": len(parsed),
            "scored_correct": sum(bool(r["scored_correct"]) for r in parsed),
            "model_correct": sum(r["model_correct"] for r in parsed),
            "changed": sum(r["changed"] for r in parsed),
        }
    return {
        "source": str(path),
        "scope": "fixed-response counterfactual, not live acceptance",
        "summary": summary,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.results], indent=2))
