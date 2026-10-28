# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
import runpy
from pathlib import Path


def test_saved_answer_selection_is_independent_of_gold():
    audit = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/audit_final_answer_authority.py"))[
        "audit_record"
    ]
    row = {
        "complete": True,
        "metrics": {
            "raw_eqa_output": json.dumps({"answer": "Two", "confidence": True}) + "\n[agentic_submit]\nanswer: One",
            "choices": ["One", "Two", "None", "Three"],
            "predicted_answer": "A",
            "gold_answer_letter": "B",
            "correct": False,
        },
    }
    result = audit(row)
    assert result["model_letter"] == "B" and result["model_correct"] and result["changed"]
    row["metrics"]["gold_answer_letter"] = "D"
    assert audit(row)["model_letter"] == "B"
    assert not audit(row)["model_correct"]
    row["metrics"]["raw_eqa_output"] = json.dumps({"answer": "Unknown", "confidence": False})
    assert audit(row)["model_letter"] == ""
    row["metrics"]["raw_eqa_output"] = json.dumps({"answer": "None", "confidence": True})
    assert audit(row)["model_letter"] == "C"
    row["metrics"]["raw_eqa_output"] = "truncated"
    assert audit(row)["status"] == "unparsed"
    row["complete"] = False
    assert audit(row)["status"] == "infrastructure"
