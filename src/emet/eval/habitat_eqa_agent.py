# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agent ↔ Habitat HM-EQA: same episode function as ``emet-habitat`` (zero-loss eval mode)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def run_hmeqa_via_shared_episode(
    *,
    question_id: int,
    method: str = "dynagraph",
    mock_llm: bool = False,
    extra_instruction: str | None = None,
    max_planning_steps: int = 20,
    max_movement_step: int = 10,
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    device: str = "cuda",
    output: Path | None = None,
) -> dict[str, Any]:
    """Run one HM-EQA episode through the shared ``run_hmeqa_episode`` (import or venv-habitat CLI).

    Agent ``--eqa-eval`` and ``emet-habitat run-episode`` must stay letter-identical when kwargs match.
    """
    try:
        from emet_habitat.runner import run_hmeqa_episode

        metrics = run_hmeqa_episode(
            question_id=int(question_id),
            method=str(method),
            mock_llm=bool(mock_llm),
            max_planning_steps=int(max_planning_steps),
            max_movement_step=int(max_movement_step),
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
            extra_instruction=extra_instruction,
        )
        payload = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics.__dict__)
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        return payload
    except ImportError:
        pass

    repo = Path(__file__).resolve().parents[3]
    habitat_bin = repo / ".venv-habitat" / "bin" / "emet-habitat"
    if not habitat_bin.is_file():
        raise RuntimeError(
            "emet_habitat is not importable and .venv-habitat/bin/emet-habitat is missing. "
            "Install with ./scripts/install_habitat.sh, then retry --eqa-eval."
        )
    out_path = Path(output) if output is not None else Path(
        os.environ.get("HOME", "/tmp")
    ) / ".cache" / "habitat_eqa" / "results" / f"agent_eqa_eval_q{int(question_id)}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(habitat_bin),
        "run-episode",
        "--question-id",
        str(int(question_id)),
        "--method",
        str(method),
        "--max-planning-steps",
        str(int(max_planning_steps)),
        "--max-movement-step",
        str(int(max_movement_step)),
        "--output",
        str(out_path),
        "--device",
        str(device),
    ]
    if mock_llm:
        cmd.append("--mock-llm")
    if extra_instruction:
        cmd.extend(["--extra-instruction", str(extra_instruction)])
    if eqa_vl_family:
        cmd.extend(["--eqa-vl-family", str(eqa_vl_family)])
    if eqa_hf_model_id:
        cmd.extend(["--eqa-hf-model-id", str(eqa_hf_model_id)])
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"emet-habitat run-episode failed (exit {proc.returncode}): {' '.join(cmd)}")
    if not out_path.is_file():
        raise RuntimeError(f"Expected episode JSONL at {out_path}")
    last: dict[str, Any] = {}
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last
