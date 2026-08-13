#!/usr/bin/env python3
"""Offline A/B of the HM-EQA location-MCQ letter-override policy on saved jsonl.

The full-113 sweep (2026-08-13) exposed a scoring bug: the geometric
equipment-distance guess ([memory-location]) overrode confident, correct VLM
letters (11 dynagraph + 14 static_graph episodes had json answer == gold but
scored wrong). The fix gates the equipment override on VLM confidence
(eqa.location_override_equip_gate).

This script re-derives each episode's scored letter from the saved jsonl under
the legacy (always-override) vs gated policy, WITHOUT re-running the GPU sweep:

    uv run python scripts/hmeqa_override_ab.py \
        ~/.cache/habitat_eqa/results/subset_paper113_*_dynagraph_qwen3_vl.jsonl

Caveat: the jsonl records the *outcome* ([memory-location] marker + appended
letter) but not which fallback branch fired (equip vs image-landmark vs
memory). Recovery is attributed to the equip gate, which is correct when the
VLM was confident with a parsed letter (the memory branch requires abstain and
the image branch would not fire when img_letter == VLM letter). Episodes where
the recovery assumption may not hold are printed with "REVIEW".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from emet.habitat.metrics import extract_mcq_letter_from_raw_eqa

_MEMORY_LOC = re.compile(r"\[memory-location\]\s*answer\s*:\s*([A-E])\s*")


def _choices(r: dict) -> list[str] | None:
    ch = r.get("choices")
    if isinstance(ch, str):
        try:
            ch = json.loads(ch)
        except Exception:
            return None
    return ch if isinstance(ch, list) else None


def _json_answer_letter(r: dict) -> str:
    raw = r.get("raw_eqa_output") or ""
    m = re.search(r"[\"']answer[\"']\s*:\s*[\"']([A-E])[\"']", raw, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _json_confidence(r: dict) -> bool:
    """Raw VLM confidence from the JSON (pre graph-coverage gate)."""
    raw = r.get("raw_eqa_output") or ""
    m = re.search(r"[\"']confidence[\"']\s*:\s*(true|false)", raw, flags=re.IGNORECASE)
    return bool(m and m.group(1).lower() == "true")


def rescore(rows: list[dict], *, equip_gate: bool, image_gate: bool) -> tuple[list[dict], int]:
    """Recompute each episode's letter + correctness under a policy."""
    n_recovered = 0
    out: list[dict] = []
    for r in rows:
        raw = r.get("raw_eqa_output") or ""
        pred = (r.get("predicted_answer") or "").strip()
        gold = r.get("gold_answer_letter")
        choices = _choices(r)
        vlm = _json_answer_letter(r)
        mem = _MEMORY_LOC.search(raw)
        mem_letter = mem.group(1).upper() if mem else ""
        confidence = _json_confidence(r)
        parsed_letter = extract_mcq_letter_from_raw_eqa(raw, choices) or ""
        vlm_clear = confidence and bool(parsed_letter)

        new_pred = pred
        recovered = False
        review = False
        if mem_letter and mem_letter != vlm:
            # A [memory-location] override replaced the VLM letter. Under the
            # gated policy the equip branch (the common case) is suppressed when
            # the VLM was confident + had a letter; the image branch stays unless
            # image_gate. We cannot tell equip vs image from the jsonl, so treat
            # equip as the assumption and flag when it changes the score.
            gated_off = (vlm_clear and equip_gate) or (vlm_clear and image_gate)
            if gated_off and vlm:
                new_pred = vlm
                if vlm == gold:
                    recovered = True
                if not (vlm_clear and equip_gate):
                    review = True
        elif not pred and vlm:
            # JSON answer field was dropped entirely (parse gap fixed).
            new_pred = vlm
            if vlm == gold:
                recovered = True

        new_correct = bool(new_pred and gold and new_pred == gold)
        out.append(
            {
                "question_id": r.get("question_id"),
                "pred": new_pred,
                "correct": new_correct,
                "recovered": recovered,
                "review": review,
            }
        )
        if recovered:
            n_recovered += 1
    return out, n_recovered


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1]).expanduser()
    rows = [json.loads(l) for l in path.open() if l.strip()]
    n = len(rows)
    base = sum(1 for r in rows if r.get("correct"))

    def _acc(res: list[dict]) -> int:
        return sum(1 for r in res if r["correct"])

    gated, n_rec = rescore(rows, equip_gate=True, image_gate=False)
    legacy, _ = rescore(rows, equip_gate=False, image_gate=False)
    img_gated, n_rec_img = rescore(rows, equip_gate=True, image_gate=True)

    print(f"file: {path.name}  n={n}")
    print(f"  as-scored (recorded): {base}/{n} = {100*base/n:.1f}%")
    print(
        f"  legacy (always-override):      {_acc(legacy)}/{n} = {100*_acc(legacy)/n:.1f}%  "
        f"(n_rec={sum(1 for r in legacy if r['recovered'])})"
    )
    print(
        f"  equip-gated (fix):              {_acc(gated)}/{n} = {100*_acc(gated)/n:.1f}%  "
        f"(recovered {n_rec})"
    )
    print(
        f"  equip+image-gated (stricter):   {_acc(img_gated)}/{n} = {100*_acc(img_gated)/n:.1f}%  "
        f"(recovered {n_rec_img})"
    )
    rec = [r for r in gated if r["recovered"]]
    rev = [r["question_id"] for r in gated if r["review"]]
    if rec:
        print(f"  recovered qids: {sorted(r['question_id'] for r in rec)}")
    if rev:
        print(f"  REVIEW (assumed equip, may be image): {sorted(rev)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
