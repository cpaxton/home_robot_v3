# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Attribute classic vs agentic HM-EQA failures (context-gap focused).

Usage::

  uv run emet hmeqa failures ~/runs/emet/hmeqa_agentic_bal32r2_...
  uv run emet hmeqa failures --from-summary paper/data/.../balanced32_summary.json \\
      --out-dir ~/runs/emet/hmeqa_agentic_bal32r2_...
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from emet.eval.hmeqa_significance import load_arm_rows, load_from_summary, paired_rows


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            out.append(row)
    return out


def _trace_path_for_row(row: dict[str, Any], qid: int) -> Path | None:
    bundle = str(row.get("debug_bundle_dir") or "").strip()
    if bundle:
        p = Path(bundle) / "agentic_trace.jsonl"
        if p.is_file():
            return p
    # Common Habitat cache layout — prefer newest mtime when several bundles exist.
    cand = Path.home() / ".cache" / "habitat_eqa" / "episodes" / f"h2h_agentic_q{qid:04d}"
    matches = list(cand.glob("*/agentic_trace.jsonl")) if cand.is_dir() else []
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _summarize_trace(path: Path | None) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "n_rows": 0,
        "tools": {},
        "n_verify": 0,
        "n_assess": 0,
        "n_explore": 0,
        "last_assess": None,
        "last_verify": None,
        "last_submit": None,
        "sync_scored": None,
    }
    if path is None or not path.is_file():
        return empty
    tools: Counter[str] = Counter()
    assesses: list[dict[str, Any]] = []
    verifies: list[dict[str, Any]] = []
    submits: list[dict[str, Any]] = []
    syncs: list[dict[str, Any]] = []
    n = 0
    for row in _jsonl_rows(path):
        n += 1
        if row.get("event") == "tool_pick":
            tools[f"pick:{row.get('tool')}"] += 1
            continue
        if row.get("event") == "sync_scored_answer":
            syncs.append(row)
            tools["sync_scored_answer"] += 1
            continue
        name = str(row.get("tool") or row.get("event") or "?")
        tools[name] += 1
        if name == "vlm_assess" or row.get("event") == "vlm_assess":
            assesses.append(row)
        elif name == "verify_siglip":
            verifies.append(row)
        elif name == "submit_answer":
            submits.append(row)
        elif name == "explore_frontier":
            pass
    return {
        "n_rows": n,
        "tools": dict(tools.most_common()),
        "n_verify": len(verifies),
        "n_assess": len(assesses),
        "n_explore": int(tools.get("explore_frontier", 0)),
        "last_assess": assesses[-1] if assesses else None,
        "last_verify": verifies[-1] if verifies else None,
        "last_submit": submits[-1] if submits else None,
        "sync_scored": syncs[-1] if syncs else None,
        "trace_path": str(path),
    }


def _stem_tokens(question: str) -> set[str]:
    q = (question or "").lower()
    cut = q.find("a)")
    if cut > 0:
        q = q[:cut]
    return {t for t in "".join(ch if ch.isalnum() else " " for ch in q).split() if len(t) > 2}


def _phrase_in_stem(phrase: str | None, question: str) -> bool | None:
    if not phrase:
        return None
    stem = _stem_tokens(question)
    if not stem:
        return None
    words = {t for t in "".join(ch if ch.isalnum() else " " for ch in phrase.lower()).split() if len(t) > 2}
    if not words:
        return None
    return bool(words & stem)


def classify_pair(
    *,
    qid: int,
    classic: dict[str, Any],
    agentic: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Assign a primary failure/context bucket for one paired question."""
    c_ok = bool(classic.get("correct"))
    a_ok = bool(agentic.get("correct"))
    c_pred = str(classic.get("predicted_answer") or "").strip()
    a_pred = str(agentic.get("predicted_answer") or "").strip()
    gold = str(
        classic.get("gold_answer_letter") or agentic.get("gold_answer_letter") or ""
    ).strip().upper()
    question = str(agentic.get("question") or classic.get("question") or "")
    pair_kind = (
        "both_correct"
        if c_ok and a_ok
        else "both_wrong"
        if (not c_ok and not a_ok)
        else "classic_only"
        if c_ok and not a_ok
        else "agentic_only"
        if a_ok and not c_ok
        else "other"
    )

    submit = trace.get("last_submit") or {}
    assess = trace.get("last_assess") or {}
    verify = trace.get("last_verify") or {}
    sync = trace.get("sync_scored") or {}
    raw_a = str(agentic.get("raw_eqa_output") or "")
    gh_c = classic.get("graph_health") if isinstance(classic.get("graph_health"), dict) else {}
    gh_a = agentic.get("graph_health") if isinstance(agentic.get("graph_health"), dict) else {}

    suggested = str(assess.get("suggested_answer") or submit.get("vlm_suggested") or "").strip()
    submit_final = str(submit.get("final_answer") or "").strip()
    sync_letter = str(sync.get("letter") or "").strip().upper()
    phrase = str(verify.get("phrase") or assess.get("target") or "").strip()
    phrase_ok = _phrase_in_stem(phrase, question)

    bucket = "ok" if a_ok else "other"
    reasons: list[str] = []

    if not a_ok and not a_pred and not (agentic.get("debug_bundle_dir") or trace.get("n_rows")):
        bucket = "infra_empty"
        reasons.append("missing agentic pred/bundle")
    elif not a_ok and (not a_pred or a_pred.lower() in {"unknown", "none"}):
        bucket = "empty_or_abstain"
        reasons.append("empty/unknown scored pred")
    elif not a_ok and suggested and suggested.upper()[:1] == gold and a_pred.upper()[:1] != gold:
        bucket = "scored_vs_submit_mismatch"
        reasons.append(
            f"assess/submit suggested {suggested[:1].upper()} but scored {a_pred[:1].upper() or '?'}"
        )
    elif not a_ok and submit_final and submit_final.upper()[:1] == gold and a_pred.upper()[:1] != gold:
        bucket = "scored_vs_submit_mismatch"
        reasons.append(f"submit final {submit_final[:1].upper()} but scored {a_pred[:1].upper()}")
    elif not a_ok and (
        bool(verify.get("fused_verified"))
        and str(verify.get("decision") or "").upper() == "ABSENT"
    ):
        bucket = "false_fused_verify"
        reasons.append("fused_verified with ABSENT decision")
    elif not a_ok and phrase_ok is False:
        bucket = "wrong_phrase_or_inventory"
        reasons.append(f"verify phrase {phrase!r} not in question stem")
    elif not a_ok and int(trace.get("n_assess") or 0) <= 1 and int(trace.get("n_explore") or 0) == 0:
        bucket = "context_thin_assess"
        reasons.append("≤1 assess and no explore before letter")
    elif not a_ok and "[salvage]" in raw_a and "agentic_submit" not in raw_a:
        bucket = "salvage_without_agentic_sync"
        reasons.append("raw_eqa salvage without agentic_submit sync")
    elif not a_ok and int(gh_a.get("prompt_obs_count") or 0) < int(gh_c.get("prompt_obs_count") or 0):
        bucket = "verified_obs_not_shown"
        reasons.append("agentic prompt_obs_count < classic")
    elif not a_ok and c_ok:
        bucket = "vlm_wrong_with_full_context"
        reasons.append("classic correct; agentic wrong under similar prompt_obs")
    elif a_ok:
        bucket = "ok"

    return {
        "question_id": qid,
        "pair_kind": pair_kind,
        "bucket": bucket,
        "reasons": reasons,
        "gold": gold,
        "classic_pred": c_pred,
        "agentic_pred": a_pred,
        "classic_correct": c_ok,
        "agentic_correct": a_ok,
        "classic_steps": classic.get("planning_steps"),
        "agentic_steps": agentic.get("planning_steps"),
        "classic_prompt_obs": gh_c.get("prompt_obs_count"),
        "agentic_prompt_obs": gh_a.get("prompt_obs_count"),
        "classic_prompt_nodes": gh_c.get("prompt_node_count"),
        "agentic_prompt_nodes": gh_a.get("prompt_node_count"),
        "classic_n_object": gh_c.get("n_object"),
        "agentic_n_object": gh_a.get("n_object"),
        "suggested_answer": suggested or None,
        "submit_final": submit_final or None,
        "submit_source": submit.get("answer_source"),
        "sync_letter": sync_letter or None,
        "verify_phrase": phrase or None,
        "phrase_in_stem": phrase_ok,
        "last_verify_decision": verify.get("decision"),
        "fused_verified": verify.get("fused_verified"),
        "force_obs_ids": submit.get("force_obs_ids"),
        "last_eqa_obs_ids": submit.get("last_eqa_obs_ids"),
        "trace": {
            "n_verify": trace.get("n_verify"),
            "n_assess": trace.get("n_assess"),
            "n_explore": trace.get("n_explore"),
            "tools": trace.get("tools"),
            "trace_path": trace.get("trace_path"),
        },
    }


def analyze_run(
    *,
    classic: dict[int, dict[str, Any]],
    agentic: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    pairs = paired_rows(classic, agentic)
    rows: list[dict[str, Any]] = []
    for p in pairs:
        qid = int(p["question_id"])
        a_row = agentic[qid]
        c_row = classic[qid]
        trace = _summarize_trace(_trace_path_for_row(a_row, qid))
        rows.append(classify_pair(qid=qid, classic=c_row, agentic=a_row, trace=trace))

    bucket_counts = Counter(r["bucket"] for r in rows)
    pair_counts = Counter(r["pair_kind"] for r in rows)
    classic_only = [r for r in rows if r["pair_kind"] == "classic_only"]
    return {
        "n_paired": len(rows),
        "pair_counts": dict(pair_counts),
        "bucket_counts": dict(bucket_counts),
        "classic_only": classic_only,
        "agentic_only": [r for r in rows if r["pair_kind"] == "agentic_only"],
        "both_wrong": [r for r in rows if r["pair_kind"] == "both_wrong"],
        "per": rows,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"n_paired={report['n_paired']}")
    print(f"pair_counts={report['pair_counts']}")
    print(f"bucket_counts={report['bucket_counts']}")
    print("classic_only:")
    for row in report.get("classic_only") or []:
        print(
            f"  q{row['question_id']}: bucket={row['bucket']} "
            f"classic={row['classic_pred']!r} agentic={row['agentic_pred']!r} "
            f"gold={row['gold']} suggested={row.get('suggested_answer')!r} "
            f"reasons={row.get('reasons')}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path, nargs="?", default=None)
    ap.add_argument("--from-summary", type=Path, default=None)
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write full report JSON (default: OUT/failure_report.json)",
    )
    args = ap.parse_args(argv)

    if args.from_summary is not None:
        summary = json.loads(args.from_summary.expanduser().resolve().read_text(encoding="utf-8"))
        classic, agentic = load_from_summary(summary)
        # Enrich from OUT if provided for traces / graph_health
        if args.out_dir is not None:
            out = args.out_dir.expanduser().resolve()
            classic_full = load_arm_rows(out, "classic")
            agentic_full = load_arm_rows(out, "agentic")
            for qid, row in classic_full.items():
                classic.setdefault(qid, {}).update(row)
            for qid, row in agentic_full.items():
                agentic.setdefault(qid, {}).update(row)
        source = {"from_summary": str(args.from_summary), "out_dir": str(args.out_dir) if args.out_dir else None}
    elif args.out_dir is not None:
        out = args.out_dir.expanduser().resolve()
        classic = load_arm_rows(out, "classic")
        agentic = load_arm_rows(out, "agentic")
        source = {"out_dir": str(out)}
    else:
        ap.error("Provide out_dir or --from-summary")

    report = analyze_run(classic=classic, agentic=agentic)
    report["source"] = source
    _print_report(report)

    json_path = args.json
    if json_path is None and args.out_dir is not None:
        json_path = args.out_dir.expanduser().resolve() / "failure_report.json"
    if json_path is not None:
        path = json_path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
