#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Summarize classic vs agentic HM-EQA H2H JSONL and write bar charts.

Usage::

  uv run python scripts/summarize_hmeqa_agentic_h2h.py ~/runs/emet/hmeqa_agentic_h2h8_...
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Coin-flip last rung of the forced-answer ladder — keep in the headline score for
# measurement integrity, but also report accuracy with these rows dropped so a
# lucky A–D prior cannot inflate the agentic story.
_GUESS_PROVENANCE = frozenset({"uniform_prior"})


def _provenance_key(row: dict) -> str:
    prov = str(row.get("answer_provenance") or "").strip()
    return prov or "unset"


def provenance_breakdown(rows: list[dict]) -> dict[str, dict]:
    """Per-channel n / correct / accuracy (sorted by channel name)."""
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for r in rows:
        key = _provenance_key(r)
        counts[key] += 1
        if r.get("correct"):
            correct[key] += 1
    out: dict[str, dict] = {}
    for key in sorted(counts):
        n = counts[key]
        ok = correct[key]
        out[key] = {"n": n, "correct": ok, "accuracy": (ok / n) if n else None}
    return out


def summarize(out: Path) -> dict:
    summary: dict = {}
    for m in ("classic", "agentic"):
        p = out / f"{m}.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        ok = sum(1 for r in rows if r.get("correct"))
        steps = [r.get("planning_steps") for r in rows if isinstance(r.get("planning_steps"), (int, float))]
        per = []
        salvage_fired = 0
        with_salvage_ok = 0
        for r in rows:
            pred = r.get("predicted_answer") or r.get("parsed_answer_letter") or ""
            salvage_pred = str(r.get("salvage_pred") or "").strip()
            gold = str(r.get("gold_answer_letter") or "")
            scored_ok = bool(r.get("correct"))
            if salvage_pred:
                salvage_fired += 1
                if r.get("salvage_correct") is not None:
                    salvage_ok = bool(r.get("salvage_correct"))
                else:
                    salvage_ok = bool(gold) and str(salvage_pred).upper()[:1] == gold.upper()[:1]
                effective_ok = salvage_ok
            else:
                salvage_ok = False
                effective_ok = scored_ok
            if effective_ok:
                with_salvage_ok += 1
            per.append(
                {
                    "q": r.get("question_id"),
                    "correct": r.get("correct"),
                    "pred": pred,
                    "gold": gold,
                    "planning_steps": r.get("planning_steps"),
                    "observations": r.get("observations"),
                    "salvage_pred": salvage_pred,
                    "salvage_correct": bool(salvage_ok) if salvage_pred else False,
                    "scored_policy": r.get("scored_policy") or "",
                    "answer_provenance": _provenance_key(r) if r.get("answer_provenance") else "",
                }
            )
        by_prov = provenance_breakdown(rows)
        kept = [r for r in rows if _provenance_key(r) not in _GUESS_PROVENANCE]
        kept_ok = sum(1 for r in kept if r.get("correct"))
        block: dict = {
            "n": len(rows),
            "correct": ok,
            "accuracy": (ok / len(rows)) if rows else None,
            "mean_planning_steps": (sum(steps) / len(steps)) if steps else None,
            "by_provenance": by_prov,
            "n_excl_uniform_prior": len(kept),
            "correct_excl_uniform_prior": kept_ok,
            "accuracy_excl_uniform_prior": (kept_ok / len(kept)) if kept else None,
            "per": per,
        }
        if m == "agentic" and rows:
            block["accuracy_no_salvage"] = block["accuracy"]
            block["accuracy_with_salvage"] = with_salvage_ok / len(rows)
            block["correct_with_salvage"] = with_salvage_ok
            block["salvage_fired"] = salvage_fired
            print(
                f"{m}: {ok}/{len(rows)} acc={block['accuracy']} "
                f"no_salvage={ok}/{len(rows)}; "
                f"with_salvage_cf={with_salvage_ok}/{len(rows)} "
                f"(fired={salvage_fired}) "
                f"mean_steps={block['mean_planning_steps']}"
            )
        else:
            print(f"{m}: {ok}/{len(rows)} acc={block['accuracy']} mean_steps={block['mean_planning_steps']}")
        if rows and by_prov:
            bits = [f"{k}={v['correct']}/{v['n']}" for k, v in by_prov.items()]
            excl = block["accuracy_excl_uniform_prior"]
            excl_s = (
                f" excl_uniform={kept_ok}/{len(kept)} ({excl:.0%})"
                if excl is not None and len(kept) != len(rows)
                else ""
            )
            print(f"  provenance: {', '.join(bits)}{excl_s}")
        summary[m] = block
    (out / "h2h_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_bars(out: Path, summary: dict, output: Path | None = None) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"figure skip: {exc}")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    names = ["classic\nDynagraph", "agentic\nverify"]
    keys = ["classic", "agentic"]
    accs = [summary[k]["accuracy"] or 0 for k in keys]
    steps = [summary[k]["mean_planning_steps"] or 0 for k in keys]
    colors = ["#4C78A8", "#F58518"]
    bars = ax[0].bar(names, accs, color=colors)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("accuracy")
    ax[0].set_title("HM-EQA holdout (Dynagraph)")
    for b, v, k in zip(bars, accs, keys, strict=True):
        ax[0].text(
            b.get_x() + b.get_width() / 2,
            v + 0.03,
            f"{summary[k]['correct']}/{summary[k]['n']}\n{v:.0%}",
            ha="center",
            fontsize=9,
        )
    ax[1].bar(names, steps, color=colors)
    ax[1].set_ylabel("mean planning steps")
    ax[1].set_title("Search cost")
    fig.suptitle("Classic vs agentic-verify Dynagraph")
    fig.tight_layout()
    if output is not None:
        path = output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fig_dir = out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        path = fig_dir / "hmeqa_agentic_h2h.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=None,
        help="H2H run directory with classic.jsonl / agentic.jsonl",
    )
    ap.add_argument(
        "--from-summary",
        type=Path,
        default=None,
        help="Load an existing h2h_summary JSON (e.g. paper/data/.../holdout8_summary.json)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Bar chart PNG path (default: OUT/figures/hmeqa_agentic_h2h.png)",
    )
    args = ap.parse_args()

    if args.from_summary is not None:
        summary_path = args.from_summary.expanduser().resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for m in ("classic", "agentic"):
            s = summary.get(m) or {}
            print(
                f"{m}: {s.get('correct')}/{s.get('n')} acc={s.get('accuracy')} "
                f"mean_steps={s.get('mean_planning_steps')}"
            )
        out = summary_path.parent
        write_bars(out, summary, output=args.output)
        return

    if args.out_dir is None:
        raise SystemExit("Provide out_dir or --from-summary")
    out = args.out_dir.expanduser().resolve()
    summary = summarize(out)
    write_bars(out, summary, output=args.output)


if __name__ == "__main__":
    main()
