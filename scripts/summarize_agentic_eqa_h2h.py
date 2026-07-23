#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Score and plot answer-only vs agentic EQA arms from a H2H run directory.

Expects::

    <out>/baseline/**/eqa_results.json
    <out>/agentic_fallback/**/eqa_results.json   (or agentic_*/**)

Writes a JSON summary and optional bar/table figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emet.memory.graph_eqa.question_bank import score_eqa_results


def _load_eqa_rows(arm_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(arm_dir.rglob("eqa_results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict) and "questions" in data:
            rows.extend(data["questions"])
        elif isinstance(data, dict) and "rows" in data:
            rows.extend(data["rows"])
    return rows


def _arm_summary(name: str, arm_dir: Path) -> dict[str, Any]:
    rows = _load_eqa_rows(arm_dir)
    scored = (
        score_eqa_results(rows, episode_dir=arm_dir)
        if rows
        else {"accuracy": None, "n_questions": 0.0, "questions": []}
    )
    # Wall times live on the raw export rows; merge by question text.
    wall_by_q = {
        str(r.get("question") or ""): r.get("eqa_wall_s") for r in rows if r.get("eqa_wall_s") is not None
    }
    q_detail = []
    for q in scored.get("questions") or []:
        qtext = str(q.get("question") or "")
        q_detail.append(
            {
                "id": q.get("id") or q.get("question_id"),
                "question": qtext[:80],
                "answer": (q.get("answer") or q.get("answer_text") or "")[:120],
                "token_pass": q.get("token_pass"),
                "pass": q.get("pass"),
                "eqa_wall_s": wall_by_q.get(qtext),
            }
        )
    trace_picks = 0
    parse_ok = 0
    for tpath in arm_dir.rglob("agentic_trace.jsonl"):
        for line in tpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "tool_pick":
                trace_picks += 1
                if row.get("router_parse_ok") or row.get("picked_by") == "fallback":
                    parse_ok += 1
    return {
        "arm": name,
        "n_questions": int(scored.get("n_questions") or len(rows)),
        "accuracy": scored.get("accuracy"),
        "n_correct": sum(1 for q in q_detail if q.get("pass")),
        "questions": q_detail,
        "n_tool_picks": trace_picks,
        "path": str(arm_dir),
    }


def _plot(summary: dict[str, Any], figure: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"matplotlib required for --figure: {e}") from e

    arms = summary["arms"]
    names = [a["arm"] for a in arms]
    accs = [float(a["accuracy"]) if a.get("accuracy") is not None else 0.0 for a in arms]
    walls = []
    for a in arms:
        ws = [q.get("eqa_wall_s") for q in a.get("questions") or [] if q.get("eqa_wall_s") is not None]
        walls.append(sum(ws) / len(ws) if ws else 0.0)

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    axes[0].bar(names, accs, color=colors[: len(names)])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("EQA accuracy (token match)")
    axes[0].set_title("Answer-only vs agentic")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)

    axes[1].bar(names, walls, color=colors[: len(names)])
    axes[1].set_ylabel("Mean EQA wall time (s)")
    axes[1].set_title("Cost per question")
    for i, v in enumerate(walls):
        axes[1].text(i, v + max(walls + [1]) * 0.02, f"{v:.0f}s", ha="center", fontsize=10)

    fig.suptitle(summary.get("title") or "GraphEQA H2H", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="H2H root with baseline/ and agentic_*/")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write JSON summary")
    parser.add_argument("--figure", type=Path, default=None, help="Write comparison PNG")
    args = parser.parse_args()
    root = args.run_dir.expanduser().resolve()
    arms: list[dict[str, Any]] = []
    for name in ("baseline", "agentic_fallback", "agentic", "agentic_router"):
        d = root / name
        if d.is_dir():
            arms.append(_arm_summary(name, d))
    if not arms:
        # Fall back: any direct child with eqa_results
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if any(d.rglob("eqa_results.json")):
                arms.append(_arm_summary(d.name, d))
    summary = {
        "title": f"Agentic vs answer-only EQA ({root.name})",
        "run_dir": str(root),
        "arms": arms,
    }
    print(json.dumps({k: summary[k] for k in ("title", "run_dir")}, indent=2))
    for a in arms:
        print(
            f"  {a['arm']}: accuracy={a.get('accuracy')} "
            f"n={a.get('n_questions')} correct={a.get('n_correct')} "
            f"tool_picks={a.get('n_tool_picks')}"
        )
        for q in a.get("questions") or []:
            flag = "✓" if q.get("pass") or q.get("token_pass") else "✗"
            print(f"    {flag} {(q.get('question') or '')[:50]!r} → {(q.get('answer') or '')[:60]!r}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    if args.figure is not None and arms:
        _plot(summary, args.figure)
        print(f"wrote {args.figure}")
    return 0 if arms else 1


if __name__ == "__main__":
    raise SystemExit(main())
