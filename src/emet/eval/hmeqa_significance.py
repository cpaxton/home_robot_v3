#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Paired significance tests for classic vs agentic HM-EQA H2H JSONL.

Reads ``OUT/classic.jsonl`` and ``OUT/agentic.jsonl`` (or an ``h2h_summary.json``
with ``classic.per`` / ``agentic.per``), joins on ``question_id``, and reports:

- Accuracy per arm with Wilson 95% CIs
- Exact McNemar (binomial on discordant pairs) for accuracy
- Bootstrap 95% CI on the accuracy difference (agentic − classic)
- Wilcoxon signed-rank on planning_steps (efficiency claim)

Usage::

  uv run emet hmeqa significance ~/runs/emet/hmeqa_agentic_bal32_...
  uv run emet hmeqa significance --from-summary paper/data/.../balanced32_summary.json \\
      --json OUT/significance.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

# Cap BLAS threads before scipy/numpy workers spawn — otherwise pytest
# subprocesses that fork after this module loads can SIGSEGV (OpenBLAS).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None, None
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def load_arm_rows(out_dir: Path, arm: str) -> dict[int, dict[str, Any]]:
    """Load per-question rows from ``{arm}.jsonl``, keyed by question_id."""
    path = out_dir / f"{arm}.jsonl"
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = r.get("question_id")
        if qid is None:
            continue
        rows[int(qid)] = r
    return rows


def load_from_summary(summary: dict[str, Any]) -> tuple[dict[int, dict], dict[int, dict]]:
    """Build arm maps from an ``h2h_summary.json``-style dict."""
    classic: dict[int, dict] = {}
    agentic: dict[int, dict] = {}
    for arm, dest in (("classic", classic), ("agentic", agentic)):
        block = summary.get(arm) or {}
        for row in block.get("per") or []:
            qid = row.get("q")
            if qid is None:
                continue
            dest[int(qid)] = {
                "question_id": int(qid),
                "correct": bool(row.get("correct")),
                "predicted_answer": row.get("pred"),
                "gold_answer_letter": row.get("gold"),
                "planning_steps": row.get("planning_steps"),
                "observations": row.get("observations"),
            }
    return classic, agentic


def paired_rows(
    classic: dict[int, dict[str, Any]],
    agentic: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join arms on shared question ids (sorted)."""
    shared = sorted(set(classic) & set(agentic))
    out: list[dict[str, Any]] = []
    for qid in shared:
        c, a = classic[qid], agentic[qid]
        out.append(
            {
                "question_id": qid,
                "classic_correct": bool(c.get("correct")),
                "agentic_correct": bool(a.get("correct")),
                "classic_steps": c.get("planning_steps"),
                "agentic_steps": a.get("planning_steps"),
                "classic_pred": c.get("predicted_answer"),
                "agentic_pred": a.get("predicted_answer"),
                "gold": c.get("gold_answer_letter") or a.get("gold_answer_letter"),
            }
        )
    return out


def mcnemar_exact(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact McNemar: binomtest on discordant pairs (b vs c).

    Contingency (rows=classic, cols=agentic):
      a = both correct, b = classic only, c = agentic only, d = both wrong.
    Under H0 (equal accuracy), P(b | b+c) = 0.5.
    """
    both = classic_only = agentic_only = neither = 0
    for p in pairs:
        c_ok, a_ok = p["classic_correct"], p["agentic_correct"]
        if c_ok and a_ok:
            both += 1
        elif c_ok and not a_ok:
            classic_only += 1
        elif a_ok and not c_ok:
            agentic_only += 1
        else:
            neither += 1
    discordant = classic_only + agentic_only
    p_value: float | None
    if discordant == 0:
        p_value = 1.0
    else:
        from scipy.stats import binomtest

        # Two-sided exact test: agentic_only successes out of discordant trials under p=0.5.
        p_value = float(binomtest(agentic_only, discordant, p=0.5, alternative="two-sided").pvalue)
    return {
        "both_correct": both,
        "classic_only": classic_only,
        "agentic_only": agentic_only,
        "both_wrong": neither,
        "discordant": discordant,
        "p_value": p_value,
    }


def wilcoxon_steps(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Wilcoxon signed-rank on planning_steps (classic − agentic)."""
    diffs: list[float] = []
    for p in pairs:
        cs, as_ = p.get("classic_steps"), p.get("agentic_steps")
        if isinstance(cs, (int, float)) and isinstance(as_, (int, float)):
            diffs.append(float(cs) - float(as_))
    if len(diffs) < 2:
        return {"n": len(diffs), "statistic": None, "p_value": None, "mean_diff": None}
    mean_diff = sum(diffs) / len(diffs)
    # All zeros → no difference to test.
    if all(abs(d) < 1e-12 for d in diffs):
        return {"n": len(diffs), "statistic": 0.0, "p_value": 1.0, "mean_diff": mean_diff}
    from scipy.stats import wilcoxon

    # alternative='greater' means classic steps > agentic (agentic more efficient).
    res = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
    return {
        "n": len(diffs),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff": mean_diff,
        "alternative": "classic_steps > agentic_steps",
    }


def bootstrap_acc_diff(
    pairs: list[dict[str, Any]],
    *,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap 95% CI on accuracy difference (agentic − classic)."""
    import numpy as np

    n = len(pairs)
    if n == 0:
        return {"n_boot": n_boot, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    c = np.array([1.0 if p["classic_correct"] else 0.0 for p in pairs])
    a = np.array([1.0 if p["agentic_correct"] else 0.0 for p in pairs])
    observed = float(a.mean() - c.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = a[idx].mean(axis=1) - c[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "n_boot": n_boot,
        "seed": seed,
        "observed": observed,
        "mean": float(boot.mean()),
        "ci95": [lo, hi],
    }


def analyze_pairs(pairs: list[dict[str, Any]], *, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Full paired analysis over shared question ids."""
    n = len(pairs)
    c_ok = sum(1 for p in pairs if p["classic_correct"])
    a_ok = sum(1 for p in pairs if p["agentic_correct"])
    c_lo, c_hi = _wilson_ci(c_ok, n)
    a_lo, a_hi = _wilson_ci(a_ok, n)
    c_steps = [p["classic_steps"] for p in pairs if isinstance(p.get("classic_steps"), (int, float))]
    a_steps = [p["agentic_steps"] for p in pairs if isinstance(p.get("agentic_steps"), (int, float))]
    return {
        "n_paired": n,
        "classic": {
            "correct": c_ok,
            "n": n,
            "accuracy": (c_ok / n) if n else None,
            "wilson_ci95": [c_lo, c_hi],
            "mean_planning_steps": (sum(c_steps) / len(c_steps)) if c_steps else None,
        },
        "agentic": {
            "correct": a_ok,
            "n": n,
            "accuracy": (a_ok / n) if n else None,
            "wilson_ci95": [a_lo, a_hi],
            "mean_planning_steps": (sum(a_steps) / len(a_steps)) if a_steps else None,
        },
        "accuracy_diff_agentic_minus_classic": ((a_ok - c_ok) / n) if n else None,
        "mcnemar": mcnemar_exact(pairs),
        "bootstrap_acc_diff": bootstrap_acc_diff(pairs, n_boot=n_boot, seed=seed),
        "wilcoxon_steps": wilcoxon_steps(pairs),
        "per": pairs,
    }


def analyze_run_dir(out_dir: Path, *, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    classic = load_arm_rows(out_dir, "classic")
    agentic = load_arm_rows(out_dir, "agentic")
    pairs = paired_rows(classic, agentic)
    result = analyze_pairs(pairs, n_boot=n_boot, seed=seed)
    result["source"] = {"out_dir": str(out_dir), "classic_n": len(classic), "agentic_n": len(agentic)}
    return result


def analyze_summary(summary: dict[str, Any], *, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    classic, agentic = load_from_summary(summary)
    pairs = paired_rows(classic, agentic)
    result = analyze_pairs(pairs, n_boot=n_boot, seed=seed)
    result["source"] = {"from_summary": True, "classic_n": len(classic), "agentic_n": len(agentic)}
    return result


def _print_report(result: dict[str, Any]) -> None:
    c, a = result["classic"], result["agentic"]
    m = result["mcnemar"]
    w = result["wilcoxon_steps"]
    b = result["bootstrap_acc_diff"]
    print(f"n_paired={result['n_paired']}")
    print(
        f"classic: {c['correct']}/{c['n']} acc={c['accuracy']:.3f} "
        f"wilson95=[{c['wilson_ci95'][0]:.3f},{c['wilson_ci95'][1]:.3f}] "
        f"mean_steps={c['mean_planning_steps']}"
    )
    print(
        f"agentic: {a['correct']}/{a['n']} acc={a['accuracy']:.3f} "
        f"wilson95=[{a['wilson_ci95'][0]:.3f},{a['wilson_ci95'][1]:.3f}] "
        f"mean_steps={a['mean_planning_steps']}"
    )
    print(
        f"McNemar: both={m['both_correct']} classic_only={m['classic_only']} "
        f"agentic_only={m['agentic_only']} both_wrong={m['both_wrong']} "
        f"p={m['p_value']}"
    )
    print(f"bootstrap Δacc (agentic−classic) obs={b.get('observed')} ci95={b.get('ci95')}")
    print(
        f"Wilcoxon steps (classic>agentic): n={w['n']} mean_diff={w['mean_diff']} "
        f"stat={w['statistic']} p={w['p_value']}"
    )


def main(argv: list[str] | None = None) -> int:
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
        help="Load an existing h2h_summary JSON instead of JSONL",
    )
    ap.add_argument("--json", type=Path, default=None, help="Write full result JSON here")
    ap.add_argument("--n-boot", type=int, default=5000, help="Bootstrap resamples (default 5000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap")
    args = ap.parse_args(argv)

    if args.from_summary is not None:
        summary = json.loads(args.from_summary.expanduser().resolve().read_text(encoding="utf-8"))
        result = analyze_summary(summary, n_boot=args.n_boot, seed=args.seed)
        result["source"]["summary_path"] = str(args.from_summary)
    elif args.out_dir is not None:
        result = analyze_run_dir(args.out_dir.expanduser().resolve(), n_boot=args.n_boot, seed=args.seed)
    else:
        ap.error("Provide out_dir or --from-summary")

    _print_report(result)
    json_path = args.json
    if json_path is None and args.out_dir is not None:
        json_path = args.out_dir.expanduser().resolve() / "significance.json"
    if json_path is not None:
        path = json_path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Drop per-question rows from the default on-disk dump if huge? Keep them — paper may cite.
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
