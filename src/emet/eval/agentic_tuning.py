# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline tuning of agentic EQA verify thresholds and budgets from traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_agentic_traces(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_agentic_traces_dir(root: str | Path) -> list[dict[str, Any]]:
    root_p = Path(root)
    rows: list[dict[str, Any]] = []
    if root_p.is_file():
        return load_agentic_traces(root_p)
    for path in sorted(root_p.rglob("agentic_trace.jsonl")):
        rows.extend(load_agentic_traces(path))
    return rows


def verify_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in traces if r.get("tool") == "verify_siglip" and "sim" in r]


def summary_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in traces if r.get("tool") == "summary"]


def recompute_sim(text_feat: list[float] | None, img_feat: list[float] | None) -> float | None:
    if not text_feat or not img_feat:
        return None
    if len(text_feat) != len(img_feat):
        return None
    return float(sum(a * b for a, b in zip(text_feat, img_feat)))


def evaluate_threshold(
    verify: list[dict[str, Any]],
    threshold: float,
) -> dict[str, float]:
    """Precision/recall/F1 of PRESENT decision vs gt_present when labeled."""
    tp = fp = tn = fn = 0
    labeled = 0
    for row in verify:
        sim = row.get("sim")
        if row.get("text_feat") and row.get("img_feat"):
            recomputed = recompute_sim(row["text_feat"], row["img_feat"])
            if recomputed is not None:
                sim = recomputed
        if sim is None:
            continue
        pred_present = float(sim) >= float(threshold)
        if "gt_present" not in row:
            continue
        labeled += 1
        gt = bool(row["gt_present"])
        if pred_present and gt:
            tp += 1
        elif pred_present and not gt:
            fp += 1
        elif (not pred_present) and gt:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": float(threshold),
        "n_labeled": float(labeled),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def sweep_verify_thresholds(
    traces: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
    verify = verify_rows(traces)
    if thresholds is None:
        thresholds = [round(0.15 + 0.01 * i, 2) for i in range(26)]  # 0.15..0.40
    return [evaluate_threshold(verify, t) for t in thresholds]


def best_threshold(sweep: list[dict[str, float]]) -> dict[str, float] | None:
    labeled = [r for r in sweep if r.get("n_labeled", 0) > 0]
    if not labeled:
        return None
    return max(labeled, key=lambda r: (r["f1"], r["precision"], -abs(r["threshold"] - 0.28)))


def simulate_early_stop_accuracy(
    traces: list[dict[str, Any]],
    threshold: float,
    *,
    max_rounds: int | None = None,
) -> dict[str, float]:
    """Re-simulate: first verify PRESENT at *threshold* early-stops that question."""
    by_q: dict[str, list[dict[str, Any]]] = {}
    for row in traces:
        q = str(row.get("question") or "")
        by_q.setdefault(q, []).append(row)

    n = 0
    correct = 0
    rounds_sum = 0.0
    for q, rows in by_q.items():
        if not q:
            continue
        summary = next((r for r in rows if r.get("tool") == "summary"), None)
        verifies = [r for r in rows if r.get("tool") == "verify_siglip"]
        stop_round = None
        for vr in verifies:
            sim = vr.get("sim")
            if vr.get("text_feat") and vr.get("img_feat"):
                recomputed = recompute_sim(vr["text_feat"], vr["img_feat"])
                if recomputed is not None:
                    sim = recomputed
            if sim is None:
                continue
            rnd = int(vr.get("round", 0))
            if max_rounds is not None and rnd >= int(max_rounds):
                break
            if float(sim) >= float(threshold):
                stop_round = rnd + 1
                break
        if summary is None:
            continue
        n += 1
        hit = stop_round is not None
        if max_rounds is not None and not hit:
            # Budget too tight: never reached a PRESENT verify → count as miss.
            rounds_sum += float(max_rounds)
            continue
        if "correct" in summary:
            correct += int(bool(summary["correct"]))
        elif summary.get("confidence"):
            correct += 1
        rounds_sum += float(stop_round if stop_round is not None else summary.get("n_rounds", 0) or 0)
    return {
        "threshold": float(threshold),
        "n_questions": float(n),
        "accuracy": (correct / n) if n else 0.0,
        "mean_rounds": (rounds_sum / n) if n else 0.0,
        "max_rounds_cap": float(max_rounds) if max_rounds is not None else -1.0,
    }


def sweep_budgets(
    traces: list[dict[str, Any]],
    threshold: float,
    max_rounds_values: list[int] | None = None,
) -> list[dict[str, float]]:
    if max_rounds_values is None:
        max_rounds_values = list(range(1, 9))
    return [simulate_early_stop_accuracy(traces, threshold, max_rounds=m) for m in max_rounds_values]


def budget_knee(sweep: list[dict[str, float]], *, tol: float = 0.02) -> dict[str, float] | None:
    if not sweep:
        return None
    best_acc = max(r["accuracy"] for r in sweep)
    for r in sorted(sweep, key=lambda x: x["max_rounds_cap"]):
        if r["accuracy"] >= best_acc - tol:
            return r
    return sweep[-1]


def nav_distance_report(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Correlational summary of nav success vs subsequent verify PRESENT."""
    navs = [r for r in traces if r.get("tool") == "navigate_to_obs"]
    verifies = verify_rows(traces)
    n_ok = sum(1 for r in navs if r.get("nav_success"))
    n_fail = len(navs) - n_ok
    present_after = 0
    for vr in verifies:
        if vr.get("decision") == "PRESENT" or (
            vr.get("sim") is not None and float(vr["sim"]) >= 0.28
        ):
            present_after += 1
    return {
        "n_nav": len(navs),
        "nav_success": n_ok,
        "nav_fail": n_fail,
        "n_verify_presentish": present_after,
        "note": "Nav-distance changes alter trajectories; confirm candidates with a real smoke.",
    }


def router_report(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """VLM-router health from tool_pick rows: parse rate, fallback usage, tool mix."""
    picks = [r for r in traces if r.get("event") == "tool_pick"]
    vlm = [r for r in picks if r.get("picked_by") == "vlm"]
    fallback = [r for r in picks if r.get("picked_by") == "fallback"]
    parse_ok = [r for r in picks if r.get("router_parse_ok")]
    tool_counts: dict[str, int] = {}
    for r in picks:
        for name in r.get("router_tool_calls") or [r.get("tool")]:
            if name:
                tool_counts[str(name)] = tool_counts.get(str(name), 0) + 1
    n = len(picks)
    return {
        "n_tool_picks": n,
        "n_vlm": len(vlm),
        "n_fallback": len(fallback),
        "parse_ok_rate": (len(parse_ok) / n) if n else None,
        "vlm_rate": (len(vlm) / n) if n else None,
        "tool_counts": tool_counts,
    }


def tune_from_traces(
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    sweep = sweep_verify_thresholds(traces)
    best = best_threshold(sweep)
    t = float(best["threshold"]) if best else 0.28
    budget = sweep_budgets(traces, t)
    knee = budget_knee(budget)
    return {
        "best_threshold": best,
        "threshold_sweep": sweep,
        "budget_sweep": budget,
        "budget_knee": knee,
        "nav_report": nav_distance_report(traces),
        "router_report": router_report(traces),
        "n_verify_rows": len(verify_rows(traces)),
        "n_summary_rows": len(summary_rows(traces)),
    }
