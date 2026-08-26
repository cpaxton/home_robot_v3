#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tier-0 audit: correlate HM-EQA slice scores with close-look geometry proxies.

Historical lean bundles may not include ``close_map_summary.json`` (added later).
This script reconstructs a **trajectory proxy** from ``trajectory.jsonl`` +
``scene_graph_report.txt``: did the robot ever get within ``r_close`` on-axis of a
relevant graph node before the episode ended?

Usage::

    uv run python scripts/audit_close_map_eqa_slice.py \\
        --jsonl ~/.cache/habitat_eqa/results/countclock_20260825_201933_dynagraph_qwen3_vl.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

_SCENE_LINE = re.compile(
    r"xyz=\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)\s+obs=(\d+)",
    re.IGNORECASE,
)

_CLOCK_KEYS = ("clock", "time", "hour", "o'clock")
_COUNT_KEYS = (
    "how many",
    "number of",
    "stool",
    "lamp",
    "utensil",
    "pillow",
    "mat",
    "bar stool",
    "potted plant",
    "bedside",
)


def _question_keywords(question: str) -> tuple[str, ...]:
    q = (question or "").lower()
    if any(k in q for k in _CLOCK_KEYS):
        return ("clock", "wall clock", "wall mounted clock")
    keys: list[str] = []
    for k in _COUNT_KEYS:
        if k in q:
            keys.append(k.split()[-1] if " " in k else k)
    if "table lamp" in q:
        keys.append("lamp")
    if "bar stool" in q or "stools" in q:
        keys.extend(("stool", "bar"))
    if not keys:
        keys.append("object")
    return tuple(dict.fromkeys(keys))


def _parse_scene_targets(report_path: Path, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    if not report_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _SCENE_LINE.search(line)
        if not m:
            continue
        x, y, _z, obs = m.group(1), m.group(2), m.group(3), int(m.group(4))
        blob = line.lower()
        if not any(k in blob for k in keywords):
            continue
        out.append({"obs_id": obs, "x": float(x), "y": float(y), "line": line.strip()[:120]})
    return out


def _trajectory_proxy(
    traj_path: Path,
    tx: float,
    ty: float,
    *,
    r_close_m: float = 0.55,
    aim_deg: float = 25.0,
) -> dict[str, Any]:
    """Best-effort close-look proxy from base pose only (no depth replay)."""
    if not traj_path.is_file():
        return {"resolved": False, "reason": "no_trajectory", "min_cam_m": None, "aimed_at_best": False}
    aim_cos = math.cos(math.radians(float(aim_deg)))
    min_dist: float | None = None
    best_aimed = False
    resolved = False
    for raw in traj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        pose = row.get("pose_xyt")
        if not isinstance(pose, (list, tuple)) or len(pose) < 3:
            continue
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if min_dist is None or dist < min_dist:
            min_dist = dist
        if dist < 1e-3:
            aimed = True
        else:
            fx, fy = math.cos(theta), math.sin(theta)
            aimed = (dx * fx + dy * fy) / dist >= aim_cos
        if aimed and dist <= r_close_m:
            resolved = True
            best_aimed = True
        elif aimed and dist == min_dist:
            best_aimed = True
    return {
        "resolved": resolved,
        "reason": "resolved" if resolved else "unresolved_proxy",
        "min_cam_m": min_dist,
        "aimed_at_best": best_aimed,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def audit_slice(
    jsonl_path: Path,
    *,
    r_close_m: float = 0.55,
    aim_deg: float = 25.0,
) -> dict[str, Any]:
    rows = _load_jsonl(jsonl_path)
    entries: list[dict[str, Any]] = []
    for row in rows:
        qid = int(row.get("question_id", -1))
        bundle = Path(str(row.get("debug_bundle_dir") or ""))
        question = str(row.get("question") or "")
        keywords = _question_keywords(question)
        report = bundle / "scene_graph_report.txt"
        traj = bundle / "trajectory.jsonl"
        cm_summary_path = bundle / "close_map_summary.json"
        cm_saved = None
        if cm_summary_path.is_file():
            cm_saved = json.loads(cm_summary_path.read_text(encoding="utf-8"))
        targets = _parse_scene_targets(report, keywords)
        proxy = {"resolved": False, "reason": "no_target", "min_cam_m": None, "aimed_at_best": False}
        if targets:
            # Closest graph node to median trajectory start — pick nearest target to any traj point min
            best = None
            for t in targets:
                p = _trajectory_proxy(traj, t["x"], t["y"], r_close_m=r_close_m, aim_deg=aim_deg)
                if best is None or (p.get("min_cam_m") is not None and (
                    best.get("min_cam_m") is None or p["min_cam_m"] < best["min_cam_m"]
                )):
                    best = {**p, "target_obs": t["obs_id"], "target_xy": (t["x"], t["y"])}
            proxy = best or proxy
        correct = bool(row.get("correct"))
        model_conf = bool(row.get("model_confident"))
        scored_conf = bool(row.get("confident"))
        entries.append(
            {
                "question_id": qid,
                "correct": correct,
                "model_confident": model_conf,
                "confident": scored_conf,
                "keywords": keywords,
                "n_targets": len(targets),
                "close_map_saved": cm_saved,
                "proxy": proxy,
                "bundle": str(bundle),
            }
        )

    def _bucket(name: str, pred) -> list[dict[str, Any]]:
        return [e for e in entries if pred(e)]

    confident_wrong = _bucket("confident_wrong", lambda e: e["model_confident"] and not e["correct"])
    confident_right = _bucket("confident_right", lambda e: e["model_confident"] and e["correct"])
    unconf_wrong = _bucket("unconf_wrong", lambda e: not e["model_confident"] and not e["correct"])
    unconf_right = _bucket("unconf_right", lambda e: not e["model_confident"] and e["correct"])

    def _unresolved(es: list[dict[str, Any]]) -> int:
        return sum(1 for e in es if not e["proxy"].get("resolved"))

    return {
        "jsonl": str(jsonl_path),
        "n": len(entries),
        "accuracy": sum(1 for e in entries if e["correct"]) / max(1, len(entries)),
        "confident_wrong": len(confident_wrong),
        "confident_wrong_unresolved_proxy": _unresolved(confident_wrong),
        "confident_right_unresolved_proxy": _unresolved(confident_right),
        "entries": entries,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"jsonl: {report['jsonl']}")
    print(f"n={report['n']} accuracy={report['accuracy']:.1%}")
    print(
        f"confident_wrong={report['confident_wrong']} "
        f"(unresolved_proxy={report['confident_wrong_unresolved_proxy']})"
    )
    print(f"confident_right_unresolved_proxy={report['confident_right_unresolved_proxy']}")
    print()
    print(f"{'qid':>4} {'ok':>3} {'mconf':>5} {'proxy_res':>9} {'min_m':>6}  question")
    for e in sorted(report["entries"], key=lambda x: x["question_id"]):
        p = e["proxy"]
        min_m = p.get("min_cam_m")
        min_s = f"{min_m:.2f}" if isinstance(min_m, (int, float)) else "  n/a"
        res = "yes" if p.get("resolved") else "no"
        q = ""
        for row in _load_jsonl(Path(report["jsonl"])):
            if int(row.get("question_id", -1)) == e["question_id"]:
                q = str(row.get("question") or "")[:56]
                break
        print(
            f"{e['question_id']:4d} {'Y' if e['correct'] else 'N':>3} "
            f"{'Y' if e['model_confident'] else 'N':>5} {res:>9} {min_s:>6}  {q}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit close-look proxy vs HM-EQA slice jsonl")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--r-close-m", type=float, default=0.55)
    ap.add_argument("--aim-deg", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report here")
    args = ap.parse_args()
    report = audit_slice(args.jsonl, r_close_m=args.r_close_m, aim_deg=args.aim_deg)
    _print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # entries only serializable subset for out file
        slim = {k: v for k, v in report.items() if k != "entries"}
        slim["entries"] = report["entries"]
        args.out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
