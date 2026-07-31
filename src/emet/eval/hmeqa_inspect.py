# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Inspect HM-EQA episode bundles (trace summary + image/video paths).

Replaces one-off ``python - <<'PY' … json.loads(agentic_trace)`` loops::

  uv run emet hmeqa inspect ~/runs/emet/hmeqa_holdout8_fix4_… --qid 105
  uv run emet hmeqa inspect OUT --misses
  uv run emet hmeqa inspect OUT --qid 105 --open frames
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def resolve_episode_bundle(out_dir: Path, qid: int, *, arm: str = "agentic") -> Path | None:
    """Prefer OUT/bundles/{arm}_q{qid}, else Habitat episode cache."""
    out = out_dir.expanduser().resolve()
    local = out / "bundles" / f"{arm}_q{qid}"
    if local.is_dir():
        return local
    cache = Path.home() / ".cache" / "habitat_eqa" / "episodes"
    for cand in (
        cache / f"h2h_{arm}_q{qid:04d}" / f"q{qid:04d}_dynagraph",
        cache / f"h2h_{arm}_q{qid:04d}" / f"q{qid:04d}_graph_eqa",
    ):
        if cand.is_dir():
            return cand
    matches = sorted(cache.glob(f"*q{qid:04d}*/**/agentic_trace.jsonl")) if cache.is_dir() else []
    return matches[0].parent if matches else None


def load_episode_row(out_dir: Path, qid: int, *, arm: str = "agentic") -> dict[str, Any] | None:
    path = out_dir.expanduser().resolve() / f"{arm}_q{qid}.jsonl"
    rows = _load_jsonl(path)
    return rows[0] if rows else None


def summarize_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tools: Counter[str] = Counter()
    assesses: list[dict[str, Any]] = []
    explores: list[dict[str, Any]] = []
    submits: list[dict[str, Any]] = []
    for row in rows:
        tool = str(row.get("tool") or row.get("event") or "")
        if tool:
            tools[tool] += 1
        if row.get("tool") == "vlm_assess":
            assesses.append(row)
        elif row.get("tool") == "explore_frontier":
            explores.append(row)
        elif row.get("tool") == "submit_answer":
            submits.append(row)
    explore_sources = Counter(str(e.get("source") or "?") for e in explores)
    present_any = any(bool(a.get("present")) for a in assesses)
    answerable_any = any(bool(a.get("answerable")) for a in assesses)
    summary = rows[-1] if rows and rows[-1].get("tool") == "summary" else {}
    return {
        "n_rows": len(rows),
        "tools": dict(tools),
        "n_assess": len(assesses),
        "n_explore": len(explores),
        "explore_sources": dict(explore_sources),
        "present_any": present_any,
        "answerable_any": answerable_any,
        "assesses": [
            {
                "obs_id": a.get("obs_id"),
                "present": a.get("present"),
                "answerable": a.get("answerable"),
                "suggested_answer": a.get("suggested_answer"),
                "proposal_status": a.get("proposal_status"),
                "reason": (str(a.get("reason") or ""))[:160],
            }
            for a in assesses
        ],
        "last_submit": (
            {
                "final_answer": submits[-1].get("final_answer"),
                "answer_source": submits[-1].get("answer_source"),
                "vlm_suggested": submits[-1].get("vlm_suggested"),
                "verified": submits[-1].get("verified"),
                "confidence": submits[-1].get("confidence"),
            }
            if submits
            else None
        ),
        "budget_hit": summary.get("budget_hit"),
        "n_nav": summary.get("n_nav"),
        "n_explore_budget": summary.get("n_explore"),
        "n_rounds": summary.get("n_rounds"),
    }


def _count_glob(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def media_paths(bundle: Path) -> dict[str, Any]:
    """Collect human-viewable media under an episode bundle."""
    frames = bundle / "frames"
    frames_all = bundle / "frames_all"
    images = bundle / "images"
    # Bundles often symlink frames_all → Habitat cache frames/; images/ may live only in cache.
    if not images.is_dir() and frames_all.is_dir():
        sibling_images = frames_all.resolve().parent / "images"
        if sibling_images.is_dir():
            images = sibling_images
    frontier = bundle / "frontier_picks"
    maps = bundle / "maps"
    rgb_mp4 = bundle / "episode_rgb.mp4"
    topdown_mp4 = bundle / "topdown_exploration.mp4"
    candidates: list[tuple[str, Path, int]] = []
    for name, path in (
        ("images", images),
        ("frames_all", frames_all),
        ("frames", frames),
    ):
        n = _count_glob(path, "*.png")
        if n:
            candidates.append((name, path, n))
    candidates.sort(key=lambda t: t[2], reverse=True)
    primary_name, primary_dir, primary_n = candidates[0] if candidates else ("", Path(), 0)
    return {
        "bundle": str(bundle),
        "frames_dir": str(frames) if frames.is_dir() else None,
        "frames_n": _count_glob(frames, "rgb_*.png") or _count_glob(frames, "*.png"),
        "frames_all_dir": str(frames_all) if frames_all.is_dir() else None,
        "frames_all_n": _count_glob(frames_all, "*.png"),
        "images_dir": str(images) if images.is_dir() else None,
        "images_n": _count_glob(images, "*.png"),
        "primary_rgb_kind": primary_name or None,
        "primary_rgb_dir": str(primary_dir) if primary_n else None,
        "primary_rgb_n": primary_n,
        "frontier_picks_dir": str(frontier) if frontier.is_dir() else None,
        "frontier_picks_n": _count_glob(frontier, "*.png"),
        "maps_dir": str(maps) if maps.is_dir() else None,
        "maps_n": _count_glob(maps, "*.png"),
        "episode_rgb_mp4": str(rgb_mp4) if rgb_mp4.is_file() else None,
        "topdown_exploration_mp4": str(topdown_mp4) if topdown_mp4.is_file() else None,
        "topdown_map": str(bundle / "topdown_map.png") if (bundle / "topdown_map.png").is_file() else None,
        "topdown_rooms": str(bundle / "topdown_rooms.png") if (bundle / "topdown_rooms.png").is_file() else None,
        "trace": str(bundle / "agentic_trace.jsonl") if (bundle / "agentic_trace.jsonl").is_file() else None,
    }


def viewer_commands(media: dict[str, Any]) -> list[str]:
    """Shell commands to browse episode media (feh / mpv / eog)."""
    cmds: list[str] = []
    primary = media.get("primary_rgb_dir")
    if primary:
        cmds.append(f"feh -g 1280x720 -. {primary}/*.png")
    frames = media.get("frames_dir")
    if frames and frames != primary:
        cmds.append(f"feh -g 1280x720 -. {frames}/rgb_*.png")
    images = media.get("images_dir")
    if images and images != primary:
        cmds.append(f"feh -g 1280x720 -. {images}/*.png")
    frontier = media.get("frontier_picks_dir")
    if frontier and int(media.get("frontier_picks_n") or 0) > 0:
        cmds.append(f"feh -g 1280x720 -. {frontier}/iter_*.png")
    maps = media.get("maps_dir")
    if maps and int(media.get("maps_n") or 0) > 0:
        cmds.append(f"feh -g 1280x720 -. {maps}/step_*.png")
    for key in ("episode_rgb_mp4", "topdown_exploration_mp4"):
        path = media.get(key)
        if path:
            player = "mpv" if shutil.which("mpv") else ("vlc" if shutil.which("vlc") else None)
            if player:
                cmds.append(f"{player} {path}")
            else:
                cmds.append(f"# video: {path}")
    return cmds


def inspect_episode(
    out_dir: Path | str,
    qid: int,
    *,
    arm: str = "agentic",
) -> dict[str, Any]:
    out = Path(out_dir).expanduser().resolve()
    row = load_episode_row(out, qid, arm=arm)
    bundle = resolve_episode_bundle(out, qid, arm=arm)
    if bundle is None and row and row.get("debug_bundle_dir"):
        cand = Path(str(row["debug_bundle_dir"])).expanduser()
        if cand.is_dir():
            bundle = cand
    trace_path = (bundle / "agentic_trace.jsonl") if bundle else None
    if trace_path is None or not trace_path.is_file():
        local = out / "bundles" / f"{arm}_q{qid}" / "agentic_trace.jsonl"
        if local.is_file():
            trace_path = local
            bundle = local.parent
    rows = _load_jsonl(trace_path) if trace_path else []
    media = media_paths(bundle) if bundle else {}
    return {
        "out_dir": str(out),
        "qid": qid,
        "arm": arm,
        "bundle": str(bundle) if bundle else None,
        "episode": (
            {
                "question": row.get("question"),
                "predicted": row.get("predicted_answer") or row.get("parsed_answer_letter"),
                "gold": row.get("gold_answer_letter"),
                "correct": row.get("correct"),
                "confident": row.get("confident"),
                "planning_steps": row.get("planning_steps"),
                "observations": row.get("observations"),
                "graph_nodes": row.get("graph_nodes"),
                "top_labels": (row.get("graph_health") or {}).get("top_labels"),
            }
            if row
            else None
        ),
        "trace": summarize_trace(rows) if rows else None,
        "media": media,
        "view": viewer_commands(media) if media else [],
    }


def list_scored_episodes(out_dir: Path | str, *, arm: str = "agentic") -> list[dict[str, Any]]:
    out = Path(out_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(out.glob(f"{arm}_q*.jsonl")):
        data = _load_jsonl(path)
        if not data:
            continue
        d = data[0]
        rows.append(
            {
                "qid": d.get("question_id"),
                "predicted": d.get("predicted_answer") or d.get("parsed_answer_letter"),
                "gold": d.get("gold_answer_letter"),
                "correct": bool(d.get("correct")),
                "question": d.get("question"),
            }
        )
    return rows


def format_inspect_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    ep = payload.get("episode") or {}
    lines.append(f"OUT={payload.get('out_dir')}  q{payload.get('qid')}  arm={payload.get('arm')}")
    if ep:
        mark = "OK" if ep.get("correct") else "MISS"
        lines.append(
            f"score: pred={ep.get('predicted')} gold={ep.get('gold')} {mark} "
            f"steps={ep.get('planning_steps')} obs={ep.get('observations')}"
        )
        if ep.get("question"):
            lines.append(f"Q: {ep['question']}")
        top = ep.get("top_labels") or []
        if top:
            labs = ", ".join(f"{t.get('label')}×{t.get('count')}" if isinstance(t, dict) else str(t) for t in top[:8])
            lines.append(f"graph top_labels: {labs}")
    tr = payload.get("trace")
    if tr:
        lines.append(
            f"trace: assess={tr.get('n_assess')} explore={tr.get('n_explore')} "
            f"present_any={tr.get('present_any')} answerable_any={tr.get('answerable_any')} "
            f"budget_hit={tr.get('budget_hit')} sources={tr.get('explore_sources')}"
        )
        for a in tr.get("assesses") or []:
            lines.append(
                f"  assess obs={a.get('obs_id')} present={a.get('present')} "
                f"ans={a.get('answerable')} sug={a.get('suggested_answer')} "
                f"prop={a.get('proposal_status')} :: {a.get('reason')}"
            )
        sub = tr.get("last_submit")
        if sub:
            lines.append(
                f"  submit final={sub.get('final_answer')!r} src={sub.get('answer_source')} "
                f"vlm_sug={sub.get('vlm_suggested')} verified={sub.get('verified')}"
            )
    media = payload.get("media") or {}
    if media.get("bundle"):
        lines.append(f"bundle: {media['bundle']}")
        lines.append(
            f"media: images={media.get('images_n')} frames={media.get('frames_n')} "
            f"frontier={media.get('frontier_picks_n')} maps={media.get('maps_n')} "
            f"rgb_mp4={'yes' if media.get('episode_rgb_mp4') else 'no'}"
        )
    view = payload.get("view") or []
    if view:
        lines.append("view (copy-paste):")
        for cmd in view:
            lines.append(f"  {cmd}")
    return "\n".join(lines)


def open_media(kind: str, media: dict[str, Any]) -> int:
    """Launch a viewer for ``frames|images|frontier|maps|video``. Returns PID."""
    kind = (kind or "").strip().lower()
    mapping = {
        "frames": ("frames_dir", "*.png"),
        "images": ("images_dir", "*.png"),
        "rgb": ("primary_rgb_dir", "*.png"),
        "frontier": ("frontier_picks_dir", "iter_*.png"),
        "maps": ("maps_dir", "step_*.png"),
        "video": ("episode_rgb_mp4", None),
    }
    if kind not in mapping:
        raise ValueError(f"unknown open kind {kind!r}; expected one of {sorted(mapping)}")
    key, glob = mapping[kind]
    path = media.get(key)
    if not path:
        if kind in ("frames", "images", "rgb"):
            path = media.get("primary_rgb_dir") or media.get("images_dir") or media.get("frames_dir")
            glob = "*.png"
        elif kind == "video":
            path = media.get("episode_rgb_mp4")
    if not path:
        raise FileNotFoundError(f"no media for kind={kind}")
    if kind == "video":
        player = "mpv" if shutil.which("mpv") else ("vlc" if shutil.which("vlc") else None)
        if not player:
            raise FileNotFoundError(f"no mpv/vlc; open manually: {path}")
        return subprocess.Popen([player, path]).pid  # noqa: S603
    if not shutil.which("feh"):
        raise FileNotFoundError(f"feh not installed; open manually: {path}/{glob}")
    pattern = str(Path(path) / (glob or "*.png"))
    # feh expands globs itself when given as one arg only if shell=True; pass via shell glob.
    return subprocess.Popen(  # noqa: S602
        f"feh -g 1280x720 -. {pattern}",
        shell=True,
    ).pid
