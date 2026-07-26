# Copyright (c) Chris Paxton 2026
"""Overnight HM-EQA ladder: holdout-8 → optional retune → bal-32.

Prefer::

    uv run emet hmeqa overnight

Inner phases call ``scripts/run_hmeqa_agentic_h2h.sh`` directly (no nested
``emet jobs``) so a single outer job owns the GPU mutex.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from emet.eval.harness import (
    DEFAULT_BAL32_IDS,
    DEFAULT_HOLDOUT8_IDS,
    evaluate_holdout_gate,
)

_LABEL = "hmeqa-overnight-ladder"


def _repo_root() -> Path:
    # src/emet/eval/hmeqa_overnight.py → repo root
    return Path(__file__).resolve().parent.parent.parent.parent


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _status_paths(base: Path) -> tuple[Path, Path]:
    repo = _repo_root().name
    log_dir = Path.home() / "runs" / "emet" / "status" / repo
    return log_dir, log_dir / "STATUS.log"


def _status_note(
    base: Path,
    state: str,
    what: str,
    next_cmd: str,
    *,
    progress: str = "-",
) -> None:
    """Append a STATUS record matching ``scripts/status_log.sh`` format."""
    log_dir, global_log = _status_paths(base)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        (log_dir / "latest").unlink(missing_ok=True)
        (log_dir / "latest").symlink_to(base)
    except OSError:
        pass
    job = os.environ.get("EMET_JOB_ID", "").strip()
    lines = [
        f"=== {_now_iso()}  {_LABEL}  {state}  {progress}",
        f"    repo: {_repo_root()}",
        f"    out:  {base}",
    ]
    if job:
        lines.append(
            f"    job:  {job}  (cd {_repo_root()} && uv run emet jobs status {job})"
        )
    lines.append(f"    what: {what}")
    lines.append(f"    next: {next_cmd}")
    record = "\n".join(lines) + "\n"
    for path in (global_log, base / "STATUS.log"):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(record)
        except OSError:
            pass


def _load_summary(out: Path) -> dict[str, Any]:
    p = out / "h2h_summary.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _summarize(out: Path) -> None:
    script = _repo_root() / "scripts" / "summarize_hmeqa_agentic_h2h.py"
    if not script.is_file():
        return
    subprocess.call(
        [sys.executable, str(script), str(out)],
        cwd=str(_repo_root()),
    )


def _run_h2h(
    out: Path,
    *,
    ids: str,
    arms: str,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    skip_kill_stale: bool,
    egl_fail_abort: int,
) -> int:
    script = _repo_root() / "scripts" / "run_hmeqa_agentic_h2h.sh"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "EMET_ALLOW_SDPA_ATTN": "1",
            "EMET_EQA_TRACE": "1",
            "EMET_EQA_AGENTIC_VERIFIER": agentic_verifier,
            "EMET_EQA_AGENTIC_REQUIRE_VERIFIED": str(int(require_verified)),
            "EMET_EQA_AGENTIC_ROUTER": str(int(agentic_router)),
            "EPISODE_COOLDOWN_SEC": str(int(cooldown)),
            "NATIVE_CRASH_POLICY": crash_policy,
            "NATIVE_CRASH_STREAK_ABORT": str(int(streak_abort)),
            "EGL_FAIL_ABORT": str(int(egl_fail_abort)),
            "ARMS": arms,
            "HOLDOUT_IDS": ids,
            "SKIP_KILL_STALE": "1" if skip_kill_stale else "0",
            "SKIP_GPU_WAIT": "0",
        }
    )
    return int(
        subprocess.call(
            ["bash", str(script), str(out)],
            cwd=str(_repo_root()),
            env=env,
        )
    )


def _append_gate_log(base: Path, line: str) -> None:
    path = base / "gate.log"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def _write_gate(base: Path, gate: Mapping[str, Any]) -> None:
    path = base / "gate.json"
    path.write_text(json.dumps(dict(gate), indent=2) + "\n", encoding="utf-8")


def run_overnight(
    *,
    base: Path,
    holdout_ids: str = DEFAULT_HOLDOUT8_IDS,
    bal32_ids: str = DEFAULT_BAL32_IDS,
    gate_min_acc: float = 0.25,
    skip_bal32: bool = False,
    agentic_verifier: str = "owlv2",
    require_verified: bool = False,
    agentic_router: bool = True,
    cooldown: int = 20,
    crash_policy: str = "skip",
    streak_abort: int = 2,
    egl_fail_abort: int = 2,
) -> int:
    """Run holdout-8 → optional retune → bal-32. Returns process exit code."""
    base = base.expanduser().resolve()
    hold_out = base / "holdout8"
    bal_out = base / "bal32"
    base.mkdir(parents=True, exist_ok=True)
    hold_out.mkdir(parents=True, exist_ok=True)
    bal_out.mkdir(parents=True, exist_ok=True)

    head = ""
    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(_repo_root()),
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"

    _status_note(
        base,
        "RUNNING",
        f"overnight ladder: holdout-8 then gated bal-32 @ {head}",
        "uv run emet jobs; bash scripts/status_log.sh tail — do not kill-stale while live",
        progress="holdout8 starting",
    )

    h2h_kwargs = {
        "agentic_verifier": agentic_verifier,
        "require_verified": require_verified,
        "agentic_router": agentic_router,
        "cooldown": cooldown,
        "crash_policy": crash_policy,
        "streak_abort": streak_abort,
        "egl_fail_abort": egl_fail_abort,
    }

    _status_note(
        base,
        "RUNNING",
        f"phase1 holdout-8 classic+agentic ids={holdout_ids} out={hold_out}",
        f"uv run emet hmeqa status {hold_out}",
        progress="holdout8 0/16",
    )
    rc1 = _run_h2h(
        hold_out,
        ids=holdout_ids,
        arms="classic,agentic",
        skip_kill_stale=False,
        **h2h_kwargs,
    )
    _summarize(hold_out)

    summary = _load_summary(hold_out)
    gate = evaluate_holdout_gate(summary, min_agentic_acc=gate_min_acc)
    gate["holdout_out"] = str(hold_out)
    _write_gate(base, gate)
    classic = gate.get("classic") or {}
    agentic = gate.get("agentic") or {}
    _append_gate_log(
        base,
        "GATE holdout "
        f"classic=[acc={classic.get('accuracy')} n={classic.get('n')}] "
        f"agentic=[acc={agentic.get('accuracy')} n={agentic.get('n')}] "
        f"need_retune={gate.get('need_retune')} reason={gate.get('reason')}",
    )
    print(json.dumps(gate, indent=2), flush=True)

    if gate.get("need_retune"):
        hold_retune = Path(str(hold_out) + "_retune")
        hold_retune.mkdir(parents=True, exist_ok=True)
        _status_note(
            base,
            "RUNNING",
            "phase1b agentic-only retune on holdout-8 (paper-router)",
            f"uv run emet hmeqa status {hold_retune}",
            progress="holdout8 retune",
        )
        _run_h2h(
            hold_retune,
            ids=holdout_ids,
            arms="agentic",
            skip_kill_stale=True,
            **h2h_kwargs,
        )
        _summarize(hold_retune)
        retune_summary = _load_summary(hold_retune)
        gate["retune_out"] = str(hold_retune)
        gate["retune_agentic"] = retune_summary.get("agentic")
        _write_gate(base, gate)
        ra = gate.get("retune_agentic") or {}
        _append_gate_log(
            base,
            f"RETUNE agentic=[acc={ra.get('accuracy')} n={ra.get('n')}] out={hold_retune}",
        )

    if skip_bal32:
        _status_note(
            base,
            "DONE",
            "holdout complete; SKIP_BAL32=1",
            f"uv run emet hmeqa summarize {hold_out}",
            progress="holdout done",
        )
        return 0 if rc1 == 0 else int(rc1)

    _status_note(
        base,
        "RUNNING",
        f"phase2 bal-32 classic+agentic out={bal_out} ids={bal32_ids}",
        f"uv run emet hmeqa status {bal_out}",
        progress="bal32 starting",
    )
    rc2 = _run_h2h(
        bal_out,
        ids=bal32_ids,
        arms="classic,agentic",
        skip_kill_stale=True,
        **h2h_kwargs,
    )
    _summarize(bal_out)
    bal_summary = _load_summary(bal_out)
    gate["bal32_out"] = str(bal_out)
    gate["bal32_classic"] = bal_summary.get("classic")
    gate["bal32_agentic"] = bal_summary.get("agentic")
    _write_gate(base, gate)
    print(
        json.dumps(
            {
                "bal32_classic": gate.get("bal32_classic"),
                "bal32_agentic": gate.get("bal32_agentic"),
            },
            indent=2,
        ),
        flush=True,
    )

    _status_note(
        base,
        "DONE",
        f"overnight ladder finished holdout→bal32 under {base}",
        (
            f"uv run emet hmeqa summarize {hold_out}; "
            f"uv run emet hmeqa summarize {bal_out}; "
            f"cat {base / 'gate.json'}"
        ),
        progress="done",
    )
    print(f"DONE overnight ladder BASE={base}", flush=True)
    if rc1 != 0:
        return int(rc1)
    return int(rc2)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HM-EQA overnight holdout→bal32 ladder")
    p.add_argument(
        "--base",
        default=os.environ.get("OVERNIGHT_BASE", ""),
        help="Overnight base dir (default: ~/runs/emet/hmeqa_overnight_<stamp>)",
    )
    p.add_argument(
        "--holdout-ids",
        default=os.environ.get("HOLDOUT8_IDS", DEFAULT_HOLDOUT8_IDS),
    )
    p.add_argument(
        "--bal32-ids",
        default=os.environ.get("BAL32_IDS", DEFAULT_BAL32_IDS),
    )
    p.add_argument(
        "--gate-min-acc",
        type=float,
        default=float(os.environ.get("GATE_MIN_AGENTIC_ACC", "0.25")),
    )
    p.add_argument(
        "--skip-bal32",
        action="store_true",
        default=_env_bool("SKIP_BAL32", False),
    )
    p.add_argument(
        "--agentic-verifier",
        choices=["none", "owlv2", "yoloe"],
        default=os.environ.get("EMET_EQA_AGENTIC_VERIFIER", "owlv2"),
    )
    verified = p.add_mutually_exclusive_group()
    verified.add_argument(
        "--require-verified",
        dest="require_verified",
        action="store_true",
        default=_env_bool("EMET_EQA_AGENTIC_REQUIRE_VERIFIED", False),
    )
    verified.add_argument(
        "--allow-unverified",
        dest="require_verified",
        action="store_false",
    )
    router = p.add_mutually_exclusive_group()
    router.add_argument(
        "--agentic-router",
        dest="agentic_router",
        action="store_true",
        default=_env_bool("EMET_EQA_AGENTIC_ROUTER", True),
    )
    router.add_argument(
        "--no-agentic-router",
        dest="agentic_router",
        action="store_false",
    )
    p.add_argument(
        "--cooldown",
        type=int,
        default=int(os.environ.get("EPISODE_COOLDOWN_SEC", "20")),
    )
    p.add_argument(
        "--crash-policy",
        choices=["skip", "abort"],
        default=os.environ.get("NATIVE_CRASH_POLICY", "skip"),
    )
    p.add_argument(
        "--streak-abort",
        type=int,
        default=int(os.environ.get("NATIVE_CRASH_STREAK_ABORT", "2")),
    )
    p.add_argument(
        "--egl-fail-abort",
        type=int,
        default=int(os.environ.get("EGL_FAIL_ABORT", "2")),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    base_s = (args.base or "").strip()
    if not base_s:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path.home() / "runs" / "emet" / f"hmeqa_overnight_{stamp}"
    else:
        base = Path(base_s)

    return run_overnight(
        base=base,
        holdout_ids=args.holdout_ids,
        bal32_ids=args.bal32_ids,
        gate_min_acc=float(args.gate_min_acc),
        skip_bal32=bool(args.skip_bal32),
        agentic_verifier=str(args.agentic_verifier),
        require_verified=bool(args.require_verified),
        agentic_router=bool(args.agentic_router),
        cooldown=int(args.cooldown),
        crash_policy=str(args.crash_policy),
        streak_abort=int(args.streak_abort),
        egl_fail_abort=int(args.egl_fail_abort),
    )


if __name__ == "__main__":
    raise SystemExit(main())
