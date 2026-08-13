# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build ``env KEY=VAL …`` parts for ``emet hmeqa h2h`` / resume job launches."""

from __future__ import annotations

import shlex

from emet.llms.remote_ops import DEFAULT_LLM_PORT, DEFAULT_VL_PORT, openai_base_for_host


def normalize_hmeqa_vl_endpoint(raw: str) -> str:
    """Return ``openai@http://…/v1`` (or pass through an already-prefixed spec)."""
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower().startswith("openai@"):
        return s
    # Bare URL or host:port → openai@base
    if "://" in s:
        base = s.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"openai@{base}"
    # host or host:port
    if ":" in s and not s.startswith("["):
        host, _, port_s = s.partition(":")
        try:
            port = int(port_s)
        except ValueError:
            port = DEFAULT_VL_PORT
        return f"openai@{openai_base_for_host(host, port)}"
    return f"openai@{openai_base_for_host(s, DEFAULT_VL_PORT)}"


def hmeqa_h2h_env_parts(
    *,
    arms: str,
    ids: str,
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    resume: bool = False,
    eqa_hf_model_id: str | None = None,
    eqa_vl_family: str | None = None,
    eqa_answer_max_new_tokens: int | None = None,
    host: str | None = None,
    vl_endpoint: str | None = None,
    vl_port: int | None = None,
    llm_port: int = DEFAULT_LLM_PORT,
) -> list[str]:
    """Explicit env assignments injected into the jobs-wrapped H2H script.

    Parent-shell exports are **not** inherited by the Habitat child unless listed here.
    """
    parts = [
        "EMET_ALLOW_SDPA_ATTN=1",
        "EMET_EQA_TRACE=1",
        f"ARMS={arms}",
        f"HOLDOUT_IDS={ids}",
        f"COVERAGE_QIDS={coverage_qids}",
        f"EPISODE_COOLDOWN_SEC={int(cooldown)}",
        f"NATIVE_CRASH_POLICY={crash_policy}",
        f"NATIVE_CRASH_STREAK_ABORT={int(streak_abort)}",
        f"EMET_EQA_AGENTIC_VERIFIER={agentic_verifier}",
        f"EMET_EQA_AGENTIC_REQUIRE_VERIFIED={int(require_verified)}",
        f"EMET_EQA_AGENTIC_ROUTER={int(agentic_router)}",
    ]
    # Do NOT force EMET_EQA_ROOM_STAMP_INVESTIGATE / ATTEMPT_LEDGER here.
    # Investigate stamps previously regressed HM-EQA letter accuracy; keep them
    # opt-in for graph-room-evidence A/Bs (see docs/experiments/graph_room_evidence.md).
    host_s = (host or "").strip()
    ep_s = (vl_endpoint or "").strip()
    if ep_s:
        parts.append(f"EMET_VL_ENDPOINT={shlex.quote(normalize_hmeqa_vl_endpoint(ep_s))}")
    if host_s:
        text_base = openai_base_for_host(host_s, llm_port)
        vl_p = int(vl_port) if vl_port is not None else DEFAULT_VL_PORT
        vl_base = openai_base_for_host(host_s, vl_p)
        parts.append(f"EMET_LLM_HOST={shlex.quote(host_s)}")
        parts.append(f"EMET_OPENAI_BASE_URL={shlex.quote(text_base)}")
        if not ep_s:
            parts.append(f"EMET_VL_ENDPOINT={shlex.quote(f'openai@{vl_base}')}")
    # Remote VL: do not force a local HF id (avoids implying a workstation weight load).
    remote_vl = bool(ep_s or host_s)
    if eqa_hf_model_id and not remote_vl:
        parts.append(f"EQA_HF_MODEL_ID={shlex.quote(eqa_hf_model_id)}")
    if eqa_vl_family:
        parts.append(f"EQA_VL_FAMILY={shlex.quote(eqa_vl_family)}")
    if eqa_answer_max_new_tokens is not None:
        parts.append(f"EMET_EQA_ANSWER_MAX_NEW_TOKENS={int(eqa_answer_max_new_tokens)}")
    if resume:
        parts.append("RESUME=1")
    return parts


def hmeqa_h2h_vl_endpoint_from_env_parts(parts: list[str]) -> str | None:
    """Return the ``EMET_VL_ENDPOINT`` value from env parts, if present."""
    for p in parts:
        if p.startswith("EMET_VL_ENDPOINT="):
            return p.split("=", 1)[1].strip("'\"")
    return None
