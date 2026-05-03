# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.

"""TTY helpers for ``EMET_AGENT_MODEL_DEBUG=1`` / ``--debug-models`` and VRAM snapshots (``EMET_VRAM_DEBUG=1``)."""

from __future__ import annotations

from typing import Any

from termcolor import colored

from emet.agent.env_flags import env_agent_model_debug
from emet.llms.base import AbstractVLLMClient
from emet.utils.vram_debug import (
    client_quantization_hint,
    format_vram_snapshot,
    vram_debug_enabled,
)


def hf_or_model_id_from_client(client: Any) -> str | None:
    for attr in ("hf_model_id", "_resolved_hf_model_id"):
        v = getattr(client, attr, None)
        if v:
            return str(v)
    m = getattr(client, "model", None)
    if m is not None and isinstance(m, str) and m:
        return m
    pipe = getattr(client, "pipe", None)
    if pipe is not None:
        mod = getattr(pipe, "model", None)
        if mod is not None:
            cfg = getattr(mod, "config", None)
            p = getattr(cfg, "name_or_path", None) if cfg is not None else None
            if p:
                return str(p)
            n = getattr(mod, "name", None)
            if n and isinstance(n, str):
                return str(n)
    return None


def format_agent_llm_one_liner(client: Any) -> str:
    parts: list[str] = [type(client).__name__]
    wid = hf_or_model_id_from_client(client)
    if wid:
        parts.append(f"weights={wid!r}")
    qh = client_quantization_hint(client)
    if qh:
        parts.append(f"quant={qh!r}")
    if isinstance(client, AbstractVLLMClient):
        parts.append(f"vl_key={client.canonical_model_key!r}")
    return " ".join(parts)


def _fmt_optional_eqa_client(c: Any) -> str:
    if c is None:
        return "None"
    s = type(c).__name__
    if isinstance(c, AbstractVLLMClient):
        q = client_quantization_hint(c)
        qpart = f" quant={q!r}" if q else ""
        return f"{s}({c.canonical_model_key!r}{qpart})"
    w = hf_or_model_id_from_client(c)
    if w:
        return f"{s}({w!r})"
    return s


def _dynmem_vlm_dedup_line(llm_client: Any, vm: Any) -> str:
    if vm is None:
        return "DynaMem VLM: (no voxel map)"
    pending = getattr(vm, "_eqa_pending", None)
    if pending is not None:
        return "DynaMem VLM: still deferred (EQA init pending — unusual at report time)"
    idc = getattr(vm, "image_description_client", None)
    eqa = getattr(vm, "eqa_client", None)
    if idc is None and eqa is None:
        return "DynaMem VLM: none (EQA off or no local caption client)"
    if isinstance(llm_client, AbstractVLLMClient) and idc is llm_client:
        return "DynaMem VLM: **shared** same object as agent chat client (caption + EQA paths)"
    if idc is not None and eqa is not None and idc is eqa:
        return "DynaMem VLM: one shared client for image_description + eqa (separate from agent --llm)"
    return "DynaMem VLM: image_description and eqa clients (see lines below; not shared with agent --llm)"


def print_offline_model_line(llm_key: str, client: Any, device: str, max_tokens: int) -> None:
    if not env_agent_model_debug():
        return
    print(
        colored(
            "[model debug] offline chat — "
            f"--llm={llm_key!r} device={device!r} max_tokens={max_tokens} | "
            f"client: {format_agent_llm_one_liner(client)}",
            "cyan",
        ),
        flush=True,
    )


def print_embodied_model_report(
    llm_key: str,
    llm_client: Any,
    device: str,
    max_tokens: int,
    executor: Any,
    vl_include_camera: bool,
    openai_tool_schemas: bool,
) -> None:
    """After the agent LLM (and optional shared EQA VLM) is ready, print a one-time stack + VRAM."""
    if not vram_debug_enabled():
        return
    ag = getattr(executor, "agent", None)
    vm = None
    if ag is not None and hasattr(ag, "get_voxel_map"):
        try:
            vm = ag.get_voxel_map()
        except Exception:
            vm = None
    dm = getattr(ag, "detection_model", None) if ag is not None else None

    idc = getattr(vm, "image_description_client", None) if vm is not None else None
    eqa = getattr(vm, "eqa_client", None) if vm is not None else None

    print(colored("=" * 60, "cyan"), flush=True)
    print(
        colored(
            "[model debug] set EMET_VRAM_DEBUG=0 and EMET_AGENT_MODEL_DEBUG=0 to hide VRAM; "
            "drop --debug-models / --debug-vram",
            "cyan",
        ),
        flush=True,
    )
    print(
        colored(
            f"[model debug] --llm / registry key: {llm_key!r}   device: {device!r}   max_tokens: {max_tokens}",
            "cyan",
        ),
        flush=True,
    )
    print(colored(f"[model debug] {_dynmem_vlm_dedup_line(llm_client, vm)}", "cyan"), flush=True)
    print(
        colored(
            "[model debug] GraphEQA / keyword helpers use ``eqa_vl:`` (shared Qwen3.5-VL process-wide) — "
            "separate from ``eqa:`` DynaMem Qwen3-VL unless you route perception manually.",
            "cyan",
        ),
        flush=True,
    )
    print(
        colored(
            "[model debug] To bind DynaMem captions to the **agent** VL, use ``--llm qwen3-vl-eqa`` "
            "(loads once from ``eqa:`` in your agent YAML) with ``--eqa --share-memory-vllm``.",
            "cyan",
        ),
        flush=True,
    )

    if env_agent_model_debug():
        print(
            colored(
                f"[model debug] Chat / tool-calling client: {format_agent_llm_one_liner(llm_client)}",
                "cyan",
            ),
            flush=True,
        )
        print(
            colored(
                f"[model debug] OpenAI-style tool schemas passed to client: {openai_tool_schemas}",
                "cyan",
            ),
            flush=True,
        )
        print(
            colored(
                f"[model debug] Head camera image to chat on first turn (if supported): {vl_include_camera}",
                "cyan",
            ),
            flush=True,
        )
        print(
            colored(
                "describe_scene: uses the robot's **detector** (YoloE / OWL), not the text chat model. "
                f"Current detector: {type(dm).__name__ if dm is not None else 'None'}",
                "cyan",
            ),
            flush=True,
        )
        print(
            colored(f"[model debug] DynaMem image_description_client: {_fmt_optional_eqa_client(idc)}", "cyan"),
            flush=True,
        )
        print(colored(f"[model debug] DynaMem eqa_client: {_fmt_optional_eqa_client(eqa)}", "cyan"), flush=True)
        print(
            colored(
                "[model debug] Head camera / black PNG: EMET_AGENT_CAMERA_DEBUG=1 or --debug-camera. "
                "Discord black PNG: try EMET_DISCORD_IMAGES_BGR=0.",
                "cyan",
            ),
            flush=True,
        )

    for ln in format_vram_snapshot("embodied_model_report (post LLM + DynaMem VLM bind/materialize)").splitlines():
        print(colored(ln, "magenta"), flush=True)
    print(colored("=" * 60, "cyan"), flush=True)


def print_llm_invoke_line(
    llm_client: Any,
    has_tools: bool,
    has_image: bool,
) -> None:
    if not env_agent_model_debug():
        return
    print(
        colored(
            "[model debug] llm_call — "
            f"{format_agent_llm_one_liner(llm_client)} | tools_param={has_tools} image={has_image}",
            "cyan",
        ),
        flush=True,
    )
