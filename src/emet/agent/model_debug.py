# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the
# root directory of this source tree.

"""TTY helpers for ``EMET_AGENT_MODEL_DEBUG=1`` / ``--debug-models`` (which weights handle what)."""

from __future__ import annotations

from typing import Any

from termcolor import colored

from emet.agent.env_flags import env_agent_model_debug
from emet.llms.base import AbstractVLLMClient


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
    if isinstance(client, AbstractVLLMClient):
        parts.append(f"vl_key={client.canonical_model_key!r}")
    return " ".join(parts)


def _fmt_optional_eqa_client(c: Any) -> str:
    if c is None:
        return "None"
    s = type(c).__name__
    if isinstance(c, AbstractVLLMClient):
        return f"{s}({c.canonical_model_key!r})"
    w = hf_or_model_id_from_client(c)
    if w:
        return f"{s}({w!r})"
    return s


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
    """After the agent LLM (and optional shared EQA VLM) is ready, print a one-time stack."""
    if not env_agent_model_debug():
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

    lines = [
        "set EMET_AGENT_MODEL_DEBUG=0 or drop --debug-models to hide this",
        f"--llm / registry key: {llm_key!r}   device: {device!r}   max_tokens: {max_tokens}",
        f"Chat / tool-calling (JSON) client: {format_agent_llm_one_liner(llm_client)}",
        f"OpenAI-style tool schemas passed to client: {openai_tool_schemas}",
        (
            f"Head camera image to chat client on the first turn (if supported): {vl_include_camera} "
            "— still separate from describe_scene; VL models can use the frame for tool choice."
        ),
        (
            "describe_scene: uses the robot's **detector** (YoloE / OWL), not the text chat model. "
            f"Current detector: {type(dm).__name__ if dm is not None else 'None (no object names from vision)'}"
        ),
        f"DynaMem image_description_client: {_fmt_optional_eqa_client(idc)}",
        f"DynaMem eqa_client: {_fmt_optional_eqa_client(eqa)}",
        "Head camera / black PNG: also set EMET_AGENT_CAMERA_DEBUG=1 (or emet run agent --debug-camera) for frame min/max. "
        "If Discord still looks black with good stats, try EMET_DISCORD_IMAGES_BGR=0 (sim may already be RGB).",
    ]
    print(colored("=" * 60, "cyan"), flush=True)
    for line in lines:
        print(colored("[model debug] " + line, "cyan"), flush=True)
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
