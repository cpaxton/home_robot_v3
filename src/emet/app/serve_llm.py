# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI entry: ``python -m emet.app.serve_llm`` / ``emet serve llm``."""

from __future__ import annotations

import click

from emet.llms.openai_server import (
    DEFAULT_LLM_SERVE_HOST,
    DEFAULT_LLM_SERVE_MODEL,
    DEFAULT_LLM_SERVE_PORT,
    DEFAULT_VL_SERVE_MODEL,
    DEFAULT_VL_SERVE_PORT,
    resolve_serve_device,
    serve_openai_llm,
)


@click.command(
    "serve-llm",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="OpenAI-compatible LLM HTTP server (remote Qwen / emet llms)",
)
@click.option(
    "--llm",
    "llm_key",
    default=None,
    help=(
        "emet llm/vlm key. Text default: qwen25-7B. With --vl default: qwen3-vl-eqa "
        "(loads via create_dynamem_vllm / eqa config)."
    ),
)
@click.option("--host", default=DEFAULT_LLM_SERVE_HOST, show_default=True, help="Bind address (0.0.0.0 for LAN).")
@click.option(
    "--port",
    default=None,
    type=int,
    help=f"HTTP port (text default {DEFAULT_LLM_SERVE_PORT}; with --vl default {DEFAULT_VL_SERVE_PORT}).",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="auto | cuda | cpu | mps. Jetson emet venv is often CPU-only until Tegra torch is wired.",
)
@click.option("--max-tokens", default=512, show_default=True, type=int, help="Default max_new_tokens.")
@click.option(
    "--api-key",
    default=None,
    help="Optional Bearer token (or EMET_LLM_SERVE_API_KEY). Empty = open LAN endpoint.",
)
@click.option(
    "--vl/--no-vl",
    "multimodal",
    default=False,
    show_default=True,
    help="Load a multimodal VLM (accepts image_url data URLs). Prefer --port 8001 beside text :8000.",
)
@click.option(
    "--multimodal/--no-multimodal",
    "multimodal_alias",
    default=None,
    hidden=True,
    help="Alias for --vl.",
)
def main(
    llm_key: str | None,
    host: str,
    port: int | None,
    device: str,
    max_tokens: int,
    api_key: str | None,
    multimodal: bool,
    multimodal_alias: bool | None,
) -> None:
    """Serve ``/v1/chat/completions`` backed by ``emet.llms.get_llm_client``.

    Text on this host (e.g. caliban)::

        emet serve llm --llm qwen25-7B --host 0.0.0.0 --port 8000

    Multimodal VL (second process / container, port 8001)::

        emet serve llm --vl --host 0.0.0.0 --port 8001

    On a workstation::

        export EMET_OPENAI_BASE_URL=http://caliban:8000/v1
        # caption/EQA: mapping.eqa.vl_endpoint: openai@http://caliban:8001/v1
        emet run agent --llm openai
    """
    use_vl = multimodal if multimodal_alias is None else bool(multimodal_alias)
    resolved_llm = llm_key or (DEFAULT_VL_SERVE_MODEL if use_vl else DEFAULT_LLM_SERVE_MODEL)
    resolved_port = int(port) if port is not None else (DEFAULT_VL_SERVE_PORT if use_vl else DEFAULT_LLM_SERVE_PORT)
    resolved = resolve_serve_device(device)
    click.echo(
        f"emet serve llm: llm={resolved_llm} device={resolved} bind={host}:{resolved_port} vl={use_vl}"
    )
    serve_openai_llm(
        llm=resolved_llm,
        host=host,
        port=resolved_port,
        device=resolved,
        max_tokens=max_tokens,
        api_key=api_key,
        multimodal=use_vl,
    )


if __name__ == "__main__":
    main()
