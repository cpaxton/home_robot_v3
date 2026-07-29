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
    default=DEFAULT_LLM_SERVE_MODEL,
    show_default=True,
    help="emet llm key (e.g. qwen25-7B, qwen25-14B, qwen35-9B). Use the largest that fits RAM.",
)
@click.option("--host", default=DEFAULT_LLM_SERVE_HOST, show_default=True, help="Bind address (0.0.0.0 for LAN).")
@click.option("--port", default=DEFAULT_LLM_SERVE_PORT, show_default=True, type=int, help="HTTP port.")
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
def main(llm_key: str, host: str, port: int, device: str, max_tokens: int, api_key: str | None) -> None:
    """Serve ``/v1/chat/completions`` backed by ``emet.llms.get_llm_client``.

    On this host (e.g. caliban)::

        emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000

    On a workstation::

        export EMET_OPENAI_BASE_URL=http://caliban:8000/v1
        # optional model id must match server --llm key (or any string the server ignores)
        emet run agent --llm openai
    """
    resolved = resolve_serve_device(device)
    click.echo(f"emet serve llm: llm={llm_key} device={resolved} bind={host}:{port}")
    serve_openai_llm(
        llm=llm_key,
        host=host,
        port=port,
        device=resolved,
        max_tokens=max_tokens,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
