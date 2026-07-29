# OpenAI-compatible LLM serve (`emet serve llm`)

Serve a local Hugging Face text model (typically **Qwen**) behind a minimal OpenAI Chat Completions API so another machine can call this host over the LAN.

## On the server host (e.g. caliban / Jetson Orin)

```bash
# After jetson install (or a normal uv sync with transformers + torch):
emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
# aliases:
#   uv run python -m emet.app.serve_llm --llm qwen25-14B --host 0.0.0.0 --port 8000
```

| Flag | Default | Notes |
|------|---------|--------|
| `--llm` | `qwen25-7B` | Any `get_llm_client` text key (`qwen25-14B`, `qwen25-32B`, `qwen35-9B`, …). Prefer **no** `-Int4` on Jetson aarch64 (no bitsandbytes). |
| `--host` | `0.0.0.0` | Bind all interfaces for LAN access |
| `--port` | `8000` | OpenAI base URL is `http://<host>:8000/v1` |
| `--device` | `auto` | `cuda` if `torch.cuda.is_available()`, else `cpu` |
| `--max-tokens` | `512` | Default generation length |
| `--api-key` | unset | Optional Bearer token (`EMET_LLM_SERVE_API_KEY`) |

Health check:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

Chat:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen25-14B","messages":[{"role":"user","content":"Say hi in one word."}]}'
```

### How large on Orin?

- emet’s Jetson venv is often **CPU-only** PyTorch; generation is slow but **RAM** (~61 GiB on AGX Orin 64GB) is the usual limit.
- CPU loads use **fp16** for Qwen2.5 text clients so larger checkpoints fit.
- Practical order to try: `qwen25-7B` → `qwen25-14B` → `qwen25-32B` (watch free RAM / swap).
- Tegra CUDA for this venv is future work (JP5 wheels are Python 3.8); until then, treat this as a **network text LLM** for workstations, not a fast edge VL server.

## On the workstation (client)

```bash
export EMET_OPENAI_BASE_URL=http://caliban:8000/v1   # or the Orin IP
export EMET_OPENAI_MODEL=qwen25-14B                  # optional; should match server --llm
# optional if the server set --api-key:
# export OPENAI_API_KEY=...

# Any path that uses --llm openai / OpenaiClient:
emet run agent --llm openai ...

# Or encode the URL in the llm key:
emet run agent --llm 'openai@http://caliban:8000/v1#qwen25-14B' ...
```

Env vars (also listed in [environment_variables.md](environment_variables.md)):

| Variable | Role |
|----------|------|
| `EMET_OPENAI_BASE_URL` | OpenAI SDK `base_url` (include `/v1`) |
| `OPENAI_BASE_URL` | Fallback if `EMET_OPENAI_BASE_URL` unset |
| `EMET_OPENAI_MODEL` | Model id sent in chat completions |
| `EMET_LLM_SERVE_API_KEY` | Server auth + client key fallback |
| `EMET_LLM_SERVE_DEVICE` | Default device for `emet serve llm` |

## Code map

| Piece | Path |
|-------|------|
| HTTP server | [`src/emet/llms/openai_server.py`](../src/emet/llms/openai_server.py) |
| Remote client | [`src/emet/llms/openai_client.py`](../src/emet/llms/openai_client.py) (`base_url`) |
| CLI | `emet serve llm` → [`src/emet/cli.py`](../src/emet/cli.py) / [`src/emet/app/serve_llm.py`](../src/emet/app/serve_llm.py) |
| Model load | [`get_llm_client`](../src/emet/llms/__init__.py) (same keys as `--llm` elsewhere) |

## Jetson aarch64 note

If imports fail with ``cannot allocate memory in static TLS block`` (sklearn/libgomp), start with::

    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
    emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000

``emet serve llm`` also re-execs once with that preload when needed.
