# OpenAI-compatible LLM serve (`emet serve llm` / Jetson container)

Serve a local Hugging Face text model (typically **Qwen**) behind a minimal OpenAI Chat Completions API so another machine can call this host over the LAN.

## Jetson AGX Orin (recommended): Tegra-CUDA container

Host emet venv on JetPack 5 is often **CPU-only** PyTorch. Use the dustynv L4T image (NVIDIA Jetson torch + CUDA):

```bash
# on the Orin (needs nvidia-container-toolkit; already present on this JP5.1.2 kit)
./scripts/run_jetson_llm_container.sh --build --detach --model Qwen/Qwen2.5-7B-Instruct
# smoke / low disk: --model Qwen/Qwen2.5-0.5B-Instruct
```

| Piece | Path |
|-------|------|
| Dockerfile | [`docker/Dockerfile.jetson-llm`](../docker/Dockerfile.jetson-llm) (`dustynv/l4t-pytorch:r35.4.1`) |
| Server | [`docker/jetson_llm_server.py`](../docker/jetson_llm_server.py) (Python 3.8–friendly) |
| Runner | [`scripts/run_jetson_llm_container.sh`](../scripts/run_jetson_llm_container.sh) |

Weights cache defaults to `~/hf-cache` (`HF_HOME` / `--hf-cache`; mounted at `/data/huggingface`). Use `--detach` for background serve; `docker logs -f emet-jetson-llm` to watch load.

Smoke:

```bash
curl -s http://127.0.0.1:8000/health
# expect "cuda": true, "ready": true
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

## Native host (CPU fallback)

Prefer the Docker path on JP5 for GPU speed. Native `emet serve llm` uses the host venv (often CPU-only torch on this Orin):

```bash
emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
```

| Flag | Default | Notes |
|------|---------|--------|
| `--llm` | `qwen25-7B` | Any `get_llm_client` text key (`qwen25-14B`, …). Prefer **no** `-Int4` on Jetson aarch64 (no bitsandbytes). |
| `--host` | `0.0.0.0` | Bind all interfaces for LAN access |
| `--port` | `8000` | OpenAI base URL is `http://<host>:8000/v1` |
| `--device` | `auto` | `cuda` if `torch.cuda.is_available()`, else `cpu` |
| `--max-tokens` | `512` | Default generation length |
| `--api-key` | unset | Optional Bearer token (`EMET_LLM_SERVE_API_KEY`) |

If imports fail with `cannot allocate memory in static TLS block` (sklearn/libgomp):

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
```

## On the workstation (client)

```bash
export EMET_OPENAI_BASE_URL=http://caliban:8000/v1   # or the Orin IP
export EMET_OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct    # match container --model
emet run agent --llm openai ...
# or: --llm 'openai@http://caliban:8000/v1#Qwen/Qwen2.5-7B-Instruct'
```

| Variable | Role |
|----------|------|
| `EMET_OPENAI_BASE_URL` | OpenAI SDK `base_url` (include `/v1`) |
| `OPENAI_BASE_URL` | Fallback if `EMET_OPENAI_BASE_URL` unset |
| `EMET_OPENAI_MODEL` | Model id sent to the remote server |
| `EMET_JETSON_LLM_IMAGE` | Docker image tag (default `emet-jetson-llm:r35.4.1`) |
| `EMET_LLM_SERVE_API_KEY` | Optional Bearer token (server + client) |
| `EMET_LLM_SERVE_DEVICE` | Default device for native `emet serve llm` |

## Code map

| Piece | Path |
|-------|------|
| Jetson container server | [`docker/jetson_llm_server.py`](../docker/jetson_llm_server.py) |
| Native HTTP server | [`src/emet/llms/openai_server.py`](../src/emet/llms/openai_server.py) |
| Remote client | [`src/emet/llms/openai_client.py`](../src/emet/llms/openai_client.py) |
| Native CLI | `emet serve llm` → [`src/emet/app/serve_llm.py`](../src/emet/app/serve_llm.py) |

See also [jetson.md](jetson.md) and [environment_variables.md](environment_variables.md).
