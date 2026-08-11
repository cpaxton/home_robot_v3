# OpenAI-compatible LLM serve + remote inference

Serve a Hugging Face text or vision-language model behind a minimal OpenAI Chat Completions API so another machine can call this host over the LAN. Clients use `openai@http://HOST:PORT/v1` (or env vars below) — **no hostname is hardcoded** in the CLI; pass `--host` or set `EMET_LLM_HOST`.

Canonical CLI flags: verify with `uv run emet llm --help`, `uv run emet run chat --help`, `uv run emet deploy llm --help`. Env list: [environment_variables.md](environment_variables.md).

---

## 1. Remote inference support

**What remotes today:** text tool-router completions and caption/EQA **JPEG frames** over HTTP. **What stays local:** voxel / Dynagraph memory on the workstation (Mars ZMQ 4401). Phase 2 (remote memory worker) is not implemented — see bottom of this page.

### Profiles (Jetson AGX Orin)

| Profile | Serve | Ports | When to use |
|---------|-------|-------|-------------|
| `unified-7b` (default) | one **Qwen2-VL-7B** | `:8000` for **text + VL** | Fits ~64 GiB Orin unified memory; eMMC can’t hold two 7B trees |
| `dual-2b` | CausalLM text + **Qwen2-VL-2B** | text `:8000`, VL `:8001` | Keep a separate small caption model beside text |

```bash
# From the workstation (rsync weights + start Jetson Docker --vl)
uv run emet deploy llm --host ORIN_HOST --profile unified-7b
# dual-2b:
uv run emet deploy llm --host ORIN_HOST --profile dual-2b
```

Shell equivalent: `./scripts/deploy_caliban_vl.sh --host ORIN_HOST --profile unified-7b`. Native desktop serve (no Jetson):

```bash
uv run emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
uv run emet serve llm --vl --host 0.0.0.0 --port 8001   # caption beside text
```

### Wire clients

| Path | How |
|------|-----|
| One-shot / interactive chat | `emet run chat --host ORIN_HOST` (text) · add `--vl` for multimodal |
| Agent text router | `--llm openai@http://ORIN_HOST:8000/v1` or `EMET_OPENAI_BASE_URL=http://ORIN_HOST:8000/v1` |
| Caption / EQA VLM | `mapping.eqa.vl_endpoint: openai@http://ORIN_HOST:8000/v1` (unified-7b) or `:8001` (dual-2b); override with `EMET_VL_ENDPOINT` |
| HM-EQA H2H answer VL | `emet hmeqa h2h --host ORIN_HOST` (or `--vl-endpoint openai@http://ORIN_HOST:8000/v1`) — injects into the jobs-wrapped env; bare shell export is ignored |
| Herman Discord preset | [`configs/agent_innate_mars.yaml`](../configs/agent_innate_mars.yaml) — `agent.llm: openai`; pass `--host ORIN` (or `EMET_LLM_HOST`) |

```bash
export EMET_LLM_HOST=ORIN_HOST          # optional default for --host-less scripts
export EMET_OPENAI_BASE_URL=http://ORIN_HOST:8000/v1
export EMET_VL_ENDPOINT=openai@http://ORIN_HOST:8000/v1   # unified-7b
# dual-2b VL:
# export EMET_VL_ENDPOINT=openai@http://ORIN_HOST:8001/v1
```

| Variable | Role |
|----------|------|
| `EMET_LLM_HOST` | LAN host for `emet run chat` / `emet llm` / `emet deploy llm` (no default) |
| `EMET_CALIBAN_HOST` | Compat alias if `EMET_LLM_HOST` unset |
| `EMET_OPENAI_BASE_URL` | Text OpenAI SDK `base_url` (include `/v1`) |
| `EMET_OPENAI_MODEL` | Model id label sent to the remote (server uses its loaded weights) |
| `EMET_VL_ENDPOINT` | Override `eqa.vl_endpoint` for `OpenaiVLLMClient` |
| `EMET_LLM_SERVE_API_KEY` | Optional Bearer on server + client |

**Code:** text → [`OpenaiClient`](../src/emet/llms/openai_client.py); VL → [`OpenaiVLLMClient`](../src/emet/llms/openai_vllm_client.py) via `create_dynamem_vllm(..., endpoint=…)`.

---

## 2. Testing LLMs (health, smoke, chat)

Use these from the **workstation** once a serve is up on `ORIN_HOST` (replace with your LAN hostname or IP).

### Health

```bash
uv run emet llm health --host ORIN_HOST
uv run emet llm health --host ORIN_HOST --text-only
uv run emet llm health --host ORIN_HOST --vl-only
# dual-2b VL on :8001:
uv run emet llm health --host ORIN_HOST --vl-port 8001 --vl-only
```

Expect `ready: true` and (for unified-7b) `multimodal: true` on `:8000`.

### Smoke (chat-completions)

```bash
# Text pong + synthetic VL image caption (same :8000 for unified-7b)
uv run emet llm smoke --host ORIN_HOST
uv run emet llm smoke --host ORIN_HOST --text-only
uv run emet llm smoke --host ORIN_HOST --vl-only
uv run emet llm smoke --host ORIN_HOST --vl-only --image /path/to.jpg
```

Shell wrappers (also need a host):

```bash
EMET_LLM_HOST=ORIN_HOST ./scripts/smoke_caliban_llm.sh
EMET_LLM_HOST=ORIN_HOST ./scripts/smoke_caliban_vl.sh
```

### Interactive / one-shot chat

```bash
# Text
uv run emet run chat --host ORIN_HOST --once "Reply with exactly: pong"
uv run emet run chat --host ORIN_HOST   # interactive REPL

# Vision-language (unified-7b: same host :8000)
uv run emet run chat --host ORIN_HOST --vl --once "Describe briefly"
uv run emet run chat --host ORIN_HOST --vl --image shot.jpg --once "What do you see?"

# dual-2b or local VL serve on another port
uv run emet run chat --host ORIN_HOST --vl --vl-port 8001 --once "Describe briefly"
uv run emet run chat --vl --vl-endpoint openai@http://127.0.0.1:8001/v1 --image shot.jpg --once "…"
```

Unit tests (no network): `uv run emet test src/test/llms/test_remote_ops.py src/test/llms/test_openai_vllm_client.py -q`.

---

## Jetson AGX Orin: Tegra-CUDA container

Host emet venv on JetPack 5 is often **CPU-only** PyTorch. Use the dustynv L4T image (NVIDIA Jetson torch + CUDA):

```bash
# on the Orin (needs nvidia-container-toolkit)
./scripts/run_jetson_llm_container.sh --build --detach --model Qwen/Qwen2.5-7B-Instruct
# multimodal / unified-7b style:
./scripts/run_jetson_llm_container.sh --vl --detach --port 8000 --model Qwen/Qwen2-VL-7B-Instruct
# smoke / low disk: --model Qwen/Qwen2.5-0.5B-Instruct
```

| Piece | Path |
|-------|------|
| Dockerfile | [`docker/Dockerfile.jetson-llm`](../docker/Dockerfile.jetson-llm) (`dustynv/l4t-pytorch:r35.4.1`) |
| Server | [`docker/jetson_llm_server.py`](../docker/jetson_llm_server.py) (Python 3.8–friendly) |
| Runner | [`scripts/run_jetson_llm_container.sh`](../scripts/run_jetson_llm_container.sh) |

Weights cache defaults to `~/hf-cache` (`HF_HOME` / `--hf-cache`; mounted at `/data/huggingface`). Use `--detach` for background serve; `docker logs -f emet-jetson-llm` to watch load.

On-device smoke:

```bash
curl -s http://127.0.0.1:8000/health
# expect "cuda": true, "ready": true
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

### Quantization on JP5

Workstation emet uses **bitsandbytes int4** for local Qwen3-VL. That path does **not** work in the Tegra-CUDA Jetson container today:

| Approach | Status on JP5 `emet-jetson-llm` |
|----------|----------------------------------|
| fp16 (current) | Works — Orin ~64 GiB unified memory fits Qwen2-VL-7B with headroom |
| bitsandbytes int4/int8 | Not installed; Tegra torch is not the PyPI CUDA build bnb expects |
| AutoAWQ / `Qwen2-VL-*-AWQ` | `pip install autoawq` pulls CPU `torch` + needs Triton |
| Quanto int8 | Installs but upgrades to CPU torch 2.4 / needs `float8` absent in nv23.05 |

Stay on **fp16** for `emet deploy llm`. Pre-quantized AWQ weights are attractive once a **JP6 / dustynv vLLM** image can load them without replacing Tegra torch. Pass `--quant awq|int4|int8` to `jetson_llm_server.py` only to get that explanation (exits non-zero on JP5).

---

## Native host (CPU fallback / multimodal VLM)

Prefer the Docker path on JP5 for **text** GPU speed. Native `emet serve llm` uses the host venv (often CPU-only torch on Jetson; on a desktop GPU it can also serve a VLM):

```bash
emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
emet serve llm --vl --host 0.0.0.0 --port 8001
```

| Flag | Default | Notes |
|------|---------|--------|
| `--llm` | `qwen25-7B` (text) / `qwen3-vl-eqa` with `--vl` | Text: any `get_llm_client` key. VL: loads via `create_dynamem_vllm` / `eqa:` config. Prefer **no** `-Int4` on Jetson aarch64 text path (no bitsandbytes). |
| `--host` | `0.0.0.0` | Bind all interfaces for LAN access |
| `--port` | `8000` text / `8001` with `--vl` | OpenAI base URL is `http://<host>:<port>/v1` |
| `--vl` / `--multimodal` | off | Accept `image_url` data URLs; route to `generate_multimodal`. Always loads **local** weights (ignores `EMET_VL_ENDPOINT` / `eqa.vl_endpoint` so the serve host is never a proxy loop). |
| `--device` | `auto` | `cuda` if `torch.cuda.is_available()`, else `cpu` |
| `--max-tokens` | `512` | Default generation length |
| `--api-key` | unset | Optional Bearer token (`EMET_LLM_SERVE_API_KEY`) |

If imports fail with `cannot allocate memory in static TLS block` (sklearn/libgomp):

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
```

### Desktop VL while Jetson hosts text only

When the Orin cannot host a second VL process, run VL on the workstation GPU:

```bash
uv run emet jobs run --name vl-serve-8001 --need-mib 12000 -- \
  uv run emet serve llm --vl --host 0.0.0.0 --port 8001
export EMET_VL_ENDPOINT=openai@http://127.0.0.1:8001/v1
uv run emet llm smoke --vl-only --vl http://127.0.0.1:8001/v1
```

---

## End-to-end example (Mars agent + LAN Orin)

Replace `ORIN_HOST` and the Mars connection name with your hosts. Port roles:

| Port | Role | Typical recipe |
|------|------|----------------|
| `:8000` | Text tool-router **or** unified text+VL | `emet-jetson-llm` (CausalLM or Qwen2-VL-7B) |
| `:8001` | Caption / EQA VLM only (dual-2b) | `emet-jetson-vl` (`Qwen2-VL-2B`); unused in unified-7b |

```bash
uv run emet deploy llm --host ORIN_HOST --profile unified-7b
uv run emet llm health --host ORIN_HOST
uv run emet llm smoke --host ORIN_HOST
uv run emet run chat --host ORIN_HOST --once "Reply with exactly: pong"
uv run emet run chat --host ORIN_HOST --vl --once "Describe briefly"
# Multi-turn REPL keeps history (conversational prompt, not Stretch pick/place).

export DISCORD_TOKEN=...
uv run emet run agent --connection mars --host ORIN_HOST

# HM-EQA answer VL on the Orin (Habitat still local). Must use --host / --vl-endpoint
# on emet hmeqa h2h — shell export EMET_VL_ENDPOINT alone is not in the jobs env.
uv run emet llm health --host ORIN_HOST && uv run emet llm smoke --host ORIN_HOST --vl-only
uv run emet hmeqa h2h --arms classic --ids 15,56,65,68 --host ORIN_HOST \
  --job-name hmeqa-json-orin --need-mib 12000 \
  -d "JSON answers; VL on ORIN_HOST; Habitat local"
```

### SSH (optional)

Use your normal SSH config / keys for the Orin. Example:

```
Host ORIN_HOST
  HostName 192.168.1.55
  User YOUR_USER
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

```bash
ssh ORIN_HOST 'cd ~/src/home_robot_v4 && ./scripts/run_jetson_llm_container.sh --vl --detach --port 8000 --model Qwen/Qwen2-VL-7B-Instruct'
```

---

## Phase 2 (not implemented): remote memory worker

Future work: a ZMQ **SUB** on Mars bridge topic **4401** (or a dedicated RPC) so voxel / graph updates can run on a remote host, with a query RPC back to the agent. Phase 1 keeps **all memory on the workstation** and only remotes caption/EQA images over HTTP. Do not change Mars bridge ports for Phase 1.

## Code map

| Piece | Path |
|-------|------|
| Jetson container server | [`docker/jetson_llm_server.py`](../docker/jetson_llm_server.py) |
| Native HTTP server | [`src/emet/llms/openai_server.py`](../src/emet/llms/openai_server.py) |
| Remote text client | [`src/emet/llms/openai_client.py`](../src/emet/llms/openai_client.py) |
| Remote VL client | [`src/emet/llms/openai_vllm_client.py`](../src/emet/llms/openai_vllm_client.py) |
| Health / smoke helpers | [`src/emet/llms/remote_ops.py`](../src/emet/llms/remote_ops.py) · `emet llm health\|smoke` |
| Chat (text / VL) | [`src/emet/app/chat.py`](../src/emet/app/chat.py) · `emet run chat --host HOST` |
| Deploy | [`src/emet/deploy_llm.py`](../src/emet/deploy_llm.py) · `emet deploy llm --host HOST` |
| Native CLI | `emet serve llm` → [`src/emet/app/serve_llm.py`](../src/emet/app/serve_llm.py) |

See also [jetson.md](jetson.md), [cli.md](cli.md), [robots/innate_mars_hardware.md](robots/innate_mars_hardware.md), and [environment_variables.md](environment_variables.md).
