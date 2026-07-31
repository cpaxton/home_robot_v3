# OpenAI-compatible LLM serve (`emet serve llm` / Jetson container)

Serve a local Hugging Face text or vision-language model behind a minimal OpenAI Chat Completions API so another machine can call this host over the LAN.

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
| Server | [`docker/jetson_llm_server.py`](../docker/jetson_llm_server.py) (Python 3.8–friendly; **text CausalLM**) |
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

## Native host (CPU fallback / multimodal VLM)

Prefer the Docker path on JP5 for **text** GPU speed. Native `emet serve llm` uses the host venv (often CPU-only torch on this Orin for text; on a desktop GPU it can also serve a VLM):

```bash
emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
# Multimodal (caption/EQA images):
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

## Caliban (LAN LLM + VLM host)

**caliban** (`192.168.1.55`, hostname `caliban`) runs OpenAI-compatible servers for workstation clients (Herman Discord agent).

| Port | Role | Typical recipe |
|------|------|----------------|
| `:8000` | Text tool-router **or** unified text+VL | Jetson container `emet-jetson-llm` (CausalLM 7B, or Qwen2-VL-7B with `--profile unified-7b`) |
| `:8001` | Caption / EQA VLM only (dual-2b) | Second container `emet-jetson-vl` (`Qwen2-VL-2B`); unused in unified-7b |

**Voxels / Dynagraph memory stay on olympia** (Mars ZMQ 4401). Only caption/EQA frames go remote.

The Jetson Docker text container loads **CausalLM** unless started with `--vl`. Image requests on a text-only `:8000` fail with a clear error. Orin eMMC cannot hold both a 7B CausalLM and a 7B VL (~15 GiB each) plus the L4T image — pick a profile:

| Profile | Command | Ports | Notes |
|---------|---------|-------|-------|
| `dual-2b` | `emet deploy llm --profile dual-2b` | text `:8000` + VL `:8001` | Fits beside CausalLM 7B; smaller captions |
| `unified-7b` | `emet deploy llm` (default) | one VL-7B on `:8000` | Uses Orin ~64 GiB unified memory; Herman points **both** endpoints at `:8000` |

```bash
./scripts/deploy_caliban_vl.sh --profile unified-7b
# preferred:
uv run emet deploy llm --profile unified-7b
```

### Dual-port recipe (text + small VL)

```bash
# on caliban — text (existing)
./scripts/run_jetson_llm_container.sh --detach --model Qwen/Qwen2.5-7B-Instruct
# port 8000, name emet-jetson-llm

# Multimodal VL (second container; same image, --vl mounts updated jetson_llm_server.py):
./scripts/run_jetson_llm_container.sh --vl --detach --port 8001 --name emet-jetson-vl \
  --model Qwen/Qwen2-VL-2B-Instruct
# or from olympia: ./scripts/deploy_caliban_vl.sh --profile dual-2b
```

### Quantization on caliban (JP5)

Workstation emet uses **bitsandbytes int4** for local Qwen3-VL. That path does **not** work in the Tegra-CUDA Jetson container today:

| Approach | Status on JP5 `emet-jetson-llm` |
|----------|----------------------------------|
| fp16 (current) | Works — Orin ~64 GiB unified memory fits Qwen2-VL-7B with headroom |
| bitsandbytes int4/int8 | Not installed; Tegra torch is not the PyPI CUDA build bnb expects |
| AutoAWQ / `Qwen2-VL-*-AWQ` | `pip install autoawq` pulls CPU `torch` + needs Triton |
| Quanto int8 | Installs but upgrades to CPU torch 2.4 / needs `float8` absent in nv23.05 |

So we stay on **fp16** for `emet deploy llm`. Pre-quantized AWQ weights (~7 GiB on disk vs ~16 GiB fp16) are attractive once a **JP6 / dustynv vLLM** image can load them without replacing Tegra torch. Until then, use `unified-7b` fp16 or `dual-2b` for a smaller second model.

Pass `--quant awq|int4|int8` to `jetson_llm_server.py` only to get that explanation (it exits non-zero on JP5).

Workstation Herman preset ([`configs/agent_innate_mars.yaml`](../configs/agent_innate_mars.yaml)):

- **unified-7b (default preset):** `agent.llm` and `mapping.eqa.vl_endpoint` both `openai@http://caliban:8000/v1` (Qwen2-VL-7B)
- **dual-2b:** text `caliban:8000` + VL `caliban:8001` — set `vl_endpoint` to `:8001` or `EMET_VL_ENDPOINT`

Override VL with `EMET_VL_ENDPOINT=openai@http://…/v1`.

### SSH from olympia

Keys live under `~/caliban_ssk_keys/` (symlink `~/caliban_ssh_keys` → same). Add to `~/.ssh/config`:

```
Host caliban
  HostName 192.168.1.55
  User cpaxton
  IdentityFile ~/caliban_ssk_keys/id_ed25519
  IdentitiesOnly yes
```

If `ssh caliban` is still `Permission denied (publickey)`, install the pubkey once (console / password on the Orin):

```bash
ssh-copy-id -i ~/caliban_ssk_keys/id_ed25519.pub cpaxton@caliban
# or append the .pub line to ~/.ssh/authorized_keys on caliban
```

Then rebuild / bump the served **text** model on the Orin:

```bash
ssh caliban 'cd ~/src/home_robot_v4 && ./scripts/run_jetson_llm_container.sh --detach --model Qwen/Qwen2.5-7B-Instruct'
# larger if VRAM allows: Qwen/Qwen2.5-14B-Instruct
# smoke: Qwen/Qwen2.5-0.5B-Instruct
```

### Workstation client (olympia)

```bash
# Health (no auth)
curl -s http://caliban:8000/health
curl -s http://caliban:8001/health
# expect ready=true

# Herman preset already sets agent.llm + mapping.eqa.vl_endpoint
export DISCORD_TOKEN=...
uv run emet run agent --connection herman

# Or any app (text only):
export EMET_OPENAI_BASE_URL=http://caliban:8000/v1
export EMET_OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct   # label only; server uses its loaded weights
emet run agent --llm openai ...
# or: --llm 'openai@http://caliban:8000/v1#Qwen/Qwen2.5-7B-Instruct'
```

Smoke:

```bash
./scripts/smoke_caliban_llm.sh
./scripts/smoke_caliban_vl.sh   # or: EMET_VL_ENDPOINT=http://127.0.0.1:8001/v1 ./scripts/smoke_caliban_vl.sh
uv run emet llm health
uv run emet llm smoke --text-only
uv run emet run chat --caliban --once "Reply with exactly: pong"
uv run emet run chat --vl --vl-endpoint openai@http://127.0.0.1:8001/v1 --once "Say hi"
```

| Variable | Role |
|----------|------|
| `EMET_OPENAI_BASE_URL` | OpenAI SDK `base_url` for **text** clients (include `/v1`) |
| `OPENAI_BASE_URL` | Fallback if `EMET_OPENAI_BASE_URL` unset |
| `EMET_OPENAI_MODEL` | Model id sent to the remote text server |
| `EMET_VL_ENDPOINT` | Override `eqa.vl_endpoint` (`openai@http://host:8001/v1`) |
| `EMET_JETSON_LLM_IMAGE` | Docker image tag (default `emet-jetson-llm:r35.4.1`) |
| `EMET_JETSON_LLM_NAME` | Container name (use a second name for a second port) |
| `EMET_LLM_SERVE_API_KEY` | Optional Bearer token (server + client) |
| `EMET_LLM_SERVE_DEVICE` | Default device for native `emet serve llm` |
| `EMET_LLM_SERVE_PORT` | Default published port for the Jetson runner script |

### Workstation VL while Orin hosts text

caliban’s Jetson container is **text CausalLM** on `:8000`. When Orin disk/VRAM cannot host a second VL process, run VL on the desktop GPU and point Herman at it:

```bash
# on olympia (or any CUDA host with emet)
uv run emet jobs run --name vl-serve-8001 --need-mib 12000 -- \
  uv run emet serve llm --vl --host 0.0.0.0 --port 8001
# clients:
export EMET_VL_ENDPOINT=openai@http://127.0.0.1:8001/v1
uv run emet llm smoke --vl-only --vl http://127.0.0.1:8001/v1
```

## Phase 2 (not implemented): remote memory worker

Future work: a ZMQ **SUB** on Mars bridge topic **4401** (or a dedicated RPC) so voxel / graph updates can run on a remote host, with a query RPC back to the agent. Phase 1 keeps **all memory on olympia** and only remotes caption/EQA images over HTTP. Do not change Mars bridge ports for Phase 1.

## Code map

| Piece | Path |
|-------|------|
| Jetson container server | [`docker/jetson_llm_server.py`](../docker/jetson_llm_server.py) |
| Native HTTP server | [`src/emet/llms/openai_server.py`](../src/emet/llms/openai_server.py) |
| Remote text client | [`src/emet/llms/openai_client.py`](../src/emet/llms/openai_client.py) |
| Remote VL client | [`src/emet/llms/openai_vllm_client.py`](../src/emet/llms/openai_vllm_client.py) |
| Health / smoke helpers | [`src/emet/llms/remote_ops.py`](../src/emet/llms/remote_ops.py) · `emet llm health|smoke` |
| Chat (text / VL) | [`src/emet/app/chat.py`](../src/emet/app/chat.py) · `emet run chat --caliban|--vl` |
| Native CLI | `emet serve llm` → [`src/emet/app/serve_llm.py`](../src/emet/app/serve_llm.py) |

See also [jetson.md](jetson.md) and [environment_variables.md](environment_variables.md).
