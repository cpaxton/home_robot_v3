# Running emet on the NVIDIA Jetson

Instructions for **Jetson AGX Orin** (and similar Tegra boards). Tested on JetPack 5.1.2 (L4T 35.4.1, Ubuntu 20.04, aarch64).

There are two options:

1. **Native install on the Jetson** (recommended for development) — [below](#native-install-jetson-profile)
2. **Docker image** — [Running in Docker](#running-emet-in-a-docker-container) (legacy Stretch AI image; rebuild for current emet)

## About Jetson

Jetson is an embedded NVIDIA Tegra platform. Trade-offs vs a workstation:

- **aarch64** — many PyPI packages ship x86_64-only wheels; emet uses PEP 508 markers and an older `open3d` pin on aarch64 so `uv sync` can resolve.
- **Tegra CUDA ≠ server CUDA** — PyPI `torch` manylinux aarch64 wheels install, but they are **not** Jetson GPU builds. `torch.cuda.is_available()` is often `False` until you install an NVIDIA Jetson wheel or build PyTorch from source for this JetPack.
- **Limited disk** — a full desktop `uv sync` (sim + SAM-2 + Molmo) can exceed eMMC. Prefer the **jetson** profile.

## Native install (jetson profile)

From the repo root on the Orin:

```bash
./scripts/install_jetson.sh -y
# equivalent:
#   ./install.sh --profile=jetson -y
#   EMET_INSTALL_PROFILE=jetson ./install.sh -y
#   uv run emet install full -y --profile jetson   # after a bootstrap sync
```

What this does:

| Step | Behavior |
|------|----------|
| Profile | `--profile=jetson` → no Robocasa clone, no SAM-2, no MolmoSpaces |
| Python | `UV_PYTHON=3.10` (system Python on JP5 is 3.8; emet needs ≥3.10) |
| Sync | `uv sync --no-default-groups --group dev --group sim` (MuJoCo CLI paths; no SAM-2 / mediapipe / DA3) |
| Deps | Skips `bitsandbytes` / Triton / FLA / `depth-anything-3` on `aarch64`; pins `open3d>=0.17,<0.19` |
| Apt | Extra build packages for aarch64 sdists (`sophuspy`, `scikit-fmm`, `PyAudio`, …) |
| Attn | Sets `EMET_ALLOW_SDPA_ATTN=1` (no flash-attn on Tegra) |

Verify:

```bash
source .venv/bin/activate
python -c "import emet, torch; print(emet.__file__); print('cuda', torch.cuda.is_available())"
emet --help
uv run emet test src/test/utils/test_platform_info.py -q
```

Optional onboard DA3 (Mars bridge): `depth-anything-3` is **not** installed on Linux aarch64 by default (its dependency `pycolmap` has no Tegra wheels). Prefer workstation DA3 via `depth_source: auto`, or build/install DA3 deps manually on the Orin.

```bash
# On x86 workstations only:
uv sync --group da3
```

### Tegra CUDA PyTorch (optional)

Official JP5 wheels target **Python 3.8**. For emet’s Python 3.10 venv you typically need to **build PyTorch from source** for this JetPack, or run inference in a container that already has a Tegra build. See:

- [NVIDIA: Install PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [PyTorch for Jetson forum thread](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

System Python 3.8 on this Orin may already have `torch 2.0.0.nv23.05` with CUDA; that stack is separate from the emet `.venv`.

### Serve a Qwen LLM for the LAN (workstation clients)

**Preferred on JP5 (this Orin): Tegra-CUDA Docker** — uses NVIDIA Jetson PyTorch with working GPU:

```bash
./scripts/run_jetson_llm_container.sh --build --detach --model Qwen/Qwen2.5-7B-Instruct
# smoke / low disk: --model Qwen/Qwen2.5-0.5B-Instruct
# larger if VRAM allows: Qwen/Qwen2.5-14B-Instruct
```

Then on another PC (e.g. olympia talking to your Orin hostname):

```bash
uv run emet llm health --host ORIN_HOST
uv run emet run chat --host ORIN_HOST --once "Reply with exactly: pong"
export EMET_OPENAI_BASE_URL=http://ORIN_HOST:8000/v1
export EMET_OPENAI_MODEL=Qwen/Qwen2-VL-7B-Instruct   # label only when unified-7b
emet run agent --llm openai
# Herman: pass --host / EMET_LLM_HOST (configs/agent_innate_mars.yaml uses agent.llm: openai) — see docs/llm_serve.md
```

Native (CPU) path without Docker: `emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000` — slower; see [llm_serve.md](llm_serve.md).

### Disk space

Aim for **≥15 GB free** before sync. Stale Docker images are a common reclaim:

```bash
docker system df
docker ps -a
# only remove images you do not need:
# docker rmi hellorobotinc/stretch-ai_jetson:latest
```

Enable swap if compiles OOM (this board ships `/mnt/4GB.swap`):

```bash
sudo swapon /mnt/4GB.swap
```

## Running emet in a Docker container

We historically shipped a Stretch AI Jetson image. Pull/run:

```bash
./scripts/run_stretch_ai_jetson.sh
```

Build locally:

```bash
./docker/build-jetson-docker.sh
```

Dockerfile: [`docker/Dockerfile.jetson`](../docker/Dockerfile.jetson). Base image must match your L4T (check `/etc/nv_tegra_release`). Prefer the **native jetson profile** for current emet unless you maintain a rebuilt image.

## Optional: `jtop`

Tegra has no `nvidia-smi`:

```bash
sudo pip3 install -U jetson-stats
```

## Related

- Mars onboard DA3: [robots/innate_mars_hardware.md](robots/innate_mars_hardware.md)
- Robot pip pins: `configs/robots/innate_mars_robot_requirements.txt`, `configs/robots/innate_mars_da3_requirements.txt`
- Platform helper: `emet.utils.platform_info`
