# SQA3D compute setup

Hardware and process layout for **real-VLM** Dynagraph / DynaMem runs on ScanNet replay (mesh or `.sens`). No cloud pricing — capacity planning only.

## Minimum GPU memory (exclusive)

| Stack | Model (default) | Quantization | Free VRAM needed |
|-------|-----------------|--------------|------------------|
| Dynagraph tuned | Qwen3-VL-4B | int4 | **~12 GiB** per episode |
| Dynagraph tuned | Qwen3-VL-9B | int4 | **~20 GiB** per episode |
| DynaMem tuned | same EQA stack | int4 | similar |

These match `eqa_vl.vram_mib_tier_4b` / `vram_mib_tier_9b` in `dynav_config.yaml`. Add **~2–4 GiB** headroom for SigLIP mapping, Open3D EGL, and GraphEQA image batches.

**Rule:** one real-VLM episode ≈ one GPU. Do not share the GPU with other training or Habitat jobs; OOM produces empty answers (sanitized as abstains), not loud crashes.

Check before a sweep:

```bash
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv
```

Aim for **≥14 GiB free** on a 24 GiB card before `run-real-sweep` with 4B int4.

## Software

```bash
cd /path/to/home_robot_v2
uv sync
uv run emet test src/test/benchmarks/sqa3d/ -q
```

Data (once per machine):

```bash
uv run python scripts/download_sqa3d_data.py --fetch-annotations
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00
# Posed RGB replay:
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00 --with-sens
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `SQA3D_DATA_DIR` | `~/.cache/sqa3d/data` | Annotations |
| `SCANNET_ROOT` | `~/.cache/scannet` | Meshes + optional `.sens` |

Disk: `.sens` is **~300–500 MiB per scene**; full SQA3D val scenes are tens of GiB if you download everything.

## Recommended: isolated real-VLM sweep

`run-real-sweep` defaults to **`--isolate-episodes`**: each question runs in a fresh subprocess so the VLM and maps are released between episodes. Slower than in-process batch, but stable on a single GPU.

```bash
uv run emet sqa3d run-real-sweep \
  --split val \
  --question-start 0 \
  --question-end 30 \
  --replay-mode sens \
  --no-download \
  --method dynagraph \
  --output-dir /tmp/sqa3d_sens_sweep
```

Preflight wrapper (checks free VRAM, optional slice):

```bash
./scripts/run_sqa3d_gpu_sweep.sh --split val --question-start 0 --question-end 30 --replay-mode sens
```

## Model / device overrides

Smaller VLM (less VRAM):

```bash
uv run emet sqa3d run-episode \
  --question-id 220602000000 --split train \
  --eqa-vl-family qwen3_vl \
  --eqa-hf-model-id Qwen/Qwen3-VL-4B-Instruct \
  --profile tuned --replay-mode sens
```

Force 4B tier via env (see `dynav_config.yaml`):

```bash
export EMET_EQA_VL_MODEL_SIZE=4B
```

CPU fallback (slow, no extra GPU):

```bash
uv run emet sqa3d run-episode ... --device cpu
```

Optional allocator hint when fragmentation is an issue:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Multi-GPU (manual sharding)

Run **non-overlapping question slices** on separate GPUs. Use `--isolate-episodes` (or `run-real-sweep`) and distinct output JSONL paths, then merge JSONL for scoring.

```bash
# Terminal / GPU 0
CUDA_VISIBLE_DEVICES=0 uv run emet sqa3d run-batch \
  --split val --question-start 0 --question-end 15 \
  --replay-mode sens --profile tuned \
  --isolate-episodes \
  -o /tmp/sqa3d_shard0.jsonl

# Terminal / GPU 1
CUDA_VISIBLE_DEVICES=1 uv run emet sqa3d run-batch \
  --split val --question-start 15 --question-end 30 \
  --replay-mode sens --profile tuned \
  --isolate-episodes \
  -o /tmp/sqa3d_shard1.jsonl

cat /tmp/sqa3d_shard0.jsonl /tmp/sqa3d_shard1.jsonl > /tmp/sqa3d_merged.jsonl
uv run emet eval-sqa3d -p /tmp/sqa3d_merged.jsonl --split val
```

Resume after interruption: add `--resume` to `run-batch` (same output path).

## Profiles (memory vs speed)

| Profile | VLM | Planning | Post-rotate updates | Nav sample cap (tuned) |
|---------|-----|----------|---------------------|-------------------------|
| `smoke` | mock | 8 | 5 | default |
| `tuned` | real | 15 | 6 | 48 (`graph_eqa_extract.navigation_samples_max`) |

In-process `run-batch` **without** `--isolate-episodes` keeps one VLM loaded and is faster only if the GPU is dedicated and episodes are small.

## Verify + score

```bash
uv run emet sqa3d verify --run-embodied-smoke
uv run emet eval-sqa3d -p /tmp/sqa3d_sens_sweep/dynagraph_val_q0-30.jsonl
uv run emet sqa3d plot-results -p /tmp/sqa3d_sens_sweep/dynagraph_val_q0-30.jsonl -o /tmp/sqa3d_figs
```

## See also

- [sqa3d.md](sqa3d.md) — benchmark overview, replay modes, metrics
- [cli.md](cli.md#emet-sqa3d-subcommand) — `emet sqa3d` / `emet eval-sqa3d` flags and defaults
- `dynav_config.yaml` — `eqa_vl`, `graph_eqa_extract`
