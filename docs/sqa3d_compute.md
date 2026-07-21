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
# Preferred (canonical CLI); bash scripts/gpu_preflight.sh delegates here:
uv run emet eval kill-stale
NEED_MIB=14000 uv run emet eval check
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

## Multi-GPU sharding (recommended for full splits)

Use **`scripts/run_sqa3d_sharded_sweep.sh`** to split a sweep across GPUs (~linear speedup when each GPU is exclusive):

```bash
# Full val dynagraph on 4 GPUs (~4× faster than one GPU)
./scripts/run_sqa3d_sharded_sweep.sh --split val --method dynagraph --all --gpus 0,1,2,3

# Wired into the large paper queue:
SQA3D_GPUS=0,1,2,3 ./scripts/run_large_paper_eval.sh sqa3d-val
```

Each shard writes `METHOD_SPLIT_qSTART-END.jsonl`; the script merges shards and writes `*_merged.csv`.

Manual slices (same idea):

```bash
CUDA_VISIBLE_DEVICES=0 uv run emet sqa3d run-real-sweep \
  --split val --method dynagraph --question-start 0 --question-end 815 \
  --no-download --resume --isolate-episodes --output-dir ~/runs/emet/sqa3d
```

Resume after interruption: `--resume` on each shard (same output JSONL path).

## Faster in-process batch (OOM risk)

`--no-isolate-episodes` loads the VLM once per sweep instead of per question (~2–3× faster) but can fragment VRAM on long runs. Try on a slice first:

```bash
uv run emet sqa3d run-real-sweep --split val --question-start 0 --question-end 100 \
  --method dynagraph --no-isolate-episodes --no-download --resume
```

Or: `SQA3D_NO_ISOLATE=1` with `run_large_paper_eval.sh` / sharded sweep.

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
