# SQA3D situated open QA

Embodied ScanNet replay with EM@1 scoring; compare `dynamem` vs `dynagraph`.

**Deep docs:**

- [sqa3d.md](../sqa3d.md) — harness, replay modes, CLI
- [sqa3d_compute.md](../sqa3d_compute.md) — wall-clock and sharding

## Paper reference

- Section: `sec:sqa3d_benchmark`
- Table: `tab:sqa3d_backend_replay`
- Metric: EM@1 on val/test splits

## Config and output

- `configs/sqa3d/benchmark.yaml`
- Default output: `~/runs/emet/sqa3d` (`EMET_SQA3D_OUTPUT`)
- Aggregate: `scripts/aggregate_sqa3d_sweep.py` → `aggregate_sqa3d.csv`

## Smoke (no GPU)

```bash
uv run emet sqa3d run-episode --mock-llm --question-id 220602000000
```

## Dev sweep (val subset)

```bash
# Data
uv run python scripts/download_sqa3d_data.py --fetch-annotations
uv run python scripts/download_scannet_data.py --accept-tos --scenes-from-sqa3d --split val --limit 10

# Paper dev sweep
uv run emet sqa3d run-real-sweep --no-download --replay-mode sens

# Score + figures
uv run emet eval-sqa3d -p ~/runs/emet/sqa3d/dynagraph_val_q0-30.jsonl
uv run emet sqa3d plot-results -p ~/runs/emet/sqa3d/dynagraph_val_q0-30.jsonl -o paper/figures/sqa3d_val30
```

## Compare backends

```bash
uv run emet sqa3d run-real-sweep --method dynamem --output-dir ~/runs/emet/sqa3d/dynamem_val30 --no-download
uv run emet sqa3d run-real-sweep --method dynagraph --output-dir ~/runs/emet/sqa3d/dynagraph_val30 --no-download
uv run python scripts/aggregate_sqa3d_sweep.py \
  ~/runs/emet/sqa3d/dynamem_val30/*.jsonl ~/runs/emet/sqa3d/dynagraph_val30/*.jsonl \
  --output-dir ~/runs/emet/sqa3d
```

## Large queue

See [large_eval_queue.md](large_eval_queue.md) for multi-GPU sharding (`SQA3D_GPUS`, `run_sqa3d_sharded_sweep.sh`).
