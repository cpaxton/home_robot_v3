# Four fixed actual TAMP input diagnostics

This is compact diagnostic evidence, not a trained controller or manipulation
qualification. Native source: `a8c9e9d7`; one CPU14 worker, no PPO or hardware.
The earlier full regression receipt belongs to source `35bb4fa6`.

- `results.json`: four outcomes, recording parity, source/actor/archive hashes,
  phase errors and actions, exact offline actor replay, named normalization
  clipping and hypothetical grip-input sensitivity. No raw histories or physics.
- `tracking.svg`: teacher versus retained-actor response on the matched pregrasp.

Raw `.json.gz` files remain ignored under `logs/eval/mars-tamp-inputs-v1`.
They are not committed and must be available to reproduce the analysis:

```bash
env -u WANDB_API_KEY CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 15 .venv/bin/python scripts/analyze_mars_tamp_inputs.py \
  --input logs/eval/mars-tamp-inputs-v1 \
  --native-log logs/eval/mars-tamp-inputs-v1-managed/job.log \
  --output /tmp/mars-tamp-input-analysis-fresh
```

Use a fresh output path and an unoccupied CPU lane. The offline script verifies
all trace hashes and the frozen actor, and steps no physics. See the
[full interpretation and limits](../../../../docs/experiments/mars_tamp_input_diagnostic_20261115.md).
