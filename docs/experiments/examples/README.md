# Example figures (local generation)

Paper experiment figures are **not** checked into git by default. Generate them on your machine and copy into `paper/figures/` when ready for LaTeX.

## Backend localization

```bash
uv run python scripts/smoke_backend_localization_figure.py \
  --quick \
  --output-dir ~/runs/emet/backend_localization_smoke
```

Outputs in the run directory:

- `backend_localization_topdown.png` — top-down map, GT (white) vs backend predictions
- `backend_localization_metrics_bar.png` — XY error and @0.5 m hit rate
- `smoke_results.json`, `summary.json`

### Validated reference run (Robocasa seed 0)

A known-good package from development:

```text
~/runs/emet/backend_localization_figures_20260617_final/
  backend_localization_topdown.png
  backend_localization_metrics_bar.png
  smoke_results.json
  summary.json
```

Reference metrics (`obj_main` jar at ~(0.57, -0.43)):

| Backend | err_xy | @0.5 m |
|---------|--------|--------|
| dynamem | 0.14 m | HIT |
| dynagraph | 0.14 m | HIT |
| graph_eqa | miss | — |
| vlm_only | miss | — |

Copy into paper figures (optional):

```bash
cp ~/runs/emet/backend_localization_smoke/backend_localization_*.png paper/figures/
```

## SQA3D

After a sweep JSONL exists:

```bash
uv run emet sqa3d plot-results \
  -p ~/runs/emet/sqa3d/dynagraph_val_q0-30.jsonl \
  -o paper/figures/sqa3d_val30
```

## Dynamic exploration

Use Rerun on a completed export, or map snapshots from the eval JSON under the output directory. No standard PNG harness yet — see [dynamic_exploration.md](../dynamic_exploration.md).
