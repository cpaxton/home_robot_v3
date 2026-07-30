# Backend localization figure

Shared-sim comparison of memory backends on one Robocasa scene: top-down map with GT vs predicted object boxes, plus a metrics bar chart.

**Script:** [`scripts/smoke_backend_localization_figure.py`](../../scripts/smoke_backend_localization_figure.py)

## Purpose

Qualitative and lightweight quantitative comparison of `dynamem`, `static_graph`, `dynagraph`, and `vlm_only` on the same Robocasa seed — useful for paper figures and debugging backend localization without a full OVMM find-phase sweep.

## Metrics

| Field | Meaning |
|-------|---------|
| `err_xy_m` | Planar distance from prediction to GT `obj_main` |
| `hit_0_5m` | `err_xy_m < 0.5` |
| `mapping_wall_s` / `query_wall_s` | Timing per backend |

## Quick smoke (recommended)

One GPU job at a time. Default uses light snap-yaw mapping (fast).

```bash
uv run python scripts/smoke_backend_localization_figure.py \
  --quick \
  --output-dir ~/runs/emet/backend_localization_smoke
```

`--quick` runs `static_graph` + `dynagraph` with `explore_steps=1`.

## Full backend row set

```bash
uv run python scripts/smoke_backend_localization_figure.py \
  --backends dynamem static_graph dynagraph vlm_only \
  --explore-steps 0 \
  --output-dir ~/runs/emet/backend_localization_metrics
```

Use `--explore-steps 0` for fastest figures. Add `--full-protocol` only when you need OVMM-identical rotate+explore mapping (much slower; frontier nav timeouts possible with `explore_steps ≥ 1` on dynamem).

## Outputs

| File | Description |
|------|-------------|
| `backend_localization_topdown.png` | Top-down map, GT (white) vs predictions |
| `backend_localization_metrics_bar.png` | Bar chart from `smoke_results.json` |
| `smoke_results.json` | Per-backend metrics |
| `summary.json` | Run metadata |

## Validated reference (Robocasa seed 0)

Development run packaged at `~/runs/emet/backend_localization_figures_20260617_final/`:

| Backend | err_xy | @0.5 m |
|---------|--------|--------|
| dynamem | 0.14 m | HIT |
| dynagraph | 0.14 m | HIT |
| static_graph | miss | — |
| vlm_only | miss | — |

GT target: `obj_main` (jar) at ~(0.57, -0.43).

## GPU notes

- Run backends **sequentially** (script default); do not launch parallel GPU sweeps.
- Kill stale `emet run dynagraph` / sim servers between runs if ports are stuck: `uv run emet kill-mujoco-server --port 4401`.

## See also

- [examples/README.md](examples/README.md) — copy figures into `paper/figures/`
- [ovmm_find_phase.md](ovmm_find_phase.md) — full find-phase benchmark
- [paper_benchmarks.md](../paper_benchmarks.md) — OVMM find-phase batch commands
