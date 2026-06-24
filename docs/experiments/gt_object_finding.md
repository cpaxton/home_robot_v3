# GT-supervised object finding

Sim oracle graph from `sim_object_placements` — upper bound for localization and fusion diagnostics.

**Paper:** `sec:gt_experiments`, appendix `04_ground_truth_graph.tex`

## Metrics

XY error, recall @ radius, `gt_graph_completeness` (via `emet eval-dynagraph`).

## Smoke (Robocasa)

```bash
# Terminal 1
uv run emet serve mujoco --scene robocasa --headless --port-offset 0

# Terminal 2
uv run emet run dynagraph --ground-truth --export runs/robocasa_gt --port-offset 0
uv run python scripts/eval_dynagraph_ground_truth.py --run-live --port-offset 0 --output metrics.json
```

## Perception vs GT alignment

```bash
uv run emet run dynagraph --compare-to-gt --export runs/<id>
```

## Batch eval

```bash
uv run emet eval-dynagraph --episode runs/<export_dir>
```

## Related

- [dynagraph_benchmarks.md](../dynagraph_benchmarks.md) — full dynagraph sim matrix
- [ovmm_find_phase.md](ovmm_find_phase.md) — `ground_truth` backend in find-phase sweeps
