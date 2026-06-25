# Dynamic exploration

Active frontier exploration (Phase 1), scripted world-change / staleness (Phase 2), and lifelong K-cycle checkpoint/fuzz/reload (Phase 3) on Stretch in Robocasa and MolmoSpaces iTHOR.

**Deep doc:** [dynamic_exploration_benchmark.md](../dynamic_exploration_benchmark.md)

## Paper references

- Section: `paper/sections/04b_dynamic_exploration.tex`
- Tables: `tab:dynamic_explore_phase1`, `tab:dynamic_explore_world_change`, `tab:dynamic_explore_lifelong`

## Primary metrics

| Phase | Aggregate CSV | Key columns |
|-------|---------------|-------------|
| P1 explore | `aggregate_dynamic_exploration.csv` | `explored_fraction`, `spatial_recall`, `eqa_accuracy`, `node_count` |
| P2 world-change | `aggregate_dynamic_exploration_world_change.csv` | `answer_correct_pre/post`, `n_stale_nodes_after_move`, `recovery_steps` |
| P3 lifelong | `aggregate_dynamic_exploration_lifelong.csv` | per-cycle `eqa_accuracy`, node churn, move adaptation flags |

Metrics are **not comparable** to OVMM find-phase or SQA3D EM@1.

## Config

- `configs/benchmarks/dynamic_exploration.yaml`
- Questions: `src/emet/config/benchmarks/dynagraph_questions.yaml`
- Default output: `~/runs/emet/dynamic_exploration` (`EMET_DYNAMIC_EXPLORE_OUTPUT`)

## Smoke commands

```bash
# Config matrix only (no GPU)
uv run python scripts/eval_dynamic_exploration.py --smoke --dry-run

# Phase 1 GPU smoke (Robocasa seed0, K=3; ~60–75 min)
uv run python scripts/eval_dynamic_exploration.py --smoke \
  --output-dir ~/runs/emet/dynamic_exploration/smoke

# Phase 2 world-change
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph --cpu-only

# Phase 3 lifelong
uv run python scripts/eval_dynamic_exploration.py \
  --phase lifelong --episode-id robocasa_seed0_lifelong \
  --backend dynagraph --cpu-only
```

`--smoke` is equivalent to `--phase explore --env robocasa --episode-id robocasa_seed0 --backend dynagraph --explore-max-iters 3 --mapping-mode explore`.

## Full matrix

```bash
./scripts/run_dynamic_exploration_full.sh
# or via large queue:
./scripts/run_large_paper_eval.sh dynamic-explore
```

## Timing and debugging

- GPU Robocasa K=3: typically **60–75 min** per run.
- Subprocess timeout scales with explore budget (~105 min for K=3); override with `EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S`.
- Each run writes `{export_dir}/dynagraph.log` for timeout debugging.

## Backend localization figure

For a quick visual of backend localization on the same Robocasa stack, see [backend_localization.md](backend_localization.md).

## Tests

```bash
uv run emet test src/test/eval/test_dynamic_exploration_config.py -q
```
