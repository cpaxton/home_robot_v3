# Habitat EQA (HM-EQA / OpenEQA)

Primary paper goal: reproduce GraphEQA-style metrics on HM-EQA and OpenEQA subsets.

**Results vs prior art (tables, JSONL paths, planned sweeps):** **[habitat_eqa_results.md](habitat_eqa_results.md)**

**Deep docs:**

- [habitat/README.md](../habitat/README.md)
- [habitat_eqa.md](../habitat_eqa.md) (operator usage)
- Parity appendix: `paper/sections/appendix/05_habitat_eqa_parity.tex`
- Fix / debias history: [plans/fable5-dynagraph-habitat.md](../plans/fable5-dynagraph-habitat.md)

## Branch note

Habitat harness development on **`feature/eval-diagnostics-smoke`** (nav, grounding, diagnostics). Merge before final paper numbers.

## Quick comparison (June 2026)

| Who | VLM | n | Accuracy |
|-----|-----|---|----------|
| GraphEQA (paper) | GPT-4o | 113 | **63.5%** |
| GraphEQA (paper) | Gemini-2.5 Pro | 113 | **67.0%** |
| **emet** graph_eqa repro | gemma-3-4b-it | 113 | **41.6%** |
| **emet** dynagraph | Qwen2.5-VL-3B | 32 | 34.4% |
| **emet** dynagraph (post-fix) | Qwen3-VL-8B | 8 hold-out | **50.0%** |

We use **weaker local VLMs** and **partial** HM3D semantics vs the reference stack. Dynagraph consistently beats graph_eqa on matched slices. See [habitat_eqa_results.md](habitat_eqa_results.md) for full tables and planned experiments.

## Metrics

MC accuracy, mean planning steps; GraphEQA vs Dynagraph vs ablations.

## Entrypoint

```bash
./scripts/install_habitat.sh
.venv-habitat/bin/emet-habitat  # see habitat/usage.md for sweep scripts
```

### Example sweeps

```bash
# Held-out random-8 (anti-overfit; excludes tuning Q3,14,17)
TAG=holdout8_postfix_20260627 IDS=15,56,65,68,79,88,104,105 METHOD=dynagraph TIMEOUT=7200 \
  ./scripts/run_habitat_iter_subset.sh

# Full 113 (planned headline run)
.venv-habitat/bin/emet-habitat run-batch \
  --method dynagraph --paper-subset \
  --question-start 0 --question-end 112 \
  --eqa-vl-family qwen3_vl --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda --resume \
  --output ~/.cache/habitat_eqa/results/full113_dynagraph_qwen3_vl_postfix.jsonl
```

## Frame / map diagnostics

Voxel-map coordinates for Habitat are documented in [habitat/README.md](../habitat/README.md) (voxel-world section). Before using `topdown_map.png` in paper figures, audit bundles with [`scripts/audit_habitat_voxel_map.py`](../../scripts/audit_habitat_voxel_map.py) — see [evaluation.md](../evaluation.md#habitat-frame-sanity-before-trusting-map-colors).

## Related sim benchmarks (this repo, no Habitat install)

- [ovmm_find_phase.md](ovmm_find_phase.md) — HM3D proxy find-phase
- [dynamic_exploration.md](dynamic_exploration.md) — Emet sim exploration
