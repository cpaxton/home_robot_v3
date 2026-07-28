# Habitat EQA (HM-EQA / OpenEQA)

Primary paper goal: reproduce GraphEQA-style metrics on HM-EQA and OpenEQA subsets.

**Central index:** [experiments/README.md](README.md) (HM-EQA baselines + run commands).  
**Results vs prior art:** [habitat_eqa_results.md](habitat_eqa_results.md)  
**CLI flags:** [habitat/usage.md](../habitat/usage.md)

**Deep docs:**

- [habitat/README.md](../habitat/README.md)
- [habitat_eqa.md](../habitat_eqa.md) (operator usage)
- Parity appendix: `paper/sections/appendix/05_habitat_eqa_parity.tex`
- Fix / debias history: [plans/fable5-dynagraph-habitat.md](../plans/fable5-dynagraph-habitat.md)

## Baselines (do not conflate)

| Method | Profile | Role |
|--------|---------|------|
| `graph_eqa` | `graph_eqa_baseline` (0/0 merge) | Internal GraphEQA reimplementation |
| `dynagraph` | `unified_eqa` (0.45 m) + tuned extras | Our method |
| Published GraphEQA 63.5–67% | API VLM + full HM3D semantics | External prior art only |

Explore-off / MCQ-debias / agentic H2H are **ablations**, not method defaults.

## Quick comparison (recorded)

| Who | VLM | n | Accuracy |
|-----|-----|---|----------|
| GraphEQA (paper) | GPT-4o | 113 | **63.5%** |
| GraphEQA (paper) | Gemini-2.5 Pro | 113 | **67.0%** |
| **emet** graph_eqa repro | gemma-3-4b-it | 113 | **41.6%** |
| **emet** dynagraph | Qwen2.5-VL-3B | 32 | 34.4% |
| **emet** dynagraph (post-fix) | Qwen3-VL-8B | 8 hold-out | **50.0%** |

We use **weaker local VLMs** and **partial** HM3D semantics vs the reference stack. See [habitat_eqa_results.md](habitat_eqa_results.md) for full tables and planned experiments.

## Metrics

MC accuracy, mean planning steps; GraphEQA vs Dynagraph vs ablations.

## Entrypoint

```bash
./scripts/install_habitat.sh
uv run emet eval recover --need-mib 12000
uv run emet habitat safe-start
uv run emet habitat run-episode --question-id 0 --method graph_eqa --mock-llm
# Full 113: ./scripts/run_hmeqa_paper113_h2h.sh via emet jobs
# Agentic H2H: uv run emet hmeqa overnight
```

Details: [experiments/README.md § HM-EQA baselines](README.md#hm-eqa-baselines), [habitat/usage.md](../habitat/usage.md).

## Frame / map diagnostics

Voxel-map coordinates for Habitat are documented in [habitat/README.md](../habitat/README.md) (voxel-world section). Before using `topdown_map.png` in paper figures, audit bundles with [`scripts/audit_habitat_voxel_map.py`](../../scripts/audit_habitat_voxel_map.py) — see [evaluation.md](../evaluation.md#habitat-frame-sanity-before-trusting-map-colors).

## Related sim benchmarks (this repo, no Habitat install)

- [ovmm_find_phase.md](ovmm_find_phase.md) — HM3D proxy find-phase
- [dynamic_exploration.md](dynamic_exploration.md) — Emet sim exploration
