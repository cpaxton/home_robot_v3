# Cross-track smoke validation

One-episode (or unit-test) sanity check per paper track before multi-day GPU sweeps.
Updated after each smoke pass; see [five-track plan](../plans/) and [experiments README](README.md).

**Last run:** 2026-06-27 (recovered after agent interrupt)

## Summary

| # | Track | Status | Pass criterion | Artifact |
|---|-------|--------|----------------|----------|
| 0 | Unit tests (all tracks) | **PASS** | pytest green | — |
| 1 | SQA3D | **PASS** | mock-LLM episode completes; prediction matches gold | train q220602000000 → `brown` |
| 2 | Habitat EQA | **PASS** | non-empty letter + episode completes | `~/.cache/habitat_eqa/results/smoke_cross_track_q17.jsonl` (Q17 → D, correct) |
| 3 | Habitat OVMM find-phase | **PASS** | find_object or find_recep success | `~/runs/emet/ovmm_habitat/smoke_track3/hm3d_lamp_bed_00006_gt.json` |
| 4 | Robocasa explore (Phase 1) | **IN PROGRESS** | explored_fraction > 0, no crash | `~/runs/emet/dynamic_exploration/smoke_cross_track_v3/` |
| 5 | Robocasa world-change (Phase 2) | **PENDING** | aggregate CSV written | after track 4 |

### Track 4 failure note (fixed)

Earlier attempts (`smoke_cross_track`, `smoke_cross_track_v2`) failed in ~1.4 s with `sim server did not bind port …` because **robocasa was not installed** in `.venv` (`ModuleNotFoundError: No module named 'robocasa'`). Fixed with `emet install sim` / editable install from `third_party/robocasa`. Retry: `smoke_cross_track_v3`.

## Commands (repro)

### Tier 0 — unit tests (~15 min)

```bash
uv run emet test src/test/benchmarks/sqa3d/ \
  src/test/habitat/test_metrics.py src/test/eval/test_dynagraph_vram.py \
  src/test/memory/test_graph_eqa_memory.py src/test/memory/test_mcq_debias.py \
  src/test/memory/test_ovmm_find_phase_metrics.py \
  src/test/memory/test_habitat_ovmm_find_loader.py \
  src/test/eval/test_dynamic_exploration_config.py \
  src/test/eval/test_dynamic_exploration_runner.py \
  src/test/memory/test_memory_backends_smoke.py -q
```

### Track 1 — SQA3D

```bash
uv run emet sqa3d run-episode --split train --mock-llm --question-id 220602000000
```

### Track 2 — Habitat EQA (Q17 regression)

```bash
.venv-habitat/bin/emet-habitat run-episode \
  --question-id 17 --method dynagraph \
  --eqa-vl-family qwen3_vl --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda --export-map \
  --output ~/.cache/habitat_eqa/results/smoke_cross_track_q17.jsonl
```

### Track 3 — Habitat OVMM

```bash
.venv-habitat/bin/emet-habitat run-ovmm-find-episode \
  --episode-id hm3d_lamp_bed_00006 \
  --backend ground_truth --cpu-only --not-rotate \
  --output ~/runs/emet/ovmm_habitat/smoke_track3/hm3d_lamp_bed_00006_gt.json
```

### Track 4 — Robocasa explore

Requires `emet install sim` (robocasa in venv). ~60–75 min with real VLM.

```bash
uv run python scripts/eval_dynamic_exploration.py --smoke \
  --output-dir ~/runs/emet/dynamic_exploration/smoke_cross_track_v3
```

### Track 5 — World-change (after track 4)

```bash
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph \
  --output-dir ~/runs/emet/dynamic_exploration/smoke_world_change
```

## Prior-art numbers (not smoke)

Full benchmark tables: [habitat_eqa_results.md](habitat_eqa_results.md). SQA3D/OVMM/dynamic exploration paper rows remain `--` until subset sweeps complete.

## Next after all smokes pass

1. SQA3D val subset aggregate
2. Habitat hold-out-8 or balanced-32 @ Qwen3-VL-8B
3. OVMM S0 ladder CSV refresh
4. Dynamic explore full matrix (robocasa + molmo lifelong)
