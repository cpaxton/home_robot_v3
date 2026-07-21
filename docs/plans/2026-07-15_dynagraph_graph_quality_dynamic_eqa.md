# Dynagraph graph quality → dynamic EQA (follow-on)

**Status (2026-07-16):** Phase 0–2 quality work landed earlier; **harness/VRAM/stale fixes for dynamic explore** landed on the feature branch (see below). Smoke gate: `scripts/run_dynagraph_dynamic_improve_smokes.sh`. Full dynamic EQA productization still gated on measured smoke numbers.

## Diagnosis snapshot (2026-07-15, existing artifacts)

| Source | Finding | Class |
|--------|---------|--------|
| Habitat `q28` dynagraph | `graph_nodes=859`, `obs=430` | **blowup** (prompt top-K required) |
| Habitat `q94` sample | ~79 nodes (not always 600+; depends on run) | borderline / prior blowup reports |
| Robocasa `explore_30` `graph.json` | obj=23, singleton_frac≈0.17, mean_support≈49 | **ok** growth; but top labels include junk (`bathroom stall`, `adapter`) → **wrong intermediate** from perception |
| Nav-fix / graph-quality verify `explore_3` | obj=17, singleton_frac≈0.17, mean_support≈14 | **ok** growth; same junk labels — structure stable, **semantics still wrong** |
| Habitat mock `q0` (no rotate, 2 steps) | frontier-only thin graph | **thin_graph** (expected without semantics/perception prelude) |

Triage command: `uv run python scripts/summarize_graph_health.py PATH/graph.json` (or Habitat `metrics.json` once `graph_health` is present on new runs).

**Implication:** Growth/blowup and prompt starvation are addressable with merge + top-K; **wrong intermediate labels** (open-vocab nonsense in kitchens) are mitigated by scene-aware [`graph_label_filter.py`](../../src/emet/memory/graph_eqa/graph_label_filter.py) (Robocasa → kitchen deny-list for bathroom ScanNet classes). Residual junk (e.g. ``adapter``) may still need detector vocab / confidence work.

## What “dynamic EQA” means here

Answer questions **while the world changes** mid-episode (objects move / appear / disappear), with:

1. Graph invalidate / remerge for moved bodies (not stale `CONFIRMED_MEMORY`)
2. Online re-grounding of Image‑N / memory summaries after each change
3. Scoring against **live** GT each cycle (prototype already exists in lifelong dynamic exploration)

Static HM-EQA and explore-then-EQA remain the primary paper tracks. Dynamic explore cycle EQA is the **stress harness** for graph quality, not yet the product milestone.

## Gates before expanding scope

- Habitat / dynamic exports include `graph_health`; blowups classified (`scripts/summarize_graph_health.py`)
- Prompt stays bounded (`eqa_max_graph_nodes`) even when memory is large
- Dedup / known-scene attach tests green; object-node growth stable across dynamic cycles
- Wrong intermediate (Image‑1 / memory) failures drop after structure fixes — not more memory-confirm patches alone

## Prototype path

Reuse [`dynamic_exploration_runner.py`](../../src/emet/eval/dynamic_exploration_runner.py) lifelong cycles: explore → EQA → world change → resume checkpoint.

**Landed (2026-07-16):**
- `flatten_eval_metrics` reads `fusion.fused` / `fusion.raw` recalls
- Phase-1 subprocess: `--benchmark-harness dynamic_explore --benchmark-method …`
- `prepare_dynagraph_vram_for_eqa` before Qwen question bank in `emet run dynagraph`
- World-change: age nodes near old pose → `maintain`; `n_stale_nodes_after_move` = nodes near old XY @ 0.75 m; refresh CONFIRMED_MEMORY before post-EQA
- Lifelong: `_invalidate_checkpoint_nodes_near_moves` after fuzz
- Kitchen deny-list adds `adapter` / `power strip` / `charger`

**EQA hang fix (2026-07-20):** Smoke of 2026-07-16 completed explore + VRAM prep + Qwen load, then hung because question-bank EQA still ran up to 5 uncover-frontier nav steps; timeout only killed `uv`, leaving an orphan on a dead sim. Fix: answer-only question bank (`allow_navigation=False`, skip look-around after explore), process-group kill on timeout.

**Still open:** full Phase-1 paper matrix (K∈{8,15,30} × seeds), lifelong table fill, residual wrong-label rate after filter.

## Related

- [dynagraph.md](../dynagraph.md#graph-health-eqa--dynamic-explore)
- [dynamic_exploration_benchmark.md](../dynamic_exploration_benchmark.md)
- [TESTING.md](../TESTING.md#known-gap-graph--eqa-on-a-known-scene-dynagraph)
- [fable5-dynagraph-habitat.md](fable5-dynagraph-habitat.md) (historical q94 blowup / top-K TODO)
