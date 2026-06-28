## Summary

Improves **Dynagraph HM-EQA exploration** in Habitat while keeping **`graph_eqa` as a clean baseline**. Adds the Habitat EQA harness wiring, MCQ position-bias debiasing, VLM model bake-off infrastructure, SigLIP2/dtype support, resume/metrics fixes, and documentation for reproducing experiments on a single 24 GB GPU.

**Scope:** ~2.4k lines across 44 files (harness + exploration + eval infra + paper appendix).

---

## Motivation

Local HM-EQA runs with Qwen2.5-VL-3B were barely above chance (~34% dynagraph vs ~31% graph_eqa on balanced-32) and far below published GraphEQA numbers (~63–67%). This branch:

1. Fixes harness bugs that silently suppressed GT semantics and mis-resumed OOM stubs
2. Adds MCQ debiasing for small-VLM position bias (letter A under-picked ~5% vs 25% expected)
3. Runs a VLM bake-off to pick a better local model (Qwen3-VL-8B int4 wins canonical-6 at **5/6**)
4. Documents HM3D-Semantics coverage limits (~**37/113** paper questions have GT labels)

---

## What's included

### Habitat EQA harness
- `emet habitat …` / `emet run graph-eqa-habitat` → `.venv-habitat` wrapper
- HM-EQA batch runner with `--resume`, compare-batch, paper subset (113 Q)
- HM3D GT semantics path when `.semantic.glb` exists; VLM fallback otherwise
- Docs: `docs/habitat/` (install, data, usage, troubleshooting, VLM bake-off)

### Dynagraph exploration
- **MCQ debias** (`src/emet/memory/graph_eqa/mcq_debias.py`) — choice-rotation vote at episode end; **dynagraph only** (`mcq_debias_enabled=True`)
- Optional **VLM frontier scoring** via `EMET_VLM_FRONTIER_SCORING=1` (off by default)
- SigLIP-guided exploration + graph frontier nodes (existing; ablation flags documented)

### VLM stack
- **Qwen3.5-9B** client (`qwen3_5_client.py`), registry/factory, `fla-core` deps
- **SigLIP2** checkpoints + `EMET_SIGLIP_VERSION` / `EMET_SIGLIP_DTYPE`
- Bake-off scripts: `scripts/run_fable5_bakeoff.sh`, `run_fable5_overnight.sh`
- Smoke/probe helpers: `smoke_vl_model.py`, `probe_vlm_boxes.py`

### Metrics & reliability
- `episode_run_completed()` now requires EQA output (`eqa_iterations > 0` or non-empty answer), not just `planning_steps > 0` — fixes infinite resume after GPU OOM
- JSONL fields: `predebias_letter`, `debias_votes`

### Semantics coverage tooling
- `--report-hmeqa-semantics` on download script
- `compute_hmeqa_semantics_coverage()` / `hmeqa_annotated_question_ids()`
- `emet habitat info` one-line semantics summary

### Paper & plans
- Appendix: `paper/sections/appendix/06_model_choice.tex` (VLM bake-off, why grounding specialist beats generalist)
- Plan docs: `docs/plans/fable5-dynagraph-habitat.md`, parity cross-refs in habitat README

---

## Key results (local, single RTX 4090)

| Eval | Dynagraph | graph_eqa | Notes |
|------|-----------|-----------|-------|
| Balanced-32 (3B) | **11/32 (34%)** | 10/32 (31%) | Pre-upsizing baseline |
| Canonical-6 bake-off (8B + debias) | **5/6 (83%)** | — | Qwen3-VL-8B int4 **winner** |
| Canonical-6 (3B control) | 2/6 | — | |
| Canonical-6 (Qwen3.5-9B) | 3/6 | — | Needs `fla-core` for fair wall-clock |
| Balanced-31 winner run (8B) | **7/14 (50%)** | — | **Partial** — stopped at 20h global deadline; 17 Q not run |

**Not comparable to GraphEQA Table 1 (63–67%)** without API VLMs, full semantics coverage, and exploration parity. See `paper/sections/appendix/05_habitat_eqa_parity.tex`.

---

## Notable fixes

- HM3D semantics path was gated incorrectly → scenes *with* GT assets built 0 object nodes (fixed in prior iteration; validated Q3: 1 node/wrong → 81 nodes/correct)
- Resume after OOM no longer treats nav-only stubs as completed episodes
- q31 / q94 excluded from bake-off sets (timeout / graph blowup)

---

## Known gaps / follow-up (not blocking merge)

- [ ] Finish balanced-31 winner run (17 remaining question ids)
- [ ] `graph_eqa` + Qwen3-VL-8B on same subset for fair 8B baseline
- [ ] Cap scene-graph text in EQA prompt (q94-style blowups)
- [ ] `--paper-subset-annotated-only` CLI flag (helper exists)
- [ ] `docs/cli.md` — add `emet habitat` / `graph-eqa-habitat` entries (currently in `docs/habitat/usage.md` only)
- [ ] Update `fable5-dynagraph-habitat.md` with final partial balanced-31 numbers

---

## Test plan

```bash
# Branch-specific unit tests
uv run pytest \
  src/test/habitat/test_hmeqa_semantics_coverage.py \
  src/test/habitat/test_metrics.py \
  src/test/memory/test_mcq_debias.py \
  src/test/llms/test_vllm_registry.py \
  src/test/agent/test_vlm_frontier_scoring.py -q

# Habitat wrapper smoke (requires ./scripts/install_habitat.sh)
uv run pytest src/test/habitat/test_smoke.py -q

# Semantics coverage report
uv run python scripts/download_habitat_eqa_data.py --report-hmeqa-semantics
uv run emet habitat info

# Mock episode (no GPU VLM)
uv run emet run graph-eqa-habitat --mock-llm --question-id 0 --method dynagraph
```

**Manual / GPU:** Re-run canonical-6 or resume balanced-31 via `scripts/run_fable5_bakeoff.sh` or `scripts/run_habitat_iter_subset.sh` (see `docs/habitat/vlm_bakeoff.md`).

---

## Docs to read

- [docs/habitat/README.md](docs/habitat/README.md) — quick start
- [docs/habitat/data.md](docs/habitat/data.md) — HM3D + **HM3D-Semantics** (why 37/113 have GT labels)
- [docs/habitat/vlm_bakeoff.md](docs/habitat/vlm_bakeoff.md) — reproduce bake-off
- [docs/plans/fable5-dynagraph-habitat.md](docs/plans/fable5-dynagraph-habitat.md) — failure analysis + backlog
- [docs/environment_variables.md](docs/environment_variables.md) — `EMET_SIGLIP_*`, `EMET_VLM_FRONTIER_SCORING`
