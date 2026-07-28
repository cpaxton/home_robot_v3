# Habitat HM-EQA VLM bake-off (reproduction)

Reproduce the **canonical-6 model comparison** and the **balanced-31 winner run**
described in the paper appendix ([`06_model_choice.tex`](../../paper/sections/appendix/06_model_choice.tex))
and the engineering log [`docs/plans/fable5-dynagraph-habitat.md`](../plans/fable5-dynagraph-habitat.md).

All runs use **Dynagraph** with **MCQ answer debiasing** (free-form match + choice-rotation
fallback; enabled in `DynagraphController`, not in baseline `graph_eqa`).

## Prerequisites

1. **Habitat harness** — follow [install.md](install.md) (`.venv-habitat`, micromamba,
   `habitat-sim` from `aihabitat-nightly`).
2. **HM-EQA data** — [data.md](data.md): question CSVs + HM3D **train** scenes for the
   question ids you will run.
3. **GPU** — one **exclusive** ~24 GB CUDA GPU. Co-tenant processes cause OOM stubs that
   `--resume` will retry forever if the VLM never loads.
4. **Qwen3.5 fast kernels** — `fla-core` + `flash-linear-attention` ship in
   [`requirements-pip.txt`](../../packages/emet_habitat/requirements-pip.txt) and are
   verified by `./scripts/install_habitat.sh`. Without them, transformers silently uses
   slower PyTorch Gated DeltaNet ops and Qwen3.5 episodes can exceed multi-hour timeouts.
   Re-run the install script (or `uv pip install --python .venv-habitat/bin/python -r packages/emet_habitat/requirements-pip.txt`) if `import fla.ops.gated_delta_rule` fails.

## Question sets

| Set | Question ids | Notes |
|-----|----------------|-------|
| **Canonical-6** | `3,14,17,28,35,81` | Bake-off comparison set (2026-06-12). |
| **Balanced-31** | see `IDS_BALANCED` in `scripts/run_fable5_bakeoff.sh` | Winner promotion set; q31 and q94 excluded (see plan doc). |

**Excluded ids (all candidates):**

- **q31** — never finishes within budget on int4 8B/9B (20 EQA iters every attempt); 0/16
  historically on the 3B control.
- **q94** — scene-graph blowup (600+ nodes) starves the EQA loop → empty-answer stubs.

## Smoke-test a VLM (no Habitat)

Verify a checkpoint loads and answers one tiny multimodal prompt:

```bash
uv run python scripts/smoke_vl_model.py qwen3_vl Qwen/Qwen3-VL-8B-Instruct int4
uv run python scripts/smoke_vl_model.py qwen3_5 Qwen/Qwen3.5-9B int4
uv run python scripts/smoke_vl_model.py gemma4 google/gemma-4-E4B-it int4
# Control: int4 unstable for Qwen2.5-VL-3B in our stack — use bf16 / omit quant arg
uv run python scripts/smoke_vl_model.py qwen2_5_vl Qwen/Qwen2.5-VL-3B-Instruct
```

Supported `vl_family` values: `qwen3_vl`, `qwen3_5`, `qwen2_5_vl`, `gemma4` (see
`src/emet/llms/vllm_registry.py`).

## Run one candidate on canonical-6

[`scripts/run_habitat_iter_subset.sh`](../../scripts/run_habitat_iter_subset.sh) wraps
`emet-habitat run-batch` with resume and a timeout:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IDS="3,14,17,28,35,81" \
METHOD=dynagraph \
TIMEOUT=14400 \
FAMILY=qwen3_vl \
HF_ID="Qwen/Qwen3-VL-8B-Instruct" \
TAG=fable5_bake_q3vl8b \
./scripts/run_habitat_iter_subset.sh
```

Results: `~/.cache/habitat_eqa/results/subset_<TAG>_<FAMILY>.jsonl`  
Episode bundles: `~/.cache/habitat_eqa/episodes/subset_<TAG>_<FAMILY>/`

Repeat with `FAMILY` / `HF_ID` for each candidate (see table below).

## Full bake-off orchestrator

[`scripts/run_fable5_bakeoff.sh`](../../scripts/run_fable5_bakeoff.sh) runs all candidates
on canonical-6, picks the winner by correct count (tie-break: earlier candidate in list),
then promotes the winner to balanced-31:

```bash
# Exclusive GPU; 4h per-attempt timeout for slow episodes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup env TIMEOUT=14400 ./scripts/run_fable5_bakeoff.sh \
  >> /tmp/fable5_bakeoff.nohup.out 2>&1 &
```

Logs: `~/.cache/habitat_eqa/overnight/bakeoff_<run_id>/`  
Skip a stuck phase: `SKIP_PHASES=gemma4e4b ./scripts/run_fable5_bakeoff.sh`

### Candidates (2026-06 bake-off)

| Phase tag | `FAMILY` | Hugging Face id | Precision |
|-----------|----------|-----------------|-----------|
| `q35_9b` | `qwen3_5` | `Qwen/Qwen3.5-9B` | int4 |
| `q3vl8b` | `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct` | int4 |
| `gemma4e4b` | `gemma4` | `google/gemma-4-E4B-it` | int4 |
| `q25vl3b` | `qwen2_5_vl` | `Qwen/Qwen2.5-VL-3B-Instruct` | bf16 |

Shared stack flags (via the subset script): `--frontier-nodes`,
`--frontier-keyword-weight 2`, `--max-planning-steps 20`, `--max-movement-step 10`,
`--device cuda`.

Optional encoder experiments (default **off**; do not mix into the bake-off without
documenting): `EMET_SIGLIP_DTYPE=bfloat16`, `EMET_SIGLIP_VERSION=siglip2_so400m`,
`EMET_VLM_FRONTIER_SCORING=1` — see [environment_variables.md](../environment_variables.md).

## Score a results file

Only rows with real VLM output count as complete (`episode_run_completed` in
`src/emet/habitat/metrics.py`):

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from emet.habitat.metrics import episode_run_completed

path = Path.home() / ".cache/habitat_eqa/results/subset_fable5_bake_q3vl8b_qwen3_vl.jsonl"
by = {}
for line in path.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        if episode_run_completed(r):
            by[str(r["question_id"])] = r
print(f"{sum(r['correct'] for r in by.values())}/{len(by)} correct")
for q, r in sorted(by.items(), key=lambda kv: int(kv[0])):
    print(f"  q{q}: {'OK' if r['correct'] else 'X'} pred={r.get('parsed_answer_letter')} gold={r.get('gold_answer_letter')}")
PY
```

Debias audit fields on each row: `predebias_letter`, `debias_votes` (JSON string).

## Expected outcomes (2026-06-12)

| Model | Canonical-6 | Notes |
|-------|-------------|-------|
| Qwen3-VL-8B int4 | **5/6** | Winner; only miss q17 (coverage) |
| Qwen3.5-9B int4 | 3/6 | Hallucination + counting; weak debias format compliance |
| Qwen2.5-VL-3B bf16 | 2/6 | Control |
| Gemma-4-E4B int4 | 2/5 | q14 OOM loop; phase skipped |

Balanced-31 with the 8B winner was **in progress** at last update; check
`subset_fable5_bake_winner_bal32_qwen3_vl.jsonl`.

## Wave: larger VLM (32B int4 candidate)

Default production VLM remains **Qwen3-VL-8B-Instruct int4**. During live 8B HM-EQA on an
RTX 4090 we see ~13 GiB used / ~11 GiB free, so **Qwen3-VL-32B-Instruct int4** is a
*candidate with OOM risk*, not ruled out a priori (paper appendix `06_model_choice.tex`).

Ladder (dogfood `emet`; **do not** start bal-32 until holdout-4 is healthy):

```bash
# 1) Smoke — no Habitat. Record peak VRAM; abort on OOM.
uv run python scripts/smoke_vl_model.py qwen3_vl Qwen/Qwen3-VL-32B-Instruct int4

# 2) Holdout-4 agentic only (paper-router). Prefer emet jobs via the CLI:
uv run emet eval recover --need-mib 12000
uv run emet hmeqa h2h --preset paper-router --arms agentic --ids 15,68,105,17 \
  --eqa-hf-model-id Qwen/Qwen3-VL-32B-Instruct --job-name hmeqa-hold4-32b
# Equivalent env into the script:
#   EQA_HF_MODEL_ID=Qwen/Qwen3-VL-32B-Instruct EQA_VL_FAMILY=qwen3_vl
```

**Abort rules:** OOM; EGL fail streak; episode wall ≫ 2× the 8B baseline → stop.
Optional holdout-8 only after holdout-4 looks OK; bal-32 only after that.
Fallbacks: larger `gemma4` HF id from `vllm_registry`, or `qwen3_5` ~27B only if `fla`
kernels are healthy. See also [agentic_scale.md](../experiments/agentic_scale.md)
(“Wave: larger VLM”).

## Related docs

- [usage.md](usage.md) — general Habitat EQA CLI
- [docs/plans/fable5-dynagraph-habitat.md](../plans/fable5-dynagraph-habitat.md) — failure
  analysis, debiasing, excluded questions, infrastructure bugs
- [docs/dynagraph.md](../dynagraph.md) — Dynagraph vs GraphEQA
- Paper appendix: `paper/sections/appendix/06_model_choice.tex`
- [agentic_scale.md](../experiments/agentic_scale.md) — H2H ladder + larger-VLM recipe
