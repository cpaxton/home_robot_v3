# Dynagraph Habitat HM-EQA: results review + improvement plan

Date: 2026-06-10. Branch: `feature/dynagraph-exploration`.
Goal: improve dynagraph exploration/answering on HM-EQA in Habitat as much as possible,
keeping `graph_eqa` as a clean baseline. VLM: Qwen2.5-VL-3B-Instruct.

## 1. Results so far (clean runs only)

| Eval | dynagraph | graph_eqa | notes |
|---|---|---|---|
| canonical-8 (`subset_cmp_*`) | **4/8** | 3/8 | post repetition-stop fixes |
| balanced-32 (`subset_bal32_*`) | **11/32 (34%)** | 10/32 (31%) | largest comparable run |
| short re-validation (Q3,Q14) | 1/2 | 1/2 | clean GPU sanity check |
| overnight `20260610_044718` | n/a | n/a | invalid — GPU contention OOM, VLM never ran (`eqa_iterations=0`) |

Random chance is 25%; with 2 "(Do not choose this option)" placeholders on yes/no
questions effective chance is ~40% on those. Both methods are barely above chance.
Dynagraph's wins concentrate on **search-style questions** (Q17 woven basket, Q18
striped towel, Q32 entrance mats, Q38 decorative owl) — SigLIP-guided frontier
exploration is doing its job. It gives ground back on counting and time questions.

## 2. Failure analysis (balanced-32 episode bundles)

### F1. Massive MCQ position bias against letter A — the dominant fixable error

Per-iteration answers across all 32 episodes x 20 EQA iterations:

| method | A | B | C | D |
|---|---|---|---|---|
| dynagraph | **29 (4.7%)** | 167 | 225 | 194 |
| graph_eqa | **41 (6.5%)** | 254 | 195 | 138 |

Expected ~25% each. Result: **gold=A questions score 0/8 for both methods** (a full
25% of the benchmark forfeited). Both final-answer and majority-over-iterations
voting inherit the bias (majority vote does NOT fix it: 11/32 vs 11/32 dynagraph,
9/32 vs 11/32 graph_eqa) — the bias is systematic, not sampling noise.

**Fix (implemented tonight): choice-rotation debiasing.** At episode end, re-ask the
final MCQ K=4 times with cyclically rotated choice orders (each original choice
appears at each letter position exactly once), terse "letter-only" directive +
the same selected EQA images, map each predicted letter back to the original
choice index, majority-vote. Gated to dynagraph (`mcq_debias_enabled`) so the
baseline stays untouched. Vote details recorded in metrics (`debias_votes`,
`predebias_letter`).

### F2. Confidence never triggers — every episode burns the full budget

`model_confident=False` on 64/64 bal32 episodes (post coverage-gate). The raw VLM
is sometimes confident but `_graph_covers_relevant_objects()` vetoes whenever any
question keyword is missing from observation labels (e.g. "clock", "time") — for
many questions the keyword can never appear, so the gate never opens and every
episode runs all 20 iterations (~12 min each). Not an accuracy bug per se, but it
caps overnight throughput and recycles low-quality "Confidence: FALSE" iterations
into the history. Candidate later fix: open the gate when SigLIP grounding says
PRESENT (dynagraph has this signal in CONFIRMED_MEMORY) — not done tonight.

### F3. Answer flip-flop across iterations

Q21 trace: b,d,B,b,c,B,d,c,c,c,c,c,d,c,D,D,D,d,d,d → graded on the last one.
Final-iteration answer is roughly as good as a majority vote, so we keep grading
the last answer but the debias vote (F1) replaces it at episode end for dynagraph.

### F4. Degenerate VLM action strings

"Navigate to Image navigate to image 4" (repetition-stop salvage tail) and
out-of-range "Navigate to Image 48". Both are handled (regex digit extraction +
range check → frontier fallback), so the impact is wasted iterations, not crashes.

### F5. Caption runaway on some episodes (e.g. Q84 "what time")

The VLM enumerates "Image 7..24 shows..." hallucinating dozens of captions; the
repetition stop kicks in and `[salvage]` recovers a letter. Salvage answers carry
the same A-bias — also addressed by F1's debias pass.

### F6. Infrastructure (fixed earlier today)

- OOM-stub resume bug: rows with `planning_steps>0` but no VLM output now retry
  on `--resume` (`episode_run_completed` requires EQA output artifacts).
- Overnight `20260610_044718` froze: GPU co-tenant (~13 GB) starved both phases.
  Lock removed; eval requires an exclusive GPU.

## 3. What was already validated as helping dynagraph

- SigLIP-guided frontier (`_eqa_explore_when_uncovered`) — search-question wins.
- SigLIP-grounded CONFIRMED_MEMORY prompt block (`memory_summary_enabled`).
- VLM repetition stop (shared with baseline) — ~12 min → ~2.5 min per EQA call.

## 4. Tonight's experiments

Orchestrator: `scripts/run_fable5_overnight.sh` (waits for the in-flight
canonical-8 rerun to release the GPU, then runs phases serially with `--resume`).

| Phase | tag | method | questions | purpose |
|---|---|---|---|---|
| 0 (in flight) | `canonical8_rerun_{dg,ge}` | both | canonical 8 | clean-GPU repro of cmp numbers |
| 1 | `fable5_dg_debias_c8` | dynagraph+debias | canonical 8 | debias delta vs phase 0 dg |
| 2 | `fable5_dg_debias_bal32` | dynagraph+debias | balanced 32 | headline number vs 11/32 |
| 3 (if time) | `fable5_ge_bal32_recheck` | graph_eqa | balanced 32 | same-code baseline recheck |

Success criteria: gold=A accuracy > 0 (was 0/8); bal32 dynagraph > 11/32;
no regression on non-A questions; `debias_votes` populated in JSONL.

## 5. Backlog (next after debiasing)

1. Open the confidence gate from SigLIP PRESENT grounding → early stop → more
   budget for hard questions / bigger eval sets.
2. Frontier-visit memory: penalize re-selecting frontiers within r of previously
   visited targets (Q29/Q40-style wandering).
3. Counting questions: CONFIRMED_MEMORY reports node counts; prompt the VLM to
   trust graph-node counts over image counting (Q12/Q28 both wrong for both methods).
4. "What time is it" questions need a clock close-up: add a question-type hint to
   navigate to + zoom on detected clock nodes (Q33/Q43/Q84).
5. Try Qwen2.5-VL-7B for the answering call only (keep 3B for captions): the 3B
   model's position bias and caption runaway are size-related.

## 6. Overnight launch state (2026-06-11 00:14)

- Implemented + unit-tested: `src/emet/memory/graph_eqa/mcq_debias.py`,
  `GraphEQAMemory.vote_mcq_letter`, dynagraph gate, metrics fields
  (`predebias_letter`, `debias_votes`), harness wiring. Mock-LLM smoke verified
  the vote path end-to-end (position-locked mock splits A,B,C,D across rotations
  and falls back to the prior, as designed).
- Orchestrator: `nohup WAIT_PID=<canonical8 rerun> OVERNIGHT_DEADLINE_HOURS=11
  ./scripts/run_fable5_overnight.sh` — waits for the in-flight canonical-8 rerun,
  then runs phases 1-3. Logs: `~/.cache/habitat_eqa/overnight/fable5_<run_id>/`,
  nohup tail: `/tmp/fable5_overnight.nohup.out`, summary written to
  `summary.txt` in the log dir at the end.
- Known pre-existing failure under load: `test_graph_eqa_default_scene_sim`
  pytest-timeout (>360 s) when the GPU/CPU is saturated by a running eval —
  navigation-step timeouts, unrelated to the debias change.

## 7. Results — overnight 2026-06-11 (run `fable5_20260611_001441`)

- **canonical-8 dynagraph+debias: 5/8** (best yet; prior dynagraph 4/8, baseline 3/8).
  Debias flipped Q14 C→D (correct); Q94 (gold A) answered A for the first time.
- balanced-32 dynagraph+debias: 6/14 partial (phase stalled on a long episode,
  hit the launch deadline). Gold-A: 2/2 correct across phases (was 0/8).
- Vote flips: fixed=1 broke=0; losses vs reference were exploration variance
  (same answer pre/post vote).

### Root cause of the letter-A bias (investigated 2026-06-11 morning)

Model-intrinsic MCQ selection bias in Qwen2.5-VL-3B, NOT prompt/context:
- Terse re-asks (minimal prompt, same images): position A picked **0/88**, B 64%.
- 12/22 episodes position-locked (same letter slot >=3/4 rotations) vs 5/22
  content-consistent — the model answers a letter prior, not the content.
- Iteration-0 (no history) A-rate equals later iterations — not history feedback.
- Placeholders are uniform across positions; bal32 has no placeholder-at-A.

### Tie-break bug found + fixed

4-way vote splits (position-locked, zero signal) defaulted to lowest index = A,
manufacturing 2 wrong A answers (Q2, Q27). Fixed: ties without a prior return
no override (keep the main answer).

## 8. Free-form answer + match (implemented 2026-06-11 afternoon)

`vote_mcq_letter` now asks free-form first ("answer in a few words", no letters
shown), then maps the reply to the closest choice via stopword-stripped token
Jaccard + containment (`match_freeform_to_choice` in `mcq_debias.py`); rotation
voting is the fallback when the free-form reply is ambiguous. Letter tokens are
never involved in the primary path, so position bias cannot act on it.
14 unit tests in `src/test/memory/test_mcq_debias.py`.

## 9. Model-size bake-off (launched 2026-06-11 13:09)

GPU budget: 24 GB total; the 3B-bf16 dynagraph stack peaks ~18 GB. int4 (bnb)
candidates verified loading + answering in `.venv-habitat`
(`scripts/smoke_vl_model.py`):

| candidate | int4 footprint | smoke |
|---|---|---|
| Qwen3.5-9B (natively multimodal, Feb 2026) | ~6-7 GB weights | OK ("Black." with thinking stripped; CPU-fallback smoke during GPU contention) |
| Qwen3-VL-8B-Instruct | full stack ~13 GB | OK (answered "black" correctly) |
| gemma-4-E4B-it | ~5 GB weights | OK (answered "Purple" — wrong but functional) |
| Qwen2.5-VL-3B (control) | bf16, int4-bnb broken for this arch | n/a |

**Qwen3.5 added 2026-06-11 17:54** (user pointed out the family is image+text->text):
trained from scratch on interleaved multimodal tokens, model card claims it
outperforms Qwen3-VL at the same scale on visual understanding. New `qwen3_5`
family (`Qwen35Client` subclasses `Qwen3VLClient`, swaps in
`Qwen3_5ForConditionalGeneration`, disables thinking via `enable_thinking=False`
+ defensive `<think>` strip). Bake-off restarted with it as preferred candidate
(prior phases resume from jsonl). 27B int4 (~15 GB) does not fit next to the
habitat stack; the MoE variants need even more. `fla`/`flash-attn` not installed,
so the Gated DeltaNet runs on slower PyTorch fallback kernels (correct, just
slower) — install both if qwen3_5 wins and throughput matters.

Orchestrator `scripts/run_fable5_bakeoff.sh`: canonical-6 per candidate
(dynagraph + free-form debias), winner by correct count (ties prefer earlier
candidate in list) auto-promotes to balanced-31. Tags `fable5_bake_*`; logs
`~/.cache/habitat_eqa/overnight/bakeoff_<run_id>/`. **Reproduction guide:**
[docs/habitat/vlm_bakeoff.md](../habitat/vlm_bakeoff.md). Note: VLM vision towers
cannot replace SigLIP for voxel grounding (no aligned text encoder; retrieval
needs a contrastive dual encoder) — candidate upgrades there are SigLIP 2
(drop-in) or VLM-as-frontier-scorer (<=12 candidates/iter), kept out of this
run to avoid confounds.

## 10. Encoder + localization upgrades (implemented 2026-06-11, gated OFF)

Both are env-gated and default-off so the in-flight bake-off stays clean; flip on
for a follow-up A/B run (see `docs/environment_variables.md`):

- **SigLIP 2** (`EMET_SIGLIP_VERSION=siglip2_so400m`): `siglip_encoder.py` now
  supports `siglip2_base` / `siglip2_so400m` (fixed-res checkpoints; NOT naflex).
  Verified drop-in on CPU: MaskSiglip pixel features (1152-d), text + image vecs
  all produced by `google/siglip2-so400m-patch14-384` with transformers 5.10.
- **VLM frontier scoring** (`EMET_VLM_FRONTIER_SCORING=1`): `_vlm_frontier_choice`
  in `controller_graph_eqa.py` asks the EQA VLM to pick among <=6 frontier views
  (reply = image number) before falling back to the SigLIP-nearest heuristic.
  Unit tests in `src/test/agent/test_vlm_frontier_scoring.py`.
- **SigLIP fp16/bf16** (`EMET_SIGLIP_DTYPE=bfloat16`): weights were loading fp32
  (3.5 GB); half precision halves that. Outputs cast back to fp32 (bf16-vs-fp32
  pixel-feature cosine 0.9994). int4 unsupported by design (MaskSiglip head
  surgery needs raw weight tensors; thresholds calibrated on fp32 geometry).
- **Encoder consolidation**: main's `siglip2_encoder.py` (registry `siglip2`)
  duplicated `siglip_encoder.py`; now a thin shim over `SiglipEncoder` /
  `MaskSiglipEncoder` with a unified `SIGLIP_CHECKPOINTS` table (SigLIP 1 +
  SigLIP 2 fixed-res + 512px variants), inheriting dtype support everywhere.
- **Bbox localization probe** (`scripts/probe_vlm_boxes.py`): Qwen-VL native
  grounding (absolute-pixel `bbox_2d` JSON) vs Gemma prompted boxes (Gemini
  0-1000 `[ymin,xmin,ymax,xmax]` convention — no dedicated loc tokens in Gemma
  chat models; PaliGemma 2 is the family's trained detector). Parser self-tests
  pass; run on real frames once the bake-off frees the GPU.

### Upsizing beyond 8B on a 24 GB card

- Qwen3-VL-32B int4 ≈ 17-18 GB and 30B-A3B (MoE) int4 ≈ 16 GB: do NOT fit next
  to the live habitat sim + SigLIP (~7 GB non-VLM stack). gemma-4-E8B does not exist.
- Practical next step if 8B wins: **final-answer judge** — explore/caption with
  4B/8B, then at episode end release the shared VLM
  (`release_shared_graph_eqa_vlm`) and load a 30B-class int4 model only for
  `vote_mcq_letter` (4-6 images, ~5 short calls), then release. Costs ~1-3 min
  load per episode; keeps all 24 GB for the judge at answer time.

**q31 dropped from both subsets (2026-06-12 08:19).** With int4 8B/9B models the
episode never finished: three 4h attempts (Qwen3.5-9B, fast kernels included)
all timed out mid-episode. q31 always maxes out 20 EQA iterations and is 0/16
correct across every historical run (3B included) — a pure time sink that
starved the gemma/control phases overnight. Canonical set is now 7 ids
(3,14,17,28,35,81,94), balanced set 31 ids. All candidates are compared on the
same reduced sets, so the comparison stays fair; q31 is noted as "unanswerable
within budget" rather than wrong.

**q94 also dropped (2026-06-12 13:30) — scene-graph node explosion bug found.**
Some scenes blow up the dynagraph graph: q94 hit 669 nodes / 335 obs
(645 KB scene-graph report) vs ~70 nodes typical; q35 hit 885, q28 516. The
giant graph text starves the EQA loop (0 iterations completed -> empty answer
stub -> infinite `--resume` retry; 13 stub rows). **Root-cause fix to implement
after the bake-off: cap scene-graph text in the EQA prompt to top-K
question-relevant nodes** (SigLIP-ranked), plus investigate why dedup/merge
fails in these scenes. Canonical set is now 6 ids (3,14,17,28,35,81).

## 11. Bake-off results (2026-06-12)

**Winner: Qwen3-VL-8B-Instruct int4** (`qwen3_vl`). Canonical-6, dynagraph + debias.
Gold letters: q3=B, q14=D, q17=D, q28=D, q35=D, q81=D.

| Model | Canonical-6 | Δ vs 3B | Per-question (q3 q14 q17 q28 q35 q81) |
|---|---|---|---|
| **Qwen3-VL-8B int4** | **5/6** | **+3** | ✓ ✓ ✗ ✓ ✓ ✓ |
| Qwen3.5-9B int4 | 3/6 | +1 | ✓ ✓ ✗ ✗ ✓ ✗ |
| Qwen2.5-VL-3B bf16 | 2/6 | — | ✗ ✓ ✗ ✗ ✗ ✓ |
| Gemma-4-E4B int4 | 2/5 | +0 | ✓ — ✗ ✗ ✗ ✓ (q14 OOM) |

**Reproduction:** [docs/habitat/vlm_bakeoff.md](../habitat/vlm_bakeoff.md)  
**Paper:** `paper/sections/appendix/06_model_choice.tex`

### Why Qwen3.5 underperformed (surprising vs benchmarks)

Qwen3.5-9B reports stronger static VQA than Qwen3-VL at similar scale, yet lost
embodied MCQ EQA 3/6 vs 5/6. The gap is task-specific, not parameter count:

1. **Hallucination under confidence (q81)** — 9B asserted a sleeping child on a red bed
   with high confidence; no person in any frame. 8B answered No correctly.
2. **Literal counting (q28)** — 9B free-form: "one red pillow"; 8B: "two red pillows".
3. **Debias format drift** — 9B replies "Caption:…" to the free-form debias query
   despite "do not caption", disabling the strong matcher; falls back to rotation
   voting with position-locked A,B,C,D pattern (same failure mode as 3B).
4. **Shared coverage miss (q17)** — neither Qwen model saw the woven basket.

Upsizing 3B→8B on the **same pipeline** recovered +3 correct — the biggest gain we
measured; the prior 3B ceiling was model-limited on this slice, not exploration-limited.

### Balanced-31 winner run

Promoted 2026-06-12 18:01. Output:
`~/.cache/habitat_eqa/results/subset_fable5_bake_winner_bal32_qwen3_vl.jsonl`  
Early progress: 1/2 episodes (q6 ✓, q2 ✗) — full set in progress overnight.

### Infrastructure notes (affect all models)

- **q31 / q94 excluded** from comparison sets (timeout / graph blowup); see §11 above.
- **Gemma q14 OOM** — deterministic after load; phase skipped when ceiling < 8B score.
- **fla-core** required for fair Qwen3.5 wall-clock (`packages/emet_habitat/requirements-pip.txt`).
- **TODO post-bake-off:** cap scene-graph text in EQA prompt (top-K SigLIP-ranked nodes).
