# Full-113 sweep (2026-08-13) — log analysis & findings

Run: `hmeqa-paper113-d1`, OUT `~/runs/emet/hmeqa_paper113/20260813_104004`,
commit `8fb7c1a5`, Qwen3-VL-8B, July nav stack. Both methods ran all 113, **0 crashes,
0 OOM, 0 flash-attn errors** (after the `--debug-run-tag` + `EMET_ALLOW_SDPA_ATTN=1` fixes).

## Results

| Method | Accuracy | Correct |
|--------|----------|---------|
| dynagraph | **44.2%** | 50/113 |
| static_graph | **37.2%** | 42/113 |

- Dynagraph **+7 pp over static_graph** on the full 113 — the memory gains hold at scale.
- Mean planning steps 50.1 (dynagraph); right (48.6) vs wrong (51.3) steps are near-equal, so failures are not budget-exhaustion.
- Gap to GraphEQA API VLMs (63.5–67.0%) is ~19–23 pp (local 8B + partial semantics).

## What we learned

### 1. Two methods disagree on 42/113 (37%) — only 62.8% consensus
`both right 25 · dyna-only 25 · sg-only 17 · both wrong 46`. The methods are not
interchangeable; the letter often depends on which evidence path wins.

### 2. Question-type accuracy is very uneven (dynagraph)
| Type | Accuracy |
|------|----------|
| existence | **72.0%** (18/25) |
| state | 37.9% (11/29) |
| count | 37.5% (6/16) |
| location | 35.1% (13/37) |
| identification | 33.3% (2/6) |

Existence (yes/no "is X there") is 2× better than spatial/counting questions — the
hard cases are **location, state, and count**, which need spatial reasoning over the
graph, not just presence detection.

### 3. Semantics gap matters more for static_graph than dynagraph
- 37/113 scenes have `.semantic.glb` (the documented ~1/3 coverage); 76 do not.
- dynagraph: 48.6% (sem) vs 42.1% (no-sem) — **+6.5 pp** from semantics.
- static_graph: 45.9% (sem) vs 32.9% (no-sem) — **+13 pp**.
- **Dynagraph's memory partially compensates for missing GT semantics** — a robustness
  advantage worth stating in the paper.

### 4. BUG (fixed): `[memory-location]` override clobbered confident correct VLM letters
**11 dynagraph + 14 static_graph episodes** had the model's JSON `"answer"` letter
== gold but the harness scored a **different wrong letter** via the
`[memory-location]` geometric override. Root cause: the equipment-distance guess
(`_equipment_letter_from_target_distances`) overrode the VLM letter even when the
VLM was confident and image-grounded. The documented "clear VLM A–D must win" rule
was only enforced for nearest-furniture memory, not the equipment branch.

**Fix (commit `f416c528`):** gate `equip_letter` on `not vlm_clear` (confident VLM
letter with a parsed letter wins). Image-landmark overrides stay (they are
image-grounded and intended to correct memory-steered letters). Regression test added.

**Config-gated (2026-08-13, commit after `f416c528`):**
- `eqa.location_override_equip_gate` (default true) — equip override gated on VLM
  confidence; env `EMET_EQA_LOCATION_OVERRIDE_EQUIP_GATE=0` restores legacy.
- `eqa.location_override_image_gate` (default **true**, flipped 2026-08-14) — image-label
  mapping gated on VLM confidence; env `EMET_EQA_LOCATION_OVERRIDE_IMAGE_GATE=0` restores
  legacy. **Live re-run correction:** the 2026-08-14 live sweep with equip-gate-only was a
  null-op (+2 pp vs baseline) — the offline A/B attributed all recoveries to the equip
  branch, but the image branch is actually the MAIN offender for the q44/q14/q25/q41/q47
  class (VLM confident + correct, Image-1 label mapping replaced the letter). Both gates
  now default on so a confident VLM letter is never clobbered.
- Legacy reference config: `configs/benchmarks/hmeqa_legacy_location_override.yaml`.
- Offline A/B (no GPU): `uv run python scripts/hmeqa_override_ab.py <jsonl>`.
- JSON `"answer":"X"` parsing is always on in `_answer_field_lines` (not a config
  toggle; the old line-start-only regex missed it in 53/107 episodes — latent,
  ~0 scored impact in this run).

**Offline-verified impact (2026-08-13, `scripts/hmeqa_override_ab.py` on the saved jsonl):**
| Method | as-scored | equip+image-gated (fix) | recovered |
|--------|-----------|-------------------------|-----------|
| dynagraph | 44.2% (50/113) | **52.2%** (59/113) | +9 (q14,24,25,31,39,44,47,94,101) |
| static_graph | 37.2% (42/113) | **44.2%** (50/113) | +8 (q21,25,44,47,67,85,104,105) |

**Live re-run (2026-08-14, `20260814_121450`, equip-gate-only — cancelled at 95/113):**
dynagraph **45.3%** (43/95) — the equip-gate-only fix was a **null-op** vs the 44.2%
baseline because the image branch (ungated) kept overriding the confident-correct VLM
letters. Partial recovery set still overridden: q14/q25/q41/q44/q47 (VLM == gold,
confident). Root cause: the offline A/B could not distinguish equip vs image branch
from the jsonl. **Fix:** `location_override_image_gate` now defaults **true**; a fresh
sweep should realize the ~50%+ recovery.

The override-on-confident-VLM bug cost ~8–10 pp on each method. (Recorded
`model_confident` is lowered by the graph-coverage gate, so the A/B reads the raw JSON
confidence field.)

### 5. Latent (not scored): JSON `"answer"` field parse gap — fixed
`_answer_field_lines` previously matched only line-start `answer:`, but Qwen3-VL emits
JSON (`"answer":"X"`). The parser missed the JSON field in **53/107** episodes; the
runner's fallback recovered most (only 6 different letters; none JSON==gold → **no
scored impact**). Hardened to always match JSON keys (always-on; not a config toggle).

### 6. Empty answers (6/113) are genuine abstains, not parse bugs
`q36,q58,q82,q84,q92,q103` returned no letter. The VLM reasoning says "None of the
images show X" (target not surfaced in attached views/graph) — a **recall/search
failure**, not a scoring bug. q103 is the one case where the model's JSON `"answer":"A"`
was dropped (overlaps with the JSON-parse gap); the rest genuinely abstained.

### 7. Letter errors are near-random (no systematic bias)
Wrong-pred letter distribution ≈ random (A:15, B:18, C:11, D:13 vs ~15.8 expected
over 63 wrong), gold is balanced (A:28/B:30/C:27/D:28) — wrong answers are guess-level,
so a better scoring/override policy (items 4–5) is the lever, not rebalancing.

## What to improve (priority)

1. **Adopt the override fix + re-run the full-113** to confirm the offline-verified
   44.2%→52.2% (dynagraph) / 37.2%→44.2% (static_graph) gains on a fresh sweep. The
   fix is default-on; `EMET_EQA_LOCATION_OVERRIDE_EQUIP_GATE=0` reproduces legacy.
2. **Improve location/state/count reasoning** (item 2): presence works (72%), spatial
   questions don't. Investigate graph-context prompts / spatial-RAG for those types.
3. **Download the missing HM3D semantics** (76 scenes) and re-run — helps static_graph
   more (+13 pp) than dynagraph (+6.5 pp); narrows the comparability gap to GraphEQA.
4. **Harden `_answer_field_lines` for JSON** (item 5) — done (always-on JSON key
   match in `emet.habitat.metrics`), latent/low priority but cheap.
5. **Quantify the methods' divergence** (item 1) in the paper — 37% disagreement is a
   strong "methods are complementary" / ensembling signal.

## Anything broken?

- **Fixed:** `[memory-location]` override (real scoring bug, ~10 pp).
- **Fixed earlier:** `run-batch --debug-run-tag` crash; missing `EMET_ALLOW_SDPA_ATTN=1`
  made the run-batch fail every episode.
- **Not broken but noted:** semantic-glb path noise (76 scenes missing — expected);
  6 empty abstains (search-limited, not crashes); JSON-parse gap (latent, no score
  impact in this run).
