# Agentic HM-EQA scale experiments

Results branch: **`exp/agentic-hmeqa-bal32-results`** (PR #81).
Balanced-32 of record: `~/runs/emet/hmeqa_agentic_bal32r2_20260726_105946`.

Goal: test whether classic vs agentic-verify Dynagraph gains hold past holdout-8.

## Ladder

| Wave | What | Status |
|------|------|--------|
| 0 | Flash-Attn in `.venv-habitat`; bundle-tag smoke; scale doc (#79) | done (SDPA fallback) |
| 1 | **Balanced-32** classic vs agentic H2H (primary) | **DONE** — agentic **11/32** vs classic **9/32** (McNemar n.s.; steps win) |
| 1b | Fail-set / explore / verify-gate fixes (q104/q105, carve, soft-recent) | done (landed on branch) |
| 2 | Paper-20 **or** annotated-semantics H2H (one night) | optional — accuracy gap on n=32 is small |
| 3 | Full 113 classic dynagraph + `graph_eqa`; agentic 113 only if claiming letter win | later (efficiency claim already supported) |

## Harness (all Habitat H2Hs)

- Method: `dynagraph`
- VLM: `Qwen/Qwen3-VL-8B-Instruct`
- `explore_when_uncovered=off`, `--no-mcq-debias`, `--memory-summary`
- Agentic: `EMET_EQA_AGENTIC_VERIFY=1`; scored bal-32 used `EMET_EQA_AGENTIC_ROUTER=0` (H2H now honors env; default still 0)
- Classic: `EMET_EQA_AGENTIC_VERIFY=0`
- Dogfood: `uv run emet hmeqa overnight` or `emet hmeqa h2h --preset paper-router`
- Always use distinct `--debug-run-tag` / `OUT/bundles/{arm}_qN` (see `scripts/run_hmeqa_agentic_h2h.sh`)
- Do **not** set `COPY_PAPER_FIGS=1` on bal-32 (overwrites holdout-8 paper figures)

## SigLIP role in agentic verify (design)

SigLIP is a **high-recall / high-false-positive** open-vocab scorer that **supports** verification — it is **not** a high-precision “object confirmed” oracle.

| Channel | Typical Habitat RGB range | Role |
|---------|---------------------------|------|
| Full-frame / dense patch (image) | three-band: **ABSENT &lt; 0.10**, **CANDIDATE [0.10, 0.12)**, **PRESENT ≥ 0.12** | High-recall proposal; ABSENT = true-negative *for this view* (move on), not scene-level absence |
| Voxel per-point (DynaMem `verify_point`) | bar **0.21** / confirm **0.28** | Stronger when dense map features exist; still not letter-level truth |

**Explicit evidence policy:** `SEARCH → APPROACH → VERIFY → ASSESS → REPLAN → ANSWER`. `VERIFY` accepts exactly one fresh observation produced by `APPROACH`; stale router requests are rejected. `EvidenceRecord` retains full-frame, dense, voxel, detector, detector-crop, graph-label, geometry, and optional VLM channels. Image SigLIP / OWLv2 `PRESENT` is only a **proposal**. Answerability is **VLM-first**: text Qwen picks `target_phrase` once; multimodal Qwen assess on fresh RGB + inventory sets `answerable` / `ANSWER`. Cheap fusion never opens the submit gate alone. Non-advancing captures (`NO_NEW_OBS`) and one VLM assess per `obs_id` block re-verify spam.

**Region-aware exploration:** frontier clusters are navigation *regions*, ranked by expected area gain per unit travel (`frontier_regions.frontier_region_utility`) rather than nearest-cell distance, so a large room several meters away beats a sliver underfoot. Frontier `GraphNode`s carry `frontier_cell_count` and `frontier_keyword_score` from clustering. After two consecutive "target not visible" view assessments the executor sets an escape floor (`agent._explore_min_travel_m = 3 m`) that demotes nearby regions and disables SigLIP-guided frontier candidates, which otherwise re-aim at the area just rejected.

**Do not:** treat ABSENT as proof of absence; use image-space 0.21 as a hard PRESENT bar (unreachable on HM-EQA RGB); force-submit after failed verifies and call that “verified”; open submit from OWL/SigLIP alone. Prefer `EMET_EQA_AGENTIC_REQUIRE_VERIFIED=1` while tuning so unverified exhaust **abstains**. Offline frame calib: `scripts/calibrate_agentic_verify_frames.py`; threshold sweeps: `scripts/tune_agentic_verify.py`.

**Hybrid bakeoff (2026-07-25):** the reproducible saved-frame pass scored 24 verify views with SigLIP, SigLIP2, OWLv2, YoloE, and detector→crop→SigLIP. Mean per-view latency was 49.8, 32.4, 60.6, and 234.6 ms respectively. Those legacy frames have no semantic-sensor masks (`n_labeled=0`), so this is a latency/score-distribution result, **not** a precision claim. OWLv2 is the live-probe backend. New traces attach HM3D semantic view visibility, pixel fraction, bbox, and range so scene-disjoint PR curves become valid.

Artifacts and commands:

```bash
uv run python scripts/build_agentic_decision_dataset.py ~/.cache/habitat_eqa/episodes \
  -o ~/runs/emet/hmeqa_decisions.jsonl
uv run python scripts/run_agentic_verifier_bakeoff.py ~/runs/emet/hmeqa_decisions.jsonl \
  -o ~/runs/emet/hmeqa_verifier_bakeoff --methods siglip1,siglip2,owlv2,yoloe \
  --detector-crop-siglip
uv run python scripts/summarize_agentic_ladder.py RUN_DIR \
  --require-balanced32-gate
```

The balanced-32 gate requires at least four probe episodes, nonzero fused verified-answer rate, and zero forced submits.

## Balanced-32 IDs

Same letter-balanced set as overnight scripts:

```text
2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84
```

## Commands

```bash
# Preferred: overnight ladder (holdout gate → bal-32) or direct H2H
uv run emet eval recover --need-mib 12000
uv run emet hmeqa overnight
# or:
OUT=~/runs/emet/hmeqa_agentic_bal32_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT" --preset paper-router --job-name hmeqa-bal32

uv run emet jobs
uv run emet hmeqa resume "$OUT" --preset paper-router

# After DONE: minimal paper data (+ significance)
uv run python scripts/summarize_hmeqa_agentic_h2h.py "$OUT"
uv run python scripts/hmeqa_significance.py "$OUT"
cp "$OUT/h2h_summary.json" paper/data/hmeqa_agentic_h2h/balanced32_summary.json
```

## Go / no-go (after Wave 1)

- **Result (2026-07-26, bal-32r2):** classic **9/32**, agentic **11/32**, mean steps **48.7 → 17.8**. McNemar p≈0.73 (letter gap n.s.); Wilcoxon steps p≈4e-7.
- **Replicate (overnight bal-32):** classic **10/32**, agentic **12/32** — same pattern.
- **Historical (2026-07-24 salvage bug):** classic 12/32, agentic 9/32 under forced 64-token answers — superseded; do not cite as current method.
- **Go for efficiency claim** (agentic uses ~3× fewer planning steps on matched Dynagraph). **No-go for letter-accuracy claim** on n=32 without a larger / semantics-richer slice.
- Wave 2/3 agentic-113 for *accuracy* remains optional until a clearer letter win; classic-113 for the paper baseline is still useful.

## Fail-set ablation (Wave 1b)

After removing the 64-token clamp (+ traces in episode bundles):

```bash
OUT=~/runs/emet/hmeqa_agentic_failset_$(date +%Y%m%d_%H%M%S)
# Holdout-8 regression + classic_only recovery IDs
IDS=15,56,65,68,79,88,104,105,11,14,28,39,47
uv run emet jobs run --name hmeqa-failset-a1 --out-dir "$OUT" -- \
  env EMET_ALLOW_SDPA_ATTN=1 EMET_EQA_TRACE=1 SKIP_KILL_STALE=1 SKIP_GPU_WAIT=1 \
  ARMS=classic,agentic HOLDOUT_IDS="$IDS" \
  ./scripts/run_hmeqa_agentic_h2h.sh "$OUT"
```

Pass: agentic recovers ≥4/5 classic_only and holdout-8 ≥7/8.

### Fail-set regressions found (2026-07-24)

Holdout-8 overnight was **agentic 8/8 vs classic 5/8**. Re-running agentic after removing the 64-token clamp regressed until:

1. **NL choice text → false abstain → wrong `[memory-location]`** (q56): VLM answered “The room with the blue curtains” (=C) but letter parse only looked for bare `A–D`, so nearest-furniture memory overwrote to **A**. Fix: use `extract_mcq_letter(answer, choices)` before override; normalize matched choice text to a letter (`[choice-text]`).
2. **`Answer: Unknown` skipped salvage** (q65): non-empty Unknown blocked the empty-answer salvage path that night’s 64-token truncations used. Fix: treat Unknown/none as emptyish for salvage.
3. **`libcuda` SIGSEGV** (orchestrator `exit=139`; kernel `segfault … in libcuda.so`): Habitat-Sim EGL + Qwen3-VL **vision** generate on one GPU. Hot scenes: `00167-yogvKWUrdnw` (q104/q105), flaky on `00094-WT4QWwXrMzs` (q68). Log ends with `timeout: the monitored command dumped core`; `agentic_qN.jsonl` stays empty. Mitigations: `torch.cuda.synchronize()` before multimodal generate in `qwen3_vl_client`; H2H `NATIVE_CRASH_ABORT=1` (default) stops the batch and writes `native_crash_<arm>_q<ID>.log`. Still flaky under `EMET_ALLOW_SDPA_ATTN=1` + int4. Distinct from Cursor/`emet` null-IP crashes — see [known_issues.md](../known_issues.md#nvidia-driver-hang--cursor-agent-crash-during-stacked-gpu-evals).
4. **Host hard freeze** (2026-07-25 bal-32 classic q48 at 26/64): whole machine dies mid-VLM decode; journal stops with no Xid/oops; empty jsonl + NUL-padded log. Incomplete `taskset -c 0-7,10-31` still left the second 6.0 GHz P-core (CPUs 10–11) online. H2H now auto-excludes all ≥6000 MHz CPUs via `emet.utils.cpu_affinity` and defaults `EPISODE_COOLDOWN_SEC=20`. See [known_issues.md Mode C](../known_issues.md#mode-c--host-hard-freeze--forced-reboot-2026-07-25) and repo-root `segfault.md`.
4. **Empty letters + `n_object=0` on q104/q105 (failfix4):** an experiment tied `EMET_EQA_AGENTIC_VERIFY=1` to skipping per-frame VLM graph label extract. On scenes without HM3D semantics that left only `["object"]` → nav samples/frontiers; answer prompts attached black 8×8 frontier placeholders; location MCQ abstain → `pred=""`. **failfix5** (`20260724_144029_f7ace2`): `n_object=8/8`, non-empty letters, 33× `max_new=128` vision extracts — but letters still wrong (B/B vs gold D/A). Checked-in: `paper/data/hmeqa_agentic_h2h/failset104105_summary.json`. Guards: never answer off frontier placeholders; phrase ranking for SigLIP; Action Image N via graph `obs_id`; agentic verify must not auto-disable label VLM.
5. **failfix5 wrong B letters (grounding):** truncated VLM (no `answer:`) → empty → `[memory-location]` invented B; agentic full MCQ string made SigLIP phrases like `table sunroom answer` instead of `fruit bowl`. Fix: `question_stem_for_keywords()` strips `A)…Answer:` before phrase/object heuristics; location MCQ empty/Unknown skips memory-location and salvage-location invent (normalize empty → `Unknown`).
6. **failfix6 round waste (fallback policy):** with the router off, `_fallback_tool` allowed only one `explore_frontier` ever, so once nav hypotheses were consumed it re-ran an identical ABSENT `verify_siglip` for 5/8 rounds and the final submit came from the `budget_hit` path, which skipped the Action:N follow entirely. The default verify phrase also ranked MCQ-option nouns (`kitchen island`) above stem phrases (`fruit bowl`) because VLM keyword extract feeds option nouns into `_relevant_objects`. Fixes: fallback explores while nav budget remains (stop when motion happened and frontiers are gone), submits as soon as budget is spent and a verify is on record; `budget_hit` submit honors one Action:N/unknown-explore follow-up; verify phrase ranking prefers phrases occurring in the question stem. Guards in `src/test/eval/test_agentic_eqa_verification.py` (`test_fallback_*`, `test_verify_phrase_prefers_question_stem_over_mcq_option`).

7. **Nearest-first frontier creep (q104/q105, 2026-07-25):** with the VLM verify gate in place, both remaining holdout failures were *search* failures, not verifier failures. The robot spawned in a yard on `00167-yogvKWUrdnw`, walked **16.7 m of path but never got further than 4.4 m from spawn**, covered ~30 m², and mapped only outdoor labels (`brick patio`, `coiled hose`, `stacked wood`) while the house interior stayed untouched. Cause: `_upsert_frontier_nodes` clusters the unexplored mask and scores clusters by area + keyword affinity, then selection threw that away — `_frontier_explore_sort_key` ranked purely on `dist + penalty`, with a 1.25 m recent-goal radius as the only anti-revisit. `_siglip_guided_frontier` made it worse by aiming at the frontier nearest the best-matching **already observed** point, which is inside the area already rejected. Fix: [`frontier_regions.py`](../../src/emet/memory/graph_eqa/frontier_regions.py) ranks regions by `sqrt(area) / (1 + dist/4 m)` with keyword bonus and nav-failure decay; `GraphNode` carries `frontier_cell_count` / `frontier_keyword_score`; after `NOT_PRESENT_ESCAPE_STREAK` (2) consecutive not-visible view assessments the executor publishes `agent._explore_min_travel_m = 3.0 m`, which demotes nearby regions and suppresses the SigLIP candidate so the robot commits to leaving. Nodes without cluster metadata score on proximity alone, so MuJoCo/Molmo ordering is unchanged. Guards: `src/test/memory/test_frontier_regions.py`, `test_not_present_streak_sets_escape_floor`.

Keep the 64-token clamp **removed** (bal-32 salvage bug); do not reintroduce `setdefault("64")`. Do **not** reintroduce agentic-verify auto-skip of `SensorGraphBuilder` VLM extract. Do **not** restore nearest-first frontier sorting: it is what kept q104/q105 circling their spawn.

## Wave 0 notes (2026-07-23)

- Bundle-tag smoke: `h2h_smoke_classic_q0015` → `~/.cache/habitat_eqa/episodes/h2h_smoke_classic_q0015/q0015_dynagraph/` (Q15 classic correct, 60 steps).
- **Flash-Attn in `.venv-habitat`:** blocked — system CUDA toolkit is 12.4 while Habitat torch is `2.12.0+cu130`. Main `.venv` already has flash-attn 2.8.3. Habitat H2Hs use `EMET_ALLOW_SDPA_ATTN=1` until toolkit/torch align (or a prebuilt cu130 wheel is available).

## Segfault modes (2026-07-24)

Do **not** conflate:

1. **Episode `libcuda` SIGSEGV (`exit=139`)** — Habitat EGL + Qwen3-VL vision generate. Hot: q104/q105 (`yogvKWUrdnw`), flaky q68. Empty `agentic_qN.jsonl` + `dumped core` in the episode log. Prefer `emet jobs`; abort/retry via Habitat H2H `NATIVE_CRASH_ABORT`.
2. **Cursor / `emet` null-IP SIGSEGV** — agent turn touches Habitat/GPU teardown; detached job may still finish.

Full write-up: [known_issues.md](../known_issues.md#nvidia-driver-hang--cursor-agent-crash-during-stacked-gpu-evals). Keep the Habitat results-branch `docs/experiments/agentic_scale.md` as the detailed failset log.

## Related

- Holdout results: [habitat_eqa_results.md](habitat_eqa_results.md) (Classic vs agentic-verify)
- Minimal checked-in summaries: [`paper/data/hmeqa_agentic_h2h/`](../../paper/data/hmeqa_agentic_h2h/), [`paper/data/README.md`](../../paper/data/README.md)
- Do not chain Robocasa + full pytest + Habitat VLM in one session.
