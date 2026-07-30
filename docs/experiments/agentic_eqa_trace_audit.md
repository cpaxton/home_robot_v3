# Agentic EQA trace audit (2026-07-29)

Full-corpus audit of the HM-EQA agentic tool loop: what the traces say is working, what is
luck, why episodes fail, and what to do next.

**Corpus:** 76 episodes with traces across 5 named runs, plus 5 additional bal-32 agentic runs
and 5 classic runs used for replication and baselining. Every question is 4-choice, so the
uniform-random floor is **25%**.

**Bottom line.** Pooled across six independent bal-32 runs the agentic arm scores **0.379**
against a classic baseline of **0.338** — an advantage of about **4 points**, not the 22 points
the single best run suggests. Roughly **45% of wins require no real search**, exploration volume
does not predict correctness at all, and — the most consequential finding — **whether the target
was actually in view does not predict correctness either** (0.355 in-view vs 0.333 never-in-view).
The loop's competence is concentrated in the first 1-3 rounds; past round 7 it scores below chance.

---

## 1. Runs audited

| Run | Dir | ids | Verifier | Confirm gate | Checkout |
|-----|-----|-----|----------|--------------|----------|
| P1 | `hmeqa_rooms_verify_probe_20260729_121035` | 11 | none | absent (pre-dates gate) | v3 dirty |
| P2 | `hmeqa_rooms_verify_probe_confirm_20260729_164006` | 11 | none | ON | v3 dirty |
| A | `hmeqa_rooms_owl_restore_20260729_182708` | 11 | owlv2 | ON | v3 dirty, **mixed** |
| B | `hmeqa_rooms_owl_noconfirm_20260729_185718` | 11 | owlv2 | OFF | v3 dirty |
| C | `hmeqa_bal32_explore_20260727_215036` | 32 | owlv2 | machinery absent | **v4** `e7d0eb20` |

All five are complete and parseable; no missing or corrupt files. Shared config across all:
`REQUIRE_VERIFIED=0`, `ROUTER=1`, `ARMS=agentic`.

### Measurement hazards found in the process

These invalidate several comparisons that were being drawn informally, and must be fixed before
the next sweep.

- **The working tree was dirty during every v3 run.** All four report `git_commit=2b073cd5`
  while running four *different* code states. `agentic_policy.py` and `agentic_tools.py` were
  edited at 16:31-16:33 (between P1 and P2); `agentic_eqa.py` and `runner.py` were edited at
  18:30-18:33 — **during run A**, between its q6 and q11 episodes. Run A is a mixed-code run and
  cannot be treated as one configuration.
- **P1 vs P2 is not a single-flag pair.** The entire answerable-confirm gate was added between
  them, confirmed by telemetry: `answerable_confirmed` / `answerable_deferred` appear 0 times in
  P1 and 53 times in P2.
- **A vs C is the only clean pair** (byte-identical env flags, differing only by checkout) and it
  shows the largest gap: 3/11 vs 7/11. **The dominant variable across this whole set is code
  state, not the flags being swept.**
- **P1's first pass failed all 11 episodes** with `exit=1` and was silently re-run under
  `RESUME=1`, which restored zero units. Its scored data is entirely from the second pass.
- At n=11, a 3-vs-4 difference is one question and is inside binomial noise.

---

## 2. What is working (protect this)

### Fast-path answering at rounds 0-1

| n_rounds | n | correct | acc |
|---|---|---|---|
| 1 | 23 | 13 | **0.565** |
| 2-3 | 15 | 7 | 0.467 |
| 4-7 | 12 | 4 | 0.333 |
| 8 (budget cap) | 26 | 5 | **0.192** |

**13 of 29 wins (45%) come from a single `investigate` → one `vlm_assess` with `present=true`
→ `submit_answer`.** The canonical case is C/q6 ("Where is the ladder?"): 1.75 m of navigation,
one assess, submit — correct in **all 6** independent runs.

Rounds 1-7 are a flat plateau near 0.45-0.50. The cliff is the budget cap, where accuracy falls
to 0.192, *below chance*.

### Replication separates robust wins from noise

Run C's 32 ids were also run by 5 other agentic runs, giving a direct replication test.

| Class | n | median rounds | budget_hit | replication |
|---|---|---|---|---|
| **Robust** (q6, 12, 15, 21, 34, 40, 49) | 7 | **1** | **0/7** | 5-6 of 6 runs |
| **Fragile** (q8, 18, 25, 27, 38, 48, 57, 80, 84) | 9 | **7** | 4/9 | 1-3 of 6 runs |

All 7 robust wins finished in ≤3 rounds; only 3 of 9 fragile wins did (Fisher exact **p=0.0114**).
**Only 7 of run C's 16 wins are reproducible.**

### `vlm_assess present==true` is the one surviving predictor

On submitted episodes: **0.583 vs 0.308** (Fisher **p=0.0411**). It is the only feature that
survives excluding abstains. Note carefully what it is *not* — see §5.

### Answering at all

All 14 abstains are structurally identical (`n_rounds=8`, `budget_hit=true`, `answer_source=null`)
and score **0/14**. Run C's zero-abstain behavior is a large part of why it looks best.

---

## 3. What is luck (stop crediting it)

- **Run C's 16/32 is the top of six draws.** Pooled agentic rate is 0.379; P(≥16/32) = **0.111**.
  Anchoring on C overstates capability by roughly 12 points.
- **The agentic-vs-classic gap is +4.1 points, not +22.** Pooled: agentic 69/182 = 0.379,
  classic 54/160 = 0.338. On the single best pairing McNemar gives p=0.0923 — suggestive, not
  significant.
- **Every round-8 budget-hit win fails to replicate** (C/q18, q27, q80, q84), all via
  `answer_source=query`. That bucket's nominal 0.192 is below the 0.25 chance floor.
- **Salvage is a letter lottery.** It fired 3 times in 32 episodes and emitted letter `A` all
  three times, correct twice because gold was A. Both wins (q80, q84) had `model_confident=false`;
  q84's own EQA had written `Answer: Unknown` with the clock never once in view (`gt_in_view=0/7`).
  Its prompt asserts *"The target object WAS observed during exploration"* to a model that just
  said it saw nothing. **C without salvage is 14/32, and 5/11 on the shared subset.**
- **The `query` channel's apparent 0.500 precision** is indistinguishable from `vlm_suggested`
  (p=0.79) and is concentrated in exactly the late, budget-hit, non-reproducing wins.
- **Letter bias:** the agent over-predicts `B` by 50% relative to base rate (predicted B=24 vs
  gold B=16).

---

## 4. Failure taxonomy

Every non-correct episode on the 11-id subset, classified into exactly one bucket.

| Bucket | P1 | P2 | A | B | C(11) | C(32) |
|---|---|---|---|---|---|---|
| CORRECT | 3 | 3 | 3 | 4 | 7 | 16 |
| (a) ABSTAIN_BUDGET | 2 | 4 | **6** | 2 | 0 | 0 |
| (b) OVERRIDE_DESTROYED | 0 | 0 | 1 | 1 | 2 | 2 |
| (c) ABSENCE_AS_ANSWER | 1 | 2 | 1 | 2 | 1 | 4 |
| (d) GENUINE_MISS | 1 | 0 | 0 | 1 | 0 | 3 |
| (e) WRONG_REASONING | 4 | 2 | 0 | 1 | 1 | 6 |
| (f) TRUNCATION | 0 | 0 | 0 | 0 | 0 | 0 |
| (g) OTHER | 0 | 0 | 0 | 0 | 0 | 1 |

**(a) Abstain by budget — 14 episodes, all guaranteed zeros.** Every one satisfies
`eqa_iterations==0` and `raw_eqa_output==""`: the four-image EQA was **never invoked**, despite
26-78 observations and 35-86 graph nodes in memory. The abstain count tracks the confirm gate
exactly — 2 (gate absent), 4 and 6 (gate ON), 2 (gate OFF), 0 (machinery absent) — and moves in
lockstep with `budget_hit` and mean planning steps.

**(b) Override destroyed a correct answer — 4 episodes.** The four-image EQA emitted the gold
letter and a different letter was scored. A/q28:

```
Answer:
D) Two                      <- gold
Confidence: TRUE
[agentic_submit]
source:vlm_suggested
answer:
C                           <- scored
```

The `C` came from a single-view assess of a *recliner* reporting `present: false`; option C is
literally "None". `query_answer` had returned `"The fan, lamp is at approximately (1.77, -1.71,
0.63) m."` — a coordinate dump about the wrong object — which is what routes control to the
single-view letter. B/q47 shows a two-stage version: `Answer: A) Above the sink` (gold) →
`[memory-location] answer: B` → `[agentic_submit] answer: D`.

**(c) Absence became an answer — 7 episodes.** The scored letter came from an assess with
`present=false`, on questions whose choice list contains an explicit absence option, so a
non-detection maps directly onto a wrong letter. q28 scores C="None" vs gold D="Two" in all five
runs; q39 scores B="No, there is none" vs gold C="Yes, next to a bed" in all five. On q39 the
robot was **0.39 m** from the fan, `gt_in_view=true`, OWL fired at 0.194, and the graph knew
`fan at (9.19, 1.80, 0.85)`.

**(f) Truncation is common but never operative.** The EQA body lacks an `Answer:` field in 2-4 of
11 episodes per v3 run and 12/32 in C, but the agentic submit layer supplies a letter regardless.
A/q6 is correct despite being truncated. This corrects an earlier hypothesis.

**(g) One scoring-format loss.** C/q43: the EQA saw the clock and reasoned to gold B (2-4pm) but
wrote `Answer: The time is 2:30 PM`, so no letter parsed and it scored as an abstain.

### Deterministic repeat errors

**10 of 32 bal-32 questions (31%) were never solved by any of 6 agentic runs**, several failing
identically every time: q28 always predicts C (gold D), q39 always predicts B (gold C). The
"what time is it now?" family is 1/7 overall and that one win does not replicate.

---

## 5. The deepest finding: grounding is not driving the wins

This reframes the whole problem and was not visible before cross-referencing ground-truth
visibility.

**Seeing the target does not predict correctness.** Across 43 GT-annotated episodes:

| | n | correct | acc |
|---|---|---|---|
| GT target ever in view | 31 | 11 | 0.355 |
| GT target never in view | 12 | 4 | 0.333 |

`vlm_assess present` vs GT-in-view: precision **0.500**, recall 0.645 — barely better than a coin
flip at tracking real visibility. So `present==true` predicts wins as a **VLM-confidence proxy,
not as a grounding signal**.

**The SigLIP/detector verify gate is inert.** Across 333 `verify_siglip` events: `ABSENT` 253,
`CANDIDATE` 51, `SKIPPED_SAME_VIEW` 21, **`PRESENT` 7 (2.1%)**. Only 5 episodes ever got a
PRESENT, and their accuracy (0.400) is indistinguishable from the rest (0.380). Meanwhile
`fused_verified` was True on **23/23** confident submits, 20 of them with `decision: ABSENT`,
because `confirm_answerable()` hardcodes `answerability_probability = 0.9, verified = True`
([agentic_policy.py:282](../../src/emet/memory/graph_eqa/agentic_policy.py)). In the flagship
robust win C/q6 the verifier returned `ABSENT` at `sim=0.088` and the episode passed anyway.

**OWL is a good finder with a miscalibrated channel** — this supports the standing intuition that
OWL should be trusted for boxes, not labels:

| signal | precision | recall |
|---|---|---|
| `owlv2 ∈ positive_channels` (as used) | **0.31** | 0.48 |
| raw `detector_score > 0.25` | **1.00** | 0.28 |
| `vlm_assess.present` | 0.86 | 0.41 |

69% of OWL firings admitted into `positive_channels` are on frames where the target is not present
at all. The raw score is monotone in real visibility and is clean at threshold 0.25. Separately,
`EvidencePolicy.assess()` fuses SigLIP, dense, voxel, and OWL and then **zeroes the result**
([agentic_policy.py:209-211](../../src/emet/memory/graph_eqa/agentic_policy.py)) — the multi-channel
fusion runs on every verify and contributes nothing to any decision.

**Exploration volume carries no signal.** Wins explored *less* map than losses (25.8 vs 27.6 m²);
at matched effort (rounds ≥6) wins and losses explored 32.0 vs 34.8 m². Nothing in the coverage
metrics predicts correctness.

The implication for planning: fixing termination and arbitration recovers the abstain and override
buckets — roughly 18 of 44 v3 episodes — but the **ceiling** is set by grounding, and grounding is
currently not contributing.

---

## 6. Plan of action

Ordered by confidence and expected value. Full task breakdown lives in the active plan; this is
the evidence-linked rationale.

### Tier 0 — Stop discarding answers (highest confidence)

Pure bug fixes; no new capability. Addresses buckets (a), (b), (c) = **25 of 44** v3 episodes.

1. **Never abstain.** Replace `_abstain_unverified`
   ([agentic_eqa.py:2466](../../src/emet/memory/graph_eqa/agentic_eqa.py)) with a forced-answer
   ladder: run the four-image EQA (currently never reached on this path), then the pending
   confirm-gate letter, then a forced-letter re-ask, then a uniform prior. Carry
   `answer_provenance` and calibrated confidence so the embodied agent can still express
   uncertainty without scoring a guaranteed zero.
2. **Fix the unreachable submit branch.** `max_rounds == max_nav_steps == 8` means `budget_left`
   is still true entering the last round, so the fallback explores, consumes the last waypoint,
   and the `for/else` abstains before `return "submit_answer"` can ever fire.
3. **Fix answer precedence.** Treat a coordinate-dump `query_answer` as *no answer* rather than
   as a trigger to fall back to the single-view letter; promote the EQA's own parsed `Answer:`
   letter above `vlm_suggested`.
4. **Absence must never become an MCQ answer.** Bar `present=false` assessments from supplying the
   scored letter, and stop counting two `present=false` frames as `two_view_agree` corroboration.
5. **Stop the runner blanking valid letters** ([runner.py:430-447](../../packages/emet_habitat/emet_habitat/runner.py)) —
   a second, trace-invisible abstain source.
6. **Parse prose answers** (C/q43) rather than scoring them blank.

### Tier 1 — Measurement integrity (do before any tuning)

Nothing here changes accuracy; everything here changes whether we can *tell*.

7. **Never run a sweep from a dirty tree again.** Record `git diff --stat` or a content hash in
   the episode record, not just `git_commit` — all four v3 runs claim the same commit while
   running different code, and one changed mid-run.
8. **The salvage counterfactual is not being computed.** `salvage_pred == ""` on all 22 rows
   because `_finalize_unknown_location_letter` early-returns on empty `relevant_images` and
   `GraphEQAMemory` defines neither `last_eqa_images` nor `last_relevant_images`. The dual-salvage
   metric currently measures nothing.
9. **`planning_steps` is `obs_count`** — mapping frames, not decisions. Report decision rounds
   separately before making any efficiency claim.
10. **`choices_are_location_mcq` misroutes by fallthrough**, sending q84's time-of-day options
    into location salvage.
11. **Report at pooled scale, not per-run.** Six draws span 0.281-0.500 on identical questions.

### Tier 2 — Reclaim the decision budget

Roughly 4 of 8 rounds are forced investigate/explore ping-pong and 1-2 more yield no assess.

12. Soften the `prefer_explore` hard-redirect (fires on any `present=false`, the common case).
13. Avoid rounds that produce nothing (`NO_NEW_OBS` → `SKIPPED_SAME_VIEW` → no assess).
14. Suppress `obs_id` revisits — 28% of navigations re-approach an already-investigated node.
15. Plumb a budget override; none exists today, so budget ablation is impossible.

### Tier 3 — Fix grounding (the actual ceiling)

Worth little until Tier 0 stops discarding answers, but this is what bounds the score.

16. **Use OWL as a finder at `detector_score >= 0.25`** and drop its label from semantic channels.
17. **Resolve the computed-then-zeroed fusion** — either use it or stop paying for it.
18. **Stop biasing exploration with the full MCQ text** including all three distractor options
    ([agentic_eqa.py:586](../../src/emet/memory/graph_eqa/agentic_eqa.py)), which contradicts the
    phrase-level rule `_siglip_phrase` enforces everywhere else.
19. **Wire rooms into navigation.** `question_target_rooms` is explicitly marked *"Diagnostic
    only"* today.
20. **Enable self-consistency.** `mcq_debias` / `vote_mcq_letter` already exists, defaults off, and
    is gated on a non-empty prediction so it can never rescue an abstain. Given the 50% B-bias,
    choice rotation is directly indicated.
21. **Attack the 31% never-solved set** as its own problem — deterministic repeat errors (q28, q39)
    and the time-of-day family are not going to be fixed by better search.

### Validation protocol

Tuning on 11 questions is noise-fitting. Deterministic replay tests for all Tier 0 logic (fixtures
from the traces already on disk for q28, q39, q47, q48, q80), then **one** bal-32 confirmation run,
compared against the pooled 0.379 baseline rather than against any single prior run.

---

## 7. Introspection gaps this audit exposed

Every finding above had to be reconstructed by hand from raw JSONL. `emet jobs report` is detailed
about *where the robot went* and silent about *how the answer was chosen*.

- **Answer provenance is invisible.** The table shows `28 agentic FAIL C/D` and nothing about the
  EQA having emitted D. One provenance line plus a red flag on disagreement would have surfaced the
  top fault immediately.
- **Abstain reads as a near-miss** (`—/A`) rather than a distinct, guaranteed-zero outcome.
- **No perception-calibration command**, though the traces already carry `gt_in_view`,
  `gt_visible_fraction`, `detector_score`, and `positive_channels` — everything needed for §5.
- **No exploration-efficiency view**: the investigate/explore alternation and the revisit rate are
  not reported.
- **No cross-run diff**, which is the only honest way to read an ablation at this noise level.
- **No replication view** — "does this win reproduce" is the single most decision-relevant question
  and requires manual cross-run joins today.
- **Visualization is motion-only** (`frontier_picks/`, `maps/`, `topdown*.png`). A per-episode
  decision card — the images actually passed to the EQA, each view's verdict, the provenance chain,
  scored vs gold — would make q28-class failures obvious at a glance.

---

## See also

- [agentic_qwen_context.md](agentic_qwen_context.md) — agentic loop context and Qwen prompt design
- [agentic_scale.md](agentic_scale.md) — SigLIP role and scale ladder
- [habitat_eqa_results.md](habitat_eqa_results.md) — scored HM-EQA result tables
- [../known_issues.md](../known_issues.md) — Mode A / Mode B segfault taxonomy
